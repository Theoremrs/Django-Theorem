"""Runs one Theorem computation inside its job directory.

Theorem starts this file with the job's own interpreter:

    <env>/bin/python -I -X faulthandler theorem_compute_runner.py <job dir>

`<job dir>/request.json` holds what the computation needs. The runner loads the
source, gives it a `theorem` module for inputs, outputs and progress, calls the
entry point with the parameters, and writes `<job dir>/result.json`. Everything
large stays a file under `<job dir>/outputs/`, and the result names those files
instead of carrying them.

Source shapes:

- `entry` names a function: it is called as `entry(**params)` and its return
  value is the result.
- `entry` is empty: the source runs as a script, and its result is whatever it
  passed to `theorem.result(...)`, or else a top-level `result` variable.

The runner holds no Theorem state and reaches nothing beyond the job directory
and the inputs it was given; the process boundary around it is Theorem's.
"""

from __future__ import annotations

import datetime as _dt
import decimal
import fractions
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
import types
from pathlib import Path

JOB_DIR = Path(sys.argv[1]).resolve()
REQUEST = json.loads((JOB_DIR / "request.json").read_text())
OUTPUTS = JOB_DIR / "outputs"
OUTPUTS.mkdir(exist_ok=True)
PROGRESS = JOB_DIR / "progress.jsonl"
_saved: dict[str, dict] = {}
_explicit_result: list = []

# A value written straight into result.json up to this many bytes; anything
# bigger is saved as an output file and referenced.
INLINE_LIMIT = int(REQUEST.get("inline_limit_bytes") or 64 * 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _record(name: str, path: Path, kind: str, media_type: str, **extra) -> dict:
    entry = {
        "name": name,
        "kind": kind,
        "path": str(path.relative_to(JOB_DIR)),
        "media_type": media_type,
        "bytes": path.stat().st_size,
        "digest": _sha256(path),
        **extra,
    }
    _saved[name] = entry
    return {"$output": name}


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name)).strip("._")
    if not cleaned:
        raise ValueError(f"output name {name!r} has no usable characters")
    return cleaned


def save(name: str, value, format: str | None = None):
    """Keep `value` as an output file and return a reference to it.

    Figures become PNG (or `format="svg"`), data frames Parquet, arrays `.npy`,
    bytes and str files of their own, paths are copied, and anything else is
    JSON when it can be and a pickle when it cannot.
    """
    stem = _safe_name(name)
    module = type(value).__module__ or ""
    cls = type(value).__name__

    if module.startswith("matplotlib") and hasattr(value, "savefig"):
        fmt = format or "png"
        path = OUTPUTS / f"{stem}.{fmt}"
        value.savefig(path, format=fmt, bbox_inches="tight", dpi=144)
        return _record(name, path, "figure", f"image/{'svg+xml' if fmt == 'svg' else fmt}")
    if module.startswith("matplotlib") and hasattr(value, "get_figure"):
        return save(name, value.get_figure(), format)
    if module.startswith("plotly") and hasattr(value, "to_json"):
        path = OUTPUTS / f"{stem}.plotly.json"
        path.write_text(value.to_json())
        return _record(name, path, "figure", "application/vnd.plotly.v1+json")
    if module.startswith("altair") and hasattr(value, "to_json"):
        path = OUTPUTS / f"{stem}.vl.json"
        path.write_text(value.to_json())
        return _record(name, path, "figure", "application/vnd.vegalite.v5+json")
    if module.startswith("PIL") and hasattr(value, "save"):
        path = OUTPUTS / f"{stem}.{format or 'png'}"
        value.save(path)
        return _record(name, path, "image", f"image/{format or 'png'}")
    if module.startswith(("pandas", "polars", "pyarrow")) and cls in {"DataFrame", "Table", "Series"}:
        path = OUTPUTS / f"{stem}.parquet"
        if module.startswith("pandas"):
            frame = value.to_frame() if cls == "Series" else value
            frame.to_parquet(path)
            rows, columns = frame.shape[0], [str(c) for c in frame.columns]
        elif module.startswith("polars"):
            frame = value.to_frame() if cls == "Series" else value
            frame.write_parquet(path)
            rows, columns = frame.height, list(frame.columns)
        else:
            import pyarrow.parquet as pq

            pq.write_table(value, path)
            rows, columns = value.num_rows, list(value.column_names)
        return _record(name, path, "dataset", "application/vnd.apache.parquet", rows=rows, columns=columns)
    if module.startswith("numpy") and cls == "ndarray":
        import numpy

        path = OUTPUTS / f"{stem}.npy"
        numpy.save(path, value, allow_pickle=False)
        return _record(
            name, path, "array", "application/x-npy", shape=list(value.shape), dtype=str(value.dtype)
        )
    if isinstance(value, (bytes, bytearray)):
        path = OUTPUTS / (f"{stem}.{format}" if format else stem)
        path.write_bytes(bytes(value))
        return _record(name, path, "file", "application/octet-stream")
    if isinstance(value, os.PathLike) or (isinstance(value, str) and format == "path"):
        source = Path(value)
        path = OUTPUTS / _safe_name(source.name if name == str(value) else name + source.suffix)
        if source.resolve() != path.resolve():
            path.write_bytes(source.read_bytes())
        return _record(name, path, "file", _media_type(path))
    if isinstance(value, str) and format:
        path = OUTPUTS / f"{stem}.{format}"
        path.write_text(value)
        return _record(name, path, "file", _media_type(path))
    try:
        text = json.dumps(_plain(value), allow_nan=False)
    except (TypeError, ValueError):
        import pickle

        path = OUTPUTS / f"{stem}.pkl"
        with path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return _record(name, path, "model", "application/x-python-pickle", python_type=f"{module}.{cls}")
    path = OUTPUTS / f"{stem}.json"
    path.write_text(text)
    return _record(name, path, "value", "application/json")


def _media_type(path: Path) -> str:
    import mimetypes

    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def output_path(name: str) -> Path:
    """A path under the job's outputs for code that writes its own file.

    Whatever is there when the computation ends is kept as output `name`."""
    path = OUTPUTS / _safe_name(name)
    _pending_paths[name] = path
    return path


_pending_paths: dict[str, Path] = {}


def input(name: str):  # noqa: A001 - the module reads as `theorem.input(...)`
    """An input by name: its value when one was given, else the path of its file."""
    inputs = REQUEST.get("inputs") or {}
    if name not in inputs:
        raise KeyError(f"no input named {name!r}; inputs are {sorted(inputs)}")
    item = inputs[name]
    if "value" in item:
        return item["value"]
    return Path(item["path"])


def inputs() -> dict:
    return {name: input(name) for name in (REQUEST.get("inputs") or {})}


def progress(fraction: float | None = None, message: str = "", **metrics):
    """Report how far the computation has got; Theorem streams it to the caller."""
    record = {"at": time.time(), "message": str(message)}
    if fraction is not None:
        record["fraction"] = max(0.0, min(1.0, float(fraction)))
    if metrics:
        record["metrics"] = _plain(metrics)
    with PROGRESS.open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def result(value):
    """Set the computation's result explicitly (script-shaped sources)."""
    _explicit_result[:] = [value]
    return value


def _plain(value, depth: int = 0):
    """`value` as JSON-ready data, converting the scientific types agents return."""
    if depth > 64:
        raise ValueError("result nests deeper than 64 levels")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"$float": repr(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v, depth + 1) for v in value]
    if isinstance(value, (decimal.Decimal, fractions.Fraction)):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    module = type(value).__module__ or ""
    if module.startswith("numpy"):
        if hasattr(value, "tolist"):
            return _plain(value.tolist(), depth + 1)
        if hasattr(value, "item"):
            return _plain(value.item(), depth + 1)
    if module.startswith("sympy"):
        return {"$sympy": str(value), "latex": __import__("sympy").latex(value)}
    if module.startswith("pandas") and hasattr(value, "to_dict"):
        if type(value).__name__ == "DataFrame":
            return _plain(value.to_dict(orient="records"), depth + 1)
        return _plain(value.to_dict(), depth + 1)
    if module.startswith("polars") and hasattr(value, "to_dicts"):
        return _plain(value.to_dicts(), depth + 1)
    if module.startswith("torch") and hasattr(value, "tolist"):
        return _plain(value.detach().cpu().tolist(), depth + 1)
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return _plain(dataclasses.asdict(value), depth + 1)
    raise TypeError(f"{module}.{type(value).__name__} is not JSON; save it with theorem.save(name, value)")


def _shape_result(value):
    """Small results inline; a large or non-JSON one becomes an output file."""
    if isinstance(value, dict) and set(value) == {"$output"}:
        return value
    try:
        plain = _plain(value)
        text = json.dumps(plain, allow_nan=False)
    except (TypeError, ValueError):
        return save("result", value)
    if len(text.encode()) > INLINE_LIMIT:
        path = OUTPUTS / "result.json"
        path.write_text(text)
        _record("result", path, "value", "application/json")
        return {"$output": "result"}
    return plain


def _seed(seed):
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy

        numpy.random.seed(seed % (2**32))
    except ImportError:
        pass
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def _module() -> types.ModuleType:
    module = types.ModuleType("theorem")
    module.__doc__ = "Inputs, outputs and progress for a Theorem computation."
    for name in ("input", "inputs", "save", "output_path", "progress", "result"):
        setattr(module, name, globals()[name])
    module.params = dict(REQUEST.get("params") or {})
    module.seed = REQUEST.get("seed")
    module.job_dir = JOB_DIR
    module.outputs_dir = OUTPUTS
    return module


_BASELINE_MODULES = {name.split(".")[0] for name in sys.modules}


def main() -> int:
    started = time.time()
    sys.modules["theorem"] = _module()
    os.chdir(JOB_DIR / "work")
    sys.path.insert(0, str(JOB_DIR / "work"))
    _seed(REQUEST.get("seed"))
    source_path = JOB_DIR / "work" / (REQUEST.get("source_name") or "main.py")
    source = source_path.read_text()
    entry = (REQUEST.get("entry") or "").strip()
    params = REQUEST.get("params") or {}
    namespace = {"__name__": "__theorem__" if entry else "__main__", "__file__": str(source_path)}
    outcome: dict = {"ok": True}
    try:
        code = compile(source, str(source_path), "exec")
        exec(code, namespace)
        if entry:
            target = namespace.get(entry)
            if not callable(target):
                raise NameError(f"entry point {entry!r} is not a function defined by the source")
            value = target(**params)
        elif _explicit_result:
            value = _explicit_result[0]
        else:
            value = namespace.get("result")
        for name, path in _pending_paths.items():
            if path.exists() and name not in _saved:
                _record(name, path, "file", _media_type(path))
        outcome["result"] = _shape_result(value)
    except BaseException as error:  # the job reports every failure, SystemExit included
        if isinstance(error, SystemExit) and error.code in (0, None):
            outcome["result"] = _shape_result(_explicit_result[0] if _explicit_result else namespace.get("result"))
        else:
            frames = traceback.extract_tb(error.__traceback__)
            user = [f for f in frames if f.filename == str(source_path)]
            outcome = {
                "ok": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "line": user[-1].lineno if user else None,
                    "traceback": "".join(traceback.format_exception(error)),
                },
            }
    outcome["outputs"] = list(_saved.values())
    outcome["runtime_seconds"] = round(time.time() - started, 6)
    outcome["imported"] = sorted({name.split(".")[0] for name in sys.modules if not name.startswith("_")} - _BASELINE_MODULES)
    (JOB_DIR / "result.json").write_text(json.dumps(outcome, allow_nan=False))
    return 0 if outcome["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
