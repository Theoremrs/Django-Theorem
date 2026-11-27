#!/usr/bin/env python3
"""RunPod worker that runs one Theorem computation bundle, on Serverless or a Pod.

The control plane sends ``theorem.scientific_python.v1``: a presigned GET for a
gzip tarball of a Theorem job directory (``request.json``, ``runner.py``,
``work/``, ``inputs/``) and a presigned PUT for the result tarball. The worker
checks the bundle's digest, extracts it without letting any member escape the
job directory, confirms the job was built for the environment baked into this
image, and runs Theorem's runner exactly as Theorem does locally:

    <python> -I -X faulthandler runner.py <job_dir>

The result tarball holds ``result.json``, ``outputs/``, ``stdout.log``,
``stderr.log`` and ``worker.json``. Failures are returned as
``{"error": "<code>: <message>", "code": <code>, "logs": <tail>}``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from functools import cache
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

CONTRACT = "theorem.scientific_python.v1"
OPERATION = "scientific_python.run"
GZIP_CONTENT_TYPE = "application/gzip"
# Baked by the Dockerfile; overridable so tests can point at fixtures.
THEOREM_ENV = Path(os.environ.get("THEOREM_ENV", "/opt/theorem/env"))
LOCK_PATH = Path(os.environ.get("THEOREM_LOCK", "/opt/theorem/project/uv.lock"))
RUNNER_PATH = Path(os.environ.get("THEOREM_RUNNER", "/opt/theorem/theorem_compute_runner.py"))
CONSTRAINTS_PATH = Path(os.environ.get("THEOREM_CONSTRAINTS", "/opt/theorem/constraints.txt"))
BAKED_EXTRAS = frozenset(
    extra for extra in os.environ.get("THEOREM_EXTRAS", "accelerated,cuda").split(",") if extra
)
UV = os.environ.get("THEOREM_UV", "uv")
MAX_ARTIFACT_BYTES = int(os.environ.get("ARTIFACT_MAX_BYTES", 64 * 1024 * 1024))
MAX_EXTRACTED_BYTES = int(os.environ.get("THEOREM_MAX_EXTRACTED_BYTES", 2 * 1024**3))
LOG_TAIL_LINES = 200
# Only these reach the computation: the worker's own environment holds RunPod
# credentials, and the computation is the agent's code, not the worker's.
PASSED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "LD_LIBRARY_PATH",
    "CUDA_VERSION",
)
NETWORK_GUARD = '''"""Installed by the Theorem worker: this computation runs without network."""
import socket as _socket

_connect = _socket.socket.connect
_connect_ex = _socket.socket.connect_ex


def _refuse(*_args, **_kwargs):
    raise OSError("network access is disabled for this Theorem computation (params.network is false)")


def _local_only(original):
    def guarded(self, address, *args, **kwargs):
        if self.family == getattr(_socket, "AF_UNIX", None):
            return original(self, address, *args, **kwargs)
        return _refuse()
    return guarded


_socket.socket.connect = _local_only(_connect)
_socket.socket.connect_ex = _local_only(_connect_ex)
_socket.socket.sendto = _local_only(_socket.socket.sendto)
_socket.getaddrinfo = _refuse
_socket.gethostbyname = _refuse
_socket.create_connection = _refuse
'''


class WorkerFailure(Exception):
    """A failure with the machine-readable code the control plane records."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _is_sha256_digest(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def validate_request(value: Any) -> str | None:
    """Return a validation error, if any, for a ``theorem.scientific_python.v1`` request."""
    if not isinstance(value, Mapping):
        return "job input must be an object"
    if value.get("contract") != CONTRACT:
        return f"contract must equal {CONTRACT!r}"
    if value.get("operation") != OPERATION:
        return f"operation must equal {OPERATION!r}"
    descriptor = value.get("input")
    if not isinstance(descriptor, Mapping):
        return "input must be a bundle descriptor object"
    if not _is_sha256_digest(descriptor.get("payload_digest")):
        return "input.payload_digest must be a sha256 digest"
    if not isinstance(descriptor.get("read_url"), str) or not descriptor["read_url"].startswith("https://"):
        return "input.read_url must be an HTTPS signed URL"
    output = value.get("output")
    if not isinstance(output, Mapping):
        return "output must be an object"
    if not isinstance(output.get("artifact_key"), str) or not output["artifact_key"]:
        return "output.artifact_key must be a non-empty string"
    if not isinstance(output.get("write_url"), str) or not output["write_url"].startswith("https://"):
        return "output.write_url must be an HTTPS signed URL"
    params = value.get("params")
    if not isinstance(params, Mapping):
        return "params must be an object"
    if not _is_sha256_digest(params.get("lock_digest")):
        return "params.lock_digest must be a sha256 digest"
    resources = params.get("resources")
    timeout = resources.get("timeout_seconds") if isinstance(resources, Mapping) else None
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        return "params.resources.timeout_seconds must be a positive integer"
    requirements = params.get("requirements", [])
    if not isinstance(requirements, list) or not all(
        isinstance(item, str) and item.strip() for item in requirements
    ):
        return "params.requirements must be a list of requirement strings"
    if not isinstance(params.get("extras", []), list):
        return "params.extras must be a list"
    try:
        Decimal(str(value.get("price_per_second_usd", "0")))
    except InvalidOperation:
        return "price_per_second_usd must be a decimal string"
    max_bytes = value.get("max_bytes")
    if max_bytes is not None and (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0):
        return "max_bytes must be a positive integer"
    return None


def _download(url: str, destination: Path, *, max_bytes: int) -> None:
    """Stream a signed object to disk, refusing anything over the artifact limit."""
    written = 0
    with urlopen(url, timeout=120) as response, destination.open("wb") as handle:  # noqa: S310 - signed capability from Django.
        for block in iter(lambda: response.read(1 << 20), b""):
            written += len(block)
            if written > max_bytes:
                raise WorkerFailure("invalid_bundle", "input bundle exceeds the artifact limit")
            handle.write(block)


def _upload(url: str, source: Path, *, max_bytes: int) -> None:
    size = source.stat().st_size
    if size > max_bytes:
        raise WorkerFailure("output_too_large", f"result bundle is {size} bytes, over the {max_bytes}-byte limit")
    with source.open("rb") as handle:
        request = Request(
            url,
            data=handle,
            method="PUT",
            headers={"Content-Type": GZIP_CONTENT_TYPE, "Content-Length": str(size)},
        )
        with urlopen(request, timeout=300):  # noqa: S310 - signed capability from Django.
            pass


def extract_bundle(bundle: Path, job_dir: Path, *, max_extracted_bytes: int = MAX_EXTRACTED_BYTES) -> None:
    """Extract regular files and directories only, all of them inside ``job_dir``."""
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            total = 0
            for member in members:
                parts = Path(member.name).parts
                if member.name.startswith(("/", "\\")) or ".." in parts:
                    raise WorkerFailure("invalid_bundle", f"bundle member {member.name!r} escapes the job directory")
                if not (member.isfile() or member.isdir()):
                    raise WorkerFailure("invalid_bundle", f"bundle member {member.name!r} is not a file or directory")
                total += member.size
                if total > max_extracted_bytes:
                    raise WorkerFailure("invalid_bundle", "bundle expands past the extraction limit")
            # The data filter is the standard library's second line: it refuses
            # anything that would still land outside the destination.
            archive.extractall(job_dir, members=members, filter="data")
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise WorkerFailure("invalid_bundle", f"bundle is not a readable gzip tarball: {exc}") from exc
    for required in ("request.json", "runner.py", "work"):
        if not (job_dir / required).exists():
            raise WorkerFailure("invalid_bundle", f"bundle has no {required}")


def bind_input_paths(job_dir: Path) -> None:
    """Point file inputs at this job directory; bundles name them relative to it."""
    request_path = job_dir / "request.json"
    try:
        request = json.loads(request_path.read_text())
    except ValueError as exc:
        raise WorkerFailure("invalid_bundle", f"request.json is not JSON: {exc}") from exc
    root = job_dir.resolve()
    for name, item in (request.get("inputs") or {}).items():
        if not isinstance(item, dict) or "path" not in item:
            continue
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkerFailure(
                "invalid_bundle",
                f"input {name!r} path {item['path']!r} must be relative to the job directory",
            )
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            raise WorkerFailure("invalid_bundle", f"input {name!r} file {item['path']!r} is not in the bundle")
        item["path"] = str(resolved)
    request_path.write_text(json.dumps(request))


@cache
def baked_lock_digest() -> str:
    return _sha256_file(LOCK_PATH)


@cache
def baked_runner_digest() -> str:
    return _sha256_file(RUNNER_PATH)


@cache
def baked_python_version() -> str:
    completed = subprocess.run(
        [str(THEOREM_ENV / "bin" / "python"), "-I", "-c", "import platform; print(platform.python_version())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def check_environment(params: Mapping[str, Any], job_dir: Path) -> None:
    """The job must have been built for exactly what this image carries."""
    if params["lock_digest"] != baked_lock_digest():
        raise WorkerFailure(
            "environment_mismatch",
            f"params.lock_digest {params['lock_digest']} does not match this image's uv.lock {baked_lock_digest()}",
        )
    wanted_python = str(params.get("python") or "")
    have_python = ".".join(baked_python_version().split(".")[:2])
    if wanted_python and wanted_python != have_python:
        raise WorkerFailure(
            "environment_mismatch", f"params.python {wanted_python} does not match this image's Python {have_python}"
        )
    missing = sorted(set(params.get("extras") or []) - BAKED_EXTRAS)
    if missing:
        raise WorkerFailure(
            "environment_mismatch", f"extras {missing} are not synced in this image (it has {sorted(BAKED_EXTRAS)})"
        )
    if _sha256_file(job_dir / "runner.py") != baked_runner_digest():
        raise WorkerFailure(
            "runner_mismatch",
            f"the bundle's runner.py is not this image's runner {baked_runner_digest()}; refresh the worker image",
        )


@cache
def network_isolation() -> str:
    """``netns`` when this container may create a network namespace, else ``python-guard``."""
    unshare = shutil.which("unshare")
    if unshare and sys.platform.startswith("linux"):
        probe = subprocess.run([unshare, "--net", "--map-root-user", "true"], capture_output=True)
        if probe.returncode == 0:
            return "netns"
    return "python-guard"


def _purelib(python: Path) -> Path:
    completed = subprocess.run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(completed.stdout.strip())


def prepare_interpreter(
    overlay: Path,
    params: Mapping[str, Any],
    *,
    deadline: float,
    stdout: BinaryIO,
    stderr: BinaryIO,
) -> Path:
    """A per-job venv layered over the baked one, holding the job's own requirements."""
    base_python = THEOREM_ENV / "bin" / "python"
    subprocess.run(
        [UV, "venv", "--quiet", "--python", str(base_python), str(overlay)],
        check=True,
        stdout=stdout,
        stderr=stderr,
        timeout=max(deadline - time.monotonic(), 1),
    )
    python = overlay / "bin" / "python"
    site = _purelib(python)
    # Overlay packages come first; the baked environment follows on sys.path.
    (site / "_theorem_base.pth").write_text(f"{_purelib(base_python)}\n")
    if not params.get("network") and network_isolation() == "python-guard":
        (site / "sitecustomize.py").write_text(NETWORK_GUARD)
    requirements = [item.strip() for item in params.get("requirements") or []]
    if requirements:
        command = [UV, "pip", "install", "--python", str(python)]
        if CONSTRAINTS_PATH.exists():
            # Shared packages keep the baked versions the rest of the env was built against.
            command += ["--constraint", str(CONSTRAINTS_PATH)]
        try:
            subprocess.run(
                [*command, "--", *requirements],
                check=True,
                stdout=stdout,
                stderr=stderr,
                timeout=max(deadline - time.monotonic(), 1),
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerFailure("timeout", "installing requirements used the job's whole timeout") from exc
        except subprocess.CalledProcessError as exc:
            raise WorkerFailure(
                "requirements_failed", f"uv could not install {requirements} over the baked environment"
            ) from exc
    return python


def _child_environment(job_dir: Path, python: Path) -> dict[str, str]:
    home = job_dir / "tmp" / "home"
    home.mkdir(parents=True, exist_ok=True)
    environment = {name: os.environ[name] for name in PASSED_ENVIRONMENT if name in os.environ}
    environment.update(
        {
            "PATH": f"{python.parent}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(home),
            "TMPDIR": str(job_dir / "tmp"),
            "LANG": "C.UTF-8",
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


class _Stream(threading.Thread):
    """Copies one pipe to its log file and the worker's console, keeping a tail."""

    def __init__(self, pipe: BinaryIO, log: BinaryIO, console: Any, tail: deque[str], label: str) -> None:
        super().__init__(daemon=True)
        self.pipe, self.log, self.console, self.tail, self.label = pipe, log, console, tail, label

    def run(self) -> None:
        for line in iter(self.pipe.readline, b""):
            self.log.write(line)
            self.log.flush()
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            self.tail.append(f"[{self.label}] {text}")
            print(f"[{self.label}] {text}", file=self.console, flush=True)


def run_runner(
    python: Path,
    job_dir: Path,
    params: Mapping[str, Any],
    *,
    deadline: float,
    stdout: BinaryIO,
    stderr: BinaryIO,
    tail: deque[str],
) -> dict[str, Any]:
    """Run Theorem's runner until it exits or the job's deadline passes."""
    command = [str(python), "-I", "-X", "faulthandler", str(job_dir / "runner.py"), str(job_dir)]
    if not params.get("network") and network_isolation() == "netns":
        command = ["unshare", "--net", "--map-root-user", "--", *command]
    process = subprocess.Popen(
        command,
        cwd=job_dir,
        env=_child_environment(job_dir, python),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    streams = [
        _Stream(process.stdout, stdout, sys.stdout, tail, "stdout"),
        _Stream(process.stderr, stderr, sys.stderr, tail, "stderr"),
    ]
    for stream in streams:
        stream.start()
    timed_out = False
    try:
        process.wait(timeout=max(deadline - time.monotonic(), 0.1))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    for stream in streams:
        stream.join(timeout=10)
    return {"exit_code": process.returncode, "timed_out": timed_out}


@cache
def gpu_identity() -> dict[str, str | None]:
    identity: dict[str, str | None] = {
        "gpu_name": None,
        "driver": None,
        "cuda_version": os.environ.get("CUDA_VERSION"),
    }
    smi = shutil.which("nvidia-smi")
    if not smi:
        return identity
    try:
        query = subprocess.run(
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        summary = subprocess.run([smi], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return identity
    first = (query.stdout.strip().splitlines() or [""])[0]
    if "," in first:
        name, driver = (part.strip() for part in first.split(",", 1))
        identity.update(gpu_name=name, driver=driver)
    # The driver's CUDA version is what the bundled CUDA wheels can use.
    match = re.search(r"CUDA Version:\s*([\d.]+)", summary.stdout)
    if match:
        identity["cuda_version"] = match.group(1)
    return identity


def _pack(job_dir: Path, logs: Mapping[str, Path], worker_json: Path, destination: Path) -> None:
    def regular_only(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return info if info.isfile() or info.isdir() else None

    with tarfile.open(destination, "w:gz") as archive:
        archive.add(job_dir / "result.json", arcname="result.json")
        archive.add(job_dir / "outputs", arcname="outputs", filter=regular_only)
        for name, path in logs.items():
            archive.add(path, arcname=name)
        archive.add(worker_json, arcname="worker.json")


def _failure(code: str, message: str, tail: deque[str]) -> dict[str, Any]:
    return {"error": f"{code}: {message}", "code": code, "logs": "\n".join(tail)}


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
    request = job.get("input")
    error = validate_request(request)
    if error:
        return _failure("invalid_input", f"invalid {CONTRACT} input: {error}", tail)
    assert isinstance(request, Mapping)
    params = request["params"]
    max_bytes = int(request.get("max_bytes") or MAX_ARTIFACT_BYTES)
    deadline = started + int(params["resources"]["timeout_seconds"])
    with tempfile.TemporaryDirectory(prefix="theorem-scientific-") as scratch:
        scratch_dir = Path(scratch)
        job_dir = scratch_dir / "job"
        logs = {"stdout.log": scratch_dir / "stdout.log", "stderr.log": scratch_dir / "stderr.log"}
        try:
            bundle = scratch_dir / "bundle.tar.gz"
            _download(str(request["input"]["read_url"]), bundle, max_bytes=max_bytes)
            if _sha256_file(bundle) != request["input"]["payload_digest"]:
                raise WorkerFailure("invalid_bundle", "input bundle digest does not match input.payload_digest")
            extract_bundle(bundle, job_dir)
            bundle.unlink()
            check_environment(params, job_dir)
            bind_input_paths(job_dir)
            with logs["stdout.log"].open("wb") as stdout, logs["stderr.log"].open("wb") as stderr:
                python = prepare_interpreter(
                    scratch_dir / "overlay", params, deadline=deadline, stdout=stdout, stderr=stderr
                )
                run = run_runner(
                    python, job_dir, params, deadline=deadline, stdout=stdout, stderr=stderr, tail=tail
                )
            if run["timed_out"]:
                raise WorkerFailure(
                    "timeout", f"the computation ran past resources.timeout_seconds={params['resources']['timeout_seconds']}"
                )
            if not (job_dir / "result.json").is_file():
                raise WorkerFailure("runner_died", f"the runner exited {run['exit_code']} without writing result.json")
            execution_ms = int((time.monotonic() - started) * 1000)
            price = Decimal(str(request.get("price_per_second_usd") or "0"))
            worker = {
                **gpu_identity(),
                "python_version": baked_python_version(),
                "executor": job.get("executor", "runpod-serverless"),
                "runpod_job_id": job.get("id"),
                "execution_ms": execution_ms,
                "cost_usd": float((price * execution_ms / 1000).quantize(Decimal("0.000001"))),
                "lock_digest": baked_lock_digest(),
                "runner_digest": baked_runner_digest(),
                "exit_code": run["exit_code"],
                "network": {"allowed": bool(params.get("network")), "isolation": network_isolation()},
            }
            worker_json = scratch_dir / "worker.json"
            worker_json.write_text(json.dumps(worker, indent=2, sort_keys=True))
            packed = scratch_dir / "result.tar.gz"
            _pack(job_dir, logs, worker_json, packed)
            payload_digest = _sha256_file(packed)
            size = packed.stat().st_size
            _upload(str(request["output"]["write_url"]), packed, max_bytes=max_bytes)
        except WorkerFailure as failure:
            return _failure(failure.code, failure.message, tail)
        except subprocess.TimeoutExpired:
            return _failure("timeout", "preparing the environment used the job's whole timeout", tail)
        except subprocess.CalledProcessError as exc:
            return _failure("environment_mismatch", f"the baked environment did not start: {exc}", tail)
        except OSError as exc:
            return _failure("transfer_failed", f"artifact transfer failed: {exc}", tail)
    return {
        "output": {
            "artifact_key": request["output"]["artifact_key"],
            "payload_digest": payload_digest,
            "bytes": size,
        },
        "worker": worker,
        "logs": "\n".join(tail),
    }


def run_local(input_path: str) -> int:
    """Run the handler against a JSON request for contract checks."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = handler({"id": "local", "input": payload})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if "error" not in result else 1


POD_REQUEST_ENV = "THEOREM_SCIENTIFIC_REQUEST"


def run_pod() -> None:
    """One job on a RunPod Pod. The request comes in the environment with a
    presigned PUT for the handler's result; the control plane reads that result
    and terminates the Pod. The process then waits, so a Pod that restarts its
    container never runs the job twice."""
    request = json.loads(os.environ.pop(POD_REQUEST_ENV))
    result_url = request.pop("result_url")
    result = handler({"id": os.environ.get("RUNPOD_POD_ID", "pod"), "executor": "runpod-pod", "input": request})
    body = json.dumps(result, sort_keys=True).encode()
    put = Request(
        result_url,
        data=body,
        method="PUT",
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    with urlopen(put, timeout=120):  # noqa: S310 - signed capability from Django.
        pass
    print("theorem scientific worker: result reported; waiting for the Pod to end", flush=True)
    while True:
        signal.pause()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--local-input":
        raise SystemExit(run_local(sys.argv[2]))
    if len(sys.argv) == 2 and sys.argv[1] == "--pod":
        run_pod()
    import runpod

    print(
        f"theorem scientific worker: python {platform.python_version()}, "
        f"lock {baked_lock_digest()}, network isolation {network_isolation()}",
        flush=True,
    )
    runpod.serverless.start({"handler": handler})
