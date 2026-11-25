"""The scientific Python RunPod worker, run for real against small bundles.

The vendored Theorem runner executes each bundle; this test interpreter's venv
stands in for the image's baked environment, and the presigned transfers are
local file copies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import socket
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parents[1] / "runpod" / "theorem_scientific_python_worker"
SPEC = importlib.util.spec_from_file_location("theorem_scientific_python_worker", WORKER_DIR / "worker.py")
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

LOCK_DIGEST = "sha256:" + hashlib.sha256((WORKER_DIR / "environment" / "uv.lock").read_bytes()).hexdigest()
PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="the worker builds job venvs with uv")


@pytest.fixture(autouse=True)
def baked_environment(monkeypatch, tmp_path):
    """This test venv plays the image's baked environment."""
    monkeypatch.setattr(worker, "THEOREM_ENV", Path(sys.prefix))
    monkeypatch.setattr(worker, "LOCK_PATH", WORKER_DIR / "environment" / "uv.lock")
    monkeypatch.setattr(worker, "RUNNER_PATH", WORKER_DIR / "theorem_compute_runner.py")
    monkeypatch.setattr(worker, "CONSTRAINTS_PATH", tmp_path / "no-constraints.txt")
    monkeypatch.setattr(worker, "BAKED_EXTRAS", frozenset({"accelerated", "cuda"}))
    monkeypatch.setenv("UV_OFFLINE", "1")
    for cached in (
        worker.baked_lock_digest,
        worker.baked_runner_digest,
        worker.baked_python_version,
        worker.network_isolation,
        worker.gpu_identity,
    ):
        cached.cache_clear()
    yield


def _job_dir(tmp_path: Path, source: str, *, inputs: dict | None = None, files: dict[str, bytes] | None = None) -> Path:
    """A job directory shaped the way Theorem writes one."""
    job = tmp_path / "theorem-job"
    (job / "work").mkdir(parents=True)
    (job / "inputs").mkdir()
    shutil.copy(WORKER_DIR / "theorem_compute_runner.py", job / "runner.py")
    (job / "work" / "main.py").write_text(source)
    for relative, payload in (files or {}).items():
        (job / relative).write_bytes(payload)
    (job / "request.json").write_text(
        json.dumps({"source_name": "main.py", "entry": "", "params": {}, "seed": 7, "inputs": inputs or {}})
    )
    return job


def _bundle(job: Path, destination: Path) -> Path:
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(job.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(job)), recursive=False)
    return destination


def _request(bundle: Path, **params) -> dict:
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    request_params = {
        "python": PYTHON,
        "lock_digest": LOCK_DIGEST,
        "extras": ["accelerated"],
        "requirements": [],
        "resources": {"gpu": "cuda", "cpu": None, "memory_mb": None, "timeout_seconds": 60},
        "budget": {"max_usd": 1.0},
        "entry": "",
        "seed": 7,
    }
    request_params.update(params)
    return {
        "contract": "theorem.scientific_python.v1",
        "operation": "scientific_python.run",
        "operation_id": "cj_0123abcd",
        "input": {
            "payload_digest": digest,
            "artifact_key": "tenants/t/content/" + digest.removeprefix("sha256:"),
            "read_url": "https://storage.example/bundle",
        },
        "output": {
            "artifact_key": "tenants/t/outputs/out.tar.gz",
            "content_type": "application/gzip",
            "write_url": "https://storage.example/result",
        },
        "params": request_params,
        "price_per_second_usd": "0.0005",
        "max_bytes": 64 * 1024 * 1024,
    }


@pytest.fixture
def transfers(monkeypatch):
    """Presigned GET/PUT as local copies; records what was uploaded."""
    state: dict = {"bundle": None, "uploaded": None}

    def download(url, destination, *, max_bytes):
        assert url == "https://storage.example/bundle"
        shutil.copy(state["bundle"], destination)

    def upload(url, source, *, max_bytes):
        assert url == "https://storage.example/result"
        state["uploaded"] = source.read_bytes()

    monkeypatch.setattr(worker, "_download", download)
    monkeypatch.setattr(worker, "_upload", upload)
    return state


def _run(transfers, bundle: Path, **params) -> dict:
    transfers["bundle"] = bundle
    return worker.handler({"id": "rp-job-1", "input": _request(bundle, **params)})


def _members(payload: bytes) -> dict[str, bytes | None]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        return {
            member.name: (archive.extractfile(member).read() if member.isfile() else None)
            for member in archive.getmembers()
        }


def test_worker_runs_a_bundle_through_the_theorem_runner(tmp_path, transfers):
    job = _job_dir(
        tmp_path,
        "import theorem\n"
        "print('hello from the job')\n"
        "data = theorem.input('data').read_text()\n"
        "theorem.save('table', {'rows': [1, 2, 3]})\n"
        "result = {'sum': 1 + 1, 'data': data}\n",
        inputs={"data": {"path": "inputs/data.txt"}},
        files={"inputs/data.txt": b"measured"},
    )
    result = _run(transfers, _bundle(job, tmp_path / "bundle.tar.gz"))

    assert "error" not in result, result
    uploaded = transfers["uploaded"]
    assert result["output"] == {
        "artifact_key": "tenants/t/outputs/out.tar.gz",
        "payload_digest": "sha256:" + hashlib.sha256(uploaded).hexdigest(),
        "bytes": len(uploaded),
    }
    members = _members(uploaded)
    assert {"result.json", "outputs", "outputs/table.json", "stdout.log", "stderr.log", "worker.json"} <= set(members)
    outcome = json.loads(members["result.json"])
    assert outcome["ok"] is True
    assert outcome["result"] == {"sum": 2, "data": "measured"}
    assert b"hello from the job" in members["stdout.log"]
    recorded = json.loads(members["worker.json"])
    assert {
        "gpu_name",
        "cuda_version",
        "driver",
        "python_version",
        "executor",
        "runpod_job_id",
        "execution_ms",
        "cost_usd",
    } <= set(recorded)
    assert recorded["executor"] == "runpod-serverless"
    assert recorded["runpod_job_id"] == "rp-job-1"
    assert recorded["python_version"].startswith(PYTHON)
    assert recorded["lock_digest"] == LOCK_DIGEST
    assert recorded["cost_usd"] == pytest.approx(recorded["execution_ms"] / 1000 * 0.0005, abs=1e-6)
    assert result["worker"] == recorded
    assert "hello from the job" in result["logs"]


def test_a_computation_that_raises_is_returned_not_failed(tmp_path, transfers):
    job = _job_dir(tmp_path, "raise ValueError('bad parameter')\n")
    result = _run(transfers, _bundle(job, tmp_path / "bundle.tar.gz"))

    assert "error" not in result, result
    outcome = json.loads(_members(transfers["uploaded"])["result.json"])
    assert outcome["ok"] is False
    assert outcome["error"]["type"] == "ValueError"
    assert outcome["error"]["line"] == 1


def _hostile_bundle(path: Path, kind: str) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in (("request.json", b"{}"), ("runner.py", b""), ("work/main.py", b"")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if kind == "symlink":
            info = tarfile.TarInfo("work/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../../etc/passwd"
            archive.addfile(info)
        else:
            name = {"parent": "../escaped.py", "nested": "work/../../escaped.py", "absolute": "/tmp/escaped.py"}[kind]
            info = tarfile.TarInfo(name)
            info.size = 5
            archive.addfile(info, io.BytesIO(b"owned"))
    return path


@pytest.mark.parametrize("kind", ["parent", "nested", "absolute", "symlink"])
def test_worker_refuses_bundles_that_escape_the_job_directory(tmp_path, transfers, kind):
    bundle = _hostile_bundle(tmp_path / "hostile.tar.gz", kind)
    result = _run(transfers, bundle)

    assert result["code"] == "invalid_bundle"
    reason = "is not a file or directory" if kind == "symlink" else "escapes the job directory"
    assert result["error"].startswith("invalid_bundle: ") and reason in result["error"]
    assert "output" not in result
    assert transfers["uploaded"] is None
    assert not (tmp_path / "escaped.py").exists()


def test_worker_refuses_a_lock_it_was_not_built_from(tmp_path, transfers):
    job = _job_dir(tmp_path, "result = 1\n")
    result = _run(transfers, _bundle(job, tmp_path / "bundle.tar.gz"), lock_digest="sha256:" + "0" * 64)

    assert result["code"] == "environment_mismatch"
    assert result["error"].startswith("environment_mismatch: params.lock_digest sha256:000")
    assert LOCK_DIGEST in result["error"]
    assert transfers["uploaded"] is None


def test_worker_refuses_extras_and_python_it_does_not_carry(tmp_path, transfers):
    job = _job_dir(tmp_path, "result = 1\n")
    bundle = _bundle(job, tmp_path / "bundle.tar.gz")

    assert _run(transfers, bundle, extras=["rocm"])["code"] == "environment_mismatch"
    assert _run(transfers, bundle, python="3.99")["code"] == "environment_mismatch"


def test_worker_refuses_a_runner_other_than_its_own(tmp_path, transfers):
    job = _job_dir(tmp_path, "result = 1\n")
    (job / "runner.py").write_text("print('not the theorem runner')\n")
    result = _run(transfers, _bundle(job, tmp_path / "bundle.tar.gz"))

    assert result["code"] == "runner_mismatch"


def test_worker_refuses_input_paths_outside_the_bundle(tmp_path, transfers):
    job = _job_dir(tmp_path, "result = 1\n", inputs={"data": {"path": "/Users/someone/data.csv"}})
    result = _run(transfers, _bundle(job, tmp_path / "bundle.tar.gz"))

    assert result["code"] == "invalid_bundle"
    assert "relative to the job directory" in result["error"]


def test_worker_refuses_a_bundle_whose_digest_differs(tmp_path, transfers):
    job = _job_dir(tmp_path, "result = 1\n")
    bundle = _bundle(job, tmp_path / "bundle.tar.gz")
    request = _request(bundle)
    request["input"]["payload_digest"] = "sha256:" + "1" * 64
    transfers["bundle"] = bundle

    result = worker.handler({"id": "rp-job-1", "input": request})

    assert result["code"] == "invalid_bundle"
    assert "digest" in result["error"]


def test_worker_rejects_other_contracts(tmp_path, transfers):
    job = _job_dir(tmp_path, "result = 1\n")
    request = _request(_bundle(job, tmp_path / "bundle.tar.gz"))
    request["contract"] = "theorem.offload.v1"

    result = worker.handler({"id": "rp-job-1", "input": request})

    assert result["code"] == "invalid_input"


def test_worker_stops_a_computation_at_its_timeout(tmp_path, transfers):
    job = _job_dir(tmp_path, "import time\nprint('started', flush=True)\ntime.sleep(60)\n")
    started = time.monotonic()
    result = _run(
        transfers,
        _bundle(job, tmp_path / "bundle.tar.gz"),
        resources={"gpu": None, "timeout_seconds": 3},
    )

    assert result["code"] == "timeout"
    assert time.monotonic() - started < 30
    assert "started" in result["logs"]
    assert transfers["uploaded"] is None


@pytest.fixture
def listener():
    server = socket.create_server(("127.0.0.1", 0))
    accepted = threading.Event()

    def accept():
        try:
            connection, _ = server.accept()
            connection.close()
            accepted.set()
        except OSError:
            pass

    threading.Thread(target=accept, daemon=True).start()
    yield server.getsockname()[1], accepted
    server.close()


NETWORK_PROBE = (
    "import socket\n"
    "try:\n"
    "    socket.create_connection(('127.0.0.1', {port}), timeout=2).close()\n"
    "    result = 'connected'\n"
    "except OSError as error:\n"
    "    result = 'refused: ' + str(error)\n"
)


def test_worker_keeps_the_computation_off_the_network_unless_allowed(tmp_path, transfers, listener):
    port, accepted = listener
    job = _job_dir(tmp_path, NETWORK_PROBE.format(port=port))
    result = _run(transfers, _bundle(job, tmp_path / "bundle.tar.gz"))

    assert "error" not in result, result
    outcome = json.loads(_members(transfers["uploaded"])["result.json"])
    assert outcome["result"].startswith("refused: ")
    if worker.network_isolation() == "python-guard":
        assert "network access is disabled" in outcome["result"]
    assert not accepted.is_set()
    assert result["worker"]["network"] == {"allowed": False, "isolation": worker.network_isolation()}


def test_worker_lets_the_computation_reach_the_network_when_allowed(tmp_path, transfers, listener):
    port, accepted = listener
    job = _job_dir(tmp_path, NETWORK_PROBE.format(port=port))
    result = _run(transfers, _bundle(job, tmp_path / "bundle.tar.gz"), network=True)

    assert "error" not in result, result
    assert json.loads(_members(transfers["uploaded"])["result.json"])["result"] == "connected"
    assert accepted.wait(5)


def _probe_wheel(directory: Path) -> Path:
    """A minimal pure-Python wheel, so the overlay install needs no index."""
    wheel = directory / "theorem_overlay_probe-1.0-py3-none-any.whl"
    dist_info = "theorem_overlay_probe-1.0.dist-info"
    files = {
        "theorem_overlay_probe.py": b"VALUE = 42\n",
        f"{dist_info}/METADATA": b"Metadata-Version: 2.1\nName: theorem-overlay-probe\nVersion: 1.0\n",
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr(f"{dist_info}/RECORD", "".join(f"{name},,\n" for name in files) + f"{dist_info}/RECORD,,\n")
    return wheel


def test_worker_installs_requirements_into_a_job_overlay(tmp_path, transfers):
    wheel = _probe_wheel(tmp_path)
    job = _job_dir(tmp_path, "import theorem_overlay_probe, pytest\nresult = theorem_overlay_probe.VALUE\n")
    result = _run(transfers, _bundle(job, tmp_path / "bundle.tar.gz"), requirements=[str(wheel)])

    assert "error" not in result, result
    outcome = json.loads(_members(transfers["uploaded"])["result.json"])
    # The overlay's own package and the baked environment's packages are both importable.
    assert outcome["result"] == 42
    with pytest.raises(ImportError):
        __import__("theorem_overlay_probe")
