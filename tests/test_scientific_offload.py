"""``scientific_python.run``: content upload, invoke, budget, status, cancel, RunPod task.

The artifact store is the real ``ArtifactStore`` over an in-memory S3 stand-in,
so key allocation, content-type rules and digest checks run as in production.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import uuid
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode

import pytest
from botocore.exceptions import ClientError
from django.test import Client, override_settings
from django.utils import timezone

from apps.billing.credit import calendar_period
from apps.billing.models import Plan, Subscription, Usage
from apps.billing.usage import accumulate
from apps.keys.mint import mint_api_key
from apps.orchestration import scientific
from apps.orchestration.artifacts import ArtifactStore
from apps.orchestration.models import Job
from apps.orchestration.tasks import run_offload_python
from apps.tenancy.models import Tenant

WORKER_PATH = (
    Path(__file__).resolve().parents[1] / "runpod" / "theorem_scientific_python_worker" / "worker.py"
)
_SPEC = importlib.util.spec_from_file_location("scientific_worker_contract", WORKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)

BUNDLE = b"\x1f\x8b gzip bytes of a theorem job directory"
BUNDLE_DIGEST = "sha256:" + hashlib.sha256(BUNDLE).hexdigest()
LOCK_DIGEST = "sha256:" + "a" * 64
PRICE = "0.0004"
RUNPOD = {
    "OFFLOAD_EXECUTION_MODE": "runpod",
    "RUNPOD_API_KEY": "test-key",
    "RUNPOD_SCIENTIFIC_ENDPOINT_ID": "scientific-endpoint",
    "RUNPOD_SCIENTIFIC_IMAGE_DIGEST": "ghcr.io/theorem/scientific@sha256:immutable",
    "RUNPOD_SCIENTIFIC_PRICE_PER_SECOND_USD": PRICE,
    "RUNPOD_POLL_INTERVAL_SECONDS": 0.1,
}


class MemoryS3:
    """Just enough of the S3 client for ArtifactStore."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def generate_presigned_url(self, operation, *, Params, ExpiresIn, HttpMethod):
        query = urlencode({k: v for k, v in Params.items() if k != "Bucket"} | {"expires": ExpiresIn})
        return f"https://storage.example/{operation}?{query}"

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        payload = self.objects[Key]
        return {"ContentLength": len(payload), "Body": io.BytesIO(payload)}


@pytest.fixture
def s3(monkeypatch):
    memory = MemoryS3()

    def store():
        return ArtifactStore(
            endpoint_url="https://storage.example",
            access_key_id="test",
            secret_access_key="test",
            region="auto",
            bucket="theorem-artifacts",
            presign_seconds=900,
            max_bytes=1024 * 1024,
            client=memory,
        )

    monkeypatch.setattr("apps.orchestration.api.ArtifactStore.from_settings", store)
    monkeypatch.setattr("apps.orchestration.tasks.ArtifactStore.from_settings", store)
    return memory


@pytest.fixture
def client(db):
    tenant = Tenant.objects.create(slug=f"scientific-{uuid.uuid4()}", display_name="Scientific")
    minted = mint_api_key(tenant, scopes=["offload:*"])
    http = Client()
    http.defaults["HTTP_AUTHORIZATION"] = f"Bearer {minted.plaintext}"
    http.tenant = tenant
    return http


@pytest.fixture
def queued(monkeypatch):
    """Record dispatches instead of running them, for invoke-only tests."""
    dispatched: list[str] = []

    class Result:
        id = "celery-task-1"

    def delay(job_id):
        dispatched.append(job_id)
        return Result()

    monkeypatch.setattr("apps.orchestration.api.dispatch_offload.delay", delay)
    return dispatched


@pytest.fixture
def provenance(monkeypatch):
    posted: list[dict] = []
    monkeypatch.setattr(
        "bridges.rust_provenance.StubProvenanceClient.post_derivation",
        lambda self, payload: posted.append(payload) or {"ok": True},
    )
    return posted


def _content_key(tenant) -> str:
    return f"tenants/{tenant.id}/content/{BUNDLE_DIGEST.removeprefix('sha256:')}"


def _invoke_body(tenant, operation_id: str | None = None, **params) -> dict:
    body_params = {
        "python": "3.13",
        "lock_digest": LOCK_DIGEST,
        "extras": ["accelerated", "cuda"],
        "requirements": ["xarray>=2025.1"],
        "resources": {"gpu": "cuda", "cpu": 4, "memory_mb": 16384, "timeout_seconds": 600},
        "budget": {"max_usd": 1.0},
        "entry": "",
        "seed": 11,
    }
    body_params.update(params)
    return {
        "operation": "scientific_python.run",
        "operation_id": operation_id or f"cj_{uuid.uuid4().hex}",
        "input": {
            "payload_digest": BUNDLE_DIGEST,
            "artifact_key": _content_key(tenant),
            "schema_json": "",
            "rows": None,
        },
        "params": body_params,
    }


def _post(client, path, body=None):
    return client.post(path, data=json.dumps(body or {}), content_type="application/json")


# --- content-upload ---------------------------------------------------------


@pytest.mark.django_db
def test_content_upload_mints_a_content_addressed_gzip_put(client, s3):
    response = _post(
        client,
        "/internal/offload/content-upload",
        {"content_type": "application/gzip", "payload_digest": BUNDLE_DIGEST, "bytes": len(BUNDLE)},
    )

    assert response.status_code == 200, response.content
    data = response.json()
    assert data["artifact_key"] == _content_key(client.tenant)
    assert data["expires_in_seconds"] == 900
    assert data["required_headers"] == {"Content-Type": "application/gzip"}
    assert data["upload_url"].startswith("https://storage.example/put_object?")
    assert "ContentType=application%2Fgzip" in data["upload_url"]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"content_type": "application/zip", "payload_digest": BUNDLE_DIGEST, "bytes": 10}, 400),
        ({"content_type": "application/gzip", "payload_digest": "abc", "bytes": 10}, 400),
        ({"content_type": "application/gzip", "payload_digest": BUNDLE_DIGEST, "bytes": 0}, 400),
        ({"content_type": "application/gzip", "payload_digest": BUNDLE_DIGEST, "bytes": 2 * 1024 * 1024}, 413),
    ],
)
def test_content_upload_refuses_what_it_cannot_store(client, s3, body, status):
    assert _post(client, "/internal/offload/content-upload", body).status_code == status


@pytest.mark.django_db
def test_content_upload_requires_the_invoke_scope(s3):
    tenant = Tenant.objects.create(slug=f"read-only-{uuid.uuid4()}", display_name="Read")
    minted = mint_api_key(tenant, scopes=["offload:read"])
    response = Client().post(
        "/internal/offload/content-upload",
        data=json.dumps({"content_type": "application/gzip", "payload_digest": BUNDLE_DIGEST, "bytes": 10}),
        content_type="application/json",
        headers={"Authorization": f"Bearer {minted.plaintext}"},
    )
    assert response.status_code == 403


# --- invoke -----------------------------------------------------------------


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_invoke_queues_an_uploaded_bundle_and_records_its_estimate(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = BUNDLE
    body = _invoke_body(client.tenant)

    response = _post(client, "/internal/offload/invoke", body)

    assert response.status_code == 200, response.content
    data = response.json()
    assert data["operation_id"] == body["operation_id"]
    assert data["status"] == "queued"
    assert data["reused"] is False
    job = Job.objects.get(id=data["job_id"])
    assert job.estimated_cost_usd == Decimal(PRICE) * 600
    assert job.kwargs_json["params"]["lock_digest"] == LOCK_DIGEST
    assert queued == [str(job.id)]


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_invoke_refuses_a_bundle_that_was_never_uploaded(client, s3, queued):
    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant))

    assert response.status_code == 400
    assert "has not been uploaded" in response.json()["detail"]
    assert queued == []


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_invoke_refuses_a_bundle_whose_bytes_differ(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = b"other bytes"

    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant))

    assert response.status_code == 400
    assert "digest does not match" in response.json()["detail"]


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_invoke_refuses_another_tenants_bundle(client, s3, queued):
    other = Tenant.objects.create(slug=f"other-{uuid.uuid4()}", display_name="Other")
    s3.objects[_content_key(other)] = BUNDLE
    body = _invoke_body(client.tenant)
    body["input"]["artifact_key"] = _content_key(other)

    response = _post(client, "/internal/offload/invoke", body)

    assert response.status_code == 400
    assert "tenant prefix" in response.json()["detail"]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("override", "detail"),
    [
        ({"lock_digest": "uv.lock"}, "lock_digest"),
        ({"resources": {"gpu": "cuda"}}, "resources.timeout_seconds"),
        ({"resources": {"gpu": "rocm", "timeout_seconds": 60}}, "resources.gpu"),
        ({"budget": {"max_usd": -1}}, "budget.max_usd"),
        ({"python": "three"}, "params.python"),
    ],
)
def test_invoke_validates_scientific_params(client, queued, override, detail):
    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant, **override))

    assert response.status_code == 400
    assert detail in response.json()["detail"]
    assert queued == []


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_invoke_refuses_an_estimate_over_the_job_budget(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = BUNDLE
    # 600 s x $0.0004 = $0.24 > $0.10
    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant, budget={"max_usd": 0.10}))

    assert response.status_code == 402
    assert "budget.max_usd" in response.json()["detail"]
    assert not Job.objects.filter(tenant=client.tenant).exists()


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_invoke_refuses_an_estimate_over_the_remaining_period_credit(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = BUNDLE
    plan = Plan.objects.create(code=f"credit-{uuid.uuid4().hex[:8]}", limits={"monthly_credit_usd": 5})
    Subscription.objects.create(tenant=client.tenant, plan=plan, status=Subscription.Status.ACTIVE)
    start, end = calendar_period(timezone.now().date())
    accumulate(client.tenant.id, "credit_usd_micros", 4_900_000, start, end)

    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant, budget={"max_usd": None}))

    assert response.status_code == 402
    assert "remaining period credit $0.100000" in response.json()["detail"]


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_running_estimates_count_against_the_period_credit(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = BUNDLE
    plan = Plan.objects.create(code=f"credit-{uuid.uuid4().hex[:8]}", limits={"monthly_credit_usd": 0.4})
    Subscription.objects.create(tenant=client.tenant, plan=plan, status=Subscription.Status.ACTIVE)

    first = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant))
    second = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant))

    # $0.24 reserved by the first leaves $0.16 for a second $0.24 estimate.
    assert first.status_code == 200, first.content
    assert second.status_code == 402


@pytest.mark.django_db
@override_settings(**(RUNPOD | {"RUNPOD_SCIENTIFIC_MAX_JOB_USD": "0.20"}))
def test_invoke_refuses_a_job_over_the_runpod_job_limit(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = BUNDLE

    # 600 s x $0.0004 = $0.24 > $0.20, though the job's own budget allows $1.
    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant))

    assert response.status_code == 402
    assert "$0.20 limit for one RunPod job" in response.json()["detail"]
    assert not Job.objects.filter(tenant=client.tenant).exists()


@pytest.mark.django_db
@override_settings(**(RUNPOD | {"RUNPOD_SCIENTIFIC_MONTHLY_USD": "1.00"}))
def test_every_tenants_month_counts_against_the_runpod_monthly_limit(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = BUNDLE
    other = Tenant.objects.create(slug=f"other-{uuid.uuid4().hex[:8]}", display_name="Other")
    for status, cost, estimate in (
        (Job.Status.SUCCEEDED, Decimal("0.50"), Decimal("0.60")),
        (Job.Status.RUNNING, None, Decimal("0.30")),
        (Job.Status.FAILED, None, Decimal("5.00")),
    ):
        Job.objects.create(
            tenant=other,
            operation="scientific_python.run",
            operation_id=f"cj_{uuid.uuid4().hex}",
            status=status,
            cost_usd=cost,
            estimated_cost_usd=estimate,
        )

    # $0.50 spent and $0.30 reserved leave $0.20 for a $0.24 estimate; the
    # failed job was never metered and holds nothing.
    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant))

    assert response.status_code == 402
    assert "$0.200000 left of this month's $1.00 RunPod limit" in response.json()["detail"]


@pytest.mark.django_db
@override_settings(**(RUNPOD | {"RUNPOD_SCIENTIFIC_PRICE_PER_SECOND_USD": ""}))
def test_invoke_refuses_paid_execution_without_a_price(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = BUNDLE

    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant))

    assert response.status_code == 503
    assert "RUNPOD_SCIENTIFIC_PRICE_PER_SECOND_USD" in response.json()["detail"]


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_invoke_is_idempotent_on_the_theorem_job_id(client, s3, queued):
    s3.objects[_content_key(client.tenant)] = BUNDLE
    body = _invoke_body(client.tenant, operation_id="cj_0123456789abcdef")

    first = _post(client, "/internal/offload/invoke", body)
    # The same job again is the same job, even with a budget it would now fail.
    body["params"]["budget"] = {"max_usd": 0.0}
    second = _post(client, "/internal/offload/invoke", body)

    assert first.status_code == 200 and second.status_code == 200
    assert second.json() == {**first.json(), "reused": True}
    assert Job.objects.filter(operation_id="cj_0123456789abcdef").count() == 1
    assert len(queued) == 1


@pytest.mark.django_db
@override_settings(CELERY_TASK_ALWAYS_EAGER=True, CELERY_TASK_EAGER_PROPAGATES=True)
def test_stub_mode_completes_like_the_other_operations(client, provenance):
    """Locally, the job completes through the deterministic stub, with no storage."""
    response = _post(client, "/internal/offload/invoke", _invoke_body(client.tenant))

    assert response.status_code == 200, response.content
    status = client.get(f"/internal/offload/{response.json()['job_id']}").json()
    assert status["status"] == "succeeded"
    assert status["output"]["payload_digest"]
    assert status["cost_usd"] is None
    assert provenance[0]["activity"]["name"] == "scientific_python.run"


# --- status and cancel -------------------------------------------------------


@pytest.mark.django_db
def test_status_returns_a_download_url_and_cost(client, s3):
    key = f"tenants/{client.tenant.id}/outputs/{'b' * 64}.tar.gz"
    job = Job.objects.create(
        tenant=client.tenant,
        operation="scientific_python.run",
        operation_id=f"cj_{uuid.uuid4().hex}",
        status=Job.Status.SUCCEEDED,
        output_artifact_key=key,
        output_payload_digest="sha256:" + "c" * 64,
        estimated_cost_usd=Decimal("0.24"),
        cost_usd=Decimal("0.004812"),
        logs="[RunPod] completed rp-1\n",
    )

    response = client.get(f"/internal/offload/{job.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "succeeded"
    assert data["cost_usd"] == pytest.approx(0.004812)
    assert data["error_code"] == ""
    assert data["logs"] == "[RunPod] completed rp-1\n"
    assert data["output"]["artifact_key"] == key
    assert data["output"]["payload_digest"] == "sha256:" + "c" * 64
    assert data["output"]["download_url"].startswith("https://storage.example/get_object?")
    assert data["output"]["rows"] is None


@pytest.mark.django_db
@override_settings(CELERY_TASK_ALWAYS_EAGER=True, **RUNPOD)
def test_cancel_cancels_the_runpod_job(client, monkeypatch):
    cancelled: list[tuple[str, str]] = []

    class RecordingRunpodClient:
        def __init__(self, **kwargs):
            self.endpoint_id = kwargs["endpoint_id"]

        def cancel(self, job_id):
            cancelled.append((self.endpoint_id, job_id))
            return {"status": "CANCELLED"}

    monkeypatch.setattr("apps.orchestration.tasks.RunpodServerlessClient", RecordingRunpodClient)
    monkeypatch.setattr("celery.app.control.Control.revoke", lambda *args, **kwargs: None)
    job = Job.objects.create(
        tenant=client.tenant,
        operation="scientific_python.run",
        operation_id=f"cj_{uuid.uuid4().hex}",
        status=Job.Status.RUNNING,
        kwargs_json={"runpod": {"endpoint_id": "scientific-endpoint", "job_id": "rp-remote-7"}},
    )

    response = _post(client, f"/internal/offload/{job.id}/cancel")

    assert response.status_code == 200
    assert response.json() == {"job_id": str(job.id), "status": "canceled", "canceled": True}
    assert cancelled == [("scientific-endpoint", "rp-remote-7")]
    job.refresh_from_db()
    assert job.status == Job.Status.CANCELED


# --- the RunPod task ---------------------------------------------------------


def _running_job(tenant, **params) -> Job:
    body = _invoke_body(tenant, **params)
    return Job.objects.create(
        tenant=tenant,
        operation="scientific_python.run",
        operation_id=body["operation_id"],
        input_payload_digest=BUNDLE_DIGEST,
        estimated_cost_usd=Decimal(PRICE) * 600,
        kwargs_json={"params": body["params"], "input_artifact_key": _content_key(tenant)},
        status=Job.Status.RUNNING,
    )


class FakeRunpod:
    """Plays RunPod and the worker: ``finish`` decides how the remote job ends."""

    SUCCESS = "COMPLETED"
    instances: list["FakeRunpod"] = []

    def __init__(self, **kwargs):
        self.endpoint_id = kwargs["endpoint_id"]
        self.submitted = None
        self.policy = None
        self.cancelled: list[str] = []
        FakeRunpod.instances.append(self)

    def submit(self, payload, *, policy=None):
        self.submitted, self.policy = payload, policy
        return type("Submitted", (), {"job_id": "rp-remote-1", "status": "IN_QUEUE"})()

    def wait(self, job_id, **kwargs):
        kwargs["on_update"]({"status": "IN_PROGRESS"})
        return self.finish(kwargs)

    def status(self, job_id):
        return {"status": "CANCELLED", "executionTime": 2_500}

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        return {"status": "CANCELLED"}


@pytest.fixture
def runpod(monkeypatch):
    FakeRunpod.instances = []
    monkeypatch.setattr("apps.orchestration.tasks.RunpodServerlessClient", FakeRunpod)
    return FakeRunpod


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_runpod_task_runs_the_bundle_and_verifies_the_result(client, s3, runpod, provenance):
    s3.objects[_content_key(client.tenant)] = BUNDLE
    result_tarball = b"\x1f\x8b result tarball"
    worker_json = {"gpu_name": "NVIDIA L4", "executor": "runpod-serverless", "runpod_job_id": "rp-remote-1"}

    def finish(fake, _kwargs):
        # The worker accepts exactly what Django sent, then uploads the result.
        assert worker.validate_request(fake.submitted) is None
        s3.objects[fake.submitted["output"]["artifact_key"]] = result_tarball
        return {
            "status": "COMPLETED",
            "executionTime": 12_000,
            "output": {
                "output": {
                    "artifact_key": fake.submitted["output"]["artifact_key"],
                    "payload_digest": "sha256:" + hashlib.sha256(result_tarball).hexdigest(),
                    "bytes": len(result_tarball),
                },
                "worker": worker_json,
                "logs": "[stdout] fitted",
            },
        }

    runpod.finish = finish
    job = _running_job(client.tenant)

    outcome = run_offload_python(str(job.id))

    job.refresh_from_db()
    fake = runpod.instances[0]
    assert outcome["status"] == "succeeded", job.error
    assert fake.endpoint_id == "scientific-endpoint"
    assert fake.policy == {"executionTimeout": (600 + scientific.TRANSFER_GRACE_SECONDS) * 1000}
    assert fake.submitted["contract"] == "theorem.scientific_python.v1"
    assert fake.submitted["input"]["read_url"].startswith("https://storage.example/get_object?")
    assert "ContentType=application%2Fgzip" in fake.submitted["output"]["write_url"]
    assert fake.submitted["price_per_second_usd"] == PRICE
    assert job.status == Job.Status.SUCCEEDED
    assert job.output_artifact_key.endswith(".tar.gz")
    assert job.output_payload_digest == "sha256:" + hashlib.sha256(result_tarball).hexdigest()
    assert job.cost_usd == Decimal("4.800000") * Decimal("0.001")  # 12 s x $0.0004
    assert job.kwargs_json["worker"] == worker_json
    assert "[stdout] fitted" in job.logs
    usage = Usage.objects.get(tenant=client.tenant)
    assert usage.counters["credit_usd_micros"] == 4_800
    assert provenance[0]["activity"]["code_ref"] == "ghcr.io/theorem/scientific@sha256:immutable"

    status = client.get(f"/internal/offload/{job.id}").json()
    assert status["cost_usd"] == pytest.approx(0.0048)
    assert status["output"]["download_url"]


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_runpod_task_records_the_workers_failure_code(client, s3, runpod, provenance):
    s3.objects[_content_key(client.tenant)] = BUNDLE

    def finish(fake, _kwargs):
        return {
            "status": "FAILED",
            "executionTime": 500,
            "error": "environment_mismatch: params.lock_digest sha256:aaa does not match this image's uv.lock",
            "output": {"code": "environment_mismatch", "logs": "[worker] checked lock"},
        }

    runpod.finish = finish
    job = _running_job(client.tenant)

    assert run_offload_python(str(job.id))["status"] == "failed"

    job.refresh_from_db()
    assert job.status == Job.Status.FAILED
    assert job.error_code == "environment_mismatch"
    assert job.error.startswith("environment_mismatch: ")
    assert "[worker] checked lock" in job.logs
    assert job.cost_usd == Decimal("0.000200")
    status = client.get(f"/internal/offload/{job.id}").json()
    assert status["error_code"] == "environment_mismatch"
    assert status["output"] is None


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_runpod_task_refuses_a_result_whose_digest_differs(client, s3, runpod, provenance):
    s3.objects[_content_key(client.tenant)] = BUNDLE

    def finish(fake, _kwargs):
        s3.objects[fake.submitted["output"]["artifact_key"]] = b"tampered"
        return {
            "status": "COMPLETED",
            "executionTime": 1_000,
            "output": {"output": {"payload_digest": "sha256:" + "d" * 64, "bytes": 8}},
        }

    runpod.finish = finish
    job = _running_job(client.tenant)

    assert run_offload_python(str(job.id))["status"] == "failed"
    job.refresh_from_db()
    assert "digest does not match" in job.error
    assert provenance == []


@pytest.mark.django_db
@override_settings(**RUNPOD)
def test_runpod_task_cancels_the_remote_job_when_cancelled_while_polling(client, s3, runpod, provenance):
    s3.objects[_content_key(client.tenant)] = BUNDLE

    def wait(self, job_id, **kwargs):
        Job.objects.filter(id=job.id).update(status=Job.Status.CANCELED)
        kwargs["on_update"]({"status": "IN_PROGRESS"})
        raise AssertionError("polling must stop once the job is cancelled")

    runpod.wait = wait
    job = _running_job(client.tenant)

    assert run_offload_python(str(job.id)) == {"status": "canceled"}

    job.refresh_from_db()
    assert job.status == Job.Status.CANCELED
    assert runpod.instances[0].cancelled == ["rp-remote-1"]
    assert job.cost_usd == Decimal("0.001000")  # 2.5 s of execution before the cancel
