"""``scientific_python.run``: Theorem computation bundles on a RunPod GPU worker.

Theorem uploads its job directory (``request.json``, ``runner.py``, ``work/``,
``inputs/``) as a gzip tarball through ``/content-upload``, then invokes this
operation with the bundle's content-addressed key. The control plane prices
the job before it is queued, hands the worker one presigned GET for the bundle
and one presigned PUT for the result tarball, and verifies the result's digest
before the job is reported as succeeded.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from uuid import UUID

from django.conf import settings
from django.db.models import Sum
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apps.billing.credit import CREDIT_METER, MICROS_PER_USD, credit_allowance
from apps.orchestration.artifacts import (
    GZIP_CONTENT_TYPE,
    ArtifactStore,
    ArtifactValidationError,
    is_sha256_digest,
)

OPERATION = "scientific_python.run"
CONTRACT = "theorem.scientific_python.v1"
OUTPUT_EXTENSION = "tar.gz"
# The worker's runner shares the job's deadline; RunPod's own execution timeout
# is a backstop past it that leaves room to fetch the bundle and upload results.
TRANSFER_GRACE_SECONDS = 120
# A Pod is billed from creation, while its image is pulled and it boots.
POD_START_GRACE_SECONDS = 600
POD_RESULT_EXTENSION = "result.json"
POD_COMMAND = ["/opt/worker/venv/bin/python", "-u", "/opt/worker/worker.py", "--pod"]
_ERROR_CODE = re.compile(r"^([a-z][a-z0-9_]{2,63}): ")


class BudgetExceeded(Exception):
    """The job's estimate is more than its own budget or the tenant's credit."""


class PricingUnavailable(Exception):
    """Paid execution is configured without a price to bound it."""


class Resources(BaseModel):
    model_config = ConfigDict(extra="allow")

    gpu: Literal["cuda", "any"] | None = None
    cpu: int | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    timeout_seconds: int = Field(gt=0, le=7 * 24 * 60 * 60)
    # Where on RunPod: the queue-based Serverless endpoint, or a Pod of its own.
    placement: Literal["serverless", "pod"] | None = None


class Budget(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_usd: float | None = Field(default=None, ge=0)


class ScientificParams(BaseModel):
    """The invoke ``params`` Theorem sends; unknown keys pass through to the worker."""

    model_config = ConfigDict(extra="allow")

    python: str = Field(pattern=r"^3\.\d+$")
    lock_digest: str
    extras: list[str] = []
    requirements: list[str] = []
    resources: Resources
    budget: Budget = Budget()
    entry: str | None = None
    seed: int | None = None
    network: bool = True


def parse_params(raw: dict[str, Any]) -> ScientificParams:
    try:
        params = ScientificParams.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise ArtifactValidationError(f"params.{where}: {first['msg']}") from exc
    if not is_sha256_digest(params.lock_digest):
        raise ArtifactValidationError("params.lock_digest must be a sha256 digest")
    if any(not requirement.strip() for requirement in params.requirements):
        raise ArtifactValidationError("params.requirements entries must be non-empty")
    return params


def validate_input(tenant_id: UUID, payload_digest: str, artifact_key: str) -> None:
    """The bundle descriptor's shape; existence is checked separately when live."""
    if not is_sha256_digest(payload_digest):
        raise ArtifactValidationError("input.payload_digest must be a sha256 digest")
    if not artifact_key:
        raise ArtifactValidationError("input.artifact_key is required")
    if not artifact_key.startswith(ArtifactStore.tenant_prefix(tenant_id)):
        raise ArtifactValidationError("artifact_key is outside the admitted tenant prefix")


def price_per_second() -> Decimal:
    try:
        price = Decimal(str(settings.RUNPOD_SCIENTIFIC_PRICE_PER_SECOND_USD).strip() or "0")
    except InvalidOperation as exc:
        raise PricingUnavailable("RUNPOD_SCIENTIFIC_PRICE_PER_SECOND_USD is not a number") from exc
    if not price.is_finite() or price < 0:
        raise PricingUnavailable("RUNPOD_SCIENTIFIC_PRICE_PER_SECOND_USD must be non-negative")
    if price == 0 and settings.OFFLOAD_EXECUTION_MODE == "runpod":
        raise PricingUnavailable(
            "RUNPOD_SCIENTIFIC_PRICE_PER_SECOND_USD is required for paid scientific execution"
        )
    return price


def placement(resources: dict[str, Any] | Resources) -> str:
    """The job's own choice, else the Serverless endpoint when one is configured."""
    chosen = resources.get("placement") if isinstance(resources, dict) else resources.placement
    if chosen:
        return str(chosen)
    return "serverless" if settings.RUNPOD_SCIENTIFIC_ENDPOINT_ID else "pod"


def job_placement(job) -> str:
    return placement(((job.kwargs_json.get("params") or {}).get("resources")) or {})


def pod_price_per_second() -> Decimal:
    hourly = _usd_setting("RUNPOD_SCIENTIFIC_POD_PRICE_PER_HOUR_USD")
    if hourly == 0 and settings.OFFLOAD_EXECUTION_MODE == "runpod":
        raise PricingUnavailable(
            "RUNPOD_SCIENTIFIC_POD_PRICE_PER_HOUR_USD is required for scientific jobs on a Pod"
        )
    return hourly / 3600


def estimate_usd(params: ScientificParams) -> Decimal:
    if placement(params.resources) == "pod":
        seconds = params.resources.timeout_seconds + POD_START_GRACE_SECONDS + TRANSFER_GRACE_SECONDS
        return (pod_price_per_second() * seconds).quantize(Decimal("0.000001"))
    return price_per_second() * params.resources.timeout_seconds


def reserved_usd(tenant_id: UUID) -> Decimal:
    """Estimates of the tenant's jobs that are still running: spend not yet metered."""
    from apps.orchestration.models import Job

    total = Job.objects.filter(
        tenant_id=tenant_id,
        status__in=(Job.Status.QUEUED, Job.Status.RUNNING),
        estimated_cost_usd__isnull=False,
    ).aggregate(total=Sum("estimated_cost_usd"))["total"]
    return total or Decimal(0)


def _usd_setting(name: str) -> Decimal:
    try:
        value = Decimal(str(getattr(settings, name)).strip())
    except InvalidOperation as exc:
        raise PricingUnavailable(f"{name} is not a number") from exc
    if not value.is_finite() or value < 0:
        raise PricingUnavailable(f"{name} must be non-negative")
    return value


def month_spend_usd() -> Decimal:
    """What scientific jobs cost on RunPod this calendar month, across every
    tenant: metered jobs at their cost, queued and running ones at their
    estimate."""
    from apps.orchestration.models import Job

    now = timezone.now()
    month = Job.objects.filter(
        operation=OPERATION,
        created_at__gte=now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
    )
    spent = month.aggregate(total=Sum("cost_usd"))["total"] or Decimal(0)
    reserved = month.filter(
        status__in=(Job.Status.QUEUED, Job.Status.RUNNING),
        cost_usd__isnull=True,
        estimated_cost_usd__isnull=False,
    ).aggregate(total=Sum("estimated_cost_usd"))["total"] or Decimal(0)
    return spent + reserved


def enforce_spend_limits(estimate: Decimal) -> None:
    """The operator's RunPod limits: one job's estimate, and the month's spend."""
    job_limit = _usd_setting("RUNPOD_SCIENTIFIC_MAX_JOB_USD")
    if estimate > job_limit:
        raise BudgetExceeded(
            f"estimated cost ${estimate:.6f} exceeds the ${job_limit:.2f} limit for one "
            "RunPod job; shorten resources.timeout_seconds"
        )
    month_limit = _usd_setting("RUNPOD_SCIENTIFIC_MONTHLY_USD")
    remaining = month_limit - month_spend_usd()
    if estimate > remaining:
        raise BudgetExceeded(
            f"estimated cost ${estimate:.6f} exceeds the ${max(remaining, Decimal(0)):.6f} "
            f"left of this month's ${month_limit:.2f} RunPod limit"
        )


def enforce_budget(tenant_id: UUID, estimate: Decimal, params: ScientificParams) -> None:
    """Refuse before queueing; the caller holds the tenant row lock."""
    max_usd = params.budget.max_usd
    if max_usd is not None and estimate > Decimal(str(max_usd)):
        raise BudgetExceeded(
            f"estimated cost ${estimate:.6f} exceeds budget.max_usd ${Decimal(str(max_usd)):.6f}"
        )
    enforce_spend_limits(estimate)
    allowance = credit_allowance(tenant_id, timezone.now().date())
    if allowance.limit_usd is None:
        return
    remaining = allowance.limit_usd - allowance.used_usd - reserved_usd(tenant_id)
    if estimate > remaining:
        raise BudgetExceeded(
            f"estimated cost ${estimate:.6f} exceeds the remaining period credit "
            f"${max(remaining, Decimal(0)):.6f}"
        )


def runpod_input(
    job, *, store: ArtifactStore, output_artifact_key: str, price: Decimal | None = None
) -> dict[str, Any]:
    """The byte-free request the scientific worker accepts."""
    if job.tenant_id is None:
        raise ArtifactValidationError("RunPod jobs require an admitted tenant")
    input_artifact_key = str(job.kwargs_json.get("input_artifact_key") or "")
    store.validate_key(job.tenant_id, input_artifact_key)
    store.validate_key(job.tenant_id, output_artifact_key)
    validate_input(job.tenant_id, job.input_payload_digest, input_artifact_key)
    return {
        "contract": CONTRACT,
        "operation": job.operation,
        "operation_id": job.operation_id,
        "input": {
            "payload_digest": job.input_payload_digest,
            "artifact_key": input_artifact_key,
            "read_url": store.presign_get(job.tenant_id, input_artifact_key),
        },
        "output": {
            "artifact_key": output_artifact_key,
            "content_type": GZIP_CONTENT_TYPE,
            "write_url": store.presign_put_content(
                job.tenant_id, output_artifact_key, content_type=GZIP_CONTENT_TYPE
            ),
        },
        "params": job.kwargs_json.get("params") or {},
        "price_per_second_usd": str(price_per_second() if price is None else price),
        "max_bytes": store.max_bytes,
    }


def pod_request(
    job, *, store: ArtifactStore, output_artifact_key: str, result_artifact_key: str
) -> dict[str, Any]:
    """A Pod of its own for one job: the worker's request rides in its
    environment, with a presigned PUT for the handler's result."""
    request = runpod_input(
        job, store=store, output_artifact_key=output_artifact_key, price=pod_price_per_second()
    )
    request["result_url"] = store.presign_put_content(
        job.tenant_id, result_artifact_key, content_type="application/json"
    )
    pod = {
        "name": f"theorem-scientific-{str(job.id)[:12]}",
        "image": settings.RUNPOD_SCIENTIFIC_IMAGE_DIGEST,
        "cloud": settings.RUNPOD_SCIENTIFIC_POD_CLOUD_TYPE,
        # torch's CUDA 13 wheels need a driver that supports CUDA 13.
        "gpu": {"id": settings.RUNPOD_SCIENTIFIC_POD_GPU_TYPE, "count": 1, "minCudaVersion": "13.0"},
        "disk": settings.RUNPOD_SCIENTIFIC_POD_DISK_GB,
        "cmd": POD_COMMAND,
        "env": {"THEOREM_SCIENTIFIC_REQUEST": json.dumps(request, sort_keys=True)},
    }
    if settings.RUNPOD_SCIENTIFIC_POD_REGISTRY_AUTH_ID:
        pod["registry"] = settings.RUNPOD_SCIENTIFIC_POD_REGISTRY_AUTH_ID
    return pod


def pod_deadline_seconds(job) -> int:
    """How long a Pod may live: the job's own time plus boot and transfers."""
    return timeout_seconds(job) + POD_START_GRACE_SECONDS + TRANSFER_GRACE_SECONDS


def pod_cost(state: dict[str, Any], seconds: float) -> Decimal:
    """The Pod's hourly rate as RunPod reports it (``cost``), else the configured
    one, for the time from creation to termination."""
    raw = state.get("cost")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        per_second = Decimal(str(raw)) / 3600
    else:
        per_second = pod_price_per_second()
    return (per_second * Decimal(str(max(seconds, 0)))).quantize(Decimal("0.000001"))


def timeout_seconds(job) -> int:
    return int(((job.kwargs_json.get("params") or {}).get("resources") or {})["timeout_seconds"])


def runpod_policy(job) -> dict[str, int]:
    """RunPod kills the job past the runner's deadline, so the estimate holds."""
    return {"executionTimeout": (timeout_seconds(job) + TRANSFER_GRACE_SECONDS) * 1000}


def control_deadline_seconds(job) -> int:
    """How long the control plane polls: the job's own time plus queue and cold start."""
    return timeout_seconds(job) + TRANSFER_GRACE_SECONDS + settings.RUNPOD_JOB_TIMEOUT_SECONDS


def cost_from_status(state: dict[str, Any]) -> Decimal | None:
    """RunPod's billed ``executionTime`` (milliseconds) times the endpoint's price."""
    raw = state.get("executionTime")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
        return None
    try:
        price = price_per_second()
    except PricingUnavailable:
        return None
    return (Decimal(str(raw)) / 1000 * price).quantize(Decimal("0.000001"))


def charge(job, cost: Decimal) -> None:
    """Record the actual cost on the job and meter it against the period credit."""
    from apps.billing.usage import accumulate

    job.cost_usd = cost
    job.save(update_fields=["cost_usd", "updated_at"])
    micros = int(cost * MICROS_PER_USD)
    if micros <= 0 or job.tenant_id is None:
        return
    allowance = credit_allowance(job.tenant_id, timezone.now().date())
    accumulate(job.tenant_id, CREDIT_METER, micros, allowance.period_start, allowance.period_end)


def output_descriptor(output: Any) -> tuple[str, int]:
    """The worker's claim about the uploaded result tarball."""
    if not isinstance(output, dict):
        raise ValueError("scientific worker output must be a descriptor object")
    payload_digest = output.get("payload_digest")
    size = output.get("bytes")
    if not isinstance(payload_digest, str) or not is_sha256_digest(payload_digest):
        raise ValueError("scientific worker output.payload_digest must be a sha256 digest")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("scientific worker output.bytes must be a positive integer")
    return payload_digest, size


def error_code(message: str) -> str:
    """Workers prefix failures with their code: ``environment_mismatch: ...``."""
    match = _ERROR_CODE.match(message or "")
    return match.group(1) if match else ""
