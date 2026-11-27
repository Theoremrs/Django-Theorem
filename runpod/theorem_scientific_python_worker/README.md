# Theorem Scientific Python Worker

The RunPod worker behind `scientific_python.run`, on Serverless or on a Pod of
its own. It runs one Theorem
computation (a job directory Theorem uploaded as a gzip tarball) in Theorem's
standard scientific environment with the `accelerated` and `cuda` extras, using
Theorem's own runner, and uploads a result tarball holding `result.json`,
`outputs/`, `stdout.log`, `stderr.log` and `worker.json`.

It accepts only `theorem.scientific_python.v1` and receives two time-limited,
tenant-scoped capabilities: a presigned GET for the bundle and a presigned PUT
for the result. It holds no object-store credential. Django re-downloads the
result and checks its digest before the job reports success.

## What it checks

- The bundle's sha256 matches `input.payload_digest`.
- Every bundle member is a regular file or directory inside the job directory
  (absolute names, `..`, links and devices are refused as `invalid_bundle`), and
  the bundle expands to at most `THEOREM_MAX_EXTRACTED_BYTES`.
- `params.lock_digest` is the sha256 of the baked `uv.lock`, `params.python` is
  the baked Python's major.minor, and `params.extras` are synced in the image.
  Any difference fails as `environment_mismatch`.
- The bundle's `runner.py` is the baked runner (`runner_mismatch` otherwise).
- File inputs in `request.json` are paths relative to the job directory, such
  as `inputs/data.csv`; the worker binds them to its own copy.

`params.requirements` are installed with `uv pip install` into a per-job venv
layered over the baked one, constrained to the baked versions of every shared
package. Install and run share `resources.timeout_seconds`.

Without `params.network`, the computation runs in its own network namespace
when the container may create one, and otherwise under a socket guard installed
in the job's interpreter. `worker.json` records which (`network.isolation`).
The computation's environment holds only GPU variables and its own home and
temp directories, never the worker's RunPod credentials.

Failures return `{"error": "<code>: <message>", "code", "logs"}` with codes
`invalid_input`, `invalid_bundle`, `environment_mismatch`, `runner_mismatch`,
`requirements_failed`, `timeout`, `runner_died`, `output_too_large` and
`transfer_failed`. A computation that raised is not a failure here: its
`result.json` says `"ok": false` and is returned like any other result.

## Vendored files

`environment/pyproject.toml` and `environment/uv.lock` are copies of Theorem's
`rustyredcore_THG/crates/theorem-compute/environments/scientific/`, and
`theorem_compute_runner.py` is a copy of
`rustyredcore_THG/crates/theorem-compute/python/theorem_compute_runner.py`.
Refresh all three together from a Theorem checkout, then rebuild the image:

```bash
THEOREM=<path to the Theorem checkout>/rustyredcore_THG/crates/theorem-compute
cp "$THEOREM/environments/scientific/pyproject.toml" "$THEOREM/environments/scientific/uv.lock" environment/
cp "$THEOREM/python/theorem_compute_runner.py" theorem_compute_runner.py
```

`requirements.txt` pins the RunPod SDK for the handler's own venv, with hashes:

```bash
uv pip compile requirements.in --python-version 3.13 --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header -o requirements.txt
```

## Hardware

torch 2.15 from PyPI ships CUDA 13 wheels, while `jax[cuda12]` and
`cupy-cuda12x` use CUDA 12; all of them bring their CUDA libraries as wheels.
The host driver must therefore support CUDA 13 (driver 580 or newer). Restrict
the RunPod endpoint's allowed CUDA versions accordingly.

## Build and deploy

The build context is this directory:

```bash
docker buildx build --platform linux/amd64 -t <registry>/theorem-scientific-python-worker:<tag> --push .
```

or, without a local builder, `fly deploy --build-only --push -c fly.image.toml`,
which creates no Fly Machine. Then:

1. Create a queue-based RunPod Serverless endpoint on that exact image digest,
   on GPUs whose driver supports CUDA 13.
2. Set on the Django control plane `RUNPOD_SCIENTIFIC_ENDPOINT_ID`,
   `RUNPOD_SCIENTIFIC_IMAGE_DIGEST` (the immutable image reference) and
   `RUNPOD_SCIENTIFIC_PRICE_PER_SECOND_USD` (the endpoint GPU class's price).

A job whose `resources.placement` is `pod`, or any job when no Serverless
endpoint is configured, runs on a Pod of its own instead: Django creates it
through RunPod's REST API v2 on this same image with `worker.py --pod`, passes
the request in `THEOREM_SCIENTIFIC_REQUEST` with a presigned PUT for the
handler's result, terminates the Pod once that result is read (or at the job's
timeout plus boot and transfer time), and meters the Pod's own hourly `cost`
for the time it lived. Set `RUNPOD_SCIENTIFIC_POD_GPU_TYPE`,
`RUNPOD_SCIENTIFIC_POD_CLOUD_TYPE` and `RUNPOD_SCIENTIFIC_POD_PRICE_PER_HOUR_USD`
(an upper bound for that GPU, which prices the estimate). An image in a private
registry such as `registry.fly.io` also needs
`RUNPOD_SCIENTIFIC_POD_REGISTRY_AUTH_ID`, the id of a RunPod container registry
credential (`POST /v2/registries`) that can pull it.

Either way a job's estimate must fit `RUNPOD_SCIENTIFIC_MAX_JOB_USD` (default
$5) and what is left of `RUNPOD_SCIENTIFIC_MONTHLY_USD` (default $50), which
every tenant's scientific jobs share in a calendar month.

To check a request without RunPod, run
`python worker.py --local-input request.json`.
