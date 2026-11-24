#!/usr/bin/env python3
"""Collect explicit release outputs with commit identity and SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["python"], required=True)
    parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("GITHUB_SHA", revision) != revision:
        raise SystemExit("The checked-out commit does not match the workflow source.")
    ref_type = os.environ.get("GITHUB_REF_TYPE", "branch")
    version = (
        os.environ["GITHUB_REF_NAME"]
        if ref_type == "tag"
        else f"snapshot-{revision[:12]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,100}", version):
        raise SystemExit("Release tag must be a safe version name.")
    out = root / "release-assets"
    wheels = list(out.glob("*.whl"))
    sdists = list(out.glob("theorem_control_plane-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected exactly one wheel and one source distribution.")
    with zipfile.ZipFile(wheels[0]) as wheel:
        required = {"theorem_control/wsgi.py", "apps/tenancy/models.py"}
        if not required.issubset(wheel.namelist()):
            raise SystemExit("Wheel is missing required application packages.")
        if wheel.testzip() is not None:
            raise SystemExit("Wheel checksum validation failed.")
    source = out / f"django-theorem-{version}-source.tar.gz"
    with source.open("wb") as stream:
        subprocess.run(
            ["git", "archive", "--format=tar.gz", f"--prefix=django-theorem-{version}/", revision],
            stdout=stream,
            check=True,
        )
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    repository = os.environ.get("GITHUB_REPOSITORY", "Theoremrs/Django-Theorem")
    metadata = {
        "schema_version": 1,
        "repository": repository,
        "source_commit": revision,
        "release_version": version,
        "python_distribution_version": project["version"],
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "artifact_kind": "python-distributions-and-source",
        "runtime_dependencies_included": False,
    }
    (out / "build-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "RELEASE-NOTES.md").write_text(
        f"Built from `{repository}@{revision}`.\n\n"
        "Contains the Python wheel, Python source distribution, and complete tracked source snapshot. "
        "The distribution version remains the version declared by pyproject.toml. "
        "Runtime dependencies, model weights, credentials, and databases are not included.\n\n"
        "This build checks Python syntax and the distribution contents. It does not certify "
        "a running Django deployment or the native rendering and extraction pipelines. "
        "No Fly service is deployed by this workflow.\n"
    )
    assets = sorted(path for path in out.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    if any(path.is_symlink() or path.stat().st_size == 0 for path in assets):
        raise SystemExit("Refusing empty or symlinked release assets.")
    with (out / "SHA256SUMS").open("w") as checksums:
        for path in assets:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            checksums.write(f"{digest}  {path.name}\n")
    print(f"Prepared {len(assets)} assets for {version} from {revision}.")


if __name__ == "__main__":
    main()
