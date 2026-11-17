"""Publish a sprite bundle for tenants' sprites to install.

    manage.py publish_environment_bundle <path-to-theorem-sprite-bundle.tar>

Theorem's ``apps/theorem-ide-host/sprite-bundle/build.sh`` builds the bundle.
This command refuses one a sprite could not run, uploads it to the private
release bucket under its sha256, and prints the digest to pin as
``ENVIRONMENT_BUNDLE_SHA256``. Publishing changes no sprite: pin the digest,
then run ``provision_tenant_environment`` for each tenant (new tenants get it
at signup). Publishing the same bundle again uploads nothing.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.tenancy.environment import (
    BUNDLE_ENTRIES,
    RELEASE_BUCKET_SETTINGS,
    bundle_object_key,
    missing_settings,
    release_bucket_client,
)

MISSING_OBJECT = {"404", "NoSuchKey", "NotFound"}
ELF_EXECUTABLE, ELF_SHARED = 2, 3  # ET_EXEC, and ET_DYN for position-independent builds


def is_linux_x86_64_executable(header: bytes) -> bool:
    """A 64-bit little-endian x86-64 ELF a sprite can run: an executable or a
    position-independent one, not an object file or a core dump."""
    return (
        header[:4] == b"\x7fELF"
        and header[4:6] == b"\x02\x01"
        and int.from_bytes(header[16:18], "little") in (ELF_EXECUTABLE, ELF_SHARED)
        and int.from_bytes(header[18:20], "little") == 62
    )


def bundle_problem(path: Path) -> str | None:
    """Why a sprite could not run this bundle, or None."""
    try:
        bundle = tarfile.open(path, "r:")
    except tarfile.ReadError:
        return f"{path} is not a tar file"
    with bundle:
        members = {m.name.removeprefix("./"): m for m in bundle.getmembers()}
        for entry in BUNDLE_ENTRIES:
            member = members.get(entry)
            if member is None or not member.isfile():
                return f"{entry} is missing"
            if entry == "Caddyfile":
                continue
            head = bundle.extractfile(member).read(20)
            executable = bool(member.mode & 0o111)
            if entry.startswith("bin/") and not (
                executable and is_linux_x86_64_executable(head)
            ):
                return f"{entry} is not a linux x86_64 executable"
            if entry.startswith("run/") and not (executable and head.startswith(b"#!")):
                return f"{entry} is not an executable script"
    return None


class Command(BaseCommand):
    help = "Upload a linux x86_64 sprite bundle and print the digest to pin."

    def add_arguments(self, parser):
        parser.add_argument("bundle", type=Path, help="Path to theorem-sprite-bundle.tar")

    def handle(self, *args, bundle: Path, **options):
        if missing := missing_settings(RELEASE_BUCKET_SETTINGS):
            raise CommandError(f"set {', '.join(missing)} first")
        if not bundle.is_file():
            raise CommandError(f"no file at {bundle}")
        if problem := bundle_problem(bundle):
            raise CommandError(f"refusing {bundle}: {problem}")
        with bundle.open("rb") as body:
            digest = hashlib.file_digest(body, "sha256").hexdigest()

        bucket = settings.ENVIRONMENT_HOST_BUCKET
        key = bundle_object_key(digest)
        client = release_bucket_client()
        try:
            client.head_object(Bucket=bucket, Key=key)
            self.stdout.write(f"{key} is already published")
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in MISSING_OBJECT:
                raise CommandError(f"reading {bucket}/{key} failed: {error}") from error
            try:
                with bundle.open("rb") as body:
                    client.put_object(
                        Bucket=bucket, Key=key, Body=body, ContentType="application/x-tar"
                    )
            except (BotoCoreError, ClientError) as upload_error:
                raise CommandError(f"uploading {key} failed: {upload_error}") from upload_error
            self.stdout.write(f"published {key}")
        except BotoCoreError as error:
            raise CommandError(f"reading {bucket}/{key} failed: {error}") from error
        self.stdout.write(f"ENVIRONMENT_BUNDLE_SHA256={digest}")
