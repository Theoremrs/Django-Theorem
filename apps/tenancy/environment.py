"""Give each tenant its own Fly Sprite, running the sprite bundle, bound to its store.

Signup creates a tenant; ``begin_environment`` records the sprite it will get
and ``provision_environment`` builds it: the sprite, the pinned sprite bundle,
the two credential files and the three services that run the bundle. The
bundle's contract, from its layout to its health checks, is Theorem's
``apps/theorem-ide-host/SPRITE-BUNDLE.md``:

- ``ide-host`` runs ``theorem-ide-host``, the workbench's environment host
  (terminal, files, language, runs), on loopback 8081.
- ``codemode`` runs ``codemode-serve`` on 8082.
- ``router`` runs Caddy on 8080, the sprite URL: ``/api/environment/*`` goes
  to the host and ``/codemode/*`` to code mode.

Provisioning waits until both refuse an anonymous request and accept the
tenant's own credential.

``bind_store`` then tells the tenant's store where its sprite is. The store
reads ``THEOREM_ENVIRONMENT_HOSTS_FILE`` and the two credential files named
there; all of them reach its machine through Fly, the credentials as app
secrets, so the machine config never holds one. The store and the sprite are
built in parallel, and whichever finishes second makes the binding.

The credentials are never stored. ``host_token`` and ``codemode_token`` derive
them from ``ENVIRONMENT_TOKEN_KEY`` and the tenant id, so Django, the sprite and
the store agree on them without a table holding them, and rotating the key
rotates every tenant's on the next provisioning run. The store is rebound only
to credentials its sprite has already accepted.

A sprite's URL is public: the host's and code mode's own credentials are its
authentication. Sprites' ``sprite`` URL auth would demand the org token in the
same ``Authorization`` header, which the store must never hold.

Bundles sit in a private bucket under their digest, so pinning
``ENVIRONMENT_BUNDLE_SHA256`` names both the object and the check it must pass.
A sprite gets a presigned link to that one object, valid for ten minutes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from time import monotonic, sleep

import boto3
import httpx
from botocore.config import Config
from django.conf import settings
from sprites import FileNotFoundError_, NotFoundError, SpritesClient, URLSettings
from sprites.exceptions import APIError

from apps.tenancy.fly import FlyMachinesClient
from apps.tenancy.models import EnvironmentHost, StorePlacement, Tenant

HOME = "/home/sprite"
THEOREM_ROOT = f"{HOME}/theorem"
STARTED_PATH = f"{THEOREM_ROOT}/started"
HOST_TOKEN_PATH = f"{HOME}/.config/theorem/environment-token"
CODEMODE_TOKEN_PATH = f"{HOME}/.config/theorem/codemode-token"
WORKSPACE_ROOT = f"{HOME}/theorem-environment"
ROUTER_PORT = 8080
# The bundle's services, in start order: the router needs the other two.
SERVICES = ("ide-host", "codemode", "router")
OPERATIONS_PATH = "/api/environment/v1/operations"
CODEMODE_HEALTH_PATH = "/codemode/healthz"
CODEMODE_PROBE_PATH = "/codemode/v1/sessions/theorem-health-probe/manifest"
# The bare host service an earlier version of this module ran on 8080, where
# the router now listens.
LEGACY_SERVICE = "theorem-ide-host"

STORE_MACHINE_NAME = "store"
STORE_HOSTS_FILE = "/run/secrets/environment-hosts.json"
STORE_TOKEN_FILE = "/run/secrets/environment-token"
STORE_TOKEN_SECRET = "THEOREM_ENVIRONMENT_TOKEN_B64"
STORE_CODEMODE_TOKEN_FILE = "/run/secrets/environment-codemode-token"
STORE_CODEMODE_TOKEN_SECRET = "THEOREM_ENVIRONMENT_CODEMODE_TOKEN_B64"

# Everything a sprite runs from a bundle; publishing refuses a bundle without
# one of them, and so does the sprite before it switches to it.
BUNDLE_ENTRIES = (
    "bin/theorem-ide-host",
    "bin/codemode-serve",
    "bin/monty",
    "bin/theorem-js-cell",
    "bin/hc",
    "bin/caddy",
    "Caddyfile",
    "run/ide-host",
    "run/codemode",
    "run/router",
)
BUNDLE_OBJECT_PREFIX = "theorem-sprite-bundle/linux-x86_64"
BUNDLE_DOWNLOAD_SECONDS = 600
RELEASE_BUCKET_SETTINGS = (
    "ENVIRONMENT_HOST_BUCKET",
    "ENVIRONMENT_HOST_S3_ACCESS_KEY_ID",
    "ENVIRONMENT_HOST_S3_SECRET_ACCESS_KEY",
)
REQUIRED_SETTINGS = (
    "SPRITES_TOKEN",
    "ENVIRONMENT_TOKEN_KEY",
    "ENVIRONMENT_BUNDLE_SHA256",
    *RELEASE_BUCKET_SETTINGS,
)

# Unpack the pinned bundle into its own release directory and point `current`
# at it. A release directory appears only complete (it is renamed into place),
# so one that exists is never downloaded again. Prints whether anything changed.
INSTALL_BUNDLE = r"""
set -euo pipefail
release="$THEOREM_ROOT/releases/$BUNDLE_SHA256"
marker="$THEOREM_ROOT/bundle.sha256"
if [ -d "$release" ] && [ "$(cat "$marker" 2>/dev/null)" = "$BUNDLE_SHA256" ] \
  && [ "$(readlink "$THEOREM_ROOT/current")" = "$release" ]; then
  echo unchanged
  exit 0
fi
mkdir -p "$THEOREM_ROOT/incoming" "$THEOREM_ROOT/releases"
if [ ! -d "$release" ]; then
  tar_file="$THEOREM_ROOT/incoming/$BUNDLE_SHA256.tar"
  staging="$(mktemp -d "$THEOREM_ROOT/releases/.incoming.XXXXXX")"
  trap 'rm -rf "$staging" "$tar_file"' EXIT
  curl -fsSL --retry 3 -o "$tar_file" "$BUNDLE_URL"
  echo "$BUNDLE_SHA256  $tar_file" | sha256sum --check --quiet -
  tar -xf "$tar_file" -C "$staging"
  for entry in $BUNDLE_ENTRIES; do
    if [ ! -f "$staging/$entry" ]; then
      echo "the bundle has no $entry" >&2
      exit 1
    fi
  done
  mv "$staging" "$release"
  rm -f "$tar_file"
  trap - EXIT
fi
previous="$(readlink "$THEOREM_ROOT/current" || true)"
ln -sfn "$release" "$THEOREM_ROOT/current"
printf '%s\n' "$BUNDLE_SHA256" > "$marker.tmp"
mv -f "$marker.tmp" "$marker"
# Keep the release just replaced, for a process still running from it; drop older ones.
for old in "$THEOREM_ROOT"/releases/*; do
  if [ "$old" != "$release" ] && [ "$old" != "$previous" ]; then
    rm -rf "$old"
  fi
done
echo installed
"""


class EnvironmentProvisioningError(RuntimeError):
    """The environment could not be built; the host row records why."""


def missing_settings(names: tuple[str, ...] = REQUIRED_SETTINGS) -> list[str]:
    return [name for name in names if not getattr(settings, name)]


def environment_configured() -> bool:
    return not missing_settings()


def bundle_object_key(sha256: str) -> str:
    return f"{BUNDLE_OBJECT_PREFIX}/{sha256}"


def release_bucket_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.ENVIRONMENT_HOST_S3_ENDPOINT_URL,
        region_name=settings.ENVIRONMENT_HOST_S3_REGION,
        aws_access_key_id=settings.ENVIRONMENT_HOST_S3_ACCESS_KEY_ID,
        aws_secret_access_key=settings.ENVIRONMENT_HOST_S3_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def bundle_download_url(client=None) -> str:
    """A link to the pinned bundle, valid for ``BUNDLE_DOWNLOAD_SECONDS``."""
    client = client or release_bucket_client()
    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.ENVIRONMENT_HOST_BUCKET,
            "Key": bundle_object_key(settings.ENVIRONMENT_BUNDLE_SHA256),
        },
        ExpiresIn=BUNDLE_DOWNLOAD_SECONDS,
        HttpMethod="GET",
    )


def sprite_name(tenant: Tenant) -> str:
    # Sprite names are lowercase and unique in the org; the slug is neither.
    return f"{settings.ENVIRONMENT_SPRITE_PREFIX}-{tenant.id.hex[:16]}"


def _derive(tenant: Tenant, purpose: str) -> str:
    digest = hmac.new(
        settings.ENVIRONMENT_TOKEN_KEY.encode(),
        f"theorem-environment-{purpose}:{tenant.id}".encode(),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def host_token(tenant: Tenant) -> str:
    """The bearer the environment host admits."""
    return _derive(tenant, "host")


def codemode_token(tenant: Tenant) -> str:
    """The bearer code mode admits."""
    return _derive(tenant, "codemode")


def credential_fingerprint(tenant: Tenant) -> str:
    """Names the pair of credentials without revealing either."""
    pair = f"{host_token(tenant)}\n{codemode_token(tenant)}"
    return hashlib.sha256(pair.encode()).hexdigest()[:16]


def begin_environment(tenant: Tenant) -> EnvironmentHost:
    """Record the sprite a tenant will get, before anything exists on Sprites."""
    host, _ = EnvironmentHost.objects.get_or_create(
        tenant=tenant, defaults={"sprite": sprite_name(tenant)}
    )
    return host


def provision_environment(
    host: EnvironmentHost,
    sprites: SpritesClient,
    *,
    http: httpx.Client | None = None,
    poll_interval_seconds: float = 2.0,
) -> EnvironmentHost:
    """Build the tenant's sprite, or bring a serving one up to the pinned bundle.

    Safe to run again, and after a run that stopped anywhere: the sprite is
    found by name, the bundle and credentials are replaced only when they
    differ, and the services restart until a start of exactly this bundle,
    these credentials and these definitions has been recorded on the sprite.
    Raises ``SpriteError`` for a failure worth retrying and
    ``EnvironmentProvisioningError`` otherwise.
    """
    host = EnvironmentHost.objects.select_related("tenant").get(pk=host.pk)
    if host.state == EnvironmentHost.State.FAILED:
        host.retry()
    tenant = host.tenant

    sprite = _ensure_sprite(sprites, host.sprite, tenant)
    _install_bundle(sprite)
    _write_secret(sprite, HOST_TOKEN_PATH, host_token(tenant))
    _write_secret(sprite, CODEMODE_TOKEN_PATH, codemode_token(tenant))
    _remove_legacy_service(sprite)
    _ensure_services(sprite, tenant)
    if not sprite.url:
        raise EnvironmentProvisioningError(f"sprite {host.sprite} has no URL")
    _wait_ready(sprite.url, tenant, http, poll_interval_seconds)

    host.mark_serving(
        url=sprite.url,
        bundle_sha256=settings.ENVIRONMENT_BUNDLE_SHA256,
        credential_fingerprint=credential_fingerprint(tenant),
    )
    return host


def store_binding(tenant: Tenant) -> dict | None:
    """What the tenant's store machine needs to reach its sprite, or None while
    there is no serving sprite holding the current credentials."""
    host = EnvironmentHost.objects.filter(
        tenant=tenant, state=EnvironmentHost.State.SERVING
    ).first()
    if host is None or not host.url or not settings.ENVIRONMENT_TOKEN_KEY:
        return None
    fingerprint = credential_fingerprint(tenant)
    if host.credential_fingerprint != fingerprint:
        # The key changed and the sprite has not been given the new
        # credentials yet; the store keeps the ones the sprite still accepts.
        return None
    hosts = {
        tenant.slug: {
            "url": host.url,
            "token_file": STORE_TOKEN_FILE,
            "codemode_token_file": STORE_CODEMODE_TOKEN_FILE,
        }
    }
    return {
        "env": {"THEOREM_ENVIRONMENT_HOSTS_FILE": STORE_HOSTS_FILE},
        # A fingerprint, so rotated credentials change the config and the
        # machine restarts with the new secrets.
        "metadata": {"theorem_environment_credentials": fingerprint},
        "files": [
            {
                "guest_path": STORE_HOSTS_FILE,
                "raw_value": base64.b64encode(json.dumps(hosts, sort_keys=True).encode()).decode(),
            },
            {"guest_path": STORE_TOKEN_FILE, "secret_name": STORE_TOKEN_SECRET},
            {"guest_path": STORE_CODEMODE_TOKEN_FILE, "secret_name": STORE_CODEMODE_TOKEN_SECRET},
        ],
    }


def with_binding(config: dict, binding: dict) -> dict:
    owned = {f["guest_path"] for f in binding["files"]}
    return {
        **config,
        "env": {**(config.get("env") or {}), **binding["env"]},
        "metadata": {**(config.get("metadata") or {}), **binding["metadata"]},
        "files": [f for f in config.get("files") or [] if f.get("guest_path") not in owned]
        + binding["files"],
    }


def set_store_tokens(client: FlyMachinesClient, app: str, tenant: Tenant) -> int:
    """Put both credentials where the store's files read them; returns the
    secrets version that carries both."""
    client.set_secret(app, STORE_TOKEN_SECRET, _b64(host_token(tenant)))
    return client.set_secret(app, STORE_CODEMODE_TOKEN_SECRET, _b64(codemode_token(tenant)))


def bind_store(tenant: Tenant, client: FlyMachinesClient) -> bool:
    """Point the tenant's store at its sprite; True when the machine changed.

    Does nothing until both the store and the sprite serve, while the sprite
    holds older credentials than the key derives, for a store an operator
    registered by hand (its fly.toml carries the binding), or when the machine
    already carries this binding.
    """
    binding = store_binding(tenant)
    placement = (
        StorePlacement.objects.select_related("store").filter(tenant=tenant).first()
    )
    if (
        binding is None
        or placement is None
        or placement.state != StorePlacement.State.SERVING
        or not placement.store.fly_app
    ):
        return False
    app = placement.store.fly_app
    machine = client.find_machine(app, STORE_MACHINE_NAME)
    if machine is None:
        return False
    config = machine.get("config") or {}
    if _carries(config, binding):
        return False
    version = set_store_tokens(client, app, tenant)
    client.update_machine(
        app, machine["id"], config=with_binding(config, binding), min_secrets_version=version
    )
    client.wait_started(app, machine["id"])
    return True


def sprites_client_from_settings() -> SpritesClient:
    return SpritesClient(settings.SPRITES_TOKEN, base_url=settings.SPRITES_API_BASE)


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def _carries(config: dict, binding: dict) -> bool:
    # Compare only what the binding owns: Fly fills in the rest of the config.
    env = config.get("env") or {}
    metadata = config.get("metadata") or {}
    files = {f.get("guest_path"): f for f in config.get("files") or []}
    return (
        all(env.get(k) == v for k, v in binding["env"].items())
        and all(metadata.get(k) == v for k, v in binding["metadata"].items())
        and all(
            all(files.get(f["guest_path"], {}).get(k) == v for k, v in f.items())
            for f in binding["files"]
        )
    )


def _ensure_sprite(sprites: SpritesClient, name: str, tenant: Tenant):
    try:
        sprite = sprites.get_sprite(name)
    except NotFoundError:
        return sprites.create_sprite(
            name,
            url_settings=URLSettings(auth="public"),
            labels=["theorem-environment", f"tenant-{tenant.id.hex[:16]}"],
            wait_for_capacity=True,
        )
    if sprite.url_settings is None or sprite.url_settings.auth != "public":
        sprite.update_url_settings(URLSettings(auth="public"))
    return sprite


def _install_bundle(sprite) -> None:
    result = sprite.run(
        "bash",
        "-c",
        INSTALL_BUNDLE,
        env={
            "THEOREM_ROOT": THEOREM_ROOT,
            "BUNDLE_URL": bundle_download_url(),
            "BUNDLE_SHA256": settings.ENVIRONMENT_BUNDLE_SHA256,
            "BUNDLE_ENTRIES": " ".join(BUNDLE_ENTRIES),
        },
        capture_output=True,
        timeout=600,
    )
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode(errors="replace")
        raise EnvironmentProvisioningError(
            f"installing the sprite bundle on {sprite.name} failed: {stderr[-500:]}"
        )


def _read(sprite, path: str) -> str | None:
    try:
        return (sprite.filesystem("/") / path).read_text().strip()
    except FileNotFoundError_:
        return None


def _write_secret(sprite, path: str, value: str) -> None:
    if _read(sprite, path) != value:
        (sprite.filesystem("/") / path).write_text(value, mode=0o600)


def _service_definitions(tenant: Tenant) -> dict[str, dict]:
    """The bundle's services, keyed in ``SERVICES`` order."""
    run = f"{THEOREM_ROOT}/current/run"
    return {
        "ide-host": {
            "cmd": f"{run}/ide-host",
            "args": [],
            "needs": [],
            "http_port": None,
            "dir": HOME,
            "env": {
                "RUST_LOG": "info",
                "SHELL": "/bin/bash",
                "THEOREM_ENVIRONMENT_LISTEN": "127.0.0.1:8081",
                "THEOREM_ENVIRONMENT_ROOT": WORKSPACE_ROOT,
                "THEOREM_ENVIRONMENT_TENANT": tenant.slug,
                "THEOREM_ENVIRONMENT_TOKEN_FILE": HOST_TOKEN_PATH,
            },
        },
        "codemode": {
            "cmd": f"{run}/codemode",
            "args": [],
            "needs": [],
            "http_port": None,
            "dir": HOME,
            "env": {
                "RUST_LOG": "info",
                "PORT": "8082",
                "THEOREM_CODEMODE_TOKEN_FILE": CODEMODE_TOKEN_PATH,
            },
        },
        "router": {
            "cmd": f"{run}/router",
            "args": [],
            "needs": ["ide-host", "codemode"],
            "http_port": ROUTER_PORT,
            "dir": HOME,
            "env": {},
        },
    }


def _started_record(tenant: Tenant) -> str:
    """What a start of the current services must have run: the bundle, the
    credentials and the definitions. Any change to one restarts them all."""
    material = {
        "bundle": settings.ENVIRONMENT_BUNDLE_SHA256,
        "credentials": credential_fingerprint(tenant),
        "services": _service_definitions(tenant),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _defined(service) -> dict | None:
    """A service as Sprites reports it, in ``_service_definitions``' shape.

    Sprites answers an empty args or needs list with null.
    """
    if service is None:
        return None
    return {
        "cmd": service.cmd,
        "args": service.args or [],
        "needs": service.needs or [],
        "http_port": service.http_port,
        "dir": service.dir,
        "env": service.env,
    }


def _service(sprite, name: str):
    try:
        return sprite.get_service(name)
    except APIError as error:
        if error.status_code != 404:
            raise
        return None


def _running(service) -> bool:
    return (
        service is not None
        and service.state is not None
        and service.state.status in ("running", "starting")
    )


def _remove_legacy_service(sprite) -> None:
    legacy = _service(sprite, LEGACY_SERVICE)
    if legacy is None:
        return
    if _running(legacy):
        sprite.stop_service(LEGACY_SERVICE).close()
    sprite.delete_service(LEGACY_SERVICE)


def _ensure_services(sprite, tenant: Tenant) -> None:
    # The restart stays pending on the sprite until it has happened, so a run
    # that stops halfway leaves it for the next run rather than forgetting it.
    record = _started_record(tenant)
    restart = _read(sprite, STARTED_PATH) != record
    definitions = _service_definitions(tenant)
    for name, desired in definitions.items():
        if _defined(_service(sprite, name)) != desired:
            sprite.create_service(name, **desired).close()
    if restart:
        for name in reversed(definitions):
            if _running(_service(sprite, name)):
                sprite.stop_service(name).close()
    for name in definitions:
        if not _running(_service(sprite, name)):
            sprite.start_service(name).close()
    if restart:
        (sprite.filesystem("/") / STARTED_PATH).write_text(record, mode=0o600)


def _wait_ready(
    url: str,
    tenant: Tenant,
    http: httpx.Client | None,
    poll_interval_seconds: float,
) -> None:
    """Wait until code mode and the host each refuse a stranger and answer
    their tenant's credential.

    The first request also wakes a sleeping sprite.
    """
    client = http or httpx.Client(timeout=30.0)
    base = url.rstrip("/")
    codemode = {"Authorization": f"Bearer {codemode_token(tenant)}"}
    bound = {"Authorization": f"Bearer {host_token(tenant)}", "x-theorem-tenant": tenant.slug}
    checks = (
        ("code mode health", CODEMODE_HEALTH_PATH, {}, 200),
        ("code mode anonymous", CODEMODE_PROBE_PATH, {}, 401),
        # The probe session does not exist: 404 shows the credential was accepted.
        ("code mode token", CODEMODE_PROBE_PATH, codemode, 404),
        ("host anonymous", OPERATIONS_PATH, {}, 401),
        ("host token", OPERATIONS_PATH, bound, 200),
    )
    deadline = monotonic() + settings.ENVIRONMENT_READY_TIMEOUT_SECONDS
    try:
        while True:
            try:
                answers = [
                    (label, client.get(base + path, headers=headers).status_code, expected)
                    for label, path, headers, expected in checks
                ]
                if all(status == expected for _, status, expected in answers):
                    return
                seen = ", ".join(f"{label} {status}" for label, status, _ in answers)
            except httpx.HTTPError as error:
                seen = str(error)
            if monotonic() >= deadline:
                raise EnvironmentProvisioningError(
                    f"{url} did not answer as {tenant.slug}'s sprite within "
                    f"{settings.ENVIRONMENT_READY_TIMEOUT_SECONDS}s ({seen})"
                )
            sleep(poll_interval_seconds)
    finally:
        if http is None:
            client.close()
