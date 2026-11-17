"""Signup gives each tenant its own Fly Sprite, running the environment host and
code mode from the sprite bundle, and binds the tenant's store to it."""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import tarfile
from hashlib import sha256
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
from botocore.stub import ANY, Stubber
from django.core.management import call_command
from django.core.management.base import CommandError
from sprites import FileNotFoundError_, NotFoundError, URLSettings
from sprites.exceptions import APIError
from sprites.exec import CompletedProcess
from sprites.services import ServiceStream
from sprites.types import ServiceState, ServiceWithState

from apps.tenancy.environment import (
    BUNDLE_ENTRIES,
    CODEMODE_PROBE_PATH,
    CODEMODE_TOKEN_PATH,
    HOST_TOKEN_PATH,
    INSTALL_BUNDLE,
    LEGACY_SERVICE,
    OPERATIONS_PATH,
    SERVICES,
    STARTED_PATH,
    STORE_CODEMODE_TOKEN_FILE,
    STORE_CODEMODE_TOKEN_SECRET,
    STORE_HOSTS_FILE,
    STORE_TOKEN_FILE,
    STORE_TOKEN_SECRET,
    THEOREM_ROOT,
    EnvironmentProvisioningError,
    begin_environment,
    bind_store,
    bundle_object_key,
    codemode_token,
    credential_fingerprint,
    host_token,
    provision_environment,
    sprite_name,
)
from apps.tenancy.models import EnvironmentHost, Store, StorePlacement, Tenant
from apps.tenancy.provisioning import provision, store_app_name
from tests.test_store_provisioning import VERIFYING_KEY, FakeFly

pytestmark = pytest.mark.django_db

BUNDLE_SHA = "1" * 64


class FakeSprite:
    """One sprite as the Sprites API and the bundle's services behave on it."""

    def __init__(self, name: str, *, auth: str = "public"):
        self.name = name
        self.url = f"https://{name}-bthwy.sprites.app"
        self.url_settings = URLSettings(auth=auth)
        self.labels: list[str] = []
        self.files: dict[str, tuple[str, int]] = {}
        self.bundle_sha: str | None = None
        self.install_url: str | None = None
        self.install_fails = False
        self.services: dict[str, dict] = {}
        self.status: dict[str, str] = {}
        # The bundle each running service was started from.
        self.running: dict[str, str | None] = {}
        self.fail_get_service = 0
        self.fail_start: set[str] = set()
        self.installs = self.writes = self.definitions = self.starts = self.stops = 0
        self.deleted: list[str] = []

    def update_url_settings(self, settings: URLSettings) -> None:
        self.url_settings = settings

    def run(self, *args, env, capture_output, timeout):
        assert args[:3] == ("bash", "-c", INSTALL_BUNDLE) and capture_output
        assert env["THEOREM_ROOT"] == THEOREM_ROOT
        assert env["BUNDLE_ENTRIES"].split() == list(BUNDLE_ENTRIES)
        self.install_url = env["BUNDLE_URL"]
        if self.bundle_sha == env["BUNDLE_SHA256"]:
            return CompletedProcess(list(args), 0, b"unchanged\n", b"")
        if self.install_fails:
            return CompletedProcess(list(args), 1, b"", b"sha256sum: WARNING: 1 did NOT match")
        self.bundle_sha = env["BUNDLE_SHA256"]
        self.installs += 1
        return CompletedProcess(list(args), 0, b"installed\n", b"")

    def filesystem(self, working_dir: str):
        assert working_dir == "/"
        return FakeFilesystem(self)

    def get_service(self, name: str) -> ServiceWithState:
        if self.fail_get_service:
            self.fail_get_service -= 1
            raise APIError("Sprites answered 502", status_code=502)
        if name not in self.services:
            raise APIError(f"Service not found: {name}", status_code=404)
        return ServiceWithState(
            name=name,
            **self.services[name],
            state=ServiceState(name, self.status.get(name, "stopped")),
        )

    def create_service(self, name: str, **definition) -> ServiceStream:
        assert name in SERVICES
        self.services[name] = {
            "args": [],
            "needs": [],
            "http_port": None,
            "env": {},
            "dir": None,
            **definition,
        }
        self.definitions += 1
        return ServiceStream([])

    def delete_service(self, name: str) -> None:
        if name not in self.services:
            raise APIError(f"Service not found: {name}", status_code=404)
        del self.services[name]
        self.status.pop(name, None)
        self.deleted.append(name)

    def stop_service(self, name: str) -> ServiceStream:
        self.status[name] = "stopped"
        self.running[name] = None
        self.stops += 1
        return ServiceStream([])

    def start_service(self, name: str) -> ServiceStream:
        assert name in self.services
        if name in self.fail_start:
            self.fail_start.discard(name)
            raise APIError(f"{name} exited during start", status_code=500)
        for need in self.services[name]["needs"]:
            assert self.status.get(need) == "running", f"{name} started before {need}"
        self.status[name] = "running"
        self.running[name] = self.bundle_sha
        self.starts += 1
        return ServiceStream([])

    def _up(self, *names: str) -> bool:
        return all(self.status.get(n) == "running" for n in ("router", *names))

    def _secret(self, service: str, variable: str) -> str:
        return self.files[self.services[service]["env"][variable]][0]

    def answer(self, request: httpx.Request) -> httpx.Response:
        """Caddy, codemode-serve and theorem-ide-host as the bundle runs them."""
        path = request.url.path
        bearer = request.headers.get("authorization")
        if path == "/codemode/healthz":
            if not self._up("codemode"):
                return httpx.Response(502, text="no service")
            return httpx.Response(200, text="ok")
        if path == CODEMODE_PROBE_PATH:
            if not self._up("codemode"):
                return httpx.Response(502, text="no service")
            token = self._secret("codemode", "THEOREM_CODEMODE_TOKEN_FILE")
            if bearer != f"Bearer {token}":
                return httpx.Response(401, json={"error": "unauthorized"})
            return httpx.Response(404, json={"error": "no such session"})
        if path == OPERATIONS_PATH:
            if not self._up("ide-host"):
                return httpx.Response(502, text="no service")
            token = self._secret("ide-host", "THEOREM_ENVIRONMENT_TOKEN_FILE")
            tenant = self.services["ide-host"]["env"]["THEOREM_ENVIRONMENT_TENANT"]
            if bearer != f"Bearer {token}" or request.headers.get("x-theorem-tenant") != tenant:
                return httpx.Response(401, json={"code": "host_unauthorized"})
            return httpx.Response(200, json={"operations": []})
        return httpx.Response(404)


class FakePath:
    def __init__(self, sprite: FakeSprite, path: str):
        self.sprite, self.path = sprite, path

    def read_text(self) -> str:
        if self.path not in self.sprite.files:
            raise FileNotFoundError_("read", self.path)
        return self.sprite.files[self.path][0]

    def write_text(self, data: str, mode: int) -> None:
        self.sprite.files[self.path] = (data, mode)
        self.sprite.writes += 1


class FakeFilesystem:
    def __init__(self, sprite: FakeSprite):
        self.sprite = sprite

    def __truediv__(self, path: str) -> FakePath:
        return FakePath(self.sprite, path)


class FakeSprites:
    def __init__(self):
        self.sprites: dict[str, FakeSprite] = {}
        self.created: list[dict] = []

    def get_sprite(self, name: str) -> FakeSprite:
        if name not in self.sprites:
            raise NotFoundError(f"Resource not found for get sprite '{name}'")
        return self.sprites[name]

    def create_sprite(self, name, *, url_settings, labels, wait_for_capacity) -> FakeSprite:
        sprite = FakeSprite(name, auth=url_settings.auth)
        sprite.labels = labels
        self.sprites[name] = sprite
        self.created.append({"name": name, "labels": labels, "wait": wait_for_capacity})
        return sprite

    def http(self) -> httpx.Client:
        def route(request: httpx.Request) -> httpx.Response:
            name = request.url.host.removesuffix("-bthwy.sprites.app")
            return self.sprites[name].answer(request)

        return httpx.Client(transport=httpx.MockTransport(route))

    def close(self) -> None:
        pass


@pytest.fixture
def env_configured(settings):
    settings.SPRITES_TOKEN = "sprites-test-token"
    settings.ENVIRONMENT_HOST_BUCKET = "theorem-ide-host-releases-test"
    settings.ENVIRONMENT_HOST_S3_ACCESS_KEY_ID = "release-test-key"
    settings.ENVIRONMENT_HOST_S3_SECRET_ACCESS_KEY = "release-test-secret"
    settings.ENVIRONMENT_BUNDLE_SHA256 = BUNDLE_SHA
    settings.ENVIRONMENT_TOKEN_KEY = "environment-test-key"
    settings.ENVIRONMENT_READY_TIMEOUT_SECONDS = 5
    settings.FLY_API_TOKEN = ""
    return settings


@pytest.fixture
def both_configured(env_configured):
    env_configured.FLY_API_TOKEN = "fly-test-token"
    env_configured.FLY_ORG_SLUG = "theorem-620"
    env_configured.STORE_IMAGE = "registry.fly.io/theorem-store-image@sha256:" + "0" * 64
    env_configured.STORE_SESSION_VERIFYING_KEYS_HEX = VERIFYING_KEY
    return env_configured


def _tenant() -> Tenant:
    return Tenant.objects.create(slug=f"Signup-{uuid4()}", display_name="New user")


def _provision(tenant: Tenant, sprites: FakeSprites) -> EnvironmentHost:
    return provision_environment(
        begin_environment(tenant), sprites, http=sprites.http(), poll_interval_seconds=0
    )


def test_signup_without_environment_config_leaves_tenant_without_one(settings):
    settings.SPRITES_TOKEN = ""
    tenant = _tenant()
    assert not EnvironmentHost.objects.filter(tenant=tenant).exists()


def test_signup_records_the_sprite_and_enqueues_both_builds(
    both_configured, django_capture_on_commit_callbacks, monkeypatch
):
    queued: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "apps.tenancy.tasks.provision_tenant_store.delay", lambda t: queued.append(("store", t))
    )
    monkeypatch.setattr(
        "apps.tenancy.tasks.provision_tenant_environment.delay",
        lambda t: queued.append(("environment", t)),
    )
    with django_capture_on_commit_callbacks(execute=True):
        tenant = _tenant()

    host = EnvironmentHost.objects.get(tenant=tenant)
    assert host.state == EnvironmentHost.State.PROVISIONING
    assert host.sprite == sprite_name(tenant) == f"theorem-env-{tenant.id.hex[:16]}"
    assert sorted(queued) == [("environment", str(tenant.pk)), ("store", str(tenant.pk))]


def test_provision_builds_a_public_sprite_running_the_bundle(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    host = _provision(tenant, sprites)

    sprite = sprites.sprites[sprite_name(tenant)]
    assert sprites.created == [
        {
            "name": sprite.name,
            "labels": ["theorem-environment", f"tenant-{tenant.id.hex[:16]}"],
            "wait": True,
        }
    ]
    assert sprite.url_settings.auth == "public"
    assert sprite.bundle_sha == BUNDLE_SHA
    assert sprite.files[HOST_TOKEN_PATH] == (host_token(tenant), 0o600)
    assert sprite.files[CODEMODE_TOKEN_PATH] == (codemode_token(tenant), 0o600)

    ide_host, codemode, router = (sprite.services[n] for n in ("ide-host", "codemode", "router"))
    assert ide_host["cmd"] == f"{THEOREM_ROOT}/current/run/ide-host"
    assert ide_host["env"]["THEOREM_ENVIRONMENT_TENANT"] == tenant.slug
    assert ide_host["env"]["THEOREM_ENVIRONMENT_TOKEN_FILE"] == HOST_TOKEN_PATH
    assert ide_host["env"]["THEOREM_ENVIRONMENT_LISTEN"] == "127.0.0.1:8081"
    assert ide_host["http_port"] is None
    assert codemode["cmd"] == f"{THEOREM_ROOT}/current/run/codemode"
    assert codemode["env"]["PORT"] == "8082"
    assert codemode["env"]["THEOREM_CODEMODE_TOKEN_FILE"] == CODEMODE_TOKEN_PATH
    assert router["cmd"] == f"{THEOREM_ROOT}/current/run/router"
    assert router["http_port"] == 8080
    assert router["needs"] == ["ide-host", "codemode"]
    assert sprite.running == {name: BUNDLE_SHA for name in SERVICES}

    assert host.state == EnvironmentHost.State.SERVING
    assert host.url == sprite.url
    assert host.bundle_sha256 == BUNDLE_SHA
    assert host.credential_fingerprint == credential_fingerprint(tenant)


def test_the_sprite_downloads_its_pinned_bundle_through_a_short_lived_link(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    _provision(tenant, sprites)

    link = sprites.sprites[sprite_name(tenant)].install_url
    url = urlparse(link)
    assert url.hostname.endswith("fly.storage.tigris.dev")
    assert url.path.endswith(f"/{bundle_object_key(BUNDLE_SHA)}")
    assert parse_qs(url.query)["X-Amz-Expires"] == ["600"]
    assert "release-test-secret" not in link


def test_the_credentials_are_derived_never_stored(env_configured):
    tenant, other = _tenant(), _tenant()
    host_secret, codemode_secret = host_token(tenant), codemode_token(tenant)
    for token in (host_secret, codemode_secret):
        assert len(token) == 43 and len(token.encode()) >= 32
    assert len({host_secret, codemode_secret, host_token(other), codemode_token(other)}) == 4
    host = _provision(tenant, FakeSprites())
    stored = json.dumps(
        {f.name: str(getattr(host, f.attname)) for f in EnvironmentHost._meta.concrete_fields}
    )
    assert host_secret not in stored and codemode_secret not in stored
    env_configured.ENVIRONMENT_TOKEN_KEY = "rotated-key"
    assert host_token(tenant) != host_secret
    assert codemode_token(tenant) != codemode_secret


def test_a_rerun_with_nothing_changed_touches_nothing(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    _provision(tenant, sprites)
    sprite = sprites.sprites[sprite_name(tenant)]
    before = (sprite.installs, sprite.writes, sprite.definitions, sprite.starts, sprite.stops)

    host = _provision(tenant, sprites)

    assert (sprite.installs, sprite.writes, sprite.definitions, sprite.starts, sprite.stops) == before
    assert len(sprites.created) == 1
    assert host.state == EnvironmentHost.State.SERVING


def test_a_new_pinned_bundle_is_installed_and_every_service_restarted(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    _provision(tenant, sprites)
    sprite = sprites.sprites[sprite_name(tenant)]

    env_configured.ENVIRONMENT_BUNDLE_SHA256 = "2" * 64
    host = _provision(tenant, sprites)

    assert sprite.bundle_sha == "2" * 64
    assert sprite.installs == 2
    assert sprite.running == {name: "2" * 64 for name in SERVICES}
    assert host.bundle_sha256 == "2" * 64


def test_an_upgrade_interrupted_before_the_restart_restarts_on_the_retry(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    _provision(tenant, sprites)
    sprite = sprites.sprites[sprite_name(tenant)]

    env_configured.ENVIRONMENT_BUNDLE_SHA256 = "2" * 64
    sprite.fail_get_service = 1
    with pytest.raises(APIError):
        _provision(tenant, sprites)
    assert sprite.bundle_sha == "2" * 64
    assert sprite.running["ide-host"] == BUNDLE_SHA

    host = _provision(tenant, sprites)

    assert sprite.running == {name: "2" * 64 for name in SERVICES}
    assert host.bundle_sha256 == "2" * 64


def test_a_rotated_key_rewrites_both_credentials_and_restarts(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    _provision(tenant, sprites)
    sprite = sprites.sprites[sprite_name(tenant)]
    starts = sprite.starts

    env_configured.ENVIRONMENT_TOKEN_KEY = "rotated-key"
    host = _provision(tenant, sprites)

    assert sprite.files[HOST_TOKEN_PATH][0] == host_token(tenant)
    assert sprite.files[CODEMODE_TOKEN_PATH][0] == codemode_token(tenant)
    assert sprite.installs == 1
    assert sprite.starts == starts + len(SERVICES)
    assert host.credential_fingerprint == credential_fingerprint(tenant)


def test_the_restart_is_recorded_only_after_the_services_start(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    _provision(tenant, sprites)
    sprite = sprites.sprites[sprite_name(tenant)]
    started = sprite.files[STARTED_PATH][0]

    env_configured.ENVIRONMENT_BUNDLE_SHA256 = "2" * 64
    sprite.fail_start = {"router"}
    with pytest.raises(APIError):
        _provision(tenant, sprites)
    assert sprite.files[STARTED_PATH][0] == started
    assert sprite.running["ide-host"] == "2" * 64 and sprite.running["router"] is None

    _provision(tenant, sprites)
    assert sprite.files[STARTED_PATH][0] != started
    assert sprite.running == {name: "2" * 64 for name in SERVICES}


def test_the_bare_host_service_of_an_earlier_sprite_is_removed(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    sprite = FakeSprite(sprite_name(tenant))
    sprite.services[LEGACY_SERVICE] = {
        "cmd": "/home/sprite/.local/bin/theorem-ide-host",
        "args": ["--http"],
        "needs": [],
        "http_port": 8080,
        "env": {},
        "dir": "/home/sprite",
    }
    sprite.status[LEGACY_SERVICE] = "running"
    sprites.sprites[sprite.name] = sprite

    host = _provision(tenant, sprites)

    assert sprite.deleted == [LEGACY_SERVICE]
    assert sorted(sprite.services) == sorted(SERVICES)
    assert host.state == EnvironmentHost.State.SERVING


def test_an_existing_sprite_behind_org_auth_is_made_public(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    sprites.sprites[sprite_name(tenant)] = FakeSprite(sprite_name(tenant), auth="sprite")
    _provision(tenant, sprites)
    assert sprites.sprites[sprite_name(tenant)].url_settings.auth == "public"
    assert sprites.created == []


@pytest.mark.parametrize(
    ("open_path", "seen"),
    [
        (CODEMODE_PROBE_PATH, "code mode anonymous 200"),
        (OPERATIONS_PATH, "host anonymous 200"),
    ],
)
def test_a_sprite_that_admits_strangers_is_not_ready(env_configured, open_path, seen):
    tenant = _tenant()
    sprites = FakeSprites()
    env_configured.ENVIRONMENT_READY_TIMEOUT_SECONDS = 0

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == open_path:
            return httpx.Response(200, json={})
        name = request.url.host.removesuffix("-bthwy.sprites.app")
        return sprites.sprites[name].answer(request)

    with pytest.raises(EnvironmentProvisioningError, match=seen):
        provision_environment(
            begin_environment(tenant),
            sprites,
            http=httpx.Client(transport=httpx.MockTransport(answer)),
            poll_interval_seconds=0,
        )
    assert EnvironmentHost.objects.get(tenant=tenant).state == EnvironmentHost.State.PROVISIONING


def test_a_bundle_that_fails_its_checksum_is_refused(env_configured):
    tenant = _tenant()
    sprites = FakeSprites()
    sprite = FakeSprite(sprite_name(tenant))
    sprite.install_fails = True
    sprites.sprites[sprite.name] = sprite
    with pytest.raises(EnvironmentProvisioningError, match="did NOT match"):
        _provision(tenant, sprites)
    assert sprite.services == {}


def test_task_records_failure(env_configured, monkeypatch):
    from apps.tenancy import tasks

    tenant = _tenant()
    sprites = FakeSprites()
    sprite = FakeSprite(sprite_name(tenant))
    sprite.install_fails = True
    sprites.sprites[sprite.name] = sprite
    monkeypatch.setattr(tasks, "sprites_client_from_settings", lambda: sprites)
    with pytest.raises(EnvironmentProvisioningError):
        tasks.provision_tenant_environment.apply(args=[str(tenant.pk)]).get()
    host = EnvironmentHost.objects.get(tenant=tenant)
    assert host.state == EnvironmentHost.State.FAILED
    assert "did NOT match" in host.last_error


def _serving_store(tenant: Tenant, fly: FakeFly) -> StorePlacement:
    return provision(StorePlacement.objects.get(tenant=tenant), fly.client(), poll_interval_seconds=0)


def _bound_hosts(config: dict) -> dict:
    files = {f["guest_path"]: f for f in config["files"]}
    return json.loads(base64.b64decode(files[STORE_HOSTS_FILE]["raw_value"]))


def _binding(host: EnvironmentHost) -> dict:
    """The entry the store's parser (theorem_environment.rs) accepts."""
    return {
        host.tenant.slug: {
            "url": host.url,
            "token_file": STORE_TOKEN_FILE,
            "codemode_token_file": STORE_CODEMODE_TOKEN_FILE,
        }
    }


def _store_secret(fly: FakeFly, tenant: Tenant, name: str) -> str:
    return base64.b64decode(fly.secrets[store_app_name(tenant)][name]).decode()


def test_an_environment_finished_after_the_store_binds_it(both_configured):
    tenant = _tenant()
    fly = FakeFly()
    _serving_store(tenant, fly)
    assert fly.updates == []
    host = _provision(tenant, FakeSprites())

    assert bind_store(tenant, fly.client()) is True

    app = store_app_name(tenant)
    assert _store_secret(fly, tenant, STORE_TOKEN_SECRET) == host_token(tenant)
    assert _store_secret(fly, tenant, STORE_CODEMODE_TOKEN_SECRET) == codemode_token(tenant)
    machine = fly.machines[app][0]
    config = machine["config"]
    assert machine["min_secrets_version"] == fly.secrets_version
    assert config["env"]["THEOREM_ENVIRONMENT_HOSTS_FILE"] == STORE_HOSTS_FILE
    assert _bound_hosts(config) == _binding(host)
    assert {"guest_path": STORE_TOKEN_FILE, "secret_name": STORE_TOKEN_SECRET} in config["files"]
    assert {
        "guest_path": STORE_CODEMODE_TOKEN_FILE,
        "secret_name": STORE_CODEMODE_TOKEN_SECRET,
    } in config["files"]
    assert host_token(tenant) not in json.dumps(config)
    assert codemode_token(tenant) not in json.dumps(config)
    # Everything else the store needs is still there.
    assert config["env"]["RUSTY_RED_TENANT_ALLOWLIST"] == tenant.slug

    assert bind_store(tenant, fly.client()) is False
    assert len(fly.updates) == 1


def test_a_store_built_after_the_environment_starts_bound(both_configured):
    tenant = _tenant()
    host = _provision(tenant, FakeSprites())
    fly = FakeFly()
    _serving_store(tenant, fly)

    machine = fly.machines[store_app_name(tenant)][0]
    assert fly.updates == []
    assert machine["min_secrets_version"] == fly.secrets_version == 2
    assert _bound_hosts(machine["config"]) == _binding(host)
    assert host_token(tenant) not in json.dumps(machine["config"])


def test_a_rotated_key_waits_for_the_sprite_before_rebinding_the_store(both_configured):
    tenant = _tenant()
    fly = FakeFly()
    sprites = FakeSprites()
    _provision(tenant, sprites)
    _serving_store(tenant, fly)
    old_host, old_codemode = host_token(tenant), codemode_token(tenant)

    both_configured.ENVIRONMENT_TOKEN_KEY = "rotated-key"
    # The sprite still holds the old credentials, so the store keeps them too.
    assert bind_store(tenant, fly.client()) is False
    _serving_store(tenant, fly)
    assert _store_secret(fly, tenant, STORE_TOKEN_SECRET) == old_host
    assert _store_secret(fly, tenant, STORE_CODEMODE_TOKEN_SECRET) == old_codemode

    _provision(tenant, sprites)
    assert bind_store(tenant, fly.client()) is True
    assert _store_secret(fly, tenant, STORE_TOKEN_SECRET) == host_token(tenant)
    assert _store_secret(fly, tenant, STORE_CODEMODE_TOKEN_SECRET) == codemode_token(tenant)


def test_a_store_registered_by_hand_is_not_rebound(both_configured):
    tenant = _tenant()
    StorePlacement.objects.filter(tenant=tenant).delete()
    store = Store.objects.create(
        slug=f"hand-{uuid4().hex[:8]}", http_url="http://personal.internal:8380",
        grpc_url="http://personal.internal:8380",
    )
    StorePlacement.objects.create(tenant=tenant, store=store)
    _provision(tenant, FakeSprites())
    fly = FakeFly()
    assert bind_store(tenant, fly.client()) is False
    assert fly.calls == []


def test_nothing_binds_before_the_environment_serves(both_configured):
    tenant = _tenant()
    fly = FakeFly()
    _serving_store(tenant, fly)
    begin_environment(tenant)
    assert bind_store(tenant, fly.client()) is False
    assert fly.secrets == {}


def test_command_provisions_and_binds(both_configured, monkeypatch):
    from apps.tenancy.management.commands import provision_tenant_environment as command

    tenant = _tenant()
    fly = FakeFly()
    _serving_store(tenant, fly)
    sprites = FakeSprites()
    monkeypatch.setattr(command, "sprites_client_from_settings", lambda: sprites)
    monkeypatch.setattr(command, "client_from_settings", fly.client)
    real = command.provision_environment
    monkeypatch.setattr(
        command,
        "provision_environment",
        lambda host, client: real(host, client, http=sprites.http(), poll_interval_seconds=0),
    )
    call_command("provision_tenant_environment", tenant.slug)
    assert EnvironmentHost.objects.get(tenant=tenant).state == EnvironmentHost.State.SERVING
    assert len(fly.updates) == 1


def test_command_refuses_without_configuration(settings):
    settings.SPRITES_TOKEN = ""
    with pytest.raises(CommandError, match="SPRITES_TOKEN"):
        call_command("provision_tenant_environment", "Travis-Gilbert")


# A minimal x86-64 ELF header: class 64, little-endian, version, then e_type at
# offset 16 and e_machine 62 at offset 18.
def _elf(e_type: int = 2, machine: int = 62) -> bytes:
    return (
        b"\x7fELF\x02\x01\x01"
        + b"\x00" * 9
        + e_type.to_bytes(2, "little")
        + machine.to_bytes(2, "little")
        + b"\x00" * 44
    )


def _bundle(
    tmp_path,
    *,
    skip: str | None = None,
    override: dict[str, tuple[bytes, int]] | None = None,
) -> tuple:
    """A sprite bundle laid out as build.sh writes it (``./`` entries)."""
    members: dict[str, tuple[bytes, int]] = {}
    for entry in BUNDLE_ENTRIES:
        if entry.startswith("bin/"):
            members[entry] = (_elf(3) + entry.encode(), 0o755)
        elif entry.startswith("run/"):
            members[entry] = (b"#!/bin/bash\nexec true\n", 0o755)
        else:
            members[entry] = (b":8080 {\n\trespond 404\n}\n", 0o644)
    members.update(override or {})
    if skip:
        del members[skip]
    path = tmp_path / "theorem-sprite-bundle.tar"
    with tarfile.open(path, "w") as tar:
        for name, (data, mode) in sorted(members.items()):
            info = tarfile.TarInfo(f"./{name}")
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    return path, sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("bash", "curl", "sha256sum", "tar")),
    reason="needs bash, curl, sha256sum and tar",
)
def test_the_install_script_unpacks_only_the_pinned_bundle(tmp_path):
    root = tmp_path / "theorem"
    bundle, pinned = _bundle(tmp_path)

    def install(digest: str, source=bundle) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", INSTALL_BUNDLE],
            env={
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "THEOREM_ROOT": str(root),
                "BUNDLE_URL": source.as_uri(),
                "BUNDLE_SHA256": digest,
                "BUNDLE_ENTRIES": " ".join(BUNDLE_ENTRIES),
            },
            capture_output=True,
            text=True,
        )

    first = install(pinned)
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip() == "installed"
    release = root / "releases" / pinned
    assert (root / "current").resolve() == release.resolve()
    assert (root / "bundle.sha256").read_text().strip() == pinned
    assert (release / "run" / "ide-host").stat().st_mode & 0o111
    assert sorted(p.name for p in (release / "bin").iterdir()) == sorted(
        e.removeprefix("bin/") for e in BUNDLE_ENTRIES if e.startswith("bin/")
    )

    assert install(pinned).stdout.strip() == "unchanged"

    refused = install("f" * 64)
    assert refused.returncode != 0
    assert (root / "current").resolve() == release.resolve()

    other = tmp_path / "incomplete"
    other.mkdir()
    incomplete, digest = _bundle(other, skip="bin/caddy")
    short = install(digest, incomplete)
    assert short.returncode != 0
    assert "bin/caddy" in short.stderr
    assert not (root / "releases" / digest).exists()
    assert (root / "current").resolve() == release.resolve()


@pytest.fixture
def release_bucket(env_configured, monkeypatch):
    from apps.tenancy.environment import release_bucket_client
    from apps.tenancy.management.commands import publish_environment_bundle as command

    client = release_bucket_client()
    monkeypatch.setattr(command, "release_bucket_client", lambda: client)
    with Stubber(client) as stub:
        yield stub
        stub.assert_no_pending_responses()


def test_publish_uploads_the_bundle_under_its_digest(release_bucket, tmp_path, capsys):
    bundle, digest = _bundle(tmp_path)
    where = {"Bucket": "theorem-ide-host-releases-test", "Key": bundle_object_key(digest)}
    release_bucket.add_client_error(
        "head_object", service_error_code="404", http_status_code=404, expected_params=where
    )
    release_bucket.add_response(
        "put_object", {}, {**where, "Body": ANY, "ContentType": "application/x-tar"}
    )
    call_command("publish_environment_bundle", str(bundle))
    out = capsys.readouterr().out
    assert f"published {bundle_object_key(digest)}" in out
    assert f"ENVIRONMENT_BUNDLE_SHA256={digest}" in out


def test_publishing_a_published_bundle_uploads_nothing(release_bucket, tmp_path, capsys):
    bundle, digest = _bundle(tmp_path)
    release_bucket.add_response(
        "head_object",
        {"ContentLength": bundle.stat().st_size},
        {"Bucket": "theorem-ide-host-releases-test", "Key": bundle_object_key(digest)},
    )
    call_command("publish_environment_bundle", str(bundle))
    assert "is already published" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        ({"skip": "bin/codemode-serve"}, "bin/codemode-serve is missing"),
        ({"skip": "Caddyfile"}, "Caddyfile is missing"),
        ({"override": {"bin/hc": (b"\xcf\xfa\xed\xfe" + b"\x00" * 60, 0o755)}}, "bin/hc is not"),
        ({"override": {"bin/caddy": (_elf(2, machine=183), 0o755)}}, "bin/caddy is not"),
        ({"override": {"bin/monty": (_elf(1), 0o755)}}, "bin/monty is not"),
        ({"override": {"bin/theorem-js-cell": (_elf(4), 0o755)}}, "bin/theorem-js-cell is not"),
        ({"override": {"bin/theorem-ide-host": (_elf(3), 0o644)}}, "bin/theorem-ide-host is not"),
        ({"override": {"run/router": (b"exec caddy\n", 0o755)}}, "run/router is not"),
        ({"override": {"run/codemode": (b"#!/bin/bash\n", 0o644)}}, "run/codemode is not"),
    ],
)
def test_publish_refuses_a_bundle_a_sprite_could_not_run(
    release_bucket, tmp_path, change, complaint
):
    bundle, _ = _bundle(tmp_path, **change)
    with pytest.raises(CommandError, match=complaint):
        call_command("publish_environment_bundle", str(bundle))


def test_publish_refuses_something_that_is_not_a_tar(release_bucket, tmp_path):
    binary = tmp_path / "theorem-ide-host"
    binary.write_bytes(_elf(3))
    with pytest.raises(CommandError, match="not a tar"):
        call_command("publish_environment_bundle", str(binary))


def test_publish_refuses_without_the_bucket(settings, tmp_path):
    settings.ENVIRONMENT_HOST_BUCKET = ""
    bundle, _ = _bundle(tmp_path)
    with pytest.raises(CommandError, match="ENVIRONMENT_HOST_BUCKET"):
        call_command("publish_environment_bundle", str(bundle))
