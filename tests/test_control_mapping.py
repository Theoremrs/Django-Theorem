"""D4 and D6: the control mapping in the front's Valkey.

``control:org:{org_id}:tenant_slug`` and ``control:user:{id}:principal`` are
written with no TTL, to FRONT_VALKEY_URL rather than Django's VALKEY_URL, and
rebuilt from Postgres on worker start, on the beat schedule and by command.
"""

from __future__ import annotations

import io
import os
import uuid

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.identity import control_mapping, tasks
from apps.identity.control_mapping import (
    mapping_client,
    org_tenant_key,
    rewarm,
    user_principal_key,
    write_org_tenant,
)
from apps.identity.models import User
from apps.identity.sync import run_sync
from apps.tenancy.models import Tenant
from tests.workos_fakes import FakeFrontValkey, FakeWorkOS, organization_data

pytestmark = pytest.mark.django_db

FRONT_URL = "redis://front-valkey.invalid:6379/0"
PRINCIPAL = "identity:0123456789ab"


@pytest.fixture
def front(settings, monkeypatch):
    settings.FRONT_VALKEY_URL = FRONT_URL
    settings.VALKEY_URL = "redis://django-valkey.invalid:6379/0"
    valkey = FakeFrontValkey()
    opened: list[str] = []

    def open_front(url):
        opened.append(url)
        return valkey

    monkeypatch.setattr(control_mapping, "_redis", open_front)
    valkey.opened = opened
    return valkey


def _bound_tenant(org_id: str = "org_bound") -> Tenant:
    tenant_id = uuid.uuid4()
    return Tenant.objects.create(
        id=tenant_id, slug=f"t-{tenant_id.hex[:12]}", display_name="Bound", workos_organization_id=org_id
    )


def test_the_sync_writes_the_org_mapping_without_a_ttl_to_the_front(front, settings, monkeypatch):
    settings.WORKOS_API_KEY = "sk_test_fake"
    Tenant.objects.filter(slug=settings.WORKOS_OWNER_TENANT_SLUG).update(workos_organization_id="org_owner")
    workos = FakeWorkOS()
    monkeypatch.setattr("apps.identity.sync.workos_client", lambda: workos)
    workos.emit("organization.created", organization_data("org_acme", "Acme Corp"))
    run_sync()

    slug = Tenant.objects.get(workos_organization_id="org_acme").slug
    key = org_tenant_key("org_acme")
    assert key == "control:org:org_acme:tenant_slug"
    assert front.get(key) == slug
    assert front.ttl(key) == -1
    assert all(options == {} for _, _, options in front.writes)
    assert front.opened == [FRONT_URL]


def test_the_user_key_has_the_shape_the_front_reads():
    assert user_principal_key("user_01ABC") == "control:user:user_01ABC:principal"


def test_a_failed_write_raises_instead_of_falling_back(front):
    front.fail_writes = ConnectionError("front valkey down")
    with pytest.raises(ConnectionError):
        write_org_tenant("org_x", "t-abc")
    assert control_mapping._memory.get(org_tenant_key("org_x")) is None


def test_rewarm_restores_every_key_after_a_flush(front):
    tenant = _bound_tenant("org_bound")
    User.objects.create(workos_user_id="user_owner", legacy_principal=PRINCIPAL)
    User.objects.create(workos_user_id="user_gone", legacy_principal="identity:ba9876543210", is_active=False)
    User.objects.create(workos_user_id="user_plain")
    front.flushdb()

    counts = rewarm()

    assert front.get(org_tenant_key("org_bound")) == tenant.slug
    assert front.get(user_principal_key("user_owner")) == PRINCIPAL
    assert front.get(user_principal_key("user_gone")) is None
    assert front.get(user_principal_key("user_plain")) is None
    assert front.ttl(org_tenant_key("org_bound")) == -1
    assert front.ttl(user_principal_key("user_owner")) == -1
    # Tenants with no organization (the owner's, before binding) map nothing.
    assert counts == {"organizations": 1, "principals": 1}


def test_the_warm_command_backfills_the_front(front):
    _bound_tenant("org_cmd")
    output = io.StringIO()
    call_command("warm_control_mapping", stdout=output)
    assert front.get(org_tenant_key("org_cmd"))
    assert "1 organization" in output.getvalue()


def test_the_warm_command_refuses_without_the_front_valkey(settings):
    settings.FRONT_VALKEY_URL = ""
    with pytest.raises(CommandError, match="FRONT_VALKEY_URL"):
        call_command("warm_control_mapping")


def test_a_starting_worker_rewarms_the_front(front):
    _bound_tenant("org_boot")
    tasks.rewarm_on_worker_start()
    assert front.get(org_tenant_key("org_boot"))


def test_a_failed_rewarm_does_not_stop_the_worker(front):
    _bound_tenant("org_boot")
    front.fail_writes = ConnectionError("front valkey down")
    tasks.rewarm_on_worker_start()  # logs, does not raise


def test_without_a_front_url_the_mapping_stays_in_memory(settings):
    settings.FRONT_VALKEY_URL = ""
    write_org_tenant("org_mem", "t-mem")
    assert mapping_client().get(org_tenant_key("org_mem")) == "t-mem"


@pytest.mark.skipif(
    not os.environ.get("FRONT_VALKEY_TEST_URL"),
    reason="set FRONT_VALKEY_TEST_URL to a disposable Valkey to run against a real one",
)
def test_a_real_valkey_holds_both_keys_with_no_ttl(settings):
    import redis

    url = os.environ["FRONT_VALKEY_TEST_URL"]
    settings.FRONT_VALKEY_URL = url
    org_id = f"org_real_{uuid.uuid4().hex[:8]}"
    user_id = f"user_real_{uuid.uuid4().hex[:8]}"
    tenant = _bound_tenant(org_id)
    User.objects.create(workos_user_id=user_id, legacy_principal=PRINCIPAL)
    real = redis.from_url(url, decode_responses=True)
    keys = [org_tenant_key(org_id), user_principal_key(user_id)]
    try:
        real.delete(*keys)
        rewarm()
        assert real.get(keys[0]) == tenant.slug
        assert real.get(keys[1]) == PRINCIPAL
        assert [real.ttl(key) for key in keys] == [-1, -1]
    finally:
        real.delete(*keys)

