"""D6 and D7: the owner keeps the tenant and the principal they had before WorkOS.

``bind_workos_owner`` gives the seeded tenant ``Travis-Gilbert`` its WorkOS
organization and binds the owner's WorkOS user to ``identity:<12 hex>``. After
that, the owner's events sync onto what exists: no second tenant, store or
sprite, and no self-serve organization.
"""

from __future__ import annotations

import io
from datetime import timedelta

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.identity import control_mapping
from apps.identity.control_mapping import org_tenant_key, rewarm, user_principal_key
from apps.identity.models import Membership, User
from apps.identity.sync import run_sync
from apps.tenancy.models import EnvironmentHost, Store, StorePlacement, Tenant
from tests.test_environment_provisioning import both_configured, env_configured  # noqa: F401 — fixtures
from tests.workos_fakes import FakeFrontValkey, FakeWorkOS, membership_data, organization_data, user_data

pytestmark = pytest.mark.django_db

OWNER_TENANT = "Travis-Gilbert"
ORG = "org_01OWNERFIXTURE"
USER = "user_01OWNERFIXTURE"
PRINCIPAL = "identity:a1b2c3d4e5f6"


@pytest.fixture
def front(settings, monkeypatch):
    settings.FRONT_VALKEY_URL = "redis://front-valkey.invalid:6379/0"
    valkey = FakeFrontValkey()
    monkeypatch.setattr(control_mapping, "_redis", lambda url: valkey)
    return valkey


@pytest.fixture
def workos(settings, monkeypatch):
    settings.WORKOS_API_KEY = "sk_test_fake"
    fake = FakeWorkOS()
    monkeypatch.setattr("apps.identity.sync.workos_client", lambda: fake)
    monkeypatch.setattr("apps.identity.sync.SELF_SERVE_GRACE", timedelta(0))
    return fake


def _bind(**overrides) -> str:
    options = {"organization": ORG, "user": USER, "principal": PRINCIPAL, **overrides}
    out = io.StringIO()
    call_command("bind_workos_owner", stdout=out, **options)
    return out.getvalue()


def _counts() -> dict[str, int]:
    return {
        "tenants": Tenant.objects.count(),
        "stores": Store.objects.count(),
        "placements": StorePlacement.objects.count(),
        "sprites": EnvironmentHost.objects.count(),
    }


def test_binding_gives_the_seeded_tenant_its_organization_and_the_owner_their_principal(front):
    _bind()
    tenant = Tenant.objects.get(slug=OWNER_TENANT)
    assert tenant.workos_organization_id == ORG
    owner = User.objects.get(workos_user_id=USER)
    assert owner.legacy_principal == PRINCIPAL
    assert owner.self_serve_pending is False
    assert Membership.objects.get(tenant=tenant, user=owner).role == Membership.Role.OWNER
    assert front.get(org_tenant_key(ORG)) == OWNER_TENANT
    assert front.get(user_principal_key(USER)) == PRINCIPAL
    assert front.ttl(org_tenant_key(ORG)) == front.ttl(user_principal_key(USER)) == -1


def test_binding_twice_changes_nothing(front):
    _bind()
    before = _counts()
    output = _bind()
    assert _counts() == before
    assert Membership.objects.filter(user__workos_user_id=USER).count() == 1
    assert "already bound" in output


def test_binding_starts_no_store_or_sprite(front, both_configured, django_capture_on_commit_callbacks):  # noqa: F811
    before = _counts()
    with django_capture_on_commit_callbacks() as callbacks:
        _bind()
    assert _counts() == before
    assert not any("provision" in getattr(cb, "__qualname__", "") for cb in callbacks)


def test_the_owners_events_sync_onto_the_existing_tenant(front, workos, both_configured):  # noqa: F811
    assert run_sync()["status"] == "waiting_for_owner_binding"
    _bind()
    before = _counts()
    workos.emit("user.created", user_data(USER, email="owner@example.com", first_name="Travis", last_name="Gilbert"))
    workos.emit("organization.created", organization_data(ORG, "Travis Gilbert"))
    workos.emit("organization_membership.created", membership_data("om_owner", ORG, USER, role="owner"))

    assert run_sync()["status"] == "ok"
    run_sync()

    assert _counts() == before
    assert workos.organizations.created == []
    assert workos.organization_membership.created == []
    owner = User.objects.get(workos_user_id=USER)
    assert owner.email == "owner@example.com"
    assert owner.self_serve_pending is False
    membership = Membership.objects.get(user=owner)
    assert membership.tenant.slug == OWNER_TENANT
    assert membership.role == Membership.Role.OWNER
    assert membership.workos_membership_id == "om_owner"
    assert front.get(org_tenant_key(ORG)) == OWNER_TENANT


def test_the_owner_tenant_takes_no_second_organization(front, workos):
    _bind()
    with pytest.raises(CommandError, match="already bound to"):
        _bind(organization="org_01SOMEONEELSE")


@pytest.mark.parametrize(
    ("setup", "overrides", "message"),
    [
        (lambda: None, {"tenant": "No-Such-Tenant"}, "no tenant"),
        (lambda: None, {"principal": "identity:XYZ"}, "identity:<12 hex>"),
        (lambda: None, {"organization": "user_01SWAPPED"}, "org_"),
        (lambda: None, {"user": "org_01SWAPPED"}, "user_"),
        (
            lambda: Tenant.objects.create(slug="t-other", display_name="Other", workos_organization_id=ORG),
            {},
            "already belongs to tenant t-other",
        ),
        (
            lambda: User.objects.create(workos_user_id=USER, legacy_principal="identity:ffffffffffff"),
            {},
            "already holds identity:ffffffffffff",
        ),
        (
            lambda: User.objects.create(workos_user_id="user_01SOMEONE", legacy_principal=PRINCIPAL),
            {},
            "already bound to user_01SOMEONE",
        ),
    ],
)
def test_binding_refuses_anything_that_would_split_an_identity(front, setup, overrides, message):
    setup()
    with pytest.raises(CommandError, match=message):
        _bind(**overrides)
    assert Tenant.objects.get(slug=OWNER_TENANT).workos_organization_id == ""
    assert front.get(org_tenant_key(ORG)) is None


def test_a_failed_front_write_leaves_nothing_bound(front):
    front.fail_writes = ConnectionError("front valkey down")
    with pytest.raises(ConnectionError):
        _bind()
    assert Tenant.objects.get(slug=OWNER_TENANT).workos_organization_id == ""
    assert not User.objects.filter(workos_user_id=USER).exists()


def test_the_principal_comes_back_from_postgres_after_a_flush(front):
    _bind()
    front.flushdb()
    rewarm()
    assert front.get(user_principal_key(USER)) == PRINCIPAL
    assert front.get(org_tenant_key(ORG)) == OWNER_TENANT


def test_a_deleted_owner_loses_the_principal_mapping(front, workos):
    _bind()
    workos.emit("user.deleted", user_data(USER))
    run_sync()
    assert front.get(user_principal_key(USER)) is None
    rewarm()
    assert front.get(user_principal_key(USER)) is None
