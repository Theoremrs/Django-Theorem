"""D5: a user who signs up and joins no organization gets their own.

The sync creates a WorkOS organization and an owner membership through the API.
Their events bring the tenant, and the tenant's signup starts its store and its
sprite through the existing provisioning.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from django.conf import settings as django_settings

from apps.identity.models import Membership, User
from apps.identity.sync import SELF_SERVE_EXTERNAL_ID_PREFIX, run_sync
from apps.tenancy.models import EnvironmentHost, StorePlacement, Tenant
from tests.test_environment_provisioning import both_configured, env_configured  # noqa: F401 — fixtures
from tests.workos_fakes import FakeWorkOS, membership_data, organization_data, user_data

pytestmark = pytest.mark.django_db


@pytest.fixture
def workos(settings, monkeypatch):
    settings.WORKOS_API_KEY = "sk_test_fake"
    Tenant.objects.filter(slug=django_settings.WORKOS_OWNER_TENANT_SLUG).update(
        workos_organization_id="org_owner"
    )
    fake = FakeWorkOS()
    monkeypatch.setattr("apps.identity.sync.workos_client", lambda: fake)
    monkeypatch.setattr("apps.identity.sync.SELF_SERVE_GRACE", timedelta(0))
    return fake


def _signup(workos: FakeWorkOS, user_id: str = "user_new", **names) -> None:
    workos.emit("user.created", user_data(user_id, email=f"{user_id}@example.com", **names))


def test_a_lone_signup_gets_an_organization_with_them_as_owner(workos):
    _signup(workos, first_name="Ada", last_name="Lovelace")
    result = run_sync()
    assert result["signups_started"] == 1

    [organization] = workos.organizations.created
    assert organization["name"] == "Ada Lovelace"
    assert organization["external_id"] == f"{SELF_SERVE_EXTERNAL_ID_PREFIX}user_new"
    assert organization["request_options"] == {"idempotency_key": f"{SELF_SERVE_EXTERNAL_ID_PREFIX}user_new"}
    [membership] = workos.organization_membership.created
    assert membership["user_id"] == "user_new"
    assert membership["role"].role_slug == "owner"
    assert User.objects.get(workos_user_id="user_new").self_serve_pending is False

    # The next run reads the events those calls made.
    run_sync()
    org_id = next(iter(workos.organizations.by_id))
    tenant = Tenant.objects.get(workos_organization_id=org_id)
    assert tenant.display_name == "Ada Lovelace"
    assert tenant.slug.startswith("t-")
    owner = Membership.objects.get(tenant=tenant, user__workos_user_id="user_new")
    assert owner.role == Membership.Role.OWNER


def test_the_new_tenant_gets_its_store_and_sprite(workos, both_configured):  # noqa: F811
    _signup(workos)
    run_sync()
    run_sync()
    tenant = Tenant.objects.get(workos_organization_id=next(iter(workos.organizations.by_id)))
    assert StorePlacement.objects.get(tenant=tenant).state == StorePlacement.State.PROVISIONING
    assert EnvironmentHost.objects.filter(tenant=tenant).exists()


def test_a_signup_is_given_one_organization_however_often_the_sync_runs(workos):
    _signup(workos)
    for _ in range(4):
        run_sync()
    assert len(workos.organizations.created) == 1
    assert len(workos.organization_membership.created) == 1
    assert Tenant.objects.filter(workos_organization_id__startswith="org_").exclude(
        workos_organization_id="org_owner"
    ).count() == 1


def test_an_invited_user_joins_their_organization_and_gets_no_other(workos):
    workos.emit("organization.created", organization_data("org_team", "Team"))
    _signup(workos, "user_invited")
    workos.organization_membership.add("org_team", "user_invited")
    run_sync()
    run_sync()
    assert workos.organizations.created == []
    assert Membership.objects.get(user__workos_user_id="user_invited").tenant.workos_organization_id == "org_team"


def test_a_membership_workos_has_but_django_has_not_synced_blocks_the_signup(workos):
    _signup(workos, "user_elsewhere")
    workos.organization_membership.add("org_unsynced", "user_elsewhere", emit=False)
    run_sync()
    assert workos.organizations.created == []
    assert User.objects.get(workos_user_id="user_elsewhere").self_serve_pending is False


def test_a_signup_inside_the_grace_period_waits(workos, monkeypatch):
    monkeypatch.setattr("apps.identity.sync.SELF_SERVE_GRACE", timedelta(minutes=2))
    _signup(workos, "user_fresh")
    run_sync()
    assert workos.organizations.created == []
    assert User.objects.get(workos_user_id="user_fresh").self_serve_pending is True


def test_a_failed_organization_create_is_retried(workos):
    _signup(workos, "user_retry")
    workos.fail_next["create_organization"] = True
    run_sync()
    assert workos.organizations.created == []
    assert User.objects.get(workos_user_id="user_retry").self_serve_pending is True

    run_sync()
    assert len(workos.organizations.created) == 1
    assert User.objects.get(workos_user_id="user_retry").self_serve_pending is False


def test_a_failed_membership_create_reuses_the_organization(workos):
    _signup(workos, "user_half")
    workos.fail_next["create_organization_membership"] = True
    run_sync()
    assert len(workos.organizations.created) == 1
    assert workos.organization_membership.created == []

    run_sync()
    assert len(workos.organizations.created) == 1
    assert len(workos.organization_membership.created) == 1


def test_a_user_holding_a_legacy_principal_gets_no_organization(workos):
    User.objects.create(workos_user_id="user_legacy", legacy_principal="identity:0123456789ab")
    _signup(workos, "user_legacy")
    run_sync()
    assert workos.organizations.created == []


def test_a_deleted_user_gets_no_organization(workos):
    _signup(workos, "user_quit")
    workos.emit("user.deleted", user_data("user_quit"))
    run_sync()
    assert workos.organizations.created == []
    assert User.objects.get(workos_user_id="user_quit").self_serve_pending is False


def test_an_organization_is_named_after_the_email_without_a_name(workos):
    _signup(workos, "user_anon")
    run_sync()
    assert workos.organizations.created[0]["name"] == "user_anon@example.com"


def test_membership_events_clear_a_pending_signup(workos, monkeypatch):
    monkeypatch.setattr("apps.identity.sync.SELF_SERVE_GRACE", timedelta(minutes=2))
    workos.emit("organization.created", organization_data("org_join", "Join Co"))
    _signup(workos, "user_late_join")
    run_sync()
    assert User.objects.get(workos_user_id="user_late_join").self_serve_pending is True
    workos.emit("organization_membership.created", membership_data("om_join", "org_join", "user_late_join"))
    run_sync()
    assert User.objects.get(workos_user_id="user_late_join").self_serve_pending is False
