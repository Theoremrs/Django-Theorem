"""D2 and D3: Django syncs accounts from the WorkOS Events API.

The sync keeps a cursor, applies each event once and in order, retries a
membership that arrives before its organization, and gives every organization
a tenant whose slug never changes.
"""

from __future__ import annotations

import re
from datetime import timedelta
from unittest.mock import patch

import pytest
from django.conf import settings as django_settings
from django.utils import timezone

from apps.identity.control_mapping import mapping_client, org_tenant_key
from apps.identity.models import Membership, User, WorkOSEvent, WorkOSSyncState
from apps.identity.sync import run_sync
from apps.tenancy.models import Tenant
from tests.workos_fakes import FakeWorkOS, membership_data, organization_data, user_data

pytestmark = pytest.mark.django_db

OWNER_ORG = "org_owner"


@pytest.fixture
def workos(settings, monkeypatch):
    settings.WORKOS_API_KEY = "sk_test_fake"
    fake = FakeWorkOS()
    monkeypatch.setattr("apps.identity.sync.workos_client", lambda: fake)
    return fake


@pytest.fixture
def owner_bound():
    Tenant.objects.filter(slug=django_settings.WORKOS_OWNER_TENANT_SLUG).update(
        workos_organization_id=OWNER_ORG
    )


def _cursor() -> str:
    return WorkOSSyncState.objects.get(pk=WorkOSSyncState.EVENTS).cursor


def test_sync_does_nothing_without_an_api_key(settings, monkeypatch):
    settings.WORKOS_API_KEY = ""
    monkeypatch.setattr("apps.identity.sync.workos_client", lambda: pytest.fail("called WorkOS"))
    assert run_sync()["status"] == "not_configured"


def test_sync_waits_for_the_owner_binding(workos):
    workos.emit("organization.created", organization_data(OWNER_ORG, "Travis Gilbert"))
    result = run_sync()
    assert result["status"] == "waiting_for_owner_binding"
    assert workos.events.calls == []
    assert Tenant.objects.filter(workos_organization_id=OWNER_ORG).count() == 0


def test_the_first_sync_starts_from_a_range_and_then_follows_the_cursor(workos, owner_bound):
    workos.emit("user.created", user_data("user_a"))
    last = workos.emit("user.created", user_data("user_b"))
    run_sync()
    first = workos.events.calls[0]
    assert first["after"] is None
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", first["range_start"])
    assert _cursor() == last["id"]

    run_sync()
    assert workos.events.calls[-1]["after"] == last["id"]
    assert "range_start" not in workos.events.calls[-1]


def test_the_sync_asks_only_for_the_account_events_it_applies(workos, owner_bound):
    run_sync()
    assert sorted(workos.events.calls[0]["events"]) == [
        "organization.created",
        "organization.updated",
        "organization_membership.created",
        "organization_membership.deleted",
        "organization_membership.updated",
        "user.created",
        "user.deleted",
        "user.updated",
    ]


def test_the_sync_pages_through_a_backlog(workos, owner_bound, monkeypatch):
    monkeypatch.setattr("apps.identity.sync.PAGE_LIMIT", 2)
    for n in range(5):
        workos.emit("organization.created", organization_data(f"org_backlog_{n}", f"Backlog {n}"))
    run_sync()
    assert Tenant.objects.filter(workos_organization_id__startswith="org_backlog_").count() == 5
    assert len(workos.events.calls) == 3


def test_an_organization_gets_a_tenant_whose_slug_is_not_its_name(workos, owner_bound):
    workos.emit("organization.created", organization_data("org_acme", "Acme Corp"))
    run_sync()
    tenant = Tenant.objects.get(workos_organization_id="org_acme")
    assert re.fullmatch(r"t-[0-9a-f]{12}", tenant.slug)
    assert tenant.display_name == "Acme Corp"
    assert mapping_client().get(org_tenant_key("org_acme")) == tenant.slug


def test_renaming_an_organization_keeps_its_tenant_slug(workos, owner_bound):
    workos.emit("organization.created", organization_data("org_acme", "Acme Corp"))
    run_sync()
    slug = Tenant.objects.get(workos_organization_id="org_acme").slug

    workos.emit("organization.updated", organization_data("org_acme", "Acme Renamed"))
    run_sync()
    tenant = Tenant.objects.get(workos_organization_id="org_acme")
    assert tenant.slug == slug
    assert tenant.display_name == "Acme Renamed"
    assert mapping_client().get(org_tenant_key("org_acme")) == slug


def test_the_owner_tenant_keeps_its_slug_when_its_organization_syncs(workos, owner_bound):
    workos.emit("organization.created", organization_data(OWNER_ORG, "Personal"))
    run_sync()
    tenant = Tenant.objects.get(workos_organization_id=OWNER_ORG)
    assert tenant.slug == "Travis-Gilbert"
    assert mapping_client().get(org_tenant_key(OWNER_ORG)) == "Travis-Gilbert"


def test_users_without_a_first_name_sync(workos, owner_bound):
    workos.emit("user.created", user_data("user_last_only", last_name="Lovelace"))
    workos.emit("user.created", user_data("user_no_name"))
    workos.emit("user.created", user_data("user_both", first_name="Ada", last_name="Lovelace"))
    run_sync()
    assert User.objects.get(workos_user_id="user_last_only").display_name == "Lovelace"
    assert User.objects.get(workos_user_id="user_no_name").display_name == ""
    assert User.objects.get(workos_user_id="user_both").display_name == "Ada Lovelace"


def test_a_deleted_user_is_deactivated(workos, owner_bound):
    workos.emit("user.created", user_data("user_gone"))
    workos.emit("user.deleted", user_data("user_gone"))
    run_sync()
    assert User.objects.get(workos_user_id="user_gone").is_active is False


def test_an_event_is_applied_once_however_often_it_is_served(workos, owner_bound):
    workos.emit("organization.created", organization_data("org_acme", "Acme Corp"))
    workos.emit("organization_membership.created", membership_data("om_1", "org_acme", "user_1", role="admin"))
    run_sync()
    workos.emit("organization_membership.updated", membership_data("om_1", "org_acme", "user_1", role="member"))
    run_sync()

    # Replay everything, as after a lost cursor.
    WorkOSSyncState.objects.filter(pk=WorkOSSyncState.EVENTS).update(cursor="")
    run_sync()

    assert Tenant.objects.filter(workos_organization_id="org_acme").count() == 1
    assert WorkOSEvent.objects.count() == 3
    membership = Membership.objects.get(workos_membership_id="om_1")
    assert membership.role == "member"


def test_a_membership_before_its_organization_is_retried_not_dropped(workos, owner_bound):
    workos.emit("organization_membership.created", membership_data("om_early", "org_late", "user_1", role="admin"))
    run_sync()
    assert Membership.objects.count() == 0
    deferred = WorkOSEvent.objects.get(event_type="organization_membership.created")
    assert deferred.status == WorkOSEvent.Status.DEFERRED
    assert "org_late" in deferred.last_error

    workos.emit("organization.created", organization_data("org_late", "Late Co"))
    run_sync()
    membership = Membership.objects.get(workos_membership_id="om_early")
    assert membership.tenant.workos_organization_id == "org_late"
    assert membership.role == "admin"
    assert WorkOSEvent.objects.exclude(status=WorkOSEvent.Status.APPLIED).count() == 0


def test_events_behind_a_deferred_one_wait_and_keep_their_order(workos, owner_bound):
    workos.emit("organization_membership.created", membership_data("om_x", "org_late", "user_1"))
    workos.emit("organization_membership.deleted", membership_data("om_x", "org_late", "user_1"))
    run_sync()
    assert WorkOSEvent.objects.filter(status=WorkOSEvent.Status.DEFERRED).count() == 2

    workos.emit("organization.created", organization_data("org_late", "Late Co"))
    run_sync()
    # Created, then deleted: applied in that order, so nothing is left.
    assert Membership.objects.filter(workos_membership_id="om_x").count() == 0
    assert WorkOSEvent.objects.exclude(status=WorkOSEvent.Status.APPLIED).count() == 0


def test_a_waiting_organization_does_not_hold_up_another(workos, owner_bound):
    workos.emit("organization_membership.created", membership_data("om_wait", "org_late", "user_1"))
    workos.emit("organization.created", organization_data("org_other", "Other Co"))
    workos.emit("organization_membership.created", membership_data("om_other", "org_other", "user_2"))
    run_sync()
    assert Membership.objects.filter(workos_membership_id="om_other").exists()
    assert not Membership.objects.filter(workos_membership_id="om_wait").exists()


def test_a_failing_event_is_kept_for_the_next_run(workos, owner_bound):
    workos.emit("organization.created", organization_data("org_flaky", "Flaky Co"))
    with patch("apps.identity.sync.write_org_tenant", side_effect=ConnectionError("valkey down")):
        run_sync()
    assert not Tenant.objects.filter(workos_organization_id="org_flaky").exists()
    failed = WorkOSEvent.objects.get(event_type="organization.created")
    assert failed.status == WorkOSEvent.Status.DEFERRED
    assert failed.attempts == 1
    assert "valkey down" in failed.last_error

    run_sync()
    tenant = Tenant.objects.get(workos_organization_id="org_flaky")
    assert mapping_client().get(org_tenant_key("org_flaky")) == tenant.slug


def test_a_second_sync_stands_down_while_one_runs(workos, owner_bound):
    state = WorkOSSyncState.load()
    assert state.claim("another-worker", timedelta(minutes=5))
    assert run_sync()["status"] == "busy"
    assert workos.events.calls == []


def test_a_lapsed_claim_does_not_stop_the_sync(workos, owner_bound):
    WorkOSSyncState.load()
    WorkOSSyncState.objects.filter(pk=WorkOSSyncState.EVENTS).update(
        claimant="dead-worker", claimed_until=timezone.now() - timedelta(seconds=1)
    )
    assert run_sync()["status"] == "ok"


def test_membership_changes_invalidate_the_membership_cache_after_commit(
    workos, owner_bound, django_capture_on_commit_callbacks
):
    workos.emit("organization.created", organization_data("org_acme", "Acme Corp"))
    workos.emit("organization_membership.created", membership_data("om_inv", "org_acme", "user_inv"))
    workos.emit("organization_membership.deleted", membership_data("om_inv", "org_acme", "user_inv"))
    with patch("apps.identity.sync.invalidate_membership") as invalidate:
        with django_capture_on_commit_callbacks(execute=False) as callbacks:
            run_sync()
        invalidate.assert_not_called()
        for callback in callbacks:
            callback()
    slug = Tenant.objects.get(workos_organization_id="org_acme").slug
    assert [c.args for c in invalidate.call_args_list] == [(slug, "user_inv"), (slug, "user_inv")]


def test_a_second_tenant_cannot_take_a_bound_organization(owner_bound):
    from django.db import IntegrityError, transaction

    with pytest.raises(IntegrityError), transaction.atomic():
        Tenant.objects.create(slug="t-duplicate", display_name="Dup", workos_organization_id=OWNER_ORG)


def test_the_beat_schedule_runs_the_sync_and_the_rewarm():
    tasks = {entry["task"] for entry in django_settings.CELERY_BEAT_SCHEDULE.values()}
    assert "apps.identity.tasks.sync_workos_events" in tasks
    assert "apps.identity.tasks.rewarm_control_mapping" in tasks
