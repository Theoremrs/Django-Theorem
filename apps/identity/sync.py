"""Keep Django's accounts in step with WorkOS by reading its Events API.

One run, under a lease so only one worker syncs at a time:

1. Fetch: page through the Events API after the stored cursor and store each
   event once (``WorkOSEvent.event_id`` is unique, so a replay is a no-op).
2. Apply: apply stored events in the order WorkOS served them. An event that
   cannot apply yet (a membership whose organization has not synced) or that
   fails is kept and retried, and later events of the same stream wait behind
   it, so their order holds.
3. Self-serve: give each new user who joined no organization their own, by
   creating a WorkOS organization and an owner membership. Their tenant then
   arrives through the same events as every other tenant's.

The webhook only nudges this; the beat schedule runs it anyway.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from workos import NotFoundError
from workos.organization_membership import RoleSingle

from apps.identity.control_mapping import delete_user_principal, write_org_tenant
from apps.identity.models import Membership, User, WorkOSEvent, WorkOSSyncState
from apps.identity.workos_api import workos_client, workos_configured
from apps.tenancy.models import Tenant
from apps.tenancy.tenant_cache import invalidate_membership

logger = logging.getLogger(__name__)

EVENT_TYPES = (
    "user.created",
    "user.updated",
    "user.deleted",
    "organization.created",
    "organization.updated",
    "organization_membership.created",
    "organization_membership.updated",
    "organization_membership.deleted",
)
PAGE_LIMIT = 100
MAX_PAGES_PER_RUN = 50
MAX_APPLY_PASSES = 10
# The Events API answers at most 30 days per request.
FIRST_RUN_LOOKBACK = timedelta(days=29)
LEASE = timedelta(minutes=5)
# A user who signs up through an invitation gets their membership a moment
# after their account; wait this long before deciding they joined nothing.
SELF_SERVE_GRACE = timedelta(minutes=2)
SELF_SERVE_BATCH = 50
SELF_SERVE_EXTERNAL_ID_PREFIX = "theorem-signup:"


class Deferred(Exception):
    """The event depends on something that has not synced yet."""


def run_sync() -> dict[str, Any]:
    if not workos_configured():
        return {"status": "not_configured"}
    if _owner_tenant_unbound():
        logger.warning(
            "WorkOS sync waits: tenant %s has no organization yet; run bind_workos_owner",
            settings.WORKOS_OWNER_TENANT_SLUG,
        )
        return {"status": "waiting_for_owner_binding"}

    state = WorkOSSyncState.load()
    claimant = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    if not state.claim(claimant, LEASE):
        return {"status": "busy"}
    try:
        client = workos_client()
        fetched = _fetch(client, state)
        applied, waiting = _apply_pending()
        signups = _reconcile_self_serve(client)
    finally:
        state.release(claimant)
    return {
        "status": "ok",
        "fetched": fetched,
        "applied": applied,
        "waiting": waiting,
        "signups_started": signups,
    }


def _owner_tenant_unbound() -> bool:
    slug = settings.WORKOS_OWNER_TENANT_SLUG
    return bool(slug) and Tenant.objects.filter(slug=slug, workos_organization_id="").exists()


# --- fetch -------------------------------------------------------------------


def _fetch(client: Any, state: WorkOSSyncState) -> int:
    cursor = state.cursor
    fetched = 0
    for _ in range(MAX_PAGES_PER_RUN):
        options: dict[str, Any] = {"events": list(EVENT_TYPES), "limit": PAGE_LIMIT}
        if cursor:
            options["after"] = cursor
        else:
            options["range_start"] = _utc_iso(timezone.now() - FIRST_RUN_LOOKBACK)
        events = [_event_dict(event) for event in client.events.list_events(**options).data]
        if not events:
            break
        with transaction.atomic():
            for event in events:
                WorkOSEvent.objects.get_or_create(
                    event_id=event["id"],
                    defaults={
                        "event_type": event.get("event") or "",
                        "occurred_at": parse_datetime(event.get("created_at") or ""),
                        "payload": event,
                    },
                )
            cursor = events[-1]["id"]
            WorkOSSyncState.objects.filter(pk=state.pk).update(cursor=cursor)
        fetched += len(events)
        if len(events) < PAGE_LIMIT:
            break
    return fetched


def _event_dict(event: Any) -> dict[str, Any]:
    raw = getattr(event, "raw_data", None)
    return raw if isinstance(raw, dict) else event.to_dict()


def _utc_iso(moment: datetime) -> str:
    return moment.astimezone(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- apply -------------------------------------------------------------------


def _apply_pending() -> tuple[int, int]:
    """Apply stored events, passing again while a pass unblocks later ones."""
    applied = 0
    for _ in range(MAX_APPLY_PASSES):
        applied_now, waiting = _apply_pass()
        applied += applied_now
        if not applied_now or not waiting:
            break
    return applied, waiting


def _apply_pass() -> tuple[int, int]:
    applied = waiting = 0
    blocked: dict[str, str] = {}
    for row in WorkOSEvent.objects.exclude(status=WorkOSEvent.Status.APPLIED).order_by("id"):
        stream = _stream(row)
        if stream in blocked:
            _hold(row, f"waiting behind event {blocked[stream]}", attempted=False)
            waiting += 1
            continue
        try:
            with transaction.atomic():
                _apply(row.event_type, (row.payload or {}).get("data") or {})
                row.status = WorkOSEvent.Status.APPLIED
                row.applied_at = timezone.now()
                row.last_error = ""
                row.save(update_fields=["status", "applied_at", "last_error"])
        except Deferred as exc:
            _hold(row, str(exc), attempted=True)
        except Exception as exc:  # noqa: BLE001 — kept and retried, never dropped
            logger.warning("WorkOS event %s (%s) failed: %s", row.event_id, row.event_type, exc)
            _hold(row, f"{type(exc).__name__}: {exc}", attempted=True)
        else:
            applied += 1
            continue
        blocked[stream] = row.event_id
        waiting += 1
    return applied, waiting


def _stream(row: WorkOSEvent) -> str:
    """Events of one stream apply in order; streams do not wait on each other."""
    data = (row.payload or {}).get("data") or {}
    if row.event_type.startswith("organization_membership."):
        return f"memberships:{data.get('organization_id') or ''}"
    if row.event_type.startswith("organization."):
        return f"organization:{data.get('id') or ''}"
    return f"user:{data.get('id') or ''}"


def _hold(row: WorkOSEvent, error: str, *, attempted: bool) -> None:
    row.status = WorkOSEvent.Status.DEFERRED
    row.last_error = error[:2000]
    fields = ["status", "last_error"]
    if attempted:
        row.attempts += 1
        fields.append("attempts")
    row.save(update_fields=fields)


def _apply(event_type: str, data: dict[str, Any]) -> None:
    if event_type in ("user.created", "user.updated"):
        _upsert_user(data, signed_up=event_type == "user.created")
    elif event_type == "user.deleted":
        _deactivate_user(data)
    elif event_type in ("organization.created", "organization.updated"):
        _upsert_organization(data)
    elif event_type in ("organization_membership.created", "organization_membership.updated"):
        _upsert_membership(data)
    elif event_type == "organization_membership.deleted":
        _delete_membership(data)


def _display_name(data: dict[str, Any]) -> str:
    parts = [(data.get(field) or "").strip() for field in ("first_name", "last_name")]
    return " ".join(part for part in parts if part) or (data.get("name") or "").strip()


def _upsert_user(data: dict[str, Any], *, signed_up: bool) -> None:
    workos_user_id = data.get("id") or ""
    if not workos_user_id:
        return
    user, _ = User.objects.get_or_create(workos_user_id=workos_user_id)
    user.email = data.get("email") or ""
    user.display_name = _display_name(data)
    user.is_active = True
    if signed_up and not user.legacy_principal and not user.memberships.exists():
        user.self_serve_pending = True
    user.save()


def _deactivate_user(data: dict[str, Any]) -> None:
    user = User.objects.filter(workos_user_id=data.get("id") or "").first()
    if user is None:
        return
    user.is_active = False
    user.self_serve_pending = False
    user.save(update_fields=["is_active", "self_serve_pending", "updated_at"])
    if user.legacy_principal:
        delete_user_principal(user.workos_user_id)


def _upsert_organization(data: dict[str, Any]) -> None:
    org_id = data.get("id") or ""
    if not org_id:
        return
    name = (data.get("name") or "").strip()
    tenant = Tenant.objects.filter(workos_organization_id=org_id).first()
    if tenant is None:
        # The slug is the tenant's identity in every keyspace and store; it is
        # never derived from the organization's name, which its owner can change.
        tenant_id = uuid.uuid4()
        tenant = Tenant.objects.create(
            id=tenant_id,
            slug=f"t-{tenant_id.hex[:12]}",
            display_name=name or org_id,
            workos_organization_id=org_id,
        )
    elif name and tenant.display_name != name:
        tenant.display_name = name
        tenant.save(update_fields=["display_name", "updated_at"])
    write_org_tenant(org_id, tenant.slug)


def _role(value: Any) -> str:
    slug = value.get("slug") if isinstance(value, dict) else value
    return slug if slug in Membership.Role.values else Membership.Role.MEMBER


def _upsert_membership(data: dict[str, Any]) -> None:
    org_id = data.get("organization_id") or ""
    workos_user_id = data.get("user_id") or ""
    if not org_id or not workos_user_id:
        return
    tenant = Tenant.objects.filter(workos_organization_id=org_id).first()
    if tenant is None:
        raise Deferred(f"organization {org_id} has not synced yet")
    if (data.get("status") or "active") != "active":
        # A pending invitation grants nothing yet; a deactivated membership no longer does.
        _delete_membership(data)
        return
    user, _ = User.objects.get_or_create(workos_user_id=workos_user_id)
    Membership.objects.update_or_create(
        tenant=tenant,
        user=user,
        defaults={"role": _role(data.get("role")), "workos_membership_id": data.get("id") or ""},
    )
    if user.self_serve_pending:
        User.objects.filter(pk=user.pk).update(self_serve_pending=False)
    _invalidate_on_commit(tenant.slug, workos_user_id)


def _delete_membership(data: dict[str, Any]) -> None:
    membership = (
        Membership.objects.filter(workos_membership_id=data.get("id") or "")
        .exclude(workos_membership_id="")
        .select_related("tenant", "user")
        .first()
    )
    if membership is None:
        membership = (
            Membership.objects.filter(
                tenant__workos_organization_id=data.get("organization_id") or "",
                user__workos_user_id=data.get("user_id") or "",
            )
            .exclude(tenant__workos_organization_id="")
            .select_related("tenant", "user")
            .first()
        )
    if membership is None:
        return
    tenant_slug, workos_user_id = membership.tenant.slug, membership.user.workos_user_id
    membership.delete()
    _invalidate_on_commit(tenant_slug, workos_user_id)


def _invalidate_on_commit(tenant_slug: str, workos_user_id: str) -> None:
    # After commit, so a peer that refills its cache reads the new membership.
    transaction.on_commit(lambda: invalidate_membership(tenant_slug, workos_user_id), robust=True)


# --- self-serve signup -------------------------------------------------------


def _reconcile_self_serve(client: Any) -> int:
    started = 0
    cutoff = timezone.now() - SELF_SERVE_GRACE
    pending = User.objects.filter(self_serve_pending=True, created_at__lte=cutoff).order_by("created_at")
    for user in pending[:SELF_SERVE_BATCH]:
        if not user.is_active or user.legacy_principal or user.memberships.exists():
            _clear_pending(user)
            continue
        try:
            existing = client.organization_membership.list_organization_memberships(
                user_id=user.workos_user_id, statuses=["active", "pending"], limit=1
            )
            if not list(existing.data):
                organization = _signup_organization(client, user)
                client.organization_membership.create_organization_membership(
                    user_id=user.workos_user_id,
                    organization_id=organization.id,
                    role=RoleSingle(role_slug=Membership.Role.OWNER),
                )
                started += 1
        except Exception as exc:  # noqa: BLE001 — the user stays pending for the next run
            logger.warning("self-serve organization for %s failed: %s", user.workos_user_id, exc)
            continue
        # Their membership reaches Django through the events.
        _clear_pending(user)
    return started


def _signup_organization(client: Any, user: User) -> Any:
    """The user's own organization, created once however often this runs."""
    external_id = f"{SELF_SERVE_EXTERNAL_ID_PREFIX}{user.workos_user_id}"
    try:
        return client.organizations.get_organization_by_external_id(external_id)
    except NotFoundError:
        pass
    return client.organizations.create_organization(
        name=user.display_name or user.email or user.workos_user_id,
        external_id=external_id,
        request_options={"idempotency_key": external_id},
    )


def _clear_pending(user: User) -> None:
    User.objects.filter(pk=user.pk).update(self_serve_pending=False)
