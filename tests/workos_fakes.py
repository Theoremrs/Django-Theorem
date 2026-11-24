"""A WorkOS double for the account sync tests.

``FakeWorkOS`` stands in for ``workos.WorkOSClient``: it serves an event log
through the SDK's own ``EventSchema`` and ``SyncPage``, so every fixture event
passes the SDK's deserializer, and the organizations and memberships it creates
append the events WorkOS would send for them. Nothing here reaches WorkOS.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import time
from types import SimpleNamespace
from typing import Any

from workos import NotFoundError
from workos._pagination import ListMetadata, SyncPage
from workos.events.models import EventSchema

TIMESTAMP = "2026-11-24T12:00:00.000Z"
_ids = itertools.count(1)


def _next_id(prefix: str) -> str:
    # WorkOS ids are ULIDs, which sort by creation time; a zero-padded counter
    # keeps the same order.
    return f"{prefix}_{next(_ids):026d}"


def event(event_type: str, data: dict[str, Any], *, event_id: str | None = None) -> dict:
    return {
        "object": "event",
        "id": event_id or _next_id("event"),
        "event": event_type,
        "data": data,
        "created_at": TIMESTAMP,
    }


def user_data(user_id: str, *, email: str = "", first_name: str | None = None, last_name: str | None = None) -> dict:
    return {
        "object": "user",
        "id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "profile_picture_url": None,
        "email": email or f"{user_id}@example.com",
        "email_verified": True,
        "external_id": None,
        "last_sign_in_at": None,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }


def organization_data(org_id: str, name: str, *, external_id: str | None = None) -> dict:
    return {
        "object": "organization",
        "id": org_id,
        "name": name,
        "domains": [],
        "metadata": {},
        "external_id": external_id,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }


def membership_data(membership_id: str, org_id: str, user_id: str, *, role: str = "member") -> dict:
    return {
        "object": "organization_membership",
        "id": membership_id,
        "user_id": user_id,
        "organization_id": org_id,
        "status": "active",
        "role": {"slug": role},
        "custom_attributes": {},
        "directory_managed": False,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }


def sign(body: bytes, secret: str, *, issued_at_ms: int | None = None) -> str:
    """A ``WorkOS-Signature`` header as WorkOS writes it: HMAC of ``t.body``."""
    timestamp = str(issued_at_ms if issued_at_ms is not None else int(time.time() * 1000))
    digest = hmac.new(
        secret.encode("utf-8"), f"{timestamp}.{body.decode('utf-8')}".encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"t={timestamp}, v1={digest}"


class FakeEvents:
    def __init__(self, log: list[dict]):
        self._log = log
        self.calls: list[dict] = []

    def list_events(self, *, events: list[str], limit: int | None = None, after: str | None = None, **kwargs):
        self.calls.append({"events": list(events), "limit": limit, "after": after, **kwargs})
        rows = [row for row in self._log if row["event"] in events]
        if after is not None:
            ids = [row["id"] for row in rows]
            rows = rows[ids.index(after) + 1 :] if after in ids else []
        page = rows[: limit or 10]
        return SyncPage(
            data=[EventSchema.from_dict(json.loads(json.dumps(row))) for row in page],
            list_metadata=ListMetadata(after=page[-1]["id"] if page else None),
        )


class FakeOrganizations:
    def __init__(self, workos: "FakeWorkOS"):
        self._workos = workos
        self.by_id: dict[str, dict] = {}
        self.created: list[dict] = []

    def create_organization(self, *, name: str, external_id=None, request_options=None, **kwargs):
        if self._workos.fail_next.pop("create_organization", None):
            raise RuntimeError("WorkOS refused create_organization")
        org = organization_data(_next_id("org"), name, external_id=external_id)
        self.by_id[org["id"]] = org
        self.created.append({"name": name, "external_id": external_id, "request_options": request_options})
        self._workos.log.append(event("organization.created", org))
        return SimpleNamespace(id=org["id"], name=name, external_id=external_id)

    def get_organization_by_external_id(self, external_id: str, **kwargs):
        for org in self.by_id.values():
            if org["external_id"] == external_id:
                return SimpleNamespace(id=org["id"], name=org["name"], external_id=external_id)
        raise NotFoundError("organization not found", status_code=404)


class FakeMemberships:
    def __init__(self, workos: "FakeWorkOS"):
        self._workos = workos
        self.rows: list[dict] = []
        self.created: list[dict] = []

    def add(self, org_id: str, user_id: str, *, role: str = "member", emit: bool = True) -> dict:
        row = membership_data(_next_id("om"), org_id, user_id, role=role)
        self.rows.append(row)
        if emit:
            self._workos.log.append(event("organization_membership.created", row))
        return row

    def list_organization_memberships(self, *, user_id=None, organization_id=None, limit=None, **kwargs):
        rows = [
            row
            for row in self.rows
            if (user_id is None or row["user_id"] == user_id)
            and (organization_id is None or row["organization_id"] == organization_id)
        ]
        return SyncPage(
            data=[SimpleNamespace(id=row["id"], organization_id=row["organization_id"]) for row in rows[: limit or 10]],
            list_metadata=ListMetadata(),
        )

    def create_organization_membership(self, *, user_id: str, organization_id: str, role=None, **kwargs):
        if self._workos.fail_next.pop("create_organization_membership", None):
            raise RuntimeError("WorkOS refused create_organization_membership")
        self.created.append({"user_id": user_id, "organization_id": organization_id, "role": role})
        row = self.add(organization_id, user_id, role=getattr(role, "role_slug", "member"))
        return SimpleNamespace(id=row["id"], organization_id=organization_id, user_id=user_id)


class FakeWorkOS:
    def __init__(self):
        self.log: list[dict] = []
        self.fail_next: dict[str, bool] = {}
        self.events = FakeEvents(self.log)
        self.organizations = FakeOrganizations(self)
        self.organization_membership = FakeMemberships(self)

    def emit(self, event_type: str, data: dict, *, event_id: str | None = None) -> dict:
        row = event(event_type, data, event_id=event_id)
        self.log.append(row)
        return row


class FakeFrontValkey:
    """The front's Valkey as the redis client sees it, recording every write."""

    def __init__(self):
        self.values: dict[str, str] = {}
        self.expiring: dict[str, int] = {}
        self.writes: list[tuple[str, str, dict]] = []
        self.fail_writes: Exception | None = None

    def set(self, key: str, value: str, ex=None, px=None, exat=None, pxat=None, keepttl=False, **kwargs):
        if self.fail_writes is not None:
            raise self.fail_writes
        options = {name: v for name, v in {"ex": ex, "px": px, "exat": exat, "pxat": pxat}.items() if v is not None}
        self.writes.append((key, value, options))
        self.values[key] = value
        if options:
            self.expiring[key] = int(ex or 1)
        elif not keepttl:
            self.expiring.pop(key, None)
        return True

    def setex(self, key: str, seconds: int, value: str):
        return self.set(key, value, ex=seconds)

    def get(self, key: str):
        return self.values.get(key)

    def delete(self, *keys: str) -> int:
        if self.fail_writes is not None:
            raise self.fail_writes
        removed = 0
        for key in keys:
            removed += self.values.pop(key, None) is not None
            self.expiring.pop(key, None)
        return removed

    def ttl(self, key: str) -> int:
        if key not in self.values:
            return -2
        return self.expiring.get(key, -1)

    def flushdb(self):
        self.values.clear()
        self.expiring.clear()
        return True
