"""The control mapping the Rust front reads from its Valkey.

Two keys, neither with a TTL, shared with theorem-mcp-server:

- ``control:org:{org_id}:tenant_slug`` holds the tenant slug of a WorkOS organization.
- ``control:user:{workos_user_id}:principal`` holds ``identity:<12 hex>`` for a
  user bound to the principal they used before WorkOS.

Postgres is the record (``Tenant.workos_organization_id`` and
``User.legacy_principal``); ``rewarm`` writes every key back from it, so a
flushed or replaced Valkey recovers without the events being replayed.

The keys live in ``FRONT_VALKEY_URL``, which is not Django's own ``VALKEY_URL``.
A failed write raises: the caller keeps its event and retries it. Only with no
URL configured (tests, a laptop) does an in-process dict stand in.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)


def org_tenant_key(org_id: str) -> str:
    return f"control:org:{org_id}:tenant_slug"


def user_principal_key(workos_user_id: str) -> str:
    return f"control:user:{workos_user_id}:principal"


class MemoryMapping:
    """The subset of the redis client this module uses, kept in process memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._values.get(key)

    def set(self, key: str, value: str) -> bool:
        with self._lock:
            self._values[key] = value
        return True

    def delete(self, *keys: str) -> int:
        with self._lock:
            return sum(self._values.pop(key, None) is not None for key in keys)

    def flushdb(self) -> bool:
        with self._lock:
            self._values.clear()
        return True


_memory = MemoryMapping()


@lru_cache(maxsize=4)
def _redis(url: str) -> Any:
    import redis

    return redis.from_url(url, decode_responses=True)


def front_valkey_configured() -> bool:
    return bool(settings.FRONT_VALKEY_URL)


def mapping_client() -> Any:
    url = settings.FRONT_VALKEY_URL
    if not url:
        return _memory
    return _redis(url)


def clear_memory_mapping() -> None:
    """Test helper: empty the in-process stand-in."""
    _memory.flushdb()


def write_org_tenant(org_id: str, tenant_slug: str) -> None:
    mapping_client().set(org_tenant_key(org_id), tenant_slug)


def write_user_principal(workos_user_id: str, principal: str) -> None:
    mapping_client().set(user_principal_key(workos_user_id), principal)


def delete_user_principal(workos_user_id: str) -> None:
    mapping_client().delete(user_principal_key(workos_user_id))


def rewarm() -> dict[str, int]:
    """Write every mapping key from Postgres. Safe to run at any time."""
    from apps.identity.models import User
    from apps.tenancy.models import Tenant

    if not front_valkey_configured():
        logger.warning("FRONT_VALKEY_URL is not set: the front cannot see the control mapping")
    client = mapping_client()
    organizations = 0
    for org_id, slug in Tenant.objects.exclude(workos_organization_id="").values_list(
        "workos_organization_id", "slug"
    ):
        client.set(org_tenant_key(org_id), slug)
        organizations += 1
    principals = 0
    for workos_user_id, principal in (
        User.objects.filter(is_active=True)
        .exclude(legacy_principal="")
        .values_list("workos_user_id", "legacy_principal")
    ):
        client.set(user_principal_key(workos_user_id), principal)
        principals += 1
    return {"organizations": organizations, "principals": principals}
