"""Membership cache invalidate (D4).

Key / channel shapes (aligned with Theorem ``theorem-valkey-client`` where
consumers exist):

- Membership cache (Django-local): ``t:{tenant_slug}:control:membership:{user_id}``
- Invalidate channel (Rust peers): ``t:{tenant_slug}:control_apikey:membership_invalidate``

The front's org and user mapping lives in ``apps.identity.control_mapping``.

Redis is optional: when ``VALKEY_URL`` / ``REDIS_URL`` is unset/empty, or the
client cannot connect, an in-process dict backs the same API for tests.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)

MEMBERSHIP_KEY_TTL_SECONDS = 300


def membership_invalidate_channel(tenant_slug: str) -> str:
    """``t:{tenant}:control_apikey:membership_invalidate`` — Rust doctrine."""
    return f"t:{tenant_slug}:control_apikey:membership_invalidate"

_memory_lock = threading.Lock()
_memory: dict[str, str] = {}


def _configured_url() -> str:
    url = getattr(settings, "VALKEY_URL", "") or ""
    if url:
        return url
    return getattr(settings, "REDIS_URL", "") or ""


def _membership_key(tenant_slug: str, user_id: str) -> str:
    return f"t:{tenant_slug}:control:membership:{user_id}"


def _redis_client() -> Any | None:
    url = _configured_url()
    if not url:
        return None
    try:
        import redis

        client = redis.from_url(url)
        client.ping()
        return client
    except Exception as exc:  # noqa: BLE001 — optional Redis
        logger.debug("tenant_cache: redis unavailable (%s); using memory", exc)
        return None


def invalidate_membership(tenant_id: str, user_id: str) -> None:
    """Drop membership cache entry and publish invalidate for Rust peers.

    ``tenant_id`` is the tenant *slug* (Valkey keyspace uses slug, not UUID).
    """
    if not tenant_id or not user_id:
        return
    key = _membership_key(tenant_id, user_id)
    payload = json.dumps(
        {
            "tenant_slug": tenant_id,
            "user_id": user_id,
            "action": "invalidate",
        }
    )
    client = _redis_client()
    if client is None:
        with _memory_lock:
            _memory.pop(key, None)
        logger.info(
            "StubMembershipInvalidate: deleted %s; publish deferred (no Valkey)",
            key,
        )
        return
    try:
        channel = membership_invalidate_channel(tenant_id)
        client.delete(key)
        client.publish(channel, payload)
        logger.info("invalidated membership %s on %s", key, channel)
    except Exception as exc:  # noqa: BLE001
        logger.warning("invalidate_membership failed for %s (%s)", key, exc)
        with _memory_lock:
            _memory.pop(key, None)


def clear_memory_cache() -> None:
    """Test helper — wipe the in-process fallback store."""
    with _memory_lock:
        _memory.clear()
