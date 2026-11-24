"""D4: invalidating a cached membership tells the Rust peers on their channel."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from apps.tenancy import tenant_cache
from apps.tenancy.tenant_cache import invalidate_membership


def test_invalidate_membership_drops_the_cached_row_and_tells_peers(settings):
    client = MagicMock()
    redis = MagicMock()
    redis.from_url.return_value = client
    settings.VALKEY_URL = "redis://django-valkey.invalid:6379/0"
    with patch.dict("sys.modules", {"redis": redis}):
        invalidate_membership("acme", "user_inv")
    client.delete.assert_called_once_with("t:acme:control:membership:user_inv")
    channel, payload = client.publish.call_args.args
    assert channel == "t:acme:control_apikey:membership_invalidate"
    assert json.loads(payload) == {"tenant_slug": "acme", "user_id": "user_inv", "action": "invalidate"}


def test_invalidate_membership_without_valkey_clears_the_memory_row(settings):
    settings.VALKEY_URL = ""
    settings.REDIS_URL = ""
    key = "t:acme:control:membership:user_inv"
    tenant_cache._memory[key] = "cached"
    invalidate_membership("acme", "user_inv")
    assert key not in tenant_cache._memory


def test_the_org_mapping_no_longer_lives_in_the_tenant_cache():
    # It moved to the front's Valkey, without a TTL (apps.identity.control_mapping).
    assert not hasattr(tenant_cache, "set_tenant_slug_for_org")
    assert not hasattr(tenant_cache, "ORG_KEY_TTL_SECONDS")
