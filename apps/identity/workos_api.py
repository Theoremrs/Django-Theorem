"""The WorkOS client, built from settings."""

from __future__ import annotations

from functools import lru_cache

from django.conf import settings
from workos import WorkOSClient


def workos_configured() -> bool:
    return bool(settings.WORKOS_API_KEY)


@lru_cache(maxsize=4)
def _client(api_key: str, client_id: str) -> WorkOSClient:
    return WorkOSClient(api_key=api_key or None, client_id=client_id or None)


def workos_client() -> WorkOSClient:
    return _client(settings.WORKOS_API_KEY, settings.WORKOS_CLIENT_ID)
