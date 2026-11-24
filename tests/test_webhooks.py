"""D1: the WorkOS webhook verifies WorkOS's real signature and only nudges a sync."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import pytest
from django.test import Client

from apps.identity.models import Membership, User, WorkOSEvent
from tests.workos_fakes import event, membership_data, sign, user_data

SECRET = "wh-test-secret"


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def secret(settings):
    settings.WORKOS_API_KEY = "sk_test_fake"
    settings.WORKOS_WEBHOOK_SECRET = SECRET
    return SECRET


@pytest.fixture
def nudges(monkeypatch):
    queued: list[tuple] = []
    monkeypatch.setattr(
        "apps.identity.tasks.sync_workos_events.delay", lambda *args: queued.append(args)
    )
    return queued


def _post(client, body: bytes, signature: str | None):
    headers = {"WorkOS-Signature": signature} if signature is not None else {}
    return client.post("/webhooks/workos", data=body, content_type="application/json", headers=headers)


def _body() -> bytes:
    return json.dumps(event("organization_membership.created", membership_data("om_1", "org_1", "user_42"))).encode()


@pytest.mark.django_db
def test_a_real_workos_signature_is_accepted_and_nudges_a_sync(client, secret, nudges):
    body = _body()
    resp = _post(client, body, sign(body, secret))
    assert resp.status_code == 200
    assert nudges == [()]
    # The webhook is only a nudge: the sync applies events, in order.
    assert User.objects.count() == 0
    assert Membership.objects.count() == 0
    assert WorkOSEvent.objects.count() == 0


@pytest.mark.django_db
def test_the_old_body_only_hmac_is_refused(client, secret, nudges):
    body = _body()
    body_only = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    resp = _post(client, body, f"v1={body_only}")
    assert resp.status_code == 401
    assert nudges == []


@pytest.mark.django_db
def test_a_signature_for_another_body_is_refused(client, secret, nudges):
    body = _body()
    other = json.dumps(event("user.created", user_data("user_x"))).encode()
    resp = _post(client, body, sign(other, secret))
    assert resp.status_code == 401
    assert nudges == []


@pytest.mark.django_db
def test_a_stale_timestamp_is_refused(client, secret, nudges):
    body = _body()
    ten_minutes_ago = int((time.time() - 600) * 1000)
    resp = _post(client, body, sign(body, secret, issued_at_ms=ten_minutes_ago))
    assert resp.status_code == 401
    assert nudges == []


@pytest.mark.django_db
def test_a_missing_signature_is_refused(client, secret, nudges):
    resp = _post(client, _body(), None)
    assert resp.status_code == 401
    assert nudges == []


@pytest.mark.django_db
def test_without_a_secret_every_delivery_is_refused(client, settings, nudges):
    settings.WORKOS_API_KEY = "sk_test_fake"
    settings.WORKOS_WEBHOOK_SECRET = ""
    body = _body()
    resp = _post(client, body, sign(body, ""))
    assert resp.status_code == 503
    assert nudges == []


def test_the_secret_has_no_default():
    from django.conf import settings

    assert settings.WORKOS_WEBHOOK_SECRET == os.environ.get("WORKOS_WEBHOOK_SECRET", "")


@pytest.mark.django_db
def test_a_broker_outage_still_answers_200(client, secret, monkeypatch):
    def refuse(*args):
        raise ConnectionError("broker down")

    monkeypatch.setattr("apps.identity.tasks.sync_workos_events.delay", refuse)
    body = _body()
    resp = _post(client, body, sign(body, secret))
    # The beat schedule syncs anyway; failing here would only make WorkOS retry.
    assert resp.status_code == 200
