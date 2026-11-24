"""WorkOS webhook: verify WorkOS's signature, then nudge the events sync.

The webhook applies nothing itself. Deliveries can arrive late, twice or out of
order; the sync reads the Events API in order, so it is the only writer.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.http import HttpRequest
from ninja import Router
from ninja.errors import HttpError

from apps.identity import tasks
from apps.identity.workos_api import workos_client, workos_configured

logger = logging.getLogger(__name__)
router = Router(tags=["webhooks"])


@router.post("/workos")
def workos_webhook(request: HttpRequest):
    secret = settings.WORKOS_WEBHOOK_SECRET
    if not secret or not workos_configured():
        raise HttpError(503, "WorkOS webhooks are not configured")
    signature = request.headers.get("WorkOS-Signature")
    if not signature:
        raise HttpError(401, "missing WorkOS-Signature")
    try:
        # WorkOS signs "<timestamp>.<body>" and refuses deliveries older than 180 s.
        workos_client().webhooks.verify_header(
            event_body=request.body, event_signature=signature, secret=secret
        )
    except ValueError as exc:
        raise HttpError(401, "invalid webhook signature") from exc
    try:
        tasks.sync_workos_events.delay()
    except Exception as exc:  # noqa: BLE001 — the beat schedule syncs regardless
        logger.warning("could not queue the WorkOS sync: %s", exc)
    return {"ok": True}
