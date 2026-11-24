"""Celery tasks for the WorkOS account sync and the front's control mapping."""

from __future__ import annotations

import logging

from celery import shared_task
from celery.signals import worker_ready

from apps.identity.control_mapping import rewarm
from apps.identity.sync import run_sync

logger = logging.getLogger(__name__)


@shared_task(name="apps.identity.tasks.sync_workos_events", ignore_result=True)
def sync_workos_events() -> dict:
    return run_sync()


@shared_task(name="apps.identity.tasks.rewarm_control_mapping", ignore_result=True)
def rewarm_control_mapping() -> dict:
    return rewarm()


@worker_ready.connect(dispatch_uid="identity.rewarm_control_mapping_on_start")
def rewarm_on_worker_start(**kwargs) -> None:
    """Restore the front's mapping from Postgres whenever a worker starts."""
    try:
        logger.info("control mapping re-warmed: %s", rewarm())
    except Exception:  # noqa: BLE001 — the beat schedule re-warms again shortly
        logger.exception("control mapping re-warm on worker start failed")
