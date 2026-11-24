"""Write the front's control mapping (org to tenant, user to principal) from Postgres."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from apps.identity.control_mapping import front_valkey_configured, rewarm


class Command(BaseCommand):
    help = (
        "Backfill control:org:{org_id}:tenant_slug and control:user:{id}:principal "
        "in the front's Valkey (FRONT_VALKEY_URL) from Postgres. Safe to rerun."
    )

    def handle(self, *args, **options):
        if not front_valkey_configured():
            raise CommandError("set FRONT_VALKEY_URL to the Valkey the front reads")
        counts = rewarm()
        self.stdout.write(
            f"wrote {counts['organizations']} organization and "
            f"{counts['principals']} principal mappings"
        )
