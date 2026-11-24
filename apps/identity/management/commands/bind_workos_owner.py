"""Bind the owner's WorkOS organization and user to the tenant and principal they
already have, so signing in through WorkOS lands on the same tenant, store and
sprite as before.

Run once at release, with the real ids, before the sync starts:

    python manage.py bind_workos_owner \\
        --organization org_... --user user_... --principal identity:<12 hex>

The sync refuses to run while the owner tenant has no organization, so the
owner's first WorkOS events can never make a second tenant. Rerunning with the
same ids changes nothing.
"""

from __future__ import annotations

import re

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.identity.control_mapping import (
    front_valkey_configured,
    write_org_tenant,
    write_user_principal,
)
from apps.identity.models import LEGACY_PRINCIPAL_PATTERN, Membership, User
from apps.tenancy.models import Tenant
from apps.tenancy.tenant_cache import invalidate_membership


class Command(BaseCommand):
    help = "Bind a WorkOS organization and user to an existing tenant and legacy principal."

    def add_arguments(self, parser):
        parser.add_argument("--organization", required=True, help="WorkOS organization id (org_...)")
        parser.add_argument("--user", required=True, help="The owner's WorkOS user id (user_...)")
        parser.add_argument("--principal", required=True, help="The owner's principal, identity:<12 hex>")
        parser.add_argument(
            "--tenant",
            default=settings.WORKOS_OWNER_TENANT_SLUG,
            help="Slug of the existing tenant (default: WORKOS_OWNER_TENANT_SLUG)",
        )

    def handle(self, *args, organization: str, user: str, principal: str, tenant: str, **options):
        if not organization.startswith("org_"):
            raise CommandError(f"--organization must be a WorkOS organization id (org_...), not {organization!r}")
        if not user.startswith("user_"):
            raise CommandError(f"--user must be a WorkOS user id (user_...), not {user!r}")
        if not re.fullmatch(LEGACY_PRINCIPAL_PATTERN, principal):
            raise CommandError(f"--principal must be identity:<12 hex>, not {principal!r}")

        with transaction.atomic():
            owner_tenant = Tenant.objects.select_for_update().filter(slug=tenant).first()
            if owner_tenant is None:
                raise CommandError(f"there is no tenant {tenant!r}")
            if owner_tenant.workos_organization_id not in ("", organization):
                raise CommandError(
                    f"tenant {tenant} is already bound to {owner_tenant.workos_organization_id}"
                )
            other_tenant = (
                Tenant.objects.filter(workos_organization_id=organization).exclude(pk=owner_tenant.pk).first()
            )
            if other_tenant is not None:
                raise CommandError(f"{organization} already belongs to tenant {other_tenant.slug}")
            owner = User.objects.select_for_update().filter(workos_user_id=user).first()
            if owner is not None and owner.legacy_principal not in ("", principal):
                raise CommandError(f"{user} already holds {owner.legacy_principal}")
            holder = User.objects.filter(legacy_principal=principal).exclude(workos_user_id=user).first()
            if holder is not None:
                raise CommandError(f"{principal} is already bound to {holder.workos_user_id}")

            already_bound = (
                owner_tenant.workos_organization_id == organization
                and owner is not None
                and owner.legacy_principal == principal
                and Membership.objects.filter(
                    tenant=owner_tenant, user=owner, role=Membership.Role.OWNER
                ).exists()
            )

            # update_fields, and an existing row: the signup signals provision
            # only on create, so the tenant keeps its store and sprite.
            owner_tenant.workos_organization_id = organization
            owner_tenant.save(update_fields=["workos_organization_id", "updated_at"])
            if owner is None:
                owner = User(workos_user_id=user)
            owner.legacy_principal = principal
            owner.self_serve_pending = False
            owner.is_active = True
            owner.save()
            Membership.objects.update_or_create(
                tenant=owner_tenant, user=owner, defaults={"role": Membership.Role.OWNER}
            )
            write_org_tenant(organization, owner_tenant.slug)
            write_user_principal(user, principal)
            slug = owner_tenant.slug
            transaction.on_commit(lambda: invalidate_membership(slug, user), robust=True)

        if already_bound:
            self.stdout.write(f"{user} and {organization} were already bound to {slug} as {principal}")
        else:
            self.stdout.write(f"bound {organization} to tenant {slug} and {user} to {principal}, as owner")
        if not front_valkey_configured():
            self.stderr.write(
                "FRONT_VALKEY_URL is not set, so the front has not seen this binding: "
                "set it and run warm_control_mapping"
            )
