"""WorkOS shadow rows (control_user, control_membership) and the event sync
that keeps them current (control_workosevent, control_workossyncstate)."""

from __future__ import annotations

import uuid
from datetime import timedelta

from django.core.validators import RegexValidator
from django.db import models
from django.utils import timezone

from apps.tenancy.models import Tenant

LEGACY_PRINCIPAL_PATTERN = r"^identity:[0-9a-f]{12}$"


class User(models.Model):
    """Shadow of a WorkOS user. Identity is theirs; authorization is ours."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workos_user_id = models.CharField(max_length=128, unique=True)
    email = models.EmailField(blank=True, default="")
    display_name = models.CharField(max_length=255, blank=True, default="")
    is_active = models.BooleanField(default=True)
    # The principal this user signed in as before WorkOS (``identity:<12 hex>``).
    # Kept here so the front's Valkey mapping can be rebuilt from Postgres.
    legacy_principal = models.CharField(
        max_length=64,
        blank=True,
        default="",
        validators=[RegexValidator(LEGACY_PRINCIPAL_PATTERN)],
    )
    # Signed up with no organization: the sync creates one for them in WorkOS.
    self_serve_pending = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "control_user"
        ordering = ["email"]
        constraints = [
            models.UniqueConstraint(
                fields=["legacy_principal"],
                condition=~models.Q(legacy_principal=""),
                name="control_user_legacy_principal_unique",
            )
        ]

    def __str__(self) -> str:
        return self.email or self.workos_user_id


class Membership(models.Model):
    class Role(models.TextChoices):
        OWNER = "owner", "Owner"
        ADMIN = "admin", "Admin"
        MEMBER = "member", "Member"
        VIEWER = "viewer", "Viewer"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        Tenant, on_delete=models.CASCADE, related_name="memberships", db_column="tenant_id"
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="memberships", db_column="user_id"
    )
    role = models.CharField(max_length=32, choices=Role.choices, default=Role.MEMBER)
    workos_membership_id = models.CharField(max_length=128, blank=True, default="", db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "control_membership"
        unique_together = [("tenant", "user")]

    def __str__(self) -> str:
        return f"{self.user} @ {self.tenant.slug} ({self.role})"


class WorkOSEvent(models.Model):
    """One event read from the WorkOS Events API, kept until it has been applied.

    Rows are applied in ``id`` order, which is the order the API served them.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        DEFERRED = "deferred", "Deferred"
        APPLIED = "applied", "Applied"

    id = models.BigAutoField(primary_key=True)
    event_id = models.CharField(max_length=128, unique=True)
    event_type = models.CharField(max_length=64)
    occurred_at = models.DateTimeField(null=True, blank=True)
    payload = models.JSONField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    attempts = models.PositiveIntegerField(default=0)
    last_error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    applied_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "control_workosevent"
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.event_type} {self.event_id} ({self.status})"


class WorkOSSyncState(models.Model):
    """The Events API cursor, and the lease that keeps the sync to one consumer."""

    EVENTS = "events"

    name = models.CharField(max_length=32, primary_key=True)
    cursor = models.CharField(max_length=128, blank=True, default="")
    claimant = models.CharField(max_length=128, blank=True, default="")
    claimed_until = models.DateTimeField(null=True, blank=True)
    last_synced_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "control_workossyncstate"

    def __str__(self) -> str:
        return f"{self.name} after {self.cursor or '(start)'}"

    @classmethod
    def load(cls) -> "WorkOSSyncState":
        state, _ = cls.objects.get_or_create(name=cls.EVENTS)
        return state

    def claim(self, claimant: str, hold: timedelta) -> bool:
        """Take the sync for ``hold``, unless another worker holds it."""
        now = timezone.now()
        taken = (
            WorkOSSyncState.objects.filter(pk=self.pk)
            .filter(
                models.Q(claimed_until__isnull=True)
                | models.Q(claimed_until__lt=now)
                | models.Q(claimant=claimant)
            )
            .update(claimant=claimant, claimed_until=now + hold, updated_at=now)
        )
        return taken == 1

    def release(self, claimant: str) -> None:
        WorkOSSyncState.objects.filter(pk=self.pk, claimant=claimant).update(
            claimant="", claimed_until=None, last_synced_at=timezone.now(), updated_at=timezone.now()
        )
