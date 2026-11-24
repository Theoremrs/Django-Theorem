"""pytest configuration for the Django-Theorem control plane.

pytest-django supplies the isolated test database and applies migrations. Do
not override its ``django_db_setup`` fixture with a no-op: database-backed
contract tests depend on the control-schema tables being present.
"""

import pytest


@pytest.fixture(autouse=True)
def _front_mapping_in_memory(settings):
    """Keep the front's control mapping in process memory unless a test points
    it at a Valkey, and start every test with it empty."""
    from apps.identity.control_mapping import clear_memory_mapping

    settings.FRONT_VALKEY_URL = ""
    clear_memory_mapping()
    yield
    clear_memory_mapping()
