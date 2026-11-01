"""Test configuration.

The test module imports Home Assistant's own helpers from ``tests.common`` and
``tests.test_util.aiohttp``, which exist in a home-assistant/core checkout. When
running against this repository instead, point those names at the copies shipped
in ``pytest_homeassistant_custom_component`` so the same file runs either way.
"""

from collections.abc import Iterator
import sys
import types

import pytest


def _alias_core_test_helpers() -> None:
    try:
        import tests.common  # noqa: F401
    except ImportError:
        pass
    else:
        return  # running inside a core checkout

    import pytest_homeassistant_custom_component.common as common
    import pytest_homeassistant_custom_component.test_util.aiohttp as aiohttp_mock

    package = sys.modules.setdefault("tests", types.ModuleType("tests"))
    package.__path__ = getattr(package, "__path__", [])
    sys.modules["tests.common"] = common

    test_util = types.ModuleType("tests.test_util")
    test_util.__path__ = []
    sys.modules["tests.test_util"] = test_util
    sys.modules["tests.test_util.aiohttp"] = aiohttp_mock


def pytest_configure(config: object) -> None:
    """Prepare the module namespace before the test module is imported."""
    _alias_core_test_helpers()


def pytest_sessionstart(session: object) -> None:
    """Import the integration so its package exposes it by attribute."""
    import custom_components.claude  # noqa: F401


@pytest.fixture(autouse=True)
def fail_on_ha_usage_report(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Fail a test when Home Assistant reports this integration's use of its API.

    For a custom integration, Home Assistant only logs a deprecated call until the
    release that breaks it, so nothing else fails while there is still time to fix.
    """
    yield
    reports = [
        record.getMessage()
        for when in ("setup", "call")
        for record in caplog.get_records(when)
        if record.name == "homeassistant.helpers.frame"
        and "custom integration 'claude'" in record.getMessage()
    ]
    assert not reports
