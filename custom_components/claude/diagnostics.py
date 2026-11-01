"""Diagnostics support for Claude."""

from typing import TYPE_CHECKING, Any

from homeassistant.components.diagnostics import async_redact_data, entity_entry_as_dict
from homeassistant.const import CONF_PROMPT
from homeassistant.helpers import entity_registry as er

from .const import CONF_API_KEY, CONF_OAUTH_TOKEN

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .coordinator import ClaudeCodeConfigEntry

TO_REDACT = {CONF_OAUTH_TOKEN, CONF_API_KEY, CONF_PROMPT}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ClaudeCodeConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    return {
        "title": entry.title,
        "entry_id": entry.entry_id,
        "entry_version": f"{entry.version}.{entry.minor_version}",
        "state": entry.state.value,
        "data": async_redact_data(entry.data, TO_REDACT),
        "options": async_redact_data(entry.options, TO_REDACT),
        "models": [
            {"id": model.id, "display_name": model.display_name}
            for model in entry.runtime_data.data or []
        ],
        "subentries": {
            subentry.subentry_id: {
                "title": subentry.title,
                "subentry_type": subentry.subentry_type,
                "data": async_redact_data(subentry.data, TO_REDACT),
            }
            for subentry in entry.subentries.values()
        },
        "entities": {
            entity_entry.entity_id: entity_entry_as_dict(entity_entry)
            for entity_entry in er.async_entries_for_config_entry(
                er.async_get(hass), entry.entry_id
            )
        },
    }
