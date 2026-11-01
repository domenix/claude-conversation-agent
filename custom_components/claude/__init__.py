"""The Claude integration.

A pure Home Assistant custom integration that runs conversation agents and AI
Tasks on your Claude subscription. It drives the Claude Agent SDK in-process
(the SDK bundles the Claude Code CLI) — there is no add-on and no external
service.
"""

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .cli import async_prewarm_cli
from .const import DOMAIN
from .coordinator import ClaudeCodeConfigEntry, ClaudeCodeCoordinator

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Claude."""
    # Pre-warm the CLI download in the background so it's cached before a config
    # entry (or the config flow) needs it.
    async_prewarm_cli(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ClaudeCodeConfigEntry) -> bool:
    """Set up Claude from a config entry."""
    coordinator = ClaudeCodeCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ClaudeCodeConfigEntry) -> bool:
    """Unload Claude."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(
    hass: HomeAssistant, entry: ClaudeCodeConfigEntry
) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)
