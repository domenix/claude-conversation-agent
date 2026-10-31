"""Coordinator for the Claude integration."""

from __future__ import annotations

from datetime import datetime
from typing import override

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .agent import ClaudeAgent, ClaudeAgentError, CredentialsInvalidError
from .const import (
    CONF_API_KEY,
    CONF_OAUTH_EXPIRES_AT,
    CONF_OAUTH_TOKEN,
    DEFAULT_RUN_TIMEOUT_S,
    DEFAULT_TOOL_TIMEOUT_S,
    DOMAIN,
    ISSUE_TOKEN_EXPIRED,
    ISSUE_TOKEN_EXPIRING,
    LOGGER,
    MODEL_DEFAULT,
    MODELS,
    TOKEN_EXPIRY_CHECK_INTERVAL,
    TOKEN_EXPIRY_WARNING,
    Model,
)

type ClaudeCodeConfigEntry = ConfigEntry[ClaudeCodeCoordinator]


class ClaudeCodeCoordinator(DataUpdateCoordinator[list[Model]]):
    """Owns the in-process agent, the session map and credential state.

    No polling: credentials are verified and the model list fetched once at
    setup (the list again whenever the model options are opened).
    ``update_interval`` is left unset so nothing runs on a timer.
    """

    config_entry: ClaudeCodeConfigEntry
    agent: ClaudeAgent

    def __init__(
        self, hass: HomeAssistant, config_entry: ClaudeCodeConfigEntry
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=config_entry,
            name=config_entry.title,
            update_method=self.async_update_data,
            always_update=False,
        )
        # Maps HA conversation IDs to Claude session IDs to resume.
        self.session_ids: dict[str, str] = {}
        self.run_timeout = DEFAULT_RUN_TIMEOUT_S
        self.tool_timeout = DEFAULT_TOOL_TIMEOUT_S

    @override
    async def _async_setup(self) -> None:
        """Set up the coordinator."""
        self.agent = ClaudeAgent(
            self.hass,
            self.config_entry.data.get(CONF_OAUTH_TOKEN, ""),
            self.config_entry.data.get(CONF_API_KEY, ""),
        )
        # Warn before the token expires, rather than letting the first 401 be
        # the notification.
        self.config_entry.async_on_unload(
            async_track_time_interval(
                self.hass,
                self._async_expiry_tick,
                TOKEN_EXPIRY_CHECK_INTERVAL,
                cancel_on_shutdown=True,
            )
        )

    @property
    def token_expires_at(self) -> datetime | None:
        """Return when the OAuth token expires, if the issuer told us."""
        raw = self.config_entry.data.get(CONF_OAUTH_EXPIRES_AT)
        return dt_util.parse_datetime(raw) if raw else None

    async def _async_expiry_tick(self, _now: datetime) -> None:
        """Re-check the token expiry on a timer."""
        self.async_check_token_expiry()

    @callback
    def async_check_token_expiry(self) -> None:
        """Raise or clear a repair issue about the OAuth token's expiry.

        An expired token also flags the entry for reauth, so the user gets the
        standard "reconfigure" prompt rather than only a repair.
        """
        expires_at = self.token_expires_at
        for issue_id in (ISSUE_TOKEN_EXPIRING, ISSUE_TOKEN_EXPIRED):
            ir.async_delete_issue(self.hass, DOMAIN, self._issue_id(issue_id))
        if expires_at is None:
            return

        remaining = expires_at - dt_util.utcnow()
        if remaining > TOKEN_EXPIRY_WARNING:
            return

        expired = remaining.total_seconds() <= 0
        issue_id = ISSUE_TOKEN_EXPIRED if expired else ISSUE_TOKEN_EXPIRING
        ir.async_create_issue(
            self.hass,
            DOMAIN,
            self._issue_id(issue_id),
            is_fixable=False,
            severity=ir.IssueSeverity.ERROR if expired else ir.IssueSeverity.WARNING,
            translation_key=issue_id,
            translation_placeholders={
                "name": self.config_entry.title,
                "expires_at": dt_util.as_local(expires_at).strftime("%Y-%m-%d %H:%M"),
                "days": str(max(0, remaining.days)),
            },
        )
        if expired:
            LOGGER.warning(
                "Claude OAuth token expired at %s; reauthentication needed",
                expires_at.isoformat(),
            )
            self.async_set_auth_failed()

    def _issue_id(self, issue: str) -> str:
        """Return an issue ID scoped to this config entry."""
        return f"{issue}_{self.config_entry.entry_id}"

    async def async_update_data(self) -> list[Model]:
        """Verify credentials once; return the available models."""
        if not self.agent.authenticated:
            raise ConfigEntryAuthFailed(
                translation_domain="claude",
                translation_key="not_authenticated",
            )
        try:
            await self.agent.async_verify()
        except CredentialsInvalidError as err:
            raise ConfigEntryAuthFailed(
                translation_domain="claude",
                translation_key="not_authenticated",
            ) from err
        except ClaudeAgentError as err:
            raise UpdateFailed(
                translation_domain="claude",
                translation_key="run_error",
                translation_placeholders={"message": str(err)},
            ) from err
        self.async_check_token_expiry()
        return await self._async_fetch_models() or list(MODELS)

    async def async_refresh_models(self) -> list[Model]:
        """Re-fetch the model list, keeping the current one if that fails."""
        if models := await self._async_fetch_models():
            self.async_set_updated_data(models)
        return self.data or list(MODELS)

    async def _async_fetch_models(self) -> list[Model] | None:
        """Return "Account default" plus the API's models, or None on failure."""
        try:
            fetched = await self.agent.async_list_models()
        except (aiohttp.ClientError, TimeoutError, KeyError, TypeError) as err:
            LOGGER.warning("Could not fetch the Claude model list: %r", err)
            return None
        if not fetched:
            LOGGER.warning("The Claude model list came back empty")
            return None
        return [MODEL_DEFAULT, *fetched]

    @callback
    def async_set_auth_failed(self) -> None:
        """Flag the entry for reauth after a run rejected the credentials."""
        self.config_entry.async_start_reauth(self.hass)

    @callback
    def get_model_info(self, model_id: str) -> Model:
        """Get model info for a given model ID."""
        for model in self.data or MODELS:
            if model.id == model_id:
                return model
        return Model(
            id=model_id,
            display_name="Account default" if model_id == "default" else model_id,
        )
