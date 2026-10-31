"""Config flow for the Claude integration."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any, override

import voluptuous as vol

from homeassistant.config_entries import (
    SOURCE_REAUTH,
    SOURCE_RECONFIGURE,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME, CONF_PROMPT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.typing import VolDictType

from . import oauth
from .agent import ClaudeAgent, ClaudeAgentError, CredentialsInvalidError
from .cli import async_prewarm_cli
from .const import (
    CONF_API_KEY,
    CONF_CHAT_MODEL,
    CONF_CODE_EXECUTION,
    CONF_MAX_TOKENS,
    CONF_OAUTH_EXPIRES_AT,
    CONF_OAUTH_TOKEN,
    CONF_RECOMMENDED,
    CONF_RUN_TIMEOUT,
    CONF_SHOW_ACTIVITY,
    CONF_THINKING_BUDGET,
    CONF_THINKING_EFFORT,
    CONF_WEB_FETCH,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_USER_LOCATION,
    DEFAULT,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    THINKING_EFFORT_OPTIONS,
)
from .coordinator import ClaudeCodeConfigEntry

_LOGGER = logging.getLogger(__name__)

CONF_CODE = "code"

_PASSWORD = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_OAUTH_TOKEN): _PASSWORD,
        vol.Optional(CONF_API_KEY): _PASSWORD,
    }
)

DEFAULT_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

DEFAULT_AI_TASK_OPTIONS = {
    CONF_RECOMMENDED: True,
}

DEFAULT_SUBENTRIES = [
    {
        "subentry_type": "conversation",
        "data": DEFAULT_CONVERSATION_OPTIONS,
        "title": DEFAULT_CONVERSATION_NAME,
        "unique_id": None,
    },
    {
        "subentry_type": "ai_task_data",
        "data": DEFAULT_AI_TASK_OPTIONS,
        "title": DEFAULT_AI_TASK_NAME,
        "unique_id": None,
    },
]


def _clean(user_input: dict[str, Any]) -> dict[str, Any]:
    """Strip all whitespace from pasted credentials, keeping any expiry.

    Tokens never contain whitespace; a paste can drag in leading/trailing
    spaces or a line-wrap break mid-token.
    """
    cleaned: dict[str, Any] = {}
    for key in (CONF_OAUTH_TOKEN, CONF_API_KEY):
        if value := user_input.get(key):
            stripped = "".join(str(value).split())
            if stripped:
                cleaned[key] = stripped
    if cleaned.get(CONF_OAUTH_TOKEN) and (
        expires_at := user_input.get(CONF_OAUTH_EXPIRES_AT)
    ):
        cleaned[CONF_OAUTH_EXPIRES_AT] = expires_at
    return cleaned


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate that the credentials work, raising on failure."""
    agent = ClaudeAgent(
        hass,
        data.get(CONF_OAUTH_TOKEN, ""),
        data.get(CONF_API_KEY, ""),
    )
    await agent.async_verify()


class ClaudeCodeConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Claude."""

    VERSION = 1

    _oauth_url: str = ""
    _oauth_verifier: str = ""

    async def _async_validate_and_create(
        self, user_input: dict[str, Any], step_id: str
    ) -> ConfigFlowResult:
        """Validate credentials, then create or update the entry."""
        errors: dict[str, str] = {}
        cleaned = _clean(user_input)

        if not cleaned:
            errors["base"] = "no_credentials"
        else:
            try:
                await validate_input(self.hass, cleaned)
            except CredentialsInvalidError as err:
                _LOGGER.warning("Claude credentials rejected: %s", err)
                errors["base"] = "invalid_auth"
            except ClaudeAgentError as err:
                _LOGGER.error("Claude verification failed: %s", err)
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        if errors:
            return self.async_show_form(
                step_id=step_id,
                data_schema=self.add_suggested_values_to_schema(
                    STEP_USER_DATA_SCHEMA, user_input
                ),
                errors=errors,
            )

        # Replace (not merge) the credential dict so switching between an OAuth
        # token and an API key doesn't leave the old one behind.
        if self.source == SOURCE_REAUTH:
            return self.async_update_reload_and_abort(
                self._get_reauth_entry(), data=cleaned
            )
        if self.source == SOURCE_RECONFIGURE:
            return self.async_update_reload_and_abort(
                self._get_reconfigure_entry(), data=cleaned
            )
        return self.async_create_entry(
            title="Claude",
            data=cleaned,
            subentries=DEFAULT_SUBENTRIES,
        )

    @override
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Offer logging in with Claude or entering credentials manually."""
        # Start the CLI download now so it's likely cached by the time the
        # user finishes picking a credential method.
        async_prewarm_cli(self.hass)
        return self.async_show_menu(
            step_id="user", menu_options=["oauth", "manual"]
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the Claude credentials on the existing entry."""
        return self.async_show_menu(
            step_id="reconfigure", menu_options=["oauth", "manual"]
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Perform reauth after the credentials were rejected."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Offer the credential methods again after a rejection."""
        return self.async_show_menu(
            step_id="reauth_confirm", menu_options=["oauth", "manual"]
        )

    async def async_step_oauth(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Log in with Claude via the OAuth device flow (no local install)."""
        errors: dict[str, str] = {}
        if user_input is not None and user_input.get(CONF_CODE):
            try:
                issued = await oauth.async_exchange_code(
                    async_get_clientsession(self.hass),
                    user_input[CONF_CODE],
                    self._oauth_verifier,
                )
            except oauth.OAuthError:
                _LOGGER.exception("OAuth token exchange failed")
                errors["base"] = "oauth_failed"
            else:
                data: dict[str, Any] = {CONF_OAUTH_TOKEN: issued.token}
                if issued.expires_at:
                    _LOGGER.debug(
                        "Claude OAuth token issued, expires %s",
                        issued.expires_at.isoformat(),
                    )
                    data[CONF_OAUTH_EXPIRES_AT] = issued.expires_at.isoformat()
                return await self._async_validate_and_create(data, "manual")

        # (Re)build the authorize URL on entry and after an error; the verifier
        # stays valid, so the same link yields a fresh code to retry with.
        self._oauth_url, self._oauth_verifier = oauth.build_login()
        return self.async_show_form(
            step_id="oauth",
            data_schema=vol.Schema({vol.Required(CONF_CODE): str}),
            description_placeholders={"url": self._oauth_url},
            errors=errors or None,
        )

    async def async_step_manual(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Enter an OAuth token or API key manually."""
        if user_input is not None:
            return await self._async_validate_and_create(user_input, "manual")
        return self.async_show_form(
            step_id="manual", data_schema=STEP_USER_DATA_SCHEMA
        )

    @classmethod
    @callback
    @override
    def async_get_supported_subentry_types(
        cls, config_entry: ClaudeCodeConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": LLMSubentryFlowHandler,
            "ai_task_data": LLMSubentryFlowHandler,
        }


class LLMSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation and AI task subentries."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        if self._subentry_type == "ai_task_data":
            self.options = DEFAULT_AI_TASK_OPTIONS.copy()
        else:
            self.options = DEFAULT_CONVERSATION_OPTIONS.copy()
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Set initial options."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]
        if suggested_llm_apis := self.options.get(CONF_LLM_HASS_API):
            if isinstance(suggested_llm_apis, str):
                suggested_llm_apis = [suggested_llm_apis]
            known_apis = {api.id for api in llm.async_get_apis(self.hass)}
            self.options[CONF_LLM_HASS_API] = [
                api for api in suggested_llm_apis if api in known_apis
            ]

        step_schema: VolDictType = {}

        if self._is_new:
            if self._subentry_type == "ai_task_data":
                default_name = DEFAULT_AI_TASK_NAME
            else:
                default_name = DEFAULT_CONVERSATION_NAME
            step_schema[vol.Required(CONF_NAME, default=default_name)] = str

        if self._subentry_type == "conversation":
            step_schema.update(
                {
                    vol.Optional(CONF_PROMPT): TemplateSelector(),
                    vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                        SelectSelectorConfig(options=hass_apis, multiple=True)
                    ),
                }
            )

        step_schema[
            vol.Required(
                CONF_RECOMMENDED, default=self.options.get(CONF_RECOMMENDED, False)
            )
        ] = bool

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)

            if user_input[CONF_RECOMMENDED]:
                if self._is_new:
                    return self.async_create_entry(
                        title=user_input.pop(CONF_NAME),
                        data=user_input,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=user_input,
                )

            self.options.update(user_input)
            if (
                CONF_LLM_HASS_API in self.options
                and CONF_LLM_HASS_API not in user_input
            ):
                self.options.pop(CONF_LLM_HASS_API)
            return await self.async_step_model()

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), self.options
            ),
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage model-specific options."""
        errors: dict[str, str] = {}

        step_schema: VolDictType = {
            vol.Optional(
                CONF_CHAT_MODEL,
                default=DEFAULT[CONF_CHAT_MODEL],
            ): SelectSelector(
                SelectSelectorConfig(
                    # Fetch when the form opens, not again on submit.
                    options=await self._async_get_model_list(
                        refresh=user_input is None
                    ),
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=DEFAULT[CONF_MAX_TOKENS],
            ): cv.positive_int,
            vol.Optional(
                CONF_THINKING_BUDGET,
                default=DEFAULT[CONF_THINKING_BUDGET],
            ): vol.All(
                NumberSelector(NumberSelectorConfig(min=0, max=64000)),
                vol.Coerce(int),
            ),
            vol.Optional(
                CONF_THINKING_EFFORT,
                default=DEFAULT[CONF_THINKING_EFFORT],
            ): SelectSelector(
                SelectSelectorConfig(
                    options=THINKING_EFFORT_OPTIONS,
                    translation_key=CONF_THINKING_EFFORT,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_CODE_EXECUTION,
                default=DEFAULT[CONF_CODE_EXECUTION],
            ): bool,
            vol.Optional(
                CONF_WEB_SEARCH,
                default=DEFAULT[CONF_WEB_SEARCH],
            ): bool,
            vol.Optional(
                CONF_WEB_SEARCH_USER_LOCATION,
                default=DEFAULT[CONF_WEB_SEARCH_USER_LOCATION],
            ): bool,
            vol.Optional(
                CONF_WEB_FETCH,
                default=DEFAULT[CONF_WEB_FETCH],
            ): bool,
            vol.Optional(
                CONF_SHOW_ACTIVITY,
                default=DEFAULT[CONF_SHOW_ACTIVITY],
            ): bool,
            vol.Optional(
                CONF_RUN_TIMEOUT,
                default=DEFAULT[CONF_RUN_TIMEOUT],
            ): vol.All(
                NumberSelector(NumberSelectorConfig(min=0, max=1800)),
                vol.Coerce(int),
            ),
        }

        if user_input is not None:
            if (
                user_input.get(CONF_THINKING_BUDGET, 0)
                and user_input.get(CONF_THINKING_BUDGET, 0)
                >= user_input.get(CONF_MAX_TOKENS, DEFAULT[CONF_MAX_TOKENS])
            ):
                errors[CONF_THINKING_BUDGET] = "thinking_budget_too_large"

            self.options.update(user_input)

            if not errors:
                if self._is_new:
                    return self.async_create_entry(
                        title=self.options.pop(CONF_NAME),
                        data=self.options,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=self.options,
                )

        return self.async_show_form(
            step_id="model",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), self.options
            ),
            errors=errors or None,
            last_step=True,
        )

    async def _async_get_model_list(self, refresh: bool) -> list[SelectOptionDict]:
        """Get the models to offer, re-fetched from Anthropic if ``refresh``."""
        coordinator = self._get_entry().runtime_data
        models = (
            await coordinator.async_refresh_models() if refresh else coordinator.data
        )
        return [
            SelectOptionDict(label=model.display_name, value=model.id)
            for model in models or []
        ]
