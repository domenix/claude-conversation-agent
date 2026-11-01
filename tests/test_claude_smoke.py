"""Smoke tests for the claude custom integration.

See tests/README.md for how to run them. The Claude Agent SDK is never really
invoked; ``async_verify`` and ``query`` are patched with fakes.
"""

from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import aiohttp
from multidict import CIMultiDict, CIMultiDictProxy
import probatio
import pytest
from yarl import URL

from homeassistant.config_entries import ConfigEntryState, ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, intent, issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util

from tests.common import MockConfigEntry, async_fire_time_changed
from tests.test_util.aiohttp import AiohttpClientMocker

DOMAIN = "claude"


# --- Fake Agent SDK message objects (matched by class name) -------------------


class SystemMessage:
    def __init__(self, subtype: str, data: dict[str, Any]) -> None:
        self.subtype = subtype
        self.data = data


class StreamEvent:
    def __init__(self, event: dict[str, Any]) -> None:
        self.event = event


class ToolUseBlock:
    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.input = input


class ToolResultBlock:
    def __init__(self, tool_use_id: str, content: Any) -> None:
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error: bool | None = False


class AssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class UserMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class ResultMessage:
    def __init__(
        self,
        *,
        subtype: str = "success",
        is_error: bool = False,
        session_id: str = "sess",
        structured_output: Any = None,
    ) -> None:
        self.subtype = subtype
        self.is_error = is_error
        self.session_id = session_id
        self.structured_output = structured_output
        self.result = None


def _text_deltas(*chunks: str) -> list[StreamEvent]:
    return [
        StreamEvent(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": c}}
        )
        for c in chunks
    ]


def _fake_query(messages: list[Any], seen_options: list[Any] | None = None):
    """Return a patch target that yields the given messages from query()."""

    def _query(prompt: Any, options: Any) -> AsyncIterator[Any]:
        if seen_options is not None:
            seen_options.append(options)

        async def _gen() -> AsyncIterator[Any]:
            # Drain the streaming prompt so the caller's generator is exercised.
            async for _ in prompt:
                pass
            for message in messages:
                yield message

        return _gen()

    return _query


@pytest.fixture(autouse=True)
def enable_custom(enable_custom_integrations: None) -> None:
    """Enable loading custom integrations."""


@pytest.fixture(autouse=True)
def mock_verify() -> Generator[AsyncMock]:
    """Make credential verification succeed without calling the CLI."""
    with patch(
        "custom_components.claude.agent.ClaudeAgent.async_verify",
        AsyncMock(return_value=None),
    ) as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_prepare() -> Generator[AsyncMock]:
    """Skip creating workspace/home directories."""
    with patch(
        "custom_components.claude.agent.ClaudeAgent.async_prepare",
        AsyncMock(return_value=None),
    ) as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_prewarm() -> Generator[None]:
    """Never download the CLI: setup and the config flow both pre-warm it."""
    with (
        patch("custom_components.claude.async_prewarm_cli"),
        patch("custom_components.claude.config_flow.async_prewarm_cli"),
    ):
        yield


MODELS_URL = "https://api.anthropic.com/v1/models"
API_MODELS = [
    {"id": "claude-new-9", "display_name": "Claude New 9"},
    {"id": "claude-old-8", "display_name": "Claude Old 8"},
]


@pytest.fixture(autouse=True)
def mock_models_api(aioclient_mock: AiohttpClientMocker) -> AiohttpClientMocker:
    """Answer Anthropic's model list for every test (no real network)."""
    aioclient_mock.get(
        MODELS_URL,
        json={"data": API_MODELS, "has_more": False, "last_id": "claude-old-8"},
    )
    return aioclient_mock


def _entry(
    conversation_data: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="Claude",
        data=data or {"oauth_token": "sk-ant-oat01-secret"},
        subentries_data=[
            ConfigSubentryData(
                data=conversation_data or {"recommended": True},
                subentry_type="conversation",
                title="Claude conversation",
                unique_id=None,
            ),
            ConfigSubentryData(
                data={"recommended": True},
                subentry_type="ai_task_data",
                title="Claude AI Task",
                unique_id=None,
            ),
        ],
    )


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    assert await async_setup_component(hass, "homeassistant", {})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def test_setup_creates_entities(hass: HomeAssistant) -> None:
    """Setting up the entry creates conversation and AI task entities."""
    entry = _entry()
    await _setup(hass, entry)

    assert hass.states.get("conversation.claude_conversation") is not None
    assert hass.states.get("ai_task.claude_ai_task") is not None


def _http_error(status: int = 500) -> aiohttp.ClientResponseError:
    """Return an HTTP error carrying its request headers, as aiohttp raises it."""
    url = URL(MODELS_URL)
    headers = CIMultiDictProxy(CIMultiDict({"Authorization": "Bearer sk-ant-oat01-secret"}))
    return aiohttp.ClientResponseError(
        aiohttp.RequestInfo(url, "GET", headers, url), (), status=status
    )


async def test_failed_model_list_keeps_credentials_out_of_the_log(
    hass: HomeAssistant,
    mock_models_api: AiohttpClientMocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An HTTP error's repr has the request's token, so it can't be logged."""
    mock_models_api.clear_requests()
    mock_models_api.get(MODELS_URL, exc=_http_error())
    await _setup(hass, _entry())

    assert "Could not fetch the Claude model list" in caplog.text
    assert "sk-ant-" not in caplog.text


def _model_ids(models: list[Any]) -> list[str]:
    return [model.id for model in models]


async def test_models_fetched_at_setup(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """Setup offers "Account default" plus Anthropic's list, OAuth as Bearer."""
    entry = _entry()
    await _setup(hass, entry)

    assert _model_ids(entry.runtime_data.data) == [
        "default",
        "claude-new-9",
        "claude-old-8",
    ]
    assert entry.runtime_data.data[1].display_name == "Claude New 9"
    _, url, _, headers = mock_models_api.mock_calls[-1]
    assert url.query["limit"] == "1000"
    assert headers["Authorization"] == f"Bearer {entry.data['oauth_token']}"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "x-api-key" not in headers


async def test_models_fetched_with_api_key(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """An API key is sent as x-api-key."""
    entry = _entry(data={"api_key": "sk-ant-api-test"})
    await _setup(hass, entry)

    _, _, _, headers = mock_models_api.mock_calls[-1]
    assert headers["x-api-key"] == "sk-ant-api-test"
    assert "Authorization" not in headers


async def test_models_follow_pagination(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """Pages are followed with after_id until has_more is false."""
    mock_models_api.clear_requests()
    # First registered match wins, so the more specific page goes first.
    mock_models_api.get(
        f"{MODELS_URL}?after_id=claude-old-8",
        json={
            "data": [{"id": "claude-ancient-1", "display_name": "Claude Ancient 1"}],
            "has_more": False,
            "last_id": "claude-ancient-1",
        },
    )
    mock_models_api.get(
        MODELS_URL,
        json={"data": API_MODELS, "has_more": True, "last_id": "claude-old-8"},
    )
    entry = _entry()
    await _setup(hass, entry)

    assert _model_ids(entry.runtime_data.data) == [
        "default",
        "claude-new-9",
        "claude-old-8",
        "claude-ancient-1",
    ]


@pytest.mark.parametrize(
    "response",
    [
        {"status": 500},
        {"exc": aiohttp.ClientError("offline")},
        {"exc": TimeoutError()},
        {"json": {"unexpected": True}},
        {"json": {"data": [], "has_more": False}},
    ],
    ids=["http-error", "network-error", "timeout", "bad-shape", "empty"],
)
async def test_models_fall_back_to_builtin_list(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker, response: dict[str, Any]
) -> None:
    """A failed fetch never fails setup; the built-in list is offered instead."""
    from custom_components.claude.const import MODELS

    mock_models_api.clear_requests()
    mock_models_api.get(MODELS_URL, **response)
    entry = _entry()
    await _setup(hass, entry)

    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.data == MODELS


def _model_entry(model: str = "claude-old-8") -> MockConfigEntry:
    return _entry(conversation_data={"recommended": False, "chat_model": model})


def _model_issue(hass: HomeAssistant, entry: MockConfigEntry, name: str) -> Any:
    return ir.async_get(hass).async_get_issue(DOMAIN, f"{name}_{entry.entry_id}")


def _mock_models(mock: AiohttpClientMocker, *extra: dict[str, Any]) -> None:
    """Answer the model list with one active model plus the given ones."""
    mock.clear_requests()
    mock.get(
        MODELS_URL,
        json={
            "data": [
                {"id": "claude-new-9", "display_name": "Claude New 9", "lifecycle": "active"},
                *extra,
            ],
            "has_more": False,
        },
    )


DEPRECATED_MODEL = {
    "id": "claude-old-8",
    "display_name": "Claude Old 8",
    "lifecycle": "deprecated",
    "retires_at": "2027-02-03T00:00:00Z",
}


async def test_deprecated_model_raises_a_warning(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """An agent on a deprecated model warns, naming the agent and the date."""
    _mock_models(mock_models_api, DEPRECATED_MODEL)
    entry = _model_entry()
    await _setup(hass, entry)

    issue = _model_issue(hass, entry, "model_deprecated")
    assert issue.severity is ir.IssueSeverity.WARNING
    assert issue.translation_placeholders == {
        "agents": "Claude conversation (Claude Old 8, until 2027-02-03)"
    }


@pytest.mark.parametrize(
    "conversation_data",
    [{"recommended": True}, {"recommended": False, "chat_model": "default"}],
    ids=["recommended", "account-default"],
)
async def test_model_warning_ignores_account_default(
    hass: HomeAssistant,
    mock_models_api: AiohttpClientMocker,
    conversation_data: dict[str, Any],
) -> None:
    """Without a specific model there is nothing to warn about."""
    _mock_models(mock_models_api, DEPRECATED_MODEL)
    entry = _entry(conversation_data=conversation_data)
    await _setup(hass, entry)

    assert _model_issue(hass, entry, "model_deprecated") is None


async def test_model_warning_without_lifecycle_data(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """A response without lifecycle data raises nothing."""
    entry = _model_entry()
    await _setup(hass, entry)

    assert _model_issue(hass, entry, "model_deprecated") is None


async def test_model_warning_resolves_dated_ids(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """An alias finds the dated model the API lists."""
    _mock_models(mock_models_api, DEPRECATED_MODEL | {"id": "claude-old-8-20250101"})
    entry = _model_entry("claude-old-8")
    await _setup(hass, entry)

    assert _model_issue(hass, entry, "model_deprecated") is not None


async def test_changing_the_model_clears_the_warning(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """Moving the agent to another model clears the warning."""
    _mock_models(mock_models_api, DEPRECATED_MODEL)
    entry = _model_entry()
    await _setup(hass, entry)
    assert _model_issue(hass, entry, "model_deprecated") is not None

    subentry = next(
        sub for sub in entry.subentries.values() if sub.subentry_type == "conversation"
    )
    hass.config_entries.async_update_subentry(
        entry,
        subentry,
        data={"recommended": False, "chat_model": "claude-new-9"},
    )
    await hass.async_block_till_done()

    assert _model_issue(hass, entry, "model_deprecated") is None


async def test_model_warning_ignores_unknown_ids(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """A CLI alias the list doesn't have raises nothing."""
    _mock_models(mock_models_api, DEPRECATED_MODEL)
    entry = _model_entry("sonnet")
    await _setup(hass, entry)

    assert _model_issue(hass, entry, "model_deprecated") is None


async def test_new_deprecation_found_on_a_timer(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """A model deprecated while Home Assistant runs is picked up within a day."""
    _mock_models(mock_models_api, DEPRECATED_MODEL | {"lifecycle": "active"})
    entry = _model_entry()
    await _setup(hass, entry)
    assert _model_issue(hass, entry, "model_deprecated") is None

    _mock_models(mock_models_api, DEPRECATED_MODEL)
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(hours=13))
    await hass.async_block_till_done()

    assert _model_issue(hass, entry, "model_deprecated") is not None


def _capabilities(
    efforts: tuple[str, ...] = (),
    thinking: tuple[str, ...] = (),
    web_search: bool = True,
    structured_outputs: bool = True,
) -> dict[str, Any]:
    """Return a model's capabilities in the shape Anthropic's API sends them."""
    return {
        "effort": {
            "supported": bool(efforts),
            **{
                level: {"supported": level in efforts}
                for level in ("low", "medium", "high", "xhigh", "max")
            },
        },
        "thinking": {
            "supported": bool(thinking),
            "types": {
                value: {"supported": value in thinking}
                for value in ("adaptive", "between_tools", "disabled", "enabled")
            },
        },
        "server_tools": {
            "supported": web_search,
            "code_execution": {"supported": False},
            "web_search": {"supported": web_search},
        },
        "structured_outputs": {"supported": structured_outputs},
    }


LIMITED_MODEL = {
    "id": "claude-limited-1",
    "display_name": "Claude Limited 1",
    "lifecycle": "active",
    "max_tokens": 8000,
    "capabilities": _capabilities(
        efforts=("low", "high"), thinking=("disabled", "enabled"), web_search=False
    ),
}
ALL_OPTIONS = {
    "max_tokens",
    "thinking_type",
    "thinking_budget",
    "effort",
    "code_execution",
    "web_search",
    "user_location",
    "web_fetch",
    "show_activity",
    "run_timeout",
}


async def _open_options(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    model: str,
    subentry_type: str = "conversation",
) -> Any:
    """Pick a model in a subentry's flow, returning the form that follows."""
    subentry_id = next(
        sid for sid, sub in entry.subentries.items() if sub.subentry_type == subentry_type
    )
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, subentry_type),
        context={"source": "reconfigure", "subentry_id": subentry_id},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"recommended": False}
    )
    return await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"chat_model": model}
    )


async def test_forms_that_dont_save_say_next(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """Only the form that saves offers to finish."""
    _mock_models(mock_models_api)
    entry = _model_entry("claude-new-9")
    await _setup(hass, entry)
    subentry_id = next(
        sid for sid, sub in entry.subentries.items() if sub.subentry_type == "conversation"
    )

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "conversation"),
        context={"source": "reconfigure", "subentry_id": subentry_id},
    )
    assert result["last_step"] is False
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"recommended": False}
    )
    assert result["step_id"] == "additional"
    assert result["last_step"] is False
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"chat_model": "claude-new-9"}
    )
    assert result["step_id"] == "model"
    assert result["last_step"] is True


async def test_options_follow_the_models_capabilities(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """The form only offers what the chosen model supports."""
    from homeassistant.helpers.selector import NumberSelectorMode

    _mock_models(mock_models_api, LIMITED_MODEL)
    entry = _model_entry("claude-new-9")
    await _setup(hass, entry)

    result = await _open_options(hass, entry, "claude-limited-1")

    assert result["step_id"] == "model"
    _assert_form_serializes(result)
    schema = result["data_schema"].schema
    assert {str(key) for key in schema} == ALL_OPTIONS - {"web_search", "user_location"}
    selectors = {str(key): value for key, value in schema.items()}
    assert selectors["effort"].config["options"] == ["default", "low", "high"]
    assert selectors["thinking_type"].config["options"] == [
        "default",
        "enabled",
        "disabled",
    ]
    for field in ("max_tokens", "thinking_budget"):
        assert selectors[field].validators[0].config["max"] == 8000
        assert selectors[field].validators[0].config["mode"] == NumberSelectorMode.BOX


async def test_options_hide_what_a_model_has_no_choice_about(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """A model without effort or thinking leaves those settings out."""
    _mock_models(
        mock_models_api,
        {
            "id": "claude-plain-1",
            "display_name": "Claude Plain 1",
            "lifecycle": "active",
            "capabilities": _capabilities(),
        },
    )
    entry = _model_entry("claude-new-9")
    await _setup(hass, entry)

    result = await _open_options(hass, entry, "claude-plain-1")

    assert {str(key) for key in result["data_schema"].schema} == ALL_OPTIONS - {
        "thinking_type",
        "thinking_budget",
        "effort",
    }


@pytest.mark.parametrize(
    "model", ["default", "claude-new-9", "sonnet"], ids=["account-default", "no-data", "alias"]
)
async def test_options_all_shown_when_capabilities_are_unknown(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker, model: str
) -> None:
    """Without capability data, every option stays available."""
    _mock_models(mock_models_api)
    entry = _model_entry("claude-new-9")
    await _setup(hass, entry)

    result = await _open_options(hass, entry, model)

    assert result["step_id"] == "model"
    assert {str(key) for key in result["data_schema"].schema} == ALL_OPTIONS


async def test_ai_task_only_offers_structured_output_models(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """An AI task needs structured output, so other models aren't offered."""
    _mock_models(
        mock_models_api,
        {
            "id": "claude-free-1",
            "display_name": "Claude Free 1",
            "lifecycle": "active",
            "capabilities": _capabilities(structured_outputs=False),
        },
    )
    entry = _model_entry("claude-new-9")
    await _setup(hass, entry)

    subentry_id = next(
        sid for sid, sub in entry.subentries.items() if sub.subentry_type == "ai_task_data"
    )
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "ai_task_data"),
        context={"source": "reconfigure", "subentry_id": subentry_id},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"recommended": False}
    )
    selector = next(
        value for key, value in result["data_schema"].schema.items() if key == "chat_model"
    )

    assert [option["value"] for option in selector.config["options"]] == [
        "default",
        "claude-new-9",
    ]


async def test_saved_options_adapt_to_the_model_at_run_time(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """A run corrects saved options the model doesn't support, without saving."""
    from homeassistant.components import conversation

    _mock_models(mock_models_api, LIMITED_MODEL)
    saved = {
        "recommended": False,
        "chat_model": "claude-limited-1",
        "max_tokens": 16000,
        "thinking_type": "adaptive",
        "effort": "max",
        "web_search": True,
        "web_fetch": True,
    }
    entry = _entry(conversation_data=saved)
    await _setup(hass, entry)

    seen: list[Any] = []
    with patch(
        "custom_components.claude.entity.query",
        _fake_query([*_text_deltas("Hi"), ResultMessage()], seen),
    ):
        await conversation.async_converse(
            hass, "hello", None, None, agent_id="conversation.claude_conversation"
        )

    assert seen[0].thinking is None
    assert seen[0].effort == "high"
    assert seen[0].tools == ["WebFetch"]
    assert seen[0].env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8000"
    subentry = next(
        sub for sub in entry.subentries.values() if sub.subentry_type == "conversation"
    )
    assert subentry.data["effort"] == "max"


@pytest.mark.parametrize(
    ("model", "options", "expected"),
    [
        (
            {"id": "claude-x-1", "capabilities": _capabilities(efforts=("low", "high"))},
            {"effort": "medium"},
            {"effort": "low"},
        ),
        (
            {"id": "claude-x-1", "capabilities": _capabilities(efforts=("high",))},
            {"effort": "low"},
            {"effort": "high"},
        ),
        (
            {"id": "claude-x-1", "capabilities": _capabilities()},
            {"effort": "high", "thinking_type": "enabled", "thinking_budget": 2048},
            {"effort": "default", "thinking_type": "default"},
        ),
        (
            {"id": "claude-x-1", "capabilities": _capabilities(thinking=("enabled",))},
            {"thinking_type": "enabled", "thinking_budget": 5000, "max_tokens": 4096},
            {"thinking_type": "enabled", "thinking_budget": 4095, "max_tokens": 4096},
        ),
        (
            {"id": "claude-x-1", "capabilities": _capabilities(thinking=("enabled",))},
            {"thinking_type": "enabled", "thinking_budget": 2000, "max_tokens": 1000},
            {"thinking_type": "enabled", "thinking_budget": 1024, "max_tokens": 1025},
        ),
        (
            {
                "id": "claude-haiku-5-5-20260101",
                "capabilities": _capabilities(
                    efforts=("low", "high", "max"), thinking=("disabled", "adaptive")
                ),
            },
            {"effort": "max", "thinking_type": "disabled"},
            {"effort": "high", "thinking_type": "disabled"},
        ),
        (
            {"id": "claude-x-1"},
            {"effort": "max", "thinking_type": "adaptive", "web_search": True},
            {"effort": "max", "thinking_type": "adaptive", "web_search": True},
        ),
    ],
    ids=[
        "effort-next-lower",
        "effort-next-higher",
        "no-effort-or-thinking",
        "budget-under-max-tokens",
        "max-tokens-over-budget",
        "known-restriction",
        "unknown-capabilities",
    ],
)
def test_reconcile_model_options(
    model: dict[str, Any], options: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Options are moved to the nearest thing the model supports."""
    from custom_components.claude.agent import _model_from_api
    from custom_components.claude.model_options import reconcile_model_options

    reconcile_model_options(options, _model_from_api(model))

    assert {key: options[key] for key in expected} == expected


@pytest.mark.parametrize(
    "messages",
    [
        [
            AssistantMessage([ToolUseBlock("call-1", "StructuredOutput", {"fruit": "Apple"})]),
            UserMessage([ToolResultBlock("call-1", "Structured output provided")]),
            ResultMessage(structured_output={"fruit": "Apple"}),
        ],
        [
            *_text_deltas('{"fruit": "Apple"}'),
            ResultMessage(),
        ],
    ],
    ids=["structured-output-tool", "json-text"],
)
async def test_ai_task_returns_structured_data(
    hass: HomeAssistant, messages: list[Any]
) -> None:
    """Claude answers through a tool with no reply text, which is still an answer."""
    await _setup(hass, _entry())

    with patch("custom_components.claude.entity.query", _fake_query(messages)):
        result = await hass.services.async_call(
            "ai_task",
            "generate_data",
            {
                "entity_id": "ai_task.claude_ai_task",
                "task_name": "test",
                "instructions": "Give me one fruit.",
                "structure": {"fruit": {"selector": {"text": {}}}},
            },
            blocking=True,
            return_response=True,
        )

    assert result["data"] == {"fruit": "Apple"}


async def test_ai_tools_show_as_a_checkbox_list(hass: HomeAssistant) -> None:
    """The AI tools field lists every option as a checkbox."""
    from homeassistant.helpers.selector import SelectSelectorMode

    entry = _entry()
    await _setup(hass, entry)
    subentry_id = next(
        sid for sid, sub in entry.subentries.items() if sub.subentry_type == "conversation"
    )

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "conversation"),
        context={"source": "reconfigure", "subentry_id": subentry_id},
    )

    selector = next(
        value for key, value in result["data_schema"].schema.items() if key == "llm_hass_api"
    )
    assert selector.config["mode"] == SelectSelectorMode.LIST
    assert selector.config["multiple"] is True


def _assert_form_serializes(result: Any) -> None:
    """Serialize a form's schema the way the flow API sends it to the frontend."""
    assert probatio.to_field_list(
        result["data_schema"], custom_serializer=cv.custom_serializer
    )


def test_credentials_form_serializes() -> None:
    """The credentials form reaches the frontend with both of its fields."""
    from custom_components.claude.config_flow import STEP_USER_DATA_SCHEMA

    fields = probatio.to_field_list(
        STEP_USER_DATA_SCHEMA, custom_serializer=cv.custom_serializer
    )

    assert {field["name"] for field in fields} == {"oauth_token", "api_key"}


async def test_model_options_refetch_and_keep_list_on_failure(
    hass: HomeAssistant, mock_models_api: AiohttpClientMocker
) -> None:
    """Opening the model options re-fetches; a failed re-fetch keeps the list."""
    entry = _entry(conversation_data={"recommended": False})
    await _setup(hass, entry)
    subentry_id = next(
        sid
        for sid, sub in entry.subentries.items()
        if sub.subentry_type == "conversation"
    )

    async def open_model_step() -> list[str]:
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, "conversation"),
            context={"source": "reconfigure", "subentry_id": subentry_id},
        )
        _assert_form_serializes(result)
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {"recommended": False}
        )
        assert result["step_id"] == "additional"
        _assert_form_serializes(result)
        selector = next(
            value
            for key, value in result["data_schema"].schema.items()
            if key == "chat_model"
        )
        return [option["value"] for option in selector.config["options"]]

    mock_models_api.clear_requests()
    mock_models_api.get(
        MODELS_URL,
        json={
            "data": [{"id": "claude-newest-10", "display_name": "Claude Newest 10"}],
            "has_more": False,
            "last_id": "claude-newest-10",
        },
    )
    assert await open_model_step() == ["default", "claude-newest-10"]
    assert mock_models_api.call_count == 1

    mock_models_api.clear_requests()
    mock_models_api.get(MODELS_URL, status=503)
    assert await open_model_step() == ["default", "claude-newest-10"]


async def test_config_flow_user(hass: HomeAssistant) -> None:
    """The manual config flow verifies credentials and creates the entry."""
    assert await async_setup_component(hass, "homeassistant", {})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "menu"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "manual"}
    )
    assert result["type"] == "form"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"oauth_token": "sk-ant-oat01-secret"},
    )
    assert result["type"] == "create_entry"
    assert result["title"] == "Claude"
    assert result["data"] == {"oauth_token": "sk-ant-oat01-secret"}
    assert len(result["subentries"]) == 2


async def test_config_flow_invalid_auth(hass: HomeAssistant) -> None:
    """Credential errors surface in the flow."""
    from custom_components.claude.agent import CredentialsInvalidError

    assert await async_setup_component(hass, "homeassistant", {})
    with patch(
        "custom_components.claude.config_flow.validate_input",
        AsyncMock(side_effect=CredentialsInvalidError("401")),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "manual"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"oauth_token": "bad"},
        )
    assert result["type"] == "form"
    assert result["errors"] == {"base": "invalid_auth"}


async def test_oauth_flow_stores_expiry(hass: HomeAssistant) -> None:
    """The OAuth login step keeps the expiry the token endpoint reported."""
    from custom_components.claude import oauth

    expires_at = datetime.now(UTC) + timedelta(days=365)
    assert await async_setup_component(hass, "homeassistant", {})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "oauth"}
    )
    with patch(
        "custom_components.claude.config_flow.oauth.async_exchange_code",
        AsyncMock(return_value=oauth.OAuthToken("sk-ant-oat01-secret", expires_at)),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"code": "the-code#the-state"}
        )

    assert result["type"] == "create_entry"
    assert result["data"] == {
        "oauth_token": "sk-ant-oat01-secret",
        "oauth_expires_at": expires_at.isoformat(),
    }


async def test_exchange_requests_token_lifetime(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """The exchange sends the PKCE fields and the requested lifetime."""
    from custom_components.claude import oauth

    aioclient_mock.post(
        oauth.TOKEN_URL,
        json={
            "access_token": "sk-ant-oat01-secret",
            "expires_in": oauth.TOKEN_LIFETIME_S,
        },
    )
    issued = await oauth.async_exchange_code(
        async_get_clientsession(hass), "the-code#the-state", "the-verifier"
    )

    body = aioclient_mock.mock_calls[0][2]
    assert body["expires_in"] == oauth.TOKEN_LIFETIME_S
    assert body["code"] == "the-code"
    assert body["state"] == "the-state"
    assert body["code_verifier"] == "the-verifier"
    assert issued.token == "sk-ant-oat01-secret"
    assert issued.expires_at is not None
    assert issued.expires_at > datetime.now(UTC)


def test_login_url() -> None:
    """The authorize URL carries the PKCE challenge and the scope."""
    from custom_components.claude import oauth

    url, verifier = oauth.build_login()

    assert url.startswith(f"{oauth.AUTHORIZE_URL}?")
    query = parse_qs(urlparse(url).query)
    assert query["scope"] == [oauth.SCOPES]
    assert query["state"] == [verifier]
    assert query["code_challenge_method"] == ["S256"]


@pytest.mark.parametrize(
    ("days", "issue_id"),
    [(3, "oauth_token_expiring"), (-1, "oauth_token_expired")],
)
async def test_token_expiry_raises_issue(
    hass: HomeAssistant, days: int, issue_id: str
) -> None:
    """A token near or past expiry raises a repair issue."""
    entry = _entry(
        data={
            "oauth_token": "sk-ant-oat01-secret",
            "oauth_expires_at": (datetime.now(UTC) + timedelta(days=days)).isoformat(),
        }
    )
    await _setup(hass, entry)

    registry = ir.async_get(hass)
    assert registry.async_get_issue(DOMAIN, f"{issue_id}_{entry.entry_id}") is not None


async def test_token_without_expiry_raises_no_issue(hass: HomeAssistant) -> None:
    """A manually pasted token has no known expiry and must not warn."""
    entry = _entry()
    await _setup(hass, entry)

    registry = ir.async_get(hass)
    assert not [
        issue for (domain, _), issue in registry.issues.items() if domain == DOMAIN
    ]


async def test_conversation_turn(hass: HomeAssistant) -> None:
    """A conversation turn streams text from the agent."""
    from homeassistant.components import conversation

    entry = _entry()
    await _setup(hass, entry)

    messages = [
        SystemMessage("init", {"session_id": "sess-1"}),
        AssistantMessage([]),
        *_text_deltas("Hello ", "there"),
        ResultMessage(session_id="sess-1"),
    ]
    with patch("custom_components.claude.entity.query", _fake_query(messages)):
        result = await conversation.async_converse(
            hass,
            "hi",
            None,
            None,
            agent_id="conversation.claude_conversation",
        )

    assert result.response.speech["plain"]["speech"] == "Hello there"
    coordinator = entry.runtime_data
    assert coordinator.session_ids[result.conversation_id] == "sess-1"


async def test_conversation_ha_tool(hass: HomeAssistant) -> None:
    """An HA tool call executed in-process is recorded and answered."""
    from homeassistant.components import conversation
    from homeassistant.helpers import llm

    from custom_components.claude.agent import ClaudeAgent

    entry = _entry(
        conversation_data={
            "recommended": True,
            "llm_hass_api": [llm.LLM_API_ASSIST],
        }
    )
    await _setup(hass, entry)

    messages = [
        SystemMessage("init", {"session_id": "sess-2"}),
        AssistantMessage([ToolUseBlock("call-1", "mcp__ha__GetLiveContext", {})]),
        UserMessage([ToolResultBlock("call-1", "some context")]),
        AssistantMessage([]),
        *_text_deltas("Done"),
        ResultMessage(session_id="sess-2"),
    ]
    with (
        patch("custom_components.claude.entity.query", _fake_query(messages)),
        patch.object(
            ClaudeAgent,
            "build_tool_server",
            autospec=True,
            side_effect=ClaudeAgent.build_tool_server,
        ) as build_tool_server,
    ):
        result = await conversation.async_converse(
            hass,
            "what's up",
            None,
            None,
            agent_id="conversation.claude_conversation",
        )

    assert result.response.speech["plain"]["speech"] == "Done"
    # Every HA tool must reach the SDK as a JSON Schema object.
    tools = build_tool_server.call_args.args[1]
    assert tools
    for tool in tools:
        assert isinstance(tool["input_schema"], dict), tool["name"]
        assert tool["input_schema"]["type"] == "object", tool["name"]


@pytest.mark.parametrize(
    ("options", "thinking", "effort"),
    [
        ({"recommended": True}, None, None),
        ({"recommended": False, "effort": "high"}, None, "high"),
        (
            {"recommended": False, "thinking_type": "adaptive"},
            {"type": "adaptive", "display": "summarized"},
            None,
        ),
        (
            {"recommended": False, "thinking_type": "enabled", "thinking_budget": 2048},
            {"type": "enabled", "budget_tokens": 2048, "display": "summarized"},
            None,
        ),
        ({"recommended": False, "thinking_type": "disabled"}, {"type": "disabled"}, None),
    ],
    ids=["recommended", "effort", "adaptive", "enabled", "disabled"],
)
async def test_thinking_and_effort_reach_the_sdk(
    hass: HomeAssistant,
    options: dict[str, Any],
    thinking: dict[str, Any] | None,
    effort: str | None,
) -> None:
    """Thinking type and effort are separate settings."""
    from homeassistant.components import conversation

    await _setup(hass, _entry(conversation_data=options))

    seen: list[Any] = []
    with patch(
        "custom_components.claude.entity.query",
        _fake_query([*_text_deltas("Hi"), ResultMessage()], seen),
    ):
        await conversation.async_converse(
            hass, "hello", None, None, agent_id="conversation.claude_conversation"
        )

    assert seen[0].thinking == thinking
    assert seen[0].effort == effort


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            {"thinking_budget": 2048, "thinking_effort": "high"},
            {"thinking_type": "enabled", "thinking_budget": 2048, "effort": "high"},
        ),
        ({"thinking_budget": 0, "thinking_effort": "default"}, {"effort": "default"}),
        ({"recommended": True}, {"recommended": True}),
    ],
    ids=["budget-and-effort", "model-decides", "recommended"],
)
async def test_thinking_options_migrate(
    hass: HomeAssistant, old: dict[str, Any], new: dict[str, Any]
) -> None:
    """Saved agents move to the separate settings and keep their behavior."""
    entry = _entry(conversation_data=old)
    await _setup(hass, entry)

    assert entry.minor_version == 2
    subentry = next(
        sub for sub in entry.subentries.values() if sub.subentry_type == "conversation"
    )
    assert dict(subentry.data) == new
    assert entry.state is ConfigEntryState.LOADED


@pytest.mark.parametrize(
    ("thinking_type", "errors"),
    [
        ("enabled", {"thinking_budget": "thinking_budget_too_large"}),
        ("adaptive", None),
    ],
)
async def test_thinking_budget_only_checked_when_it_applies(
    hass: HomeAssistant, thinking_type: str, errors: dict[str, str] | None
) -> None:
    """A budget over the token limit only matters when manual thinking is on."""
    entry = _entry(conversation_data={"recommended": False})
    await _setup(hass, entry)
    subentry_id = next(
        sid for sid, sub in entry.subentries.items() if sub.subentry_type == "conversation"
    )
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "conversation"),
        context={"source": "reconfigure", "subentry_id": subentry_id},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"recommended": False}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"chat_model": "default"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {"max_tokens": 2000, "thinking_type": thinking_type, "thinking_budget": 4000},
    )

    assert result.get("errors") == errors


@pytest.mark.parametrize(
    ("options", "tools"),
    [
        ({"recommended": True}, []),
        ({"recommended": False, "web_search": True}, ["WebSearch"]),
        ({"recommended": False, "web_fetch": True}, ["WebFetch"]),
        (
            {"recommended": False, "code_execution": True},
            ["Bash", "Write", "Edit", "Read", "Glob", "Grep", "NotebookEdit", "Task"],
        ),
    ],
    ids=["recommended", "web-search", "web-fetch", "code-execution"],
)
async def test_builtin_tools_come_from_options(
    hass: HomeAssistant, options: dict[str, Any], tools: list[str]
) -> None:
    """Claude only gets the built-in tools the subentry options enable."""
    from homeassistant.components import conversation

    await _setup(hass, _entry(conversation_data=options))

    seen: list[Any] = []
    with patch(
        "custom_components.claude.entity.query",
        _fake_query([*_text_deltas("Hi"), ResultMessage()], seen),
    ):
        await conversation.async_converse(
            hass, "hello", None, None, agent_id="conversation.claude_conversation"
        )

    assert seen[0].tools == tools
    assert seen[0].disallowed_tools == []


def _message_delta(stop_reason: str) -> StreamEvent:
    return StreamEvent(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


async def _collect(messages: list[Any]) -> list[Any]:
    """Run messages through the delta stream, as a conversation turn does."""
    from custom_components.claude.entity import SDKDeltaStream

    async def _gen() -> AsyncIterator[Any]:
        for message in messages:
            yield message

    stream = SDKDeltaStream(MagicMock(), _gen(), lambda _: None)
    return [delta async for delta in stream]


@pytest.mark.parametrize(
    ("stop_reason", "translation_key"),
    [
        ("max_tokens", "response_max_tokens"),
        ("model_context_window_exceeded", "response_context_window_exceeded"),
    ],
)
async def test_failed_run_names_the_limit_it_hit(
    stop_reason: str, translation_key: str
) -> None:
    """The error names the limit, not the CLI's advice to set a variable."""
    # The final result reports stop_sequence; only the stream has the real reason.
    failed = ResultMessage(is_error=True)
    failed.result = "API Error: set the CLAUDE_CODE_MAX_OUTPUT_TOKENS variable"
    failed.stop_reason = "stop_sequence"

    with pytest.raises(HomeAssistantError) as excinfo:
        await _collect([*_text_deltas("Partial"), _message_delta(stop_reason), failed])

    assert excinfo.value.translation_key == translation_key


async def test_run_recovered_from_a_limit_succeeds() -> None:
    """The CLI retries a cut-off turn, so a limit alone is not a failure."""
    deltas = await _collect(
        [
            _message_delta("max_tokens"),
            *_text_deltas("Done"),
            _message_delta("end_turn"),
            ResultMessage(),
        ]
    )

    assert {"content": "Done"} in deltas


async def test_unrelated_failure_keeps_its_message() -> None:
    """A failed run that hit no limit still reports the CLI's own message."""
    failed = ResultMessage(is_error=True)
    failed.result = "There's an issue with the selected model"

    with pytest.raises(HomeAssistantError) as excinfo:
        await _collect([_message_delta("end_turn"), failed])

    assert excinfo.value.translation_key == "run_error"
    assert excinfo.value.translation_placeholders == {"message": failed.result}


async def test_refusal_raises() -> None:
    """A refused request is an error, not an empty or partial reply."""
    with pytest.raises(HomeAssistantError) as excinfo:
        await _collect([*_text_deltas("Sorry"), _message_delta("refusal")])

    assert excinfo.value.translation_key == "api_refusal"


async def test_usage_trace_counts_every_input_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The trace counts cache writes and reads, and keeps the full usage."""
    from custom_components.claude.entity import SDKDeltaStream

    caplog.set_level(logging.DEBUG, logger="custom_components.claude")
    chat_log = MagicMock()
    usage = {
        "input_tokens": 2,
        "cache_creation_input_tokens": 300,
        "cache_read_input_tokens": 4000,
        "output_tokens": 50,
    }

    async def _gen() -> AsyncIterator[Any]:
        yield StreamEvent(
            {
                "type": "message_start",
                "message": {"id": "msg_1", "usage": usage | {"output_tokens": 1}},
            }
        )
        yield StreamEvent(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 50},
            }
        )

    async for _ in SDKDeltaStream(chat_log, _gen(), lambda _: None):
        pass

    chat_log.async_trace.assert_called_once_with(
        {
            "stats": {
                "input_tokens": 4302,
                "uncached_input_tokens": 2,
                "cache_creation_input_tokens": 300,
                "cached_input_tokens": 4000,
                "output_tokens": 50,
                "stop_reason": "end_turn",
            },
            "usage": usage,
            "message_id": "msg_1",
        }
    )
    assert "Response usage" in caplog.text
    assert "'input_tokens': 4302" in caplog.text


async def test_failed_run_closes_the_sdk_stream(hass: HomeAssistant) -> None:
    """A run that stops early closes the SDK stream, which stops the CLI process."""
    from homeassistant.components import conversation

    await _setup(hass, _entry())

    closed = False

    def _query(prompt: Any, options: Any) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            nonlocal closed
            try:
                yield ResultMessage(is_error=True, subtype="error_during_execution")
                yield ResultMessage()
            finally:
                closed = True

        return _gen()

    with patch("custom_components.claude.entity.query", _query):
        result = await conversation.async_converse(
            hass, "hello", None, None, agent_id="conversation.claude_conversation"
        )
        assert closed

    assert result.response.response_type is intent.IntentResponseType.ERROR


@pytest.mark.parametrize("error", [False, True])
async def test_ha_tool_handler_unwraps_tool_result(error: bool) -> None:
    """Claude gets the tool's own data, with a failed tool flagged as an error."""
    from homeassistant.helpers import llm
    from homeassistant.util.json import json_loads

    from custom_components.claude.agent import ClaudeAgent

    data = {"speech": "Turned on the light"}
    llm_api = MagicMock()
    llm_api.async_call_tool = AsyncMock(
        return_value=llm.ToolResult(data=data, error=error)
    )
    handler = ClaudeAgent._make_handler(MagicMock(), "intent__HassTurnOn", llm_api, 5)

    result = await handler({})

    assert json_loads(result["content"][0]["text"]) == data
    assert result.get("is_error", False) is error


@pytest.mark.parametrize(("is_error", "error"), [(None, False), (True, True)])
async def test_tool_result_delta_uses_tool_result(
    is_error: bool | None, error: bool
) -> None:
    """A tool result reaches the chat log as a ToolResult, keeping its error flag."""
    from homeassistant.helpers import llm

    from custom_components.claude.entity import SDKDeltaStream

    block = ToolResultBlock("call-1", "some context")
    block.is_error = is_error
    stream = SDKDeltaStream(MagicMock(), MagicMock(), lambda _: None)

    deltas = [delta async for delta in stream._handle_user(UserMessage([block]))]

    assert deltas == [
        {
            "role": "tool_result",
            "tool_call_id": "call-1",
            "tool_name": "unknown",
            "result": llm.ToolResult(data={"content": "some context"}, error=error),
        }
    ]


def test_structured_output_schema_is_json_schema() -> None:
    """An AI Task structure converts to a JSON Schema object."""
    from homeassistant.helpers import llm, selector

    from custom_components.claude.entity import _to_json_schema

    structure = probatio.Schema(
        {
            probatio.Required("answer"): selector.TextSelector(),
            probatio.Optional("count"): selector.NumberSelector(),
        }
    )
    schema = _to_json_schema(structure, llm.selector_serializer)

    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"answer", "count"}
    assert schema["required"] == ["answer"]


def test_result_classification_network_beats_auth_wording() -> None:
    """Offline errors phrased in auth terms must stay retryable, not fatal."""
    from custom_components.claude.agent import (
        ClaudeAgentError,
        CredentialsInvalidError,
        _raise_for_result,
    )

    for message in (
        "unable to refresh authentication token: fetch failed",
        "401 from proxy :: getaddrinfo ENOTFOUND api.anthropic.com",
        "authentication check timed out",
    ):
        with pytest.raises(ClaudeAgentError) as excinfo:
            _raise_for_result(message)
        assert not isinstance(excinfo.value, CredentialsInvalidError), message

    with pytest.raises(CredentialsInvalidError):
        _raise_for_result("401 invalid bearer token")


class TestVerifyReal:
    """Tests that need the real async_verify/async_prepare, not the mocks."""

    @pytest.fixture(autouse=True)
    def mock_verify(self) -> Generator[None]:
        """Keep the real async_verify."""
        yield

    @pytest.fixture(autouse=True)
    def mock_prepare(self) -> Generator[None]:
        """Keep the real async_prepare."""
        yield

    async def test_verify_wraps_prepare_failure(self) -> None:
        """A raw network error during CLI preparation becomes ClaudeAgentError."""
        from custom_components.claude.agent import ClaudeAgent, ClaudeAgentError

        agent = ClaudeAgent.__new__(ClaudeAgent)
        with patch.object(
            ClaudeAgent, "async_prepare", AsyncMock(side_effect=OSError("boom"))
        ):
            with pytest.raises(ClaudeAgentError, match="OSError: boom"):
                await agent.async_verify()


async def test_brand_images_shipped(hass: HomeAssistant) -> None:
    """HA 2026.3+ serves a custom integration's brand/ folder as its icon."""
    from pathlib import Path

    from homeassistant.loader import async_get_integration

    integration = await async_get_integration(hass, DOMAIN)
    assert integration.has_branding
    brand = Path(integration.file_path) / "brand"
    for name in ("icon.png", "icon@2x.png", "dark_icon.png", "logo.png"):
        assert (brand / name).read_bytes().startswith(b"\x89PNG"), name
