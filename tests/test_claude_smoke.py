"""Smoke tests for the claude custom integration.

See tests/README.md for how to run them. The Claude Agent SDK is never really
invoked; ``async_verify`` and ``query`` are patched with fakes.
"""

from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import aiohttp
import probatio
import pytest

from homeassistant.config_entries import ConfigEntryState, ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, intent, issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.setup import async_setup_component

from tests.common import MockConfigEntry
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
        assert result["step_id"] == "model"
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
