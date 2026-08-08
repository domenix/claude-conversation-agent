"""Smoke tests for the claude custom integration.

See tests/README.md for how to run them. The Claude Agent SDK is never really
invoked; ``async_verify`` and ``query`` are patched with fakes.
"""

from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
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
        self.is_error = False


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


def _fake_query(messages: list[Any]):
    """Return a patch target that yields the given messages from query()."""

    def _query(prompt: Any, options: Any) -> AsyncIterator[Any]:
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
    with patch("custom_components.claude.entity.query", _fake_query(messages)):
        result = await conversation.async_converse(
            hass,
            "what's up",
            None,
            None,
            agent_id="conversation.claude_conversation",
        )

    assert result.response.speech["plain"]["speech"] == "Done"


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
