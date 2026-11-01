"""In-process Claude agent.

Wraps the Python Claude Agent SDK (``claude_agent_sdk``), which drives the
bundled Claude Code CLI as a subprocess. Everything runs inside Home
Assistant — there is no add-on and no network hop. Home Assistant LLM tools are
exposed to the model through an in-process SDK MCP server and executed directly
via the conversation's LLM API.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import aiohttp
from claude_agent_sdk import (
    ClaudeAgentOptions,
    SdkMcpTool,
    create_sdk_mcp_server,
    query,
)
from claude_agent_sdk.types import McpSdkServerConfig

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.json import json_dumps

from .cli import CliError, async_ensure_cli
from .const import (
    ANTHROPIC_VERSION,
    HA_SERVER_NAME,
    HOME_SUBDIR,
    LOGGER,
    MODELS_FETCH_TIMEOUT_S,
    MODELS_PAGE_LIMIT,
    MODELS_URL,
    WORKSPACE_SUBDIR,
    Model,
)

# Re-exported so callers stream messages without importing the SDK directly.
__all__ = [
    "ClaudeAgent",
    "ClaudeAgentError",
    "CredentialsInvalidError",
    "query",
    "raise_for_result",
]

VERIFY_MODEL = "claude-haiku-4-5"


class ClaudeAgentError(Exception):
    """The Claude Code CLI failed to run a turn."""


class CredentialsInvalidError(ClaudeAgentError):
    """The configured credentials were rejected."""


class ClaudeAgent:
    """Owns credentials, workspace paths and Agent SDK option construction."""

    def __init__(
        self, hass: HomeAssistant, oauth_token: str, api_key: str
    ) -> None:
        """Initialize the agent."""
        self._hass = hass
        self._oauth_token = oauth_token
        self._api_key = api_key
        self.home = Path(hass.config.path(HOME_SUBDIR))
        self.workspace = Path(hass.config.path(WORKSPACE_SUBDIR))
        # Path to the provisioned musl/glibc CLI; set by async_prepare().
        self._cli_path: str | None = None

    @property
    def authenticated(self) -> bool:
        """Return whether any credential is configured."""
        return bool(self._oauth_token or self._api_key)

    async def async_list_models(self) -> list[Model]:
        """Fetch the models these credentials can use, newest first.

        Raises ``aiohttp.ClientError`` / ``TimeoutError`` on network or HTTP
        errors and ``KeyError`` / ``TypeError`` on an unexpected response.
        """
        session = async_get_clientsession(self._hass)
        params = {"limit": str(MODELS_PAGE_LIMIT)}
        models: list[Model] = []
        while True:
            async with session.get(
                MODELS_URL,
                headers=self._api_headers(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=MODELS_FETCH_TIMEOUT_S),
            ) as response:
                response.raise_for_status()
                page = await response.json()
            models.extend(_model_from_api(item) for item in page["data"])
            if not page.get("has_more") or not page.get("last_id"):
                return models
            params["after_id"] = page["last_id"]

    def _api_headers(self) -> dict[str, str]:
        """Return the headers for Anthropic's API."""
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if self._oauth_token:
            headers["Authorization"] = f"Bearer {self._oauth_token}"
        else:
            headers["x-api-key"] = self._api_key
        return headers

    async def async_prepare(self) -> None:
        """Create the CLI home/workspace dirs and provision the CLI binary."""

        def _mkdirs() -> None:
            self.home.mkdir(parents=True, exist_ok=True)
            self.workspace.mkdir(parents=True, exist_ok=True)

        await self._hass.async_add_executor_job(_mkdirs)
        try:
            self._cli_path = await async_ensure_cli(self._hass)
        except CliError as err:
            raise ClaudeAgentError(f"Claude Code CLI unavailable: {err}") from err

    def base_env(self, max_output_tokens: int | None = None) -> dict[str, str]:
        """Build the environment for the Claude subprocess.

        Merged over ``os.environ`` by the SDK, so only overrides are set.
        """
        env: dict[str, str] = {
            # The CLI stores its credentials/cache under HOME; keep it inside
            # the HA config dir so it survives restarts and stays writable.
            "HOME": str(self.home),
            # Don't send non-essential telemetry from a home server.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            # Home Assistant Core runs as root, where the CLI requires this to
            # accept the permission mode set below. Re-check on CLI upgrades.
            "IS_SANDBOX": "1",
        }
        if self._oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self._oauth_token
        elif self._api_key:
            env["ANTHROPIC_API_KEY"] = self._api_key
        if max_output_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
        return env

    def build_tool_server(
        self,
        tools: list[dict[str, Any]],
        llm_api: llm.APIInstance | None,
        tool_timeout: int,
    ) -> McpSdkServerConfig | None:
        """Build an in-process MCP server exposing the HA LLM tools.

        ``tools`` are the wire-format tool definitions (name/description/
        input_schema). Calls are dispatched to ``llm_api`` and executed inline;
        a slow tool is bounded by ``tool_timeout`` seconds (0 disables).
        """
        if not tools or llm_api is None:
            return None

        sdk_tools = [
            SdkMcpTool(
                name=tool["name"],
                description=tool["description"],
                input_schema=tool["input_schema"],
                handler=self._make_handler(tool["name"], llm_api, tool_timeout),
            )
            for tool in tools
        ]
        return create_sdk_mcp_server(name=HA_SERVER_NAME, tools=sdk_tools)

    def _make_handler(
        self, tool_name: str, llm_api: llm.APIInstance, tool_timeout: int
    ) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        """Return an MCP handler that runs one HA tool via the LLM API."""

        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            tool_input = llm.ToolInput(tool_name=tool_name, tool_args=args)
            try:
                call = llm_api.async_call_tool(tool_input)
                result = (
                    await asyncio.wait_for(call, tool_timeout)
                    if tool_timeout > 0
                    else await call
                )
                text = json_dumps(result.data)
            except TimeoutError:
                LOGGER.warning("HA tool %s timed out after %ss", tool_name, tool_timeout)
                return _error(f"Tool '{tool_name}' did not respond within {tool_timeout}s")
            except Exception as err:  # noqa: BLE001 - surface to the model, not a crash
                LOGGER.warning("HA tool %s failed: %s", tool_name, err)
                return _error(str(err))
            return {"content": [{"type": "text", "text": text}], "is_error": result.error}

        return handler

    def build_options(
        self,
        *,
        model: str | None,
        resume: str | None,
        system_prompt: str,
        builtin_tools: list[str],
        tool_server: McpSdkServerConfig | None,
        thinking_type: str,
        thinking_budget: int,
        effort: str | None,
        max_output_tokens: int | None,
        structure: dict[str, Any] | None,
        max_turns: int,
        stderr: Callable[[str], None],
    ) -> ClaudeAgentOptions:
        """Assemble ClaudeAgentOptions for one run."""
        mcp_servers: dict[str, Any] = {}
        if tool_server is not None:
            mcp_servers[HA_SERVER_NAME] = tool_server

        options = ClaudeAgentOptions(
            model=model or None,
            resume=resume or None,
            cwd=str(self.workspace),
            cli_path=self._cli_path,
            env=self.base_env(max_output_tokens),
            system_prompt=system_prompt or None,
            # Never load rules from disk (~/.claude, project .claude); the turn
            # must be fully driven by what HA passes.
            setting_sources=[],
            # A message can't read files with "@path" or run a slash command.
            verbatim_prompts=True,
            include_partial_messages=True,
            permission_mode="bypassPermissions",
            max_turns=max_turns,
            tools=builtin_tools,
            mcp_servers=mcp_servers,
            strict_mcp_config=True,
            stderr=stderr,
        )
        match thinking_type:
            case "adaptive":
                options.thinking = {"type": "adaptive", "display": "summarized"}
            case "enabled":
                options.thinking = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                    "display": "summarized",
                }
            case "disabled":
                options.thinking = {"type": "disabled"}
        if effort:
            options.effort = effort
        if structure:
            options.output_format = {"type": "json_schema", "schema": structure}
        return options

    async def async_verify(self) -> None:
        """Confirm the credentials work with a minimal query.

        Raises CredentialsInvalidError / ClaudeAgentError on failure.
        """
        try:
            await self.async_prepare()
        except ClaudeAgentError:
            raise
        except Exception as err:  # noqa: BLE001
            # Preparation is dominated by the CLI download — a raw network
            # error here must stay a retryable ClaudeAgentError, not escape
            # as an unexpected exception (which ends as a permanent
            # setup_error when this runs during config entry setup).
            raise ClaudeAgentError(f"{type(err).__name__}: {err}") from err
        # Capture the CLI's stderr so a spawn/auth failure gives a real reason
        # instead of a bare exit code.
        stderr_lines: list[str] = []

        def _stderr(line: str) -> None:
            line = line.rstrip()
            if line:
                stderr_lines.append(line)
                LOGGER.debug("claude: %s", line)

        options = ClaudeAgentOptions(
            model=VERIFY_MODEL,
            cwd=str(self.workspace),
            cli_path=self._cli_path,
            env=self.base_env(),
            system_prompt="Reply with the single word: ok",
            setting_sources=[],
            max_turns=1,
            tools=[],
            strict_mcp_config=True,
            stderr=_stderr,
        )

        def _detail(base: str) -> str:
            tail = " | ".join(stderr_lines[-5:])
            return f"{base} :: {tail}" if tail else base

        api_error: str | None = None
        try:
            async for message in query(prompt="ok", options=options):
                if type(message).__name__ == "AssistantMessage":
                    api_error = message.error
                if type(message).__name__ != "ResultMessage":
                    continue
                if getattr(message, "is_error", False):
                    errors = getattr(message, "result", None) or message.subtype
                    raise_for_result(
                        _detail(str(errors)), api_error, match_wording=True
                    )
                return
            # Stream ended without a ResultMessage — treat as a failure and
            # surface whatever the CLI wrote to stderr.
            raise ClaudeAgentError(_detail("no result from Claude Code CLI"))
        except ClaudeAgentError:
            raise
        except Exception as err:  # noqa: BLE001
            raise ClaudeAgentError(_detail(f"{type(err).__name__}: {err}")) from err


def _model_from_api(item: dict[str, Any]) -> Model:
    """Convert a model from Anthropic's API."""
    return Model(
        item["id"],
        item.get("display_name") or item["id"],
        lifecycle=item.get("lifecycle"),
        retires_at=item.get("retires_at"),
        capabilities=item.get("capabilities"),
        max_tokens=item.get("max_tokens"),
    )


def _error(message: str) -> dict[str, Any]:
    """Return an MCP tool-error result the model can read and react to."""
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "is_error": True}


def raise_for_result(
    message: str, error: str | None = None, *, match_wording: bool = False
) -> None:
    """Classify a failing result message into an auth vs generic error.

    ``error`` is the SDK's ``AssistantMessage.error``. It spots rejected credentials
    the wording doesn't ("Invalid API key", a 403) and tells them apart from an
    account that only needs topping up, which must not ask for new credentials.

    ``match_wording`` falls back on the message when the CLI doesn't say what failed.
    That suits the setup check, whose message is short. A chat run's can quote
    anything, such as a session ID containing "401", so it relies on ``error`` alone.

    Network trouble wins over auth wording: an offline CLI frequently phrases
    its failure in auth terms ("unable to refresh authentication token", 401s
    from a captive portal, …). Classifying those as bad credentials turns a
    transient outage into ConfigEntryAuthFailed → setup_error, which never
    retries — the entry then stays dead long after the network heals. A
    generic ClaudeAgentError instead becomes ConfigEntryNotReady on the first
    refresh, and Home Assistant keeps retrying until connectivity is back.
    """
    lowered = message.lower()
    if any(
        token in lowered
        for token in (
            "getaddrinfo",
            "enotfound",
            "econnrefused",
            "econnreset",
            "etimedout",
            "enetunreach",
            "ehostunreach",
            "fetch failed",
            "network",
            "dns",
            "timed out",
            "timeout",
            "connection error",
            "connect error",
        )
    ):
        raise ClaudeAgentError(message)
    if error == "authentication_failed":
        raise CredentialsInvalidError(message)
    if (
        match_wording
        and error in (None, "unknown")
        and any(
            token in lowered
            for token in ("401", "invalid bearer", "authentication", "unauthorized")
        )
    ):
        raise CredentialsInvalidError(message)
    raise ClaudeAgentError(message)
