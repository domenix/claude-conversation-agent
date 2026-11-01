"""Base entity for Claude."""

import asyncio
import base64
from collections.abc import AsyncIterator, Callable
from mimetypes import guess_file_type
from pathlib import Path
from typing import Any

import probatio

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .agent import ClaudeAgent, ClaudeAgentError, CredentialsInvalidError, query
from .const import (
    ALWAYS_DISALLOWED,
    CODE_TOOLS,
    CONF_CHAT_MODEL,
    CONF_CODE_EXECUTION,
    CONF_MAX_TOKENS,
    CONF_RUN_TIMEOUT,
    CONF_SHOW_ACTIVITY,
    CONF_THINKING_BUDGET,
    CONF_THINKING_EFFORT,
    CONF_WEB_FETCH,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_USER_LOCATION,
    DEFAULT,
    DOMAIN,
    HA_TOOL_PREFIX,
    LOGGER,
    MAX_TURNS,
)
from .coordinator import ClaudeCodeConfigEntry, ClaudeCodeCoordinator


def _to_json_schema(
    schema: Any, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Convert an HA schema (tool parameters, AI Task structure) to JSON Schema."""
    return probatio.to_openapi(
        schema, custom_serializer=custom_serializer, openapi_version="3.1.0"
    )


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format an HA LLM tool for the Agent SDK MCP server."""
    unsupported_keys = {"oneOf", "anyOf", "allOf"}
    schema = _to_json_schema(tool.parameters, custom_serializer)
    schema = {k: v for k, v in schema.items() if k not in unsupported_keys}
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": schema,
    }


def _render_transcript(contents: list[conversation.Content]) -> str:
    """Render prior chat log content as plain text context.

    Used when there is no Claude session to resume (e.g. after a Home
    Assistant restart) so the model still sees the conversation so far.
    """
    lines: list[str] = []
    for content in contents:
        if isinstance(content, conversation.UserContent) and content.content:
            lines.append(f"User: {content.content}")
        elif isinstance(content, conversation.AssistantContent) and content.content:
            lines.append(f"Assistant: {content.content}")
    return "\n".join(lines)


class SDKDeltaStream:
    """Convert Agent SDK messages into HA chat_log deltas.

    The SDK yields partial ``StreamEvent`` for text / thinking deltas,
    ``AssistantMessage`` for tool-use blocks, ``UserMessage`` for tool-result
    blocks, and a final ``ResultMessage``.
    """

    def __init__(
        self,
        chat_log: conversation.ChatLog,
        messages: AsyncIterator[Any],
        on_session: Callable[[str], None],
        show_activity: bool = False,
    ) -> None:
        """Initialize the delta stream."""
        self._chat_log = chat_log
        self._messages = messages
        self._on_session = on_session
        self._show_activity = show_activity
        self._need_role = True
        self._claude_tool_names: dict[str, str] = {}
        self.result_error: str | None = None
        self.structured_output: Any = None

    def __aiter__(
        self,
    ) -> AsyncIterator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        """Return the delta iterator."""
        return self._stream()

    async def _stream(
        self,
    ) -> AsyncIterator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        async for message in self._messages:
            LOGGER.debug("Agent message: %s", type(message).__name__)
            kind = type(message).__name__
            if kind == "SystemMessage":
                if message.subtype == "init":
                    if session_id := message.data.get("session_id"):
                        self._on_session(session_id)
            elif kind == "StreamEvent":
                async for delta in self._handle_stream_event(message.event):
                    yield delta
            elif kind == "AssistantMessage":
                async for delta in self._handle_assistant(message):
                    yield delta
            elif kind == "UserMessage":
                async for delta in self._handle_user(message):
                    yield delta
            elif kind == "ResultMessage":
                self._handle_result(message)

    async def _handle_stream_event(
        self, event: dict[str, Any]
    ) -> AsyncIterator[conversation.AssistantContentDeltaDict]:
        etype = event.get("type")
        if etype == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta" and delta.get("text"):
                if self._need_role:
                    self._need_role = False
                    yield {"role": "assistant"}
                yield {"content": delta["text"]}
            elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                if self._need_role:
                    self._need_role = False
                    yield {"role": "assistant"}
                yield {"thinking_content": delta["thinking"]}
        elif etype == "message_delta":
            usage = event.get("usage") or {}
            if usage:
                self._chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": usage.get("input_tokens", 0),
                            "cached_input_tokens": usage.get(
                                "cache_read_input_tokens", 0
                            ),
                            "output_tokens": usage.get("output_tokens", 0),
                        }
                    }
                )

    async def _handle_assistant(
        self, message: Any
    ) -> AsyncIterator[conversation.AssistantContentDeltaDict]:
        # Text/thinking arrive via StreamEvent; here only surface tool-use so
        # every tool call shows up in the conversation trace. All tools —
        # HA tools (run in-process by the SDK MCP server) and Claude built-ins
        # (run by the CLI) — are already executed by the agent loop, so they are
        # marked external and chat_log records them without re-executing.
        for block in message.content:
            if type(block).__name__ != "ToolUseBlock":
                continue
            if self._need_role:
                self._need_role = False
                yield {"role": "assistant"}
            self._claude_tool_names[block.id] = block.name
            # Optionally surface the action live as reasoning text so the user
            # can see what the agent is doing (tool calls, subagent spawns)
            # instead of a silent spinner.
            if self._show_activity:
                yield {"thinking_content": _activity_line(block)}
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=block.id,
                        tool_name=_strip_prefix(block.name),
                        tool_args=block.input or {},
                        external=True,
                    )
                ]
            }
            self._need_role = True

    async def _handle_user(
        self, message: Any
    ) -> AsyncIterator[conversation.ToolResultContentDeltaDict]:
        content = message.content
        if not isinstance(content, list):
            return
        for block in content:
            if type(block).__name__ != "ToolResultBlock":
                continue
            yield {
                "role": "tool_result",
                "tool_call_id": block.tool_use_id,
                "tool_name": _strip_prefix(
                    self._claude_tool_names.get(block.tool_use_id, "unknown")
                ),
                "result": llm.ToolResult(
                    data={"content": _stringify(block.content)},
                    error=bool(block.is_error),
                ),
            }
            self._need_role = True

    def _handle_result(self, message: Any) -> None:
        if session_id := getattr(message, "session_id", None):
            self._on_session(session_id)
        self.structured_output = getattr(message, "structured_output", None)
        if getattr(message, "is_error", False):
            self.result_error = str(
                getattr(message, "result", None) or message.subtype
            )
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="run_error",
                translation_placeholders={"message": self.result_error},
            )


def _strip_prefix(name: str) -> str:
    """Strip the mcp__ha__ prefix from an HA tool name for the trace."""
    return name[len(HA_TOOL_PREFIX) :] if name.startswith(HA_TOOL_PREFIX) else name


def _activity_line(block: Any) -> str:
    """A short human-readable status line for a tool-use block."""
    name = _strip_prefix(block.name)
    if block.name == "Task":
        desc = (block.input or {}).get("description") or "working"
        return f"\n🧠 Delegating a subagent: {desc}…\n"
    return f"\n🔧 {name}…\n"


def _stringify(content: Any) -> str:
    """Coerce a tool-result content block to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


class ClaudeCodeBaseLLMEntity(CoordinatorEntity[ClaudeCodeCoordinator]):
    """Claude base LLM entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: ClaudeCodeConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry.runtime_data)
        self.entry = entry
        self.subentry = subentry
        model_info = entry.runtime_data.get_model_info(
            subentry.data.get(CONF_CHAT_MODEL, DEFAULT[CONF_CHAT_MODEL])
        )
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Anthropic",
            model=model_info.display_name,
            model_id=None if model_info.id == "default" else model_info.id,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    def _user_location(self) -> str | None:
        """Compose an approximate location string from the HA config."""
        hass = self.hass
        parts: list[str] = []
        if hass.config.latitude and hass.config.longitude:
            parts.append(f"{hass.config.latitude:.2f},{hass.config.longitude:.2f}")
        if hass.config.country:
            parts.append(f"country {hass.config.country}")
        if hass.config.time_zone:
            parts.append(f"timezone {hass.config.time_zone}")
        return "; ".join(parts) or None

    def _build_system_prompt(
        self, chat_log: conversation.ChatLog, resume: str | None
    ) -> str:
        """Compose the system prompt, folding in transcript when not resuming."""
        system = chat_log.content[0]
        if not isinstance(system, conversation.SystemContent):
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="system_message_not_found"
            )
        prompt = system.content or ""
        options = DEFAULT | dict(self.subentry.data)
        if options[CONF_WEB_SEARCH] and options[CONF_WEB_SEARCH_USER_LOCATION]:
            if location := self._user_location():
                prompt += f"\n\nThe user's approximate location: {location}"
        if not resume and len(chat_log.content) > 2:
            transcript = _render_transcript(chat_log.content[1:-1])
            if transcript:
                prompt += f"\n\nEarlier conversation with the user:\n{transcript}"
        return prompt

    def _disallowed_tools(self) -> list[str]:
        """Gate Claude built-in tools by the subentry options."""
        options = DEFAULT | dict(self.subentry.data)
        disallowed = list(ALWAYS_DISALLOWED)
        if not options[CONF_WEB_SEARCH]:
            disallowed.append("WebSearch")
        if not options[CONF_WEB_FETCH]:
            disallowed.append("WebFetch")
        if not options[CONF_CODE_EXECUTION]:
            disallowed.extend(CODE_TOOLS)
        return disallowed

    async def _build_prompt(
        self, chat_log: conversation.ChatLog, session_id: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        """Build the streaming prompt (one user turn) for the Agent SDK."""
        last_content = chat_log.content[-1]
        if not isinstance(last_content, conversation.UserContent):
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="user_message_not_found"
            )

        blocks: list[dict[str, Any]] = []
        if last_content.content:
            blocks.append({"type": "text", "text": last_content.content})
        if last_content.attachments:
            for att in await async_prepare_files_for_prompt(
                self.hass,
                [(a.path, a.mime_type) for a in last_content.attachments],
            ):
                is_pdf = att["media_type"] == "application/pdf"
                blocks.append(
                    {
                        "type": "document" if is_pdf else "image",
                        "source": {
                            "type": "base64",
                            "media_type": att["media_type"],
                            "data": att["data"],
                        },
                    }
                )
        if not blocks:
            blocks.append({"type": "text", "text": ""})

        message = {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": None,
            "session_id": session_id or "",
        }

        async def _gen() -> AsyncIterator[dict[str, Any]]:
            yield message

        return _gen()

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: probatio.Schema | None = None,
    ) -> Any:
        """Generate an answer for the chat log with the in-process agent.

        Returns the structured output object when a structure was requested and
        the model produced one, else None.
        """
        coordinator = self.entry.runtime_data
        agent: ClaudeAgent = coordinator.agent
        options = DEFAULT | dict(self.subentry.data)

        session_id = coordinator.session_ids.get(chat_log.conversation_id)

        tools: list[dict[str, Any]] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        req_model = self.subentry.data.get(CONF_CHAT_MODEL, DEFAULT[CONF_CHAT_MODEL])
        model = req_model if req_model and req_model != "default" else None

        effort = options[CONF_THINKING_EFFORT]
        structure_dict = None
        if structure:
            structure_dict = _to_json_schema(
                structure,
                chat_log.llm_api.custom_serializer
                if chat_log.llm_api
                else llm.selector_serializer,
            )

        await agent.async_prepare()
        tool_server = agent.build_tool_server(
            tools, chat_log.llm_api, coordinator.tool_timeout
        )
        agent_options = agent.build_options(
            model=model,
            resume=session_id,
            system_prompt=self._build_system_prompt(chat_log, session_id),
            disallowed_tools=self._disallowed_tools(),
            tool_server=tool_server,
            thinking_budget=options[CONF_THINKING_BUDGET],
            effort=effort if effort != "default" else None,
            max_output_tokens=options[CONF_MAX_TOKENS] or None,
            structure=structure_dict,
            max_turns=MAX_TURNS,
            stderr=lambda line: LOGGER.debug("claude: %s", line.rstrip()),
        )

        def on_session(sid: str) -> None:
            coordinator.session_ids[chat_log.conversation_id] = sid

        run_timeout = options[CONF_RUN_TIMEOUT]
        prompt = await self._build_prompt(chat_log, session_id)
        messages = query(prompt=prompt, options=agent_options)
        stream = SDKDeltaStream(
            chat_log, messages, on_session, options[CONF_SHOW_ACTIVITY]
        )

        LOGGER.info(
            "Chat run started (model: %s%s)",
            model or "default",
            f", resume: {session_id}" if session_id else "",
        )
        try:
            async with asyncio.timeout(run_timeout if run_timeout > 0 else None):
                async for _content in chat_log.async_add_delta_content_stream(
                    self.entity_id, stream
                ):
                    pass
        except TimeoutError as err:
            LOGGER.error("Chat run timed out after %ss", run_timeout)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="run_error",
                translation_placeholders={
                    "message": f"run timed out after {run_timeout}s"
                },
            ) from err
        except CredentialsInvalidError as err:
            coordinator.async_set_auth_failed()
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="not_authenticated",
            ) from err
        except ClaudeAgentError as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="run_error",
                translation_placeholders={"message": str(err)},
            ) from err

        return stream.structured_output


async def async_prepare_files_for_prompt(
    hass: HomeAssistant, files: list[tuple[Path, str | None]]
) -> list[dict[str, str]]:
    """Encode attachment files for the prompt.

    Caller needs to ensure that the files are allowed.
    """

    def encode_files() -> list[dict[str, str]]:
        encoded: list[dict[str, str]] = []
        for file_path, mime_type in files:
            if not file_path.exists():
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="wrong_file_path",
                    translation_placeholders={"file_path": file_path.as_posix()},
                )
            if mime_type is None:
                mime_type = guess_file_type(file_path)[0]
            if not mime_type or not mime_type.startswith(
                ("image/", "application/pdf")
            ):
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="wrong_file_type",
                    translation_placeholders={
                        "file_path": file_path.as_posix(),
                        "mime_type": mime_type or "unknown",
                    },
                )
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"
            encoded.append(
                {
                    "media_type": mime_type,
                    "data": base64.b64encode(file_path.read_bytes()).decode("utf-8"),
                }
            )
        return encoded

    return await hass.async_add_executor_job(encode_files)
