"""AI Task integration for Claude."""

from json import JSONDecodeError
import re
from typing import override

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .const import DOMAIN, LOGGER
from .coordinator import ClaudeCodeConfigEntry
from .entity import ClaudeCodeBaseLLMEntity

PARALLEL_UPDATES = 0

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ClaudeCodeConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [ClaudeCodeTaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ClaudeCodeTaskEntity(
    ai_task.AITaskEntity,
    ClaudeCodeBaseLLMEntity,
):
    """Claude AI Task entity."""

    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )
    _attr_translation_key = "ai_task_data"

    @override
    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        structured_output = await self._async_handle_chat_log(
            chat_log, task.name, task.structure
        )

        # Claude answers a structure through a tool, often with no reply text.
        if task.structure and structured_output is not None:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=structured_output,
            )

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="response_not_found"
            )

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )
        try:
            data = json_loads(_JSON_FENCE.sub("", text).strip())
        except JSONDecodeError as err:
            LOGGER.error(
                "Failed to parse JSON response: %s. Response: %s", err, text
            )
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="json_parse_error"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
