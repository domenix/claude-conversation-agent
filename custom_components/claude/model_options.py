"""Model-specific options for the Claude integration."""

from typing import Any

from .const import (
    CONF_EFFORT,
    CONF_MAX_TOKENS,
    CONF_THINKING_BUDGET,
    CONF_THINKING_TYPE,
    CONF_WEB_SEARCH,
    DEFAULT,
    EFFORT_OPTIONS,
    MIN_THINKING_BUDGET,
    THINKING_TYPE_OPTIONS,
    Model,
    model_alias,
)


def _supported(capability: Any) -> bool:
    """Return whether a capability from Anthropic's API is supported."""
    return isinstance(capability, dict) and capability.get("supported") is True


def get_thinking_types(model: Model) -> list[str]:
    """Return the thinking types to offer for a model."""
    if model.capabilities is None:
        return THINKING_TYPE_OPTIONS
    thinking = model.capabilities.get("thinking") or {}
    types = thinking.get("types") or {}
    return [
        value
        for value in THINKING_TYPE_OPTIONS
        if value == "default"
        or (
            (value == "disabled" or _supported(thinking))
            and _supported(types.get(value))
        )
    ]


def get_efforts(model: Model) -> list[str]:
    """Return the effort levels to offer for a model."""
    if model.capabilities is None:
        return EFFORT_OPTIONS
    effort = model.capabilities.get("effort") or {}
    return [
        value
        for value in EFFORT_OPTIONS
        if value == "default" or (_supported(effort) and _supported(effort.get(value)))
    ]


def supports_web_search(model: Model) -> bool:
    """Return whether a model can search the web."""
    if model.capabilities is None:
        return True
    server_tools = model.capabilities.get("server_tools") or {}
    return _supported(server_tools.get("web_search"))


def supports_structured_output(model: Model) -> bool:
    """Return whether a model can return structured output, as an AI task needs."""
    if model.capabilities is None:
        return True
    return _supported(model.capabilities.get("structured_outputs"))


def _is_effort_compatible(model: Model, thinking_type: str, effort: str) -> bool:
    """Return whether an effort works with a thinking type on a model."""
    # Anthropic's API does not describe restrictions between supported settings.
    return not (
        effort in ("xhigh", "max")
        and thinking_type == "disabled"
        and model_alias(model.id) in ("claude-haiku-5-5", "claude-opus-5")
    )


def reconcile_model_options(options: dict[str, Any], model: Model) -> None:
    """Move options to the nearest thing a model supports."""
    if model.max_tokens and options.get(CONF_MAX_TOKENS, 0) > model.max_tokens:
        options[CONF_MAX_TOKENS] = model.max_tokens

    if options.get(CONF_THINKING_TYPE, "default") not in get_thinking_types(model):
        options[CONF_THINKING_TYPE] = "default"

    if options.get(CONF_THINKING_TYPE) == "enabled":
        max_tokens = options.get(CONF_MAX_TOKENS, DEFAULT[CONF_MAX_TOKENS])
        if max_tokens <= MIN_THINKING_BUDGET:
            max_tokens = options[CONF_MAX_TOKENS] = MIN_THINKING_BUDGET + 1
        budget = options.get(CONF_THINKING_BUDGET, DEFAULT[CONF_THINKING_BUDGET])
        options[CONF_THINKING_BUDGET] = min(
            max(budget, MIN_THINKING_BUDGET), max_tokens - 1
        )

    effort = options.get(CONF_EFFORT, "default")
    if effort != "default":
        levels = [
            value
            for value in get_efforts(model)
            if value != "default"
            and _is_effort_compatible(
                model, options.get(CONF_THINKING_TYPE, "default"), value
            )
        ]
        if effort not in levels:
            # The nearest lower level, or else the lowest there is.
            order = EFFORT_OPTIONS[1:]
            lower = order[: order.index(effort)]
            options[CONF_EFFORT] = next(
                (value for value in reversed(lower) if value in levels),
                levels[0] if levels else "default",
            )

    if not supports_web_search(model):
        options[CONF_WEB_SEARCH] = False
