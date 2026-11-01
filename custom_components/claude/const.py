"""Constants for the Claude integration."""

from dataclasses import dataclass
from datetime import timedelta
import logging

DOMAIN = "claude"
LOGGER = logging.getLogger(__package__)

DEFAULT_CONVERSATION_NAME = "Claude conversation"
DEFAULT_AI_TASK_NAME = "Claude AI Task"

# Credentials (entry data).
CONF_OAUTH_TOKEN = "oauth_token"
CONF_API_KEY = "api_key"
# ISO 8601 UTC timestamp of when CONF_OAUTH_TOKEN stops working, as reported by
# the token endpoint. Absent for manually pasted tokens and API keys.
CONF_OAUTH_EXPIRES_AT = "oauth_expires_at"

# Raise a repair issue this far ahead of expiry.
TOKEN_EXPIRY_WARNING = timedelta(days=14)
TOKEN_EXPIRY_CHECK_INTERVAL = timedelta(hours=12)
ISSUE_TOKEN_EXPIRING = "oauth_token_expiring"
ISSUE_TOKEN_EXPIRED = "oauth_token_expired"

# Subentry options.
CONF_RECOMMENDED = "recommended"
CONF_CHAT_MODEL = "chat_model"
CONF_CODE_EXECUTION = "code_execution"
CONF_MAX_TOKENS = "max_tokens"
CONF_THINKING_BUDGET = "thinking_budget"
CONF_THINKING_EFFORT = "thinking_effort"
CONF_WEB_FETCH = "web_fetch"
CONF_WEB_SEARCH = "web_search"
CONF_WEB_SEARCH_USER_LOCATION = "user_location"
CONF_RUN_TIMEOUT = "run_timeout"
CONF_SHOW_ACTIVITY = "show_activity"

DEFAULT = {
    CONF_CHAT_MODEL: "default",
    CONF_CODE_EXECUTION: False,
    CONF_MAX_TOKENS: 4096,
    CONF_THINKING_BUDGET: 0,
    CONF_THINKING_EFFORT: "default",
    CONF_WEB_FETCH: False,
    CONF_WEB_SEARCH: False,
    CONF_WEB_SEARCH_USER_LOCATION: False,
    CONF_RUN_TIMEOUT: 290,
    CONF_SHOW_ACTIVITY: False,
}

THINKING_EFFORT_OPTIONS = ["default", "low", "medium", "high", "xhigh", "max"]

MAX_TURNS = 50

# In-process MCP server name; HA LLM tools are exposed to Claude as
# mcp__ha__<tool_name> and executed in-process (no round-trip).
HA_SERVER_NAME = "ha"
HA_TOOL_PREFIX = f"mcp__{HA_SERVER_NAME}__"

# Claude built-in tools are off unless a subentry option turns them on, so a tool
# added by a new CLI release never reaches a conversation by accident.
WEB_SEARCH_TOOL = "WebSearch"
WEB_FETCH_TOOL = "WebFetch"
# Task (subagent fan-out) and the code tools are unlocked together by the
# "Code execution" option — subagents are only useful with tools to work with.
CODE_TOOLS = ["Bash", "Write", "Edit", "Read", "Glob", "Grep", "NotebookEdit", "Task"]

# A run with no timeout can hang forever if an upstream stream stalls; the
# conversation framework would otherwise block a long time. Bound each run and
# each in-process HA tool call so a stall fails fast.
#
# Assist wraps a run in its own 300 s pipeline timeout, which cancels the run
# and reports a generic error. Staying just under it means our own, specific
# error wins the race and says what actually timed out.
DEFAULT_RUN_TIMEOUT_S = 290
DEFAULT_TOOL_TIMEOUT_S = 180

# Directories, relative to the HA config dir, used by the bundled Claude
# CLI: HOME holds its credentials/cache, WORKSPACE is the agent's cwd.
HOME_SUBDIR = ".claude_home"
WORKSPACE_SUBDIR = ".claude_workspace"


@dataclass(slots=True, frozen=True)
class Model:
    """A model the integration can offer for the chat-model picker."""

    id: str
    display_name: str


# Anthropic's model list; works with both an API key and a subscription token.
MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"
MODELS_PAGE_LIMIT = 1000
MODELS_FETCH_TIMEOUT_S = 10

MODEL_DEFAULT = Model("default", "Account default")

# Offered only when the model list can't be fetched (offline, API error).
MODELS = [
    MODEL_DEFAULT,
    Model("claude-opus-4-8", "Claude Opus 4.8"),
    Model("claude-opus-4-7", "Claude Opus 4.7"),
    Model("claude-sonnet-5", "Claude Sonnet 5"),
    Model("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    Model("claude-haiku-4-5", "Claude Haiku 4.5"),
]
