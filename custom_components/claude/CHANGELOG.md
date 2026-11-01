# Changelog

## Unreleased

- Update the Claude Agent SDK to 0.2.176 and the Claude Code CLI to 2.1.316. The CLI
  download grows to about 310 MB.
- Allow `mcp` 2.x, as the SDK now does and Home Assistant 2026.11 requires. A custom
  integration that still needs 1.x has to pin it itself.

## 1.1.0

- Keep the login's PKCE verifier out of the login link and the code you paste back, so
  seeing that code is no longer enough to redeem it. Refuse a code from a different
  login. Reported by [@mattabate](https://github.com/mattabate).
- Keep the Claude token out of the log: if a warning said "Could not fetch the Claude
  model list", log in again to replace it. Say that code execution has no sandbox and
  reaches your passwords and tokens, and warn in Repairs while it's on.
- Recognize a rejected API key during setup. Ask to log in again when a conversation or
  AI task finds the credentials rejected, and stop asking for new credentials when an
  account is out of credit; setup retries until it's topped up. Reported by
  [@mattabate](https://github.com/mattabate) in
  [#1](https://github.com/domenix/claude-conversation-agent/issues/1).
- List models live from Anthropic, warn in Repairs when an agent uses a deprecated one,
  ask for the model first and show only the options it supports, separate the thinking
  type from the effort (saved agents migrate and keep their behavior), and rename
  "Control Home Assistant" to "AI tools" as a checkbox list.
- Give Claude only the built-in tools the options enable, send it a tool's own data,
  flag failed tools as errors, and stop the `tool_result` deprecation warning. Say when
  a reply hit the output or context limit, report refused requests as errors, and fix
  the "Last content in chat log is not an AssistantContent" failure on AI tasks with a
  structure. Stop the Claude CLI within seconds when a run ends early, add the
  integration's icon and logo, and add cache usage, the stop reason, the message ID and
  failed tools to the conversation trace.

## 1.0.2

- Keep transient network failures retryable during setup. Previously a Home
  Assistant boot without internet could leave the integration permanently
  dead (`setup_error`, which never retries) until a manual reload: the CLI
  download ran outside the error-classification path, and offline errors
  phrased in auth terms ("unable to refresh authentication token") were
  misclassified as invalid credentials. Network failures now map to
  `ConfigEntryNotReady`, so setup retries automatically until connectivity
  returns. Genuine credential problems still trigger reauthentication.

## 1.0.1

- Raise the default run timeout to 290 s (was 90 s) and the per-tool timeout to
  180 s (was 45 s), so multi-step turns and slow tools have room to finish.
  290 s sits just under Assist's own 300 s pipeline timeout, so a run that
  overruns reports what timed out instead of a generic pipeline error.

## 1.0.0

Initial release.

- Conversation agent and AI Task entities backed by your Claude subscription,
  with Assist device control, streaming, attachments, extended thinking, web
  search / fetch and structured output.
- **Log in with Claude** in the config flow, or enter an OAuth token / API key
  manually. The login expiry is watched: a repair issue is raised 14 days ahead
  of it and again once it lapses.
- Home Assistant LLM tools are exposed to the model through an in-process MCP
  server and executed directly via the conversation's LLM API.
- Per-subentry options for model, thinking budget / effort, code execution,
  subagents, web search / fetch, activity streaming and run timeout.
