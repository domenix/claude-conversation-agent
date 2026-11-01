# Claude

Run Home Assistant's conversation agent and AI Tasks through the Claude Agent
SDK. The integration runs the SDK in-process (the SDK drives the Claude Code CLI,
`claude -p`) and offers features similar to the stock Anthropic integration.

## How it works

```
Assist / AI Task
      │  (conversation entity, streaming)
claude integration        ── in-process, inside Home Assistant
      │  Claude Agent SDK (query) ── bundled claude CLI subprocess
      │  in-process MCP server "ha"  ← HA LLM tools, executed directly
your Claude subscription
```

- Home Assistant builds the prompt and exposes the selected LLM API tools (e.g.
  Assist) to the model as an in-process MCP server (`mcp__ha__*`).
- The agent loop runs inside Home Assistant. When Claude calls an HA tool, the
  integration executes it directly via the conversation's LLM API, with no
  round-trip. It respects your exposed entities exactly like the stock
  integration. The chat log still records tool calls and results.
- Conversations map to Claude sessions (`resume`), so context carries
  across turns. After a Home Assistant restart, the integration folds the prior
  transcript into the system prompt instead.

## Setup

**Settings → Devices & services → Add integration → Claude.** The first time
you submit credentials, the integration downloads the Claude Code CLI matching
your platform (~250 MB, cached under `/config`), so allow a few minutes. Then
pick one of:

### Log in with Claude (recommended)

Open the link the flow shows, approve access, and paste the code back. Home
Assistant notifies you before the login expires, and again if it does.

### Enter a token or API key manually

- **OAuth token:** from `claude setup-token` (the `sk-ant-oat01-...` value).
- **API key:** an Anthropic API key, used only when no OAuth token is set.

The integration verifies credentials with a minimal query before it creates the
entry. To change them later, use **Reconfigure**.

The integration creates a conversation agent (with Assist control enabled) and
an AI Task entity. Add more, or reconfigure options (model, thinking, web search,
web fetch, code execution), from the integration's subentries.

## Options (per subentry)

| Option | Description |
|---|---|
| Model | Model for responses, listed live from Anthropic (re-fetched each time the options open; a built-in list if that fails). "Account default" uses your Claude default; any other model ID can be typed in. |
| Maximum tokens | Cap on response tokens per turn, up to the model's limit. |
| Thinking type | Default (Claude decides), adaptive, enabled or disabled. |
| Thinking budget | Token budget for the "enabled" thinking type. |
| Effort | Effort level (default/low/medium/high/xhigh/max). |
| Code execution | Allow Bash / file tools on this system, with no sandbox. See [Known limitations](#known-limitations). |
| Web search / Web fetch | Enable Claude's WebSearch / WebFetch tools. |
| Include home location | Localize web search to your home location. |

The options follow the model you pick: settings it doesn't support aren't offered,
and a saved setting is moved to the nearest supported one at run time. With "Account
default" or a model Anthropic can't describe, every option stays available.

Credentials (OAuth token / API key) live on the integration entry itself and are
changed via **Reconfigure** or the reauth prompt.

## Feature mapping vs. the Anthropic integration

| Anthropic integration | Claude |
|---|---|
| API key | Claude subscription (OAuth token) or API key |
| Model selection | Same (fetched from Anthropic + custom value) |
| Assist / LLM API tools | Same (executed in-process) |
| Streaming responses | Same |
| Thinking type / budget / effort | Same, plus "Default" to let Claude decide |
| Web search / web fetch | Claude WebSearch / WebFetch tools |
| Code execution | Claude Bash/file tools, on this system with no sandbox |
| Attachments (images, PDF) | Same |
| AI Task structured output | Native JSON-schema output |
| Prompt caching strategy | Managed automatically by Claude |
| Per-tool max uses | Not available (bounded by max turns instead) |
| Repair flow for deprecated models | A warning in Repairs; change the model by reconfiguring the agent |

## Known limitations

- Latency is higher than the raw API: each turn spins the Claude agent
  loop. Expect a few seconds, which is noticeable on voice.
- Headless runs consume the Agent SDK credit pool of your subscription; when
  exhausted, requests fail until it resets (or configure an API key).
- The CLI is fetched for your platform (`linux-arm64`/`linux-x64`, glibc or
  musl) from Anthropic's release server on first setup; it needs outbound
  internet and ~250 MB free under `/config`.
- Code execution has no sandbox. It runs with Home Assistant's own rights, which is
  root on Home Assistant OS, so it can read and change everything, including your
  passwords and tokens. Anyone who can talk to the agent, or any text it reads, could
  make use of this. Leave it off unless you trust all of them. Repairs shows a warning
  while it's on.
- Citations are not surfaced.

## Troubleshooting

- **"Claude has no valid credentials"** / **"401 Invalid authentication
  credentials"**: the login expired or was revoked. Reconfigure the integration
  and log in again.
- **Setup hangs on first install:** the `claude-agent-sdk` wheel is large. Give
  it time, and check that Home Assistant has outbound internet for pip.
- **Model errors:** leave the model on "Account default" or pick one your
  subscription can access.
- Set the `claude` logger to `debug` to see the full event flow and the
  CLI's stderr.
