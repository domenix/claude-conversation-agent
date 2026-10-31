# Claude

Claude, but for Home Assistant.

A conversation agent and AI Tasks powered by your **Claude subscription** — no
API key required. Feature parity with the stock
[Anthropic integration](https://www.home-assistant.io/integrations/anthropic/):
Assist device control, streaming replies, attachments, extended thinking, web
search / fetch, and structured AI Task output.

## Installation

### HACS

1. HACS → ⋮ → **Custom repositories**, add
   `https://github.com/domenix/claude-conversation-agent` as an **Integration**.
2. Install **Claude**, then restart Home Assistant.

### Manual

Copy [`custom_components/claude`](./custom_components/claude) into your
`/config/custom_components/` and restart Home Assistant.

## Setup

1. **Settings → Devices & services → Add integration → Claude.**
2. Choose **Log in with Claude** — open the link, approve, paste the code back —
   or enter a Claude OAuth token / Anthropic API key manually.
3. A conversation agent (with Assist control) and an AI Task entity are created.
   Tune model, thinking, web search, and more from the integration's subentries.

Full setup, feature mapping and limitations:
[DOCS](./custom_components/claude/DOCS.md).

## Development

Tests run inside a home-assistant/core checkout — see [`tests/README.md`](./tests/README.md).

## Credits

This integration mirrors the capabilities of the stock
[Anthropic integration](https://www.home-assistant.io/integrations/anthropic/),
created and maintained by
[Denis Shulyaka](https://github.com/Shulyaka).
