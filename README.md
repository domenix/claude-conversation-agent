# Claude

Claude, but for Home Assistant.

The **Claude** integration adds a conversation agent and an AI task entity powered
by your Claude subscription. It strives to have feature parity with the stock
[Anthropic integration](https://www.home-assistant.io/integrations/anthropic/).
This integration is not affiliated with the Open Home Foundation,
Nabu Casa or Anthropic.

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
2. Choose **Log in with Claude**, open the link, approve, and paste the code
   back. Or enter a Claude OAuth token / Anthropic API key manually.
3. The integration creates a conversation agent (with Assist control) and an AI
   Task entity. Tune model, thinking, web search, and more from the integration's subentries.

Full setup, feature mapping and limitations:
[DOCS](./custom_components/claude/DOCS.md).

## Development

Tests run inside a home-assistant/core checkout. See [`tests/README.md`](./tests/README.md).

## Credits

This integration mirrors the capabilities of the stock
[Anthropic integration](https://www.home-assistant.io/integrations/anthropic/),
created and maintained by
[Denis Shulyaka](https://github.com/Shulyaka).
