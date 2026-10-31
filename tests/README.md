# Tests

Smoke tests for the integration. The Claude Agent SDK is patched out, so no CLI,
network or credentials are required.

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv pytest-homeassistant-custom-component claude-agent-sdk
.venv/bin/python -m pytest tests -q
```

Pin the interpreter to the one current Home Assistant requires. On an older
Python, the resolver silently walks back to an ancient
`pytest-homeassistant-custom-component` — and its Home Assistant pin with it —
so the tests quietly run against a version you don't ship on.

Home Assistant pulls extra requirements for the platforms under test
(`conversation` needs `hassil` and `home-assistant-intents`, plus
`gazetteer-matcher` since 2026.10; `ai_task` needs `PyTurboJPEG`). Install them
at the versions pinned in the installed Home Assistant's
`homeassistant/package_constraints.txt`.

Tool and structured-output schemas go through `probatio` on Home Assistant
2026.9+ and through `voluptuous-openapi` before that, so run the suite against
both a current release and the oldest supported one (`hacs.json`) after
touching schema handling.

`conftest.py` maps the `tests.*` helper imports onto
`pytest_homeassistant_custom_component`, so the same test file also runs
unchanged inside a home-assistant/core checkout.
