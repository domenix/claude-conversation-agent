# Tests

Smoke tests for the integration.

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv pytest-homeassistant-custom-component ruff
constraints=$(.venv/bin/python -c "import homeassistant, os; print(os.path.dirname(homeassistant.__file__))")/package_constraints.txt
uv pip install --python .venv -c "$constraints" $(jq -r '.requirements[]' custom_components/claude/manifest.json) hassil home-assistant-intents gazetteer-matcher PyTurboJPEG
.venv/bin/ruff check custom_components tests
.venv/bin/python -m pytest tests -q
```

Pin the interpreter to the one current Home Assistant requires. On an older
Python, the resolver silently walks back to an ancient
`pytest-homeassistant-custom-component`, and its Home Assistant pin with it, so
the tests run against a version you don't ship on.

The second install adds the integration's requirements and what the platforms
under test need (`conversation` needs `hassil`, `home-assistant-intents` and
`gazetteer-matcher`; `ai_task` needs `PyTurboJPEG`) at the versions the installed
Home Assistant pins, which is how Home Assistant installs them for users.

## Staying current

The integration supports the latest Home Assistant release only, so it carries
no compatibility code for older ones. When a release is out, drop whatever only
that older version needed. Three things flag drift, and CI runs them weekly as
well as on every push:

- `ruff.toml` bans what core bans for its own integrations.
- `conftest.py` fails a test when Home Assistant reports a deprecated call by this
  integration. Otherwise, a custom integration only gets a log line until the
  release that breaks it.
- `tests/import_all.py` imports every module on the newest beta.
  `pytest-homeassistant-custom-component` only follows stable releases, so the
  full suite can't run there until the release is out.

Renovate (`.github/renovate.json`) opens a PR when Home Assistant stable or beta,
the Claude Agent SDK or the Claude Code CLI has a new release, and CI runs against
the new version on that PR. Nothing merges by itself: the tests fake the SDK, so an
SDK bump also needs one real conversation with a tool call before it's merged.

To ship, run `script/release <version>`. It sets the version and raises the minimum
in `hacs.json` to the Home Assistant release CI tests against, then commits and
tags. Pushing the tag runs CI and, if that passes, publishes the GitHub release
HACS offers to users.

`conftest.py` also maps the `tests.*` helper imports onto
`pytest_homeassistant_custom_component`, so the same test file also runs
unchanged inside a home-assistant/core checkout.
