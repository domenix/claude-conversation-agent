"""Import every module of the integration against the installed Home Assistant.

Used on Home Assistant betas, where the test suite's plugin may not work yet.
"""

from importlib import import_module
from pathlib import Path

from homeassistant.const import __version__

import custom_components.claude as integration

for path in sorted(Path(integration.__file__).parent.glob("*.py")):
    if path.stem != "__init__":
        import_module(f"{integration.__name__}.{path.stem}")

print(f"All modules import on Home Assistant {__version__}")
