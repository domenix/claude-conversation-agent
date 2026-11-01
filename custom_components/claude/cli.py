"""Provision the Claude Code CLI binary.

The ``claude-agent-sdk`` wheel bundles a glibc CLI, but Home Assistant Core runs
on Alpine/musl. So we fetch the matching musl ``claude`` release, cache it under
the config dir, and point the SDK at it via ``cli_path``.

The version is taken from the installed SDK (``_cli_version``) so the CLI and
SDK always match. Downloads are checksum-verified against the release manifest.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import urllib.request

from claude_agent_sdk._cli_version import __cli_version__ as CLI_VERSION

from homeassistant.core import HomeAssistant

from .const import DOMAIN, HOME_SUBDIR, LOGGER

RELEASES_BASE = "https://downloads.claude.ai/claude-code-releases"
_CHUNK = 1 << 20  # 1 MiB

# Serialize provisioning so a pre-warm and an on-demand call don't both download.
_LOCK = asyncio.Lock()
_DATA_CLI_PATH = "cli_path"


class CliError(Exception):
    """The Claude Code CLI could not be provisioned."""


def _release_platform() -> str:
    """Return the release platform tag for this host, e.g. linux-arm64-musl."""
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64"):
        arch = "x64"
    else:
        raise CliError(f"unsupported architecture: {machine}")

    system = platform.system().lower()
    if system != "linux":
        # HA Core is Linux; anything else, let the SDK's own bundled CLI apply.
        raise CliError(f"unsupported OS for managed CLI: {system}")

    musl = bool(list(Path("/lib").glob("ld-musl-*.so.1"))) or any(
        Path(p).exists()
        for p in ("/lib/libc.musl-aarch64.so.1", "/lib/libc.musl-x86_64.so.1")
    )
    return f"linux-{arch}{'-musl' if musl else ''}"


def _installed_version(path: Path) -> str | None:
    """Return the ``x.y.z`` version the binary reports, or None."""
    try:
        out = subprocess.run(  # noqa: S603
            [str(path), "-v"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.split()
    return parts[0] if parts and parts[0][:1].isdigit() else None


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        return resp.read()


def _download_verified(url: str, dest: Path, checksum: str) -> None:
    """Stream ``url`` to ``dest``, verifying its sha256, then mark executable."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:  # noqa: S310
            while chunk := resp.read(_CHUNK):
                digest.update(chunk)
                fh.write(chunk)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        raise CliError(f"download failed: {err}") from err

    if digest.hexdigest() != checksum:
        tmp.unlink(missing_ok=True)
        raise CliError("checksum mismatch on downloaded CLI")

    tmp.chmod(tmp.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, dest)


def _ensure_cli_blocking(dest: Path) -> str:
    """Ensure the CLI exists at ``dest`` at the SDK-matched version (blocking)."""
    if dest.exists() and _installed_version(dest) == CLI_VERSION:
        return str(dest)

    tag = _release_platform()
    manifest_url = f"{RELEASES_BASE}/{CLI_VERSION}/manifest.json"
    try:
        manifest = json.loads(_fetch(manifest_url))
    except (OSError, ValueError) as err:
        raise CliError(f"could not fetch CLI manifest: {err}") from err

    entry = manifest.get("platforms", {}).get(tag)
    if not entry or "checksum" not in entry:
        raise CliError(f"no {tag} build in manifest for CLI {CLI_VERSION}")

    LOGGER.info(
        "Downloading Claude Code CLI %s (%s) — this happens once", CLI_VERSION, tag
    )
    _download_verified(
        f"{RELEASES_BASE}/{CLI_VERSION}/{tag}/claude", dest, entry["checksum"]
    )
    LOGGER.info("Claude Code CLI ready at %s", dest)
    return str(dest)


async def async_ensure_cli(hass: HomeAssistant) -> str:
    """Provision the CLI (cached) and return its path. Runs off the event loop."""
    store = hass.data.setdefault(DOMAIN, {})
    if path := store.get(_DATA_CLI_PATH):
        return path
    async with _LOCK:
        if path := store.get(_DATA_CLI_PATH):
            return path
        dest = Path(hass.config.path(HOME_SUBDIR, "bin", "claude"))
        path = await hass.async_add_executor_job(_ensure_cli_blocking, dest)
        store[_DATA_CLI_PATH] = path
        return path


def async_prewarm_cli(hass: HomeAssistant) -> None:
    """Kick off the CLI download in the background so it's cached when needed.

    Fire-and-forget: failures (e.g. no internet yet) are only logged; the
    on-demand ``async_ensure_cli`` in the setup path will retry and surface a
    real error if it still can't provision.
    """

    async def _run() -> None:
        try:
            await async_ensure_cli(hass)
        except CliError as err:
            LOGGER.debug("Claude Code CLI pre-warm deferred: %s", err)

    hass.async_create_background_task(_run(), "claude_cli_prewarm")
