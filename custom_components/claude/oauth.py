"""Claude OAuth (PKCE) login.

The user opens the authorize URL, approves, and pastes the returned
``code#state`` back into the config flow, which exchanges it for a token.
"""

import base64
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import secrets
from urllib.parse import urlencode

import aiohttp
from claude_agent_sdk._cli_version import __cli_version__ as CLI_VERSION

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
SCOPES = "user:inference"

TOKEN_LIFETIME_S = 365 * 24 * 3600

TOKEN_HEADERS = {
    "anthropic-beta": "oauth-2025-04-20",
    "User-Agent": f"claude-cli/{CLI_VERSION} (external, cli)",
}

EXCHANGE_TIMEOUT = aiohttp.ClientTimeout(total=30)


class OAuthError(Exception):
    """The OAuth login flow failed."""


@dataclass(slots=True, frozen=True)
class OAuthToken:
    """A token issued by the login flow. ``expires_at`` is None if unknown."""

    token: str
    expires_at: datetime | None


@dataclass(slots=True, frozen=True)
class OAuthLogin:
    """A login in progress: the link to open, and what completes it."""

    url: str
    verifier: str
    state: str


def _pkce_pair() -> tuple[str, str]:
    """Return a (verifier, challenge) PKCE pair (S256)."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_login() -> OAuthLogin:
    """Start a login.

    The verifier stays here until the exchange. ``state`` is separate because it
    travels in the link and comes back with the code, and whoever holds the code and
    the verifier can redeem the code.
    """
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return OAuthLogin(f"{AUTHORIZE_URL}?{urlencode(params)}", verifier, state)


async def async_exchange_code(
    session: aiohttp.ClientSession, code_input: str, login: OAuthLogin
) -> OAuthToken:
    """Exchange the pasted ``code#state`` for a token. Raises OAuthError."""
    code_input = "".join(code_input.split())
    code, _, state = code_input.partition("#")
    if not code:
        raise OAuthError("empty authorization code")
    if state and state != login.state:
        raise OAuthError("the code comes from a different login")
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "state": login.state,
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": login.verifier,
        "expires_in": TOKEN_LIFETIME_S,
    }
    try:
        async with session.post(
            TOKEN_URL, json=body, headers=TOKEN_HEADERS, timeout=EXCHANGE_TIMEOUT
        ) as resp:
            if resp.status != 200:
                text = (await resp.text())[:200]
                raise OAuthError(f"token exchange failed (HTTP {resp.status}): {text}")
            data = await resp.json()
    except aiohttp.ClientError as err:
        raise OAuthError(str(err)) from err
    except TimeoutError as err:
        raise OAuthError("token exchange timed out") from err
    token = data.get("access_token")
    if not token:
        raise OAuthError("no access_token in token response")
    return OAuthToken(str(token), _expires_at(data.get("expires_in")))


def _expires_at(expires_in: object) -> datetime | None:
    """Turn an ``expires_in`` seconds value into an absolute UTC datetime."""
    try:
        seconds = int(expires_in)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return datetime.now(UTC) + timedelta(seconds=seconds)
