"""
Operator Authentication
Verifies that a request comes from a logged-in admin/operator of the
Sentra admin dashboard.

The token is the same Supabase JWT the dashboard uses for the Flask backend.
We validate it by calling the backend's GET /api/auth/me, so the AI service
needs no Supabase credentials of its own. Results are cached briefly to
avoid a backend round-trip on every request.
"""

import time
from typing import Optional

import httpx
from fastapi import Header, HTTPException, WebSocket

from config import settings

OPERATOR_ROLES = ("admin", "operator")
_CACHE_TTL_SECONDS = 60

# token -> (is_operator, expires_at)
_cache: dict[str, tuple[bool, float]] = {}


async def verify_operator_token(token: Optional[str]) -> bool:
    """Return True if the token belongs to an active admin/operator."""
    if not token:
        return False

    now = time.time()
    cached = _cache.get(token)
    if cached and cached[1] > now:
        return cached[0]

    try:
        async with httpx.AsyncClient(
            base_url=settings.PARKING_API_URL, timeout=5.0
        ) as client:
            response = await client.get(
                "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
            )
        ok = (
            response.status_code == 200
            and response.json().get("user", {}).get("role") in OPERATOR_ROLES
        )
    except (httpx.RequestError, ValueError):
        # Backend unreachable: fail closed, and don't cache the failure
        return False

    # Drop expired entries so the cache can't grow without bound
    for key in [k for k, (_, exp) in _cache.items() if exp <= now]:
        del _cache[key]
    _cache[token] = (ok, now + _CACHE_TTL_SECONDS)
    return ok


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    return parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else None


async def require_operator(authorization: Optional[str] = Header(default=None)) -> str:
    """
    FastAPI dependency: require an admin/operator Bearer token.
    Returns the token so handlers can forward it to the backend.
    """
    token = _bearer(authorization)
    if not await verify_operator_token(token):
        raise HTTPException(status_code=401, detail="Operator login required")
    return token


async def websocket_operator_token(websocket: WebSocket) -> Optional[str]:
    """
    Read and verify the operator token for a WebSocket connection.
    Browsers cannot set headers on WebSockets, so the token is passed
    as the ?token= query parameter. Returns the token, or None if invalid.
    """
    token = websocket.query_params.get("token")
    return token if await verify_operator_token(token) else None


def clear_cache():
    """Clear cached token checks (used by tests)."""
    _cache.clear()
