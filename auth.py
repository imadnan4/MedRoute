"""Authentication dependencies for Neon Auth JWTs."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config import settings


class AuthenticatedUser(BaseModel):
    """The trusted identity attached to a request by Neon Auth."""

    user_id: str
    claims: dict[str, Any]


_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks_client() -> jwt.PyJWKClient:
    if not settings.auth_jwks_url:
        raise RuntimeError("MEDROUTE_AUTH_JWKS_URL is not configured")
    return jwt.PyJWKClient(settings.auth_jwks_url, cache_jwk_set=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthenticatedUser:
    """Verify a Neon Auth bearer token and return its subject."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not settings.auth_jwks_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the API",
        )

    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(credentials.credentials)
        decode_options: dict[str, Any] = {"algorithms": [
            "RS256", "RS384", "RS512", "PS256", "PS384", "PS512",
            "ES256", "ES384", "ES512", "EdDSA",
        ]}
        if settings.auth_audience:
            decode_options["audience"] = settings.auth_audience
        if settings.auth_issuer:
            decode_options["issuer"] = settings.auth_issuer
        claims = jwt.decode(
            credentials.credentials,
            signing_key.key,
            **decode_options,
        )
    except (jwt.PyJWTError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has no user subject",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthenticatedUser(user_id=user_id, claims=claims)
