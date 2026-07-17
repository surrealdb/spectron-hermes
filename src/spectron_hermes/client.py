"""Spectron client construction and error-class resolution.

Imports are deliberately lazy: the SurrealDB SDK is only imported when we
actually build a client, so importing this package never fails just because
``surrealdb`` (>=3.0.0a4, which bundles Spectron) isn't installed.
``is_available()`` relies on this to do a cheap, dependency-only readiness check.
"""

from __future__ import annotations

from typing import Any, Tuple

from .config import SpectronConfig


def spectron_installed() -> bool:
    """True if the Spectron SDK can be imported. No network, no client build."""
    try:
        import importlib.util

        return importlib.util.find_spec("surrealdb.spectron") is not None
    except Exception:
        return False


def spectron_errors() -> Tuple[type, ...]:
    """Return the Spectron exception classes to treat as fail-open, broadest first.

    Falls back to ``(Exception,)`` when the SDK isn't importable so callers can
    always use the result in an ``except`` clause.
    """
    try:
        from surrealdb.spectron import SpectronError  # type: ignore

        return (SpectronError,)
    except Exception:
        return (Exception,)


def is_auth_error(exc: BaseException) -> bool:
    """True when the exception is a Spectron auth/authorization failure (401/403)."""
    try:
        from surrealdb.spectron import (  # type: ignore
            SpectronAuthError,
            SpectronScopeError,
        )

        return isinstance(exc, (SpectronAuthError, SpectronScopeError))
    except Exception:
        return False


def build_client(config: SpectronConfig) -> Any:
    """Construct a blocking Spectron client from resolved config.

    Constructing the client does not perform network I/O — the SDK validates and
    stores connection settings; requests happen on the first method call.
    """
    from surrealdb.spectron import Spectron  # type: ignore

    return Spectron(
        config.context,
        endpoint=config.endpoint,
        api_key=config.api_key,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )
