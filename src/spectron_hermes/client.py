"""Spectron client construction and error-class resolution.

Imports are deliberately lazy: the SurrealDB SDK is only imported when we
actually build a client, so importing this package never fails just because
``surrealdb`` (>=3.0.0a4, which bundles Spectron) isn't installed.
``is_available()`` relies on this to do a cheap, dependency-only readiness check.
"""

from __future__ import annotations

from typing import Any, Tuple

from .config import SpectronConfig


# The SDK requirement, kept in one place so diagnostics and docs agree.
SDK_REQUIREMENT = "surrealdb>=3.0.0a4"
_INSTALL_HINT = f'uv pip install "{SDK_REQUIREMENT}"'


def spectron_installed() -> bool:
    """True if the Spectron SDK can be imported. No network, no client build."""
    try:
        import importlib.util

        return importlib.util.find_spec("surrealdb.spectron") is not None
    except Exception:
        return False


def spectron_status() -> Tuple[bool, str]:
    """Return ``(ok, reason)`` describing SDK readiness.

    Distinguishes "no ``surrealdb`` at all" from "a ``surrealdb`` that predates
    Spectron", because the two need the same command but look nothing alike to a
    user. The second case is easy to land in by accident: Hermes decides whether
    a declared dependency is satisfied by importing its top-level name, so an
    existing ``surrealdb`` 2.x satisfies ``surrealdb>=3.0.0a4`` and is never
    upgraded — leaving the provider configured but permanently inert.

    ``reason`` is empty when *ok* is True.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("surrealdb") is None:
            return False, f"the surrealdb SDK is not installed — run: {_INSTALL_HINT}"
    except Exception as exc:
        return False, f"could not probe the surrealdb SDK ({exc}) — run: {_INSTALL_HINT}"

    try:
        if importlib.util.find_spec("surrealdb.spectron") is not None:
            return True, ""
    except Exception as exc:
        return False, f"could not probe surrealdb.spectron ({exc}) — run: {_INSTALL_HINT}"

    return False, (
        f"surrealdb {_installed_sdk_version()} is installed but does not bundle "
        f"Spectron; Spectron ships only in surrealdb v3 — run: {_INSTALL_HINT}"
    )


def _installed_sdk_version() -> str:
    """Best-effort version of the installed ``surrealdb``, for diagnostics only."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("surrealdb")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        import surrealdb  # type: ignore

        return str(getattr(surrealdb, "__version__", "") or "(unknown version)")
    except Exception:
        return "(unknown version)"


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


def _status_of(exc: BaseException) -> int:
    """HTTP status carried by a Spectron API error, or 0 for anything else."""
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else 0


def _with_trace(reason: str, exc: BaseException) -> str:
    trace_id = getattr(exc, "trace_id", None)
    return f"{reason} (trace {trace_id})" if trace_id else reason


def verify_credentials(client: Any) -> Tuple[bool, str]:
    """Probe the endpoint and credentials, returning ``(ok, reason)``.

    A rejected key surfaces as a single ``[401] InvalidToken`` on the first
    recall, which says nothing about *which* of the three settings is wrong —
    and by then memory has already disabled itself for the session. Splitting
    the check across the SDK's two probes tells them apart:

    * ``health()`` is unauthenticated, so a failure there is the endpoint.
    * ``whoami()`` is context-scoped (``GET /api/v1/{context}/me``), so it
      exercises the key *and* its binding to this context.

    ``reason`` is empty when *ok* is True. Never raises.
    """
    endpoint = getattr(client, "endpoint", "the configured endpoint")
    context = getattr(client, "context_id", "") or "(unset)"

    try:
        client.health()
    except Exception as exc:
        status = _status_of(exc)
        if status == 0:
            return False, _with_trace(
                f"cannot reach {endpoint} ({exc}) — SPECTRON_ENDPOINT should be "
                "the API origin, with no path and no trailing slash",
                exc,
            )
        # A reachable server that dislikes an unauthenticated probe is not
        # evidence of bad credentials; let whoami() have the final say.

    try:
        client.whoami()
    except Exception as exc:
        status = _status_of(exc)
        if status == 401:
            return False, _with_trace(
                f"API key rejected by {endpoint} for context '{context}' — check "
                "SPECTRON_API_KEY, and that the key was issued for this context",
                exc,
            )
        if status == 403:
            return False, _with_trace(
                f"API key does not authorize context '{context}' at {endpoint}",
                exc,
            )
        if status == 404:
            return False, _with_trace(
                f"context '{context}' does not exist at {endpoint} — check "
                "SPECTRON_CONTEXT",
                exc,
            )
        return False, _with_trace(f"could not verify credentials: {exc}", exc)

    return True, ""


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
