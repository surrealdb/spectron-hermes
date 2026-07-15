"""The Spectron memory provider for Hermes.

Implements Hermes' ``MemoryProvider`` interface backed by SurrealDB Spectron:

* ``prefetch``    — recall relevant memory before each turn
* ``sync_turn``   — write the completed turn back to Spectron (non-blocking)
* ``on_session_end`` — trigger background consolidation
* six explicit tools (``spectron_recall`` / ``remember`` / ``context`` /
  ``forget`` / ``reflect`` / ``upload``)

Design rules borrowed from the Honcho/Cognee reference providers:

* **Never block the agent.** Writes go through a daemon worker thread; recall is
  bounded by the client timeout.
* **Fail open.** Every SDK call is wrapped; failures are logged and degrade to
  empty results, never raised into the agent loop.
* **Circuit breaker.** After repeated failures the provider disables itself for
  the rest of the session and stops hitting the backend.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any, Dict, List, Optional

# Use Hermes' real base class when running inside Hermes; fall back to a local
# mirror otherwise so the package imports and tests without a Hermes install.
# (The literal "MemoryProvider" also satisfies the drop-in loader heuristic.)
try:  # pragma: no cover - import path depends on runtime environment
    from agent.memory_provider import MemoryProvider  # type: ignore
except Exception:  # pragma: no cover
    from ._compat import MemoryProvider

from . import tools as _tools
from .client import build_client, is_auth_error, spectron_errors, spectron_installed
from .config import SpectronConfig, config_schema, load_config, save_config_file

logger = logging.getLogger("spectron_hermes")

# Disable the provider for the session after this many consecutive failures.
_FAILURE_THRESHOLD = 3
# Rough character budget for recalled context injected into a turn.
_PREFETCH_BUDGET_CHARS = 2000
# Sentinel pushed onto the write queue to stop the worker.
_STOP = object()


class SpectronMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by SurrealDB Spectron."""

    def __init__(self) -> None:
        self._config: SpectronConfig = SpectronConfig()
        self._client: Any = None
        self._hermes_home: str = ""
        self._session_id: str = ""
        self._default_scope: Optional[str] = None

        self._write_q: "queue.Queue[Any]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._consecutive_failures = 0
        self._disabled = False
        self._errors = spectron_errors()

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "spectron"

    # -- readiness -----------------------------------------------------------

    def is_available(self) -> bool:
        """Config- and dependency-only readiness check. No network calls."""
        if not spectron_installed():
            return False
        cfg = load_config(self._hermes_home or None)
        return cfg.is_configured()

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", "") or self._hermes_home
        self._config = load_config(self._hermes_home or None)
        self._errors = spectron_errors()

        user_id = kwargs.get("user_id") or kwargs.get("user_id_alt")
        self._default_scope = self._config.default_scope or (
            f"user/{user_id}" if user_id else None
        )

        # Building the client is local (no I/O); guard anyway to fail open.
        try:
            self._client = build_client(self._config)
        except Exception as exc:  # pragma: no cover - depends on SDK/env
            logger.warning("Spectron client init failed; memory disabled: %s", exc)
            self._client = None
            self._disabled = True
            return

        self._start_worker()

    def shutdown(self) -> None:
        """Flush pending writes and stop the worker."""
        worker = self._worker
        if worker and worker.is_alive():
            self._write_q.put(_STOP)
            worker.join(timeout=5.0)
        client = self._client
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # -- circuit breaker helpers --------------------------------------------

    def _record_ok(self) -> None:
        self._consecutive_failures = 0

    def _record_fail(self, where: str, exc: BaseException) -> None:
        if is_auth_error(exc):
            logger.warning("Spectron auth error during %s; disabling memory: %s", where, exc)
            self._disabled = True
            return
        self._consecutive_failures += 1
        logger.warning(
            "Spectron %s failed (%d/%d): %s",
            where,
            self._consecutive_failures,
            _FAILURE_THRESHOLD,
            exc,
        )
        if self._consecutive_failures >= _FAILURE_THRESHOLD:
            logger.warning("Spectron failure threshold reached; disabling memory for session.")
            self._disabled = True

    def _active(self) -> bool:
        return bool(self._client) and not self._disabled

    # -- recall (prefetch) ---------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._active():
            return ""
        return (
            "You have persistent long-term memory backed by SurrealDB Spectron. "
            "Relevant memories are recalled automatically before each turn. "
            "Use `spectron_recall` to search memory, `spectron_context` for a "
            "synthesised answer, `spectron_remember` to store durable facts, "
            "`spectron_forget` to remove them, `spectron_reflect` for insights, "
            "and `spectron_upload` to ingest documents."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active() or not query.strip():
            return ""
        if self._config.recall_mode == "tools":
            # Model recalls explicitly via tools; no automatic injection.
            return ""
        try:
            lens = [self._default_scope] if self._default_scope else None
            if self._config.recall_mode == "context":
                resp = self._call_context(query, lens)
            else:  # hybrid
                resp = self._call_recall(query, lens)
            self._record_ok()
            return self._format_recall(resp)
        except self._errors as exc:  # type: ignore[misc]
            self._record_fail("prefetch", exc)
            return ""
        except Exception as exc:  # pragma: no cover - unexpected
            self._record_fail("prefetch", exc)
            return ""

    def _call_recall(self, query: str, lens: Optional[List[str]]) -> Any:
        if lens:
            return self._client.recall(query, k=self._config.top_k, lens=lens)
        return self._client.recall(query, k=self._config.top_k)

    def _call_context(self, query: str, lens: Optional[List[str]]) -> Any:
        if lens:
            return self._client.query_context(query, k=self._config.top_k, lens=lens)
        return self._client.query_context(query, k=self._config.top_k)

    @staticmethod
    def _format_recall(resp: Any) -> str:
        """Turn a recall/context response into a bounded plain-text block."""
        data = _tools.to_jsonable(resp)

        # A synthesised context answer.
        if isinstance(data, dict):
            for key in ("answer", "context", "summary", "text"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return _truncate(
                        "## Recalled from memory (Spectron)\n" + val.strip()
                    )

        items = _extract_items(data)
        if not items:
            return ""

        lines = ["## Recalled from memory (Spectron)"]
        for item in items:
            text = _item_text(item)
            if text:
                lines.append(f"- {text}")
        if len(lines) == 1:
            return ""
        return _truncate("\n".join(lines))

    # -- writes (sync_turn) --------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self._active() or self._config.write_frequency != "turn":
            return
        items = []
        if user_content:
            items.append({"role": "user", "content": user_content})
        if assistant_content:
            items.append({"role": "assistant", "content": assistant_content})
        if items:
            self._write_q.put({"items": items, "session_id": session_id or self._session_id})

    def _start_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._write_loop, name="spectron-writer", daemon=True
            )
            self._worker.start()

    def _write_loop(self) -> None:
        while True:
            job = self._write_q.get()
            try:
                if job is _STOP:
                    return
                if not self._active():
                    continue
                self._flush_write(job)
            except self._errors as exc:  # type: ignore[misc]
                self._record_fail("write", exc)
            except Exception as exc:  # pragma: no cover - unexpected
                self._record_fail("write", exc)
            finally:
                self._write_q.task_done()

    def _flush_write(self, job: Dict[str, Any]) -> None:
        kwargs: Dict[str, Any] = {"session_id": job["session_id"]}
        if self._default_scope:
            kwargs["scopes"] = self._default_scope
        self._client.remember_many(job["items"], **kwargs)
        self._record_ok()

    # -- session end ---------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._active():
            return
        # In "session" write mode, persist the whole conversation now.
        if self._config.write_frequency == "session":
            items = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in (messages or [])
                if m.get("content")
            ]
            if items:
                self._write_q.put({"items": items, "session_id": self._session_id})

        if self._config.consolidate_on_end:
            threading.Thread(
                target=self._consolidate, name="spectron-consolidate", daemon=True
            ).start()

    def _consolidate(self) -> None:
        try:
            self._client.consolidate()
            self._record_ok()
        except self._errors as exc:  # type: ignore[misc]
            self._record_fail("consolidate", exc)
        except Exception as exc:  # pragma: no cover
            self._record_fail("consolidate", exc)

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(_tools.ALL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name not in _tools.TOOL_NAMES:
            return json.dumps(
                {"error": f"Unknown tool: {tool_name}", "provider": self.name}
            )
        if not self._active():
            return json.dumps(
                {"error": "Spectron memory is unavailable.", "provider": self.name}
            )
        try:
            result = _tools.dispatch(
                self._client, self._config, self._default_scope, tool_name, args
            )
            self._record_ok()
            return json.dumps({"success": True, "result": result})
        except self._errors as exc:  # type: ignore[misc]
            self._record_fail(tool_name, exc)
            return json.dumps({"error": str(exc), "provider": self.name})
        except Exception as exc:  # pragma: no cover - unexpected
            self._record_fail(tool_name, exc)
            return json.dumps({"error": str(exc), "provider": self.name})

    # -- config --------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return config_schema()

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        self._hermes_home = hermes_home or self._hermes_home
        save_config_file(values, self._hermes_home or None)


# -- module-level formatting helpers -----------------------------------------


def _truncate(text: str, limit: int = _PREFETCH_BUDGET_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _extract_items(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("memories", "results", "hits", "items", "matches", "recall"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("text", "content", "summary", "fact", "statement", "value"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return json.dumps(item, default=str)
    return str(item)
