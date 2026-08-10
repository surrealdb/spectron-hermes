"""The Spectron memory provider for Hermes.

Implements Hermes' ``MemoryProvider`` interface backed by SurrealDB Spectron:

* ``prefetch``    — recall relevant memory before each turn
* ``sync_turn``   — write the completed turn back to Spectron (non-blocking)
* ``on_memory_write`` — mirror the built-in memory tool's MEMORY.md / USER.md
  writes into Spectron, so the two memories cannot silently diverge
* ``on_session_switch`` — retarget writes when Hermes rotates the session id
* ``on_delegation`` — record what a subagent was asked and what it returned
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
from pathlib import Path
from typing import Any, Dict, List, Optional

# Use Hermes' real base class when running inside Hermes; fall back to a local
# mirror otherwise so the package imports and tests without a Hermes install.
# (The literal "MemoryProvider" also satisfies the drop-in loader heuristic.)
try:  # pragma: no cover - import path depends on runtime environment
    from agent.memory_provider import MemoryProvider  # type: ignore
except Exception:  # pragma: no cover
    from ._compat import MemoryProvider

from . import tools as _tools
from .client import (
    SDK_REQUIREMENT,
    build_client,
    is_auth_error,
    spectron_errors,
    spectron_installed,
    spectron_status,
    verify_credentials,
)
from .config import (
    RECALL_MODES,
    WRITE_FREQUENCIES,
    SpectronConfig,
    config_schema,
    load_config,
    save_config_file,
)

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
        self._warned: set = set()

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "spectron"

    # -- readiness -----------------------------------------------------------

    def is_available(self) -> bool:
        """Config- and dependency-only readiness check. No network calls."""
        if not spectron_installed():
            # spectron_status() only supplies the human-readable reason here;
            # spectron_installed() stays the gate so the check is a single
            # cheap find_spec.
            self._warn_once("sdk", "Spectron memory unavailable: %s", spectron_status()[1])
            return False
        cfg = load_config(self._hermes_home or None)
        if not cfg.is_configured():
            self._warn_once(
                "config",
                "Spectron memory unavailable: %s. Run `hermes memory setup` "
                "and choose spectron.",
                _missing_settings_text(cfg),
            )
            return False
        return True

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        """Log *msg* the first time *key* is seen.

        ``is_available()`` is called on every discovery pass, so an unguarded
        warning would repeat on each one.
        """
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(msg, *args)

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
            # Name the endpoint and context, not just the status code: on its
            # own "[401] InvalidToken" doesn't say which of the three settings
            # is wrong, and memory is about to go quiet for the whole session.
            logger.warning(
                "Spectron auth error during %s; disabling memory for this "
                "session: %s (endpoint=%s, context=%s%s). Run `hermes memory "
                "status` to check the credentials.",
                where,
                exc,
                self._config.endpoint or "(not set)",
                self._config.context or "(not set)",
                f", trace={exc.trace_id}" if getattr(exc, "trace_id", None) else "",
            )
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
            self._write_q.put(
                {
                    "kind": "turn",
                    "items": items,
                    "session_id": session_id or self._session_id,
                }
            )

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
        """Execute one queued write.

        Every backend write goes through here — turns, mirrored built-in memory
        entries, and supersessions alike — so they all inherit the worker's
        non-blocking guarantee and the circuit breaker in ``_write_loop``.
        """
        kind = job.get("kind", "turn")

        if kind == "forget":
            # purge=False: supersede, keeping history. Spectron is tri-temporal
            # and the `spectron_forget` tool defaults the same way; a built-in
            # memory edit should never be destructive here.
            self._client.forget(job["query"], purge=False)
        elif kind == "fact":
            kwargs: Dict[str, Any] = {"session_id": job["session_id"]}
            if job.get("scopes"):
                kwargs["scopes"] = job["scopes"]
            if job.get("labels"):
                kwargs["labels"] = job["labels"]
            self._client.remember(job["text"], **kwargs)
        else:
            kwargs = {"session_id": job["session_id"]}
            if self._default_scope:
                kwargs["scopes"] = self._default_scope
            self._client.remember_many(job["items"], **kwargs)

        self._record_ok()

    # -- built-in memory mirror ----------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror a write made by Hermes' built-in memory tool.

        Without this, "remember X" through the built-in tool lands in MEMORY.md
        and nowhere in Spectron, and the two memories drift apart.

        Hermes only calls this for writes that actually committed (staged and
        failed tool calls are filtered upstream), with *action* already narrowed
        to add/replace/remove. On ``replace`` it also hands us the superseded
        text as ``metadata["old_text"]``.
        """
        if not self._active():
            return
        if action not in ("add", "replace", "remove"):
            return

        metadata = metadata or {}

        # Cron and subagent runs write system-shaped content that would corrupt
        # the user's representation; the MemoryProvider contract says to skip
        # them. An absent value means an older Hermes that predates the field.
        execution_context = metadata.get("execution_context")
        if execution_context and execution_context != "primary":
            return

        text = (content or "").strip()
        old_text = str(metadata.get("old_text") or "").strip()

        # Deliberately NOT gated on write_frequency. That setting decides how
        # conversational turns are batched; this is an explicit, durable fact
        # and has to reach Spectron even in "session" mode.
        if action == "remove":
            if text:
                self._write_q.put({"kind": "forget", "query": text})
            return

        # Supersede the old wording first — the queue has a single consumer, so
        # the forget is guaranteed to land before the replacement fact.
        if action == "replace" and old_text:
            self._write_q.put({"kind": "forget", "query": old_text})

        if not text:
            return
        self._write_q.put(
            {
                "kind": "fact",
                "text": text,
                "session_id": metadata.get("session_id") or self._session_id,
                "scopes": self._default_scope,
                # Labels keep mirrored entries distinguishable from turn writes
                # at recall time. memory_category is deliberately left alone —
                # it is a server-side enum and a wrong value would 400 on every
                # mirrored write.
                "labels": ["hermes", f"memory:{target}"],
            }
        )

    # -- session lifecycle ----------------------------------------------------

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Retarget writes when Hermes rotates the session id mid-process.

        Fires on ``/reset``, ``/new``, ``/branch``, ``/resume`` and context
        compression. Without it, ``_session_id`` keeps the value it was given in
        ``initialize()`` and every later write is filed under a dead session.

        Nothing to flush on *reset*: we hold no per-session buffer, and jobs
        already on the queue carry their own explicit ``session_id``, so
        in-flight writes stay attributed to the session they came from.
        """
        if new_session_id:
            self._session_id = new_session_id

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        """Record a completed subagent delegation on the parent's memory.

        The subagent has no provider session of its own, so this pair is the
        only trace of the work. It is conversational content, so it follows
        ``sync_turn``'s write_frequency gating rather than bypassing it.
        """
        if not self._active() or self._config.write_frequency != "turn":
            return
        items = []
        if task:
            items.append({"role": "user", "content": task})
        if result:
            items.append({"role": "assistant", "content": result})
        if items:
            self._write_q.put(
                {"kind": "turn", "items": items, "session_id": self._session_id}
            )

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
                self._write_q.put(
                    {"kind": "turn", "items": items, "session_id": self._session_id}
                )

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

    # -- interactive setup ---------------------------------------------------
    #
    # ``post_setup`` and ``get_status_config`` are optional hooks Hermes probes
    # with ``hasattr``; they are not part of the MemoryProvider ABC, so older
    # Hermes releases simply ignore them and fall back to the generic
    # ``get_config_schema()`` walk.

    def post_setup(self, hermes_home: str, config: Dict[str, Any]) -> None:
        """Own the whole ``hermes memory setup`` flow for Spectron.

        Hermes' generic wizard treats an already-exported secret as a reason to
        *skip* writing it: the prompt degrades to "blank to keep", and pressing
        enter persists nothing. The exported value then dies with the shell and
        the next session has no API key — a provider that reports success and is
        silently inert. It also never enforces ``required``.

        So we do it ourselves: an existing environment value is a *default*, not
        a skip, and activation is refused outright when the API key is still
        missing afterwards.
        """
        self._hermes_home = hermes_home or self._hermes_home
        current = load_config(self._hermes_home or None)

        try:
            values = self._prompt_for_settings(current)
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled; Spectron was not activated.\n")
            return

        api_key = values.pop("api_key", "") or ""
        missing = [
            label
            for label, value in (
                ("API key", api_key),
                ("endpoint", values.get("endpoint")),
                ("context", values.get("context")),
            )
            if not value
        ]
        if missing:
            # Leave memory.provider untouched rather than activating a provider
            # that cannot talk to Spectron.
            print(
                f"\n  Missing required setting(s): {', '.join(missing)}."
                "\n  Spectron was NOT activated — re-run `hermes memory setup` "
                "once you have them.\n"
            )
            return

        save_config_file(values, self._hermes_home or None)
        env_written = self._write_api_key(api_key)

        # Report what was persisted before the activation outcome, so a failure
        # to write config.yaml doesn't leave the user unsure whether their
        # credentials were saved.
        print(f"\n  Settings saved to {Path(self._resolved_home()) / 'spectron.json'}")
        if env_written:
            print(f"  API key saved to {env_written}")

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = self.name
        activated = self._save_hermes_config(config)

        ok, reason = spectron_status()
        if not ok:
            print(f"\n  ⚠ {reason}")
        else:
            # Warn, don't refuse: the settings are all present, and the network
            # simply being down at setup time is not a reason to leave the user
            # unconfigured.
            creds_ok, creds_reason = self._check_credentials()
            print(
                "\n  Credentials verified against Spectron."
                if creds_ok
                else f"\n  ⚠ {creds_reason}"
            )

        if activated:
            print(f"\n  Memory provider: {self.name}")
            print("\n  Start a new session to activate.\n")
        else:
            print(
                "\n  Could not write config.yaml — activate manually by setting "
                f"memory.provider to '{self.name}'.\n"
            )

    def _check_credentials(self) -> tuple[bool, str]:
        """Build a throwaway client from the saved config and probe it.

        Used by the setup flow and ``hermes memory status`` — both are explicit,
        interactive commands where a network round-trip is expected. It is
        deliberately not called from ``is_available()``, which must stay
        network-free.
        """
        cfg = load_config(self._hermes_home or None)
        if not cfg.is_configured():
            return False, _missing_settings_text(cfg)
        try:
            client = build_client(cfg)
        except Exception as exc:
            return False, f"could not build a Spectron client: {exc}"
        try:
            return verify_credentials(client)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _prompt_for_settings(self, current: SpectronConfig) -> Dict[str, Any]:
        """Prompt for each setting, defaulting to whatever is already resolved."""
        print("\n  Configuring spectron:\n")
        if not current.api_key:
            print("  Get an API key at https://surrealdb.com/pricing/spectron")

        values: Dict[str, Any] = {
            "api_key": _ask_secret("Spectron API key", current.api_key),
            "endpoint": _ask(
                "Spectron endpoint origin (e.g. https://your-instance.spectron.dev)",
                current.endpoint,
            ),
            "context": _ask("Spectron context", current.context),
            "recall_mode": _ask_choice(
                "Recall mode", RECALL_MODES, current.recall_mode
            ),
            "write_frequency": _ask_choice(
                "Write frequency", WRITE_FREQUENCIES, current.write_frequency
            ),
            "top_k": _coerce_int(
                _ask("Memories recalled per turn", str(current.top_k)), current.top_k
            ),
            "default_scope": _ask(
                "Default scope, optional (e.g. user/tobie)", current.default_scope
            ),
        }
        return {k: v for k, v in values.items() if v not in (None, "")}

    def _write_api_key(self, api_key: str) -> Optional[str]:
        """Write the API key to ``$HERMES_HOME/.env``, returning the path written."""
        env_path = Path(self._resolved_home()) / ".env"
        try:
            return str(_upsert_env_var(env_path, "SPECTRON_API_KEY", api_key))
        except OSError as exc:
            print(f"  Could not write {env_path}: {exc}")
            print("  Export SPECTRON_API_KEY in your environment instead.")
            return None

    def _resolved_home(self) -> str:
        from .config import default_hermes_home

        return default_hermes_home(self._hermes_home or None)

    @staticmethod
    def _save_hermes_config(config: Dict[str, Any]) -> bool:
        """Persist Hermes' own config.yaml. Returns False if unavailable.

        ``post_setup`` short-circuits the wizard's own activation write, so the
        provider is responsible for saving ``memory.provider`` itself.
        """
        try:
            from hermes_cli.config import save_config as _save  # type: ignore
        except Exception:
            return False
        try:
            _save(config)
            return True
        except Exception as exc:  # pragma: no cover - depends on Hermes internals
            logger.warning("Failed to save Hermes config: %s", exc)
            return False

    def get_status_config(self, provider_config: Dict[str, Any]) -> Dict[str, Any]:
        """Redacted settings for ``hermes memory status``.

        Reports whether the API key actually resolved — the generic status view
        cannot tell "configured" apart from "key silently missing", which is the
        exact state a skipped secret prompt leaves behind.
        """
        cfg = load_config(self._hermes_home or None)
        merged: Dict[str, Any] = dict(provider_config or {})
        merged.update(
            {
                "endpoint": cfg.endpoint or "(not set)",
                "context": cfg.context or "(not set)",
                "api_key": _redact(cfg.api_key),
                "recall_mode": cfg.recall_mode,
                "write_frequency": cfg.write_frequency,
                "top_k": cfg.top_k,
            }
        )
        if cfg.default_scope:
            merged["default_scope"] = cfg.default_scope
        ok, reason = spectron_status()
        merged["sdk"] = SDK_REQUIREMENT if ok else f"⚠ {reason}"
        if ok:
            # "Configured" and "actually works" are different questions, and the
            # generic status view only ever answered the first.
            creds_ok, creds_reason = self._check_credentials()
            merged["connection"] = "ok" if creds_ok else f"⚠ {creds_reason}"
        return merged


# -- setup helpers -----------------------------------------------------------


def _missing_settings_text(cfg: SpectronConfig) -> str:
    missing = [
        label
        for label, value in (
            ("API key", cfg.api_key),
            ("endpoint", cfg.endpoint),
            ("context", cfg.context),
        )
        if not value
    ]
    if not missing:
        return "configuration incomplete"
    return "missing " + ", ".join(missing)


def _coerce_int(value: Any, fallback: int) -> int:
    """Keep numeric settings numeric in spectron.json.

    Prompts hand back strings; writing ``"5"`` where an int belongs still loads
    (``load_config`` coerces on read) but looks wrong in the saved file.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def _ask(label: str, default: Optional[str] = None) -> str:
    """Prompt for a plain value, offering *default* when the user just hits enter."""
    suffix = f" [{default}]" if default else ""
    _flush()
    raw = input(f"  {label}{suffix}: ").strip()
    return raw or (default or "")


def _flush() -> None:
    """Flush stdout so preceding print() output lands before a prompt.

    getpass writes its prompt to the tty/stderr rather than stdout, so without
    this the hint lines can appear after the prompt they describe.
    """
    import sys

    try:
        sys.stdout.flush()
    except Exception:
        pass


def _ask_choice(label: str, choices: "tuple[str, ...]", default: Optional[str]) -> str:
    """Prompt for one of *choices*, re-asking rather than persisting junk.

    ``load_config`` silently falls back on an unrecognised value when reading, so
    without this an unnoticed typo would sit in spectron.json looking effective.
    """
    options = " | ".join(choices)
    for _ in range(3):
        answer = _ask(f"{label} ({options})", default)
        if answer in choices:
            return answer
        print(f"    '{answer}' is not one of: {options}")
    print(f"    Keeping {default!r}.")
    return default or choices[0]


def _ask_secret(label: str, default: Optional[str] = None) -> str:
    """Prompt for a secret without echoing it.

    Unlike Hermes' generic wizard, an existing value becomes a *default* that is
    re-persisted on enter, rather than a reason to write nothing.
    """
    import getpass

    hint = f" [keep {_redact(default)}]" if default else ""
    _flush()
    try:
        raw = getpass.getpass(f"  {label}{hint}: ").strip()
    except (OSError, ValueError):  # no tty available for masked input
        raw = input(f"  {label}{hint}: ").strip()
    return raw or (default or "")


def _redact(value: Optional[str]) -> str:
    if not value:
        return "(not set)"
    return f"…{value[-4:]}" if len(value) > 4 else "set"


def _upsert_env_var(env_path: Path, key: str, value: str) -> Path:
    """Insert or replace ``key=value`` in a line-oriented ``.env`` file.

    ``.env`` is one ``KEY=VALUE`` per line and values are interpolated straight
    into that line, so every separator ``str.splitlines()`` recognises (plus NUL)
    is stripped — otherwise a pasted secret containing a newline would be re-read
    as an additional, attacker-chosen variable.
    """
    safe = "".join(str(value).replace("\x00", "").splitlines())

    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    replaced = False
    for i, line in enumerate(lines):
        if "=" in line and line.split("=", 1)[0].strip() == key:
            lines[i] = f"{key}={safe}"
            replaced = True
    if not replaced:
        lines.append(f"{key}={safe}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        env_path.chmod(0o600)  # holds credentials
    except OSError:
        pass
    return env_path


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
