"""Simulate a full Hermes session against the Spectron provider — no credentials.

This drives ``SpectronMemoryProvider`` through the exact lifecycle Hermes uses
(is_available → initialize → system_prompt_block → prefetch → handle_tool_call →
sync_turn → on_session_end → shutdown), but with a fake in-memory Spectron
client so it runs anywhere with nothing installed but this package.

Run:

    python examples/simulate_session.py

For the real thing against a live Spectron instance, see ``live_session.py``.
"""

from __future__ import annotations

import os

from spectron_hermes import provider as provider_mod
from spectron_hermes.provider import SpectronMemoryProvider


# --- a tiny fake Spectron client (mirrors the methods the provider calls) ----


class _Resp:
    def __init__(self, **data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class _Docs:
    def upload(self, path, *, title=None, **kwargs):
        return _Resp(document_id="doc:1", title=title or path)


class FakeSpectron:
    """In-memory stand-in — stores 'facts' and echoes them back on recall."""

    def __init__(self):
        self._facts: list[str] = []
        self.documents = _Docs()

    def remember(self, text, *, scope=None, **kwargs):
        self._facts.append(text)
        return _Resp(stored=True, scope=scope)

    def remember_many(self, items, *, session_id=None, scope=None, **kwargs):
        for m in items:
            self._facts.append(f"[{m['role']}] {m['content']}")
        return _Resp(count=len(items))

    def recall(self, query, *, k=None, lens=None):
        hits = [{"text": f} for f in self._facts][: (k or 5)]
        return _Resp(memories=hits)

    def query_context(self, query, *, k=None, lens=None):
        return _Resp(answer="; ".join(self._facts) or "nothing recalled yet")

    def forget(self, query, *, purge=False, **kwargs):
        self._facts.clear()
        return _Resp(forgotten=True, purge=purge)

    def reflect(self, query, *, persist=False, **kwargs):
        return _Resp(reflection=f"{len(self._facts)} facts on record", persisted=persist)

    def consolidate(self, **kwargs):
        return _Resp(ok=True)


def main() -> None:
    # 1. Configure via env (Hermes would resolve these from spectron.json/.env).
    os.environ.setdefault("SPECTRON_ENDPOINT", "https://demo.spectron.local")
    os.environ.setdefault("SPECTRON_CONTEXT", "demo")
    os.environ.setdefault("SPECTRON_API_KEY", "sk-demo")

    # Inject the fake client instead of building a real one.
    fake = FakeSpectron()
    provider_mod.build_client = lambda config: fake
    provider_mod.spectron_installed = lambda: True

    provider = SpectronMemoryProvider()

    print("is_available:", provider.is_available())

    # 2. Hermes initializes the provider once at session start.
    provider.initialize("example-session", user_id="tobie")
    print("\nsystem prompt block:\n ", provider.system_prompt_block())

    # 3. The agent stores a fact via a tool call.
    print("\nremember →", provider.handle_tool_call(
        "spectron_remember", {"text": "Tobie was promoted to CTO"}
    ))

    # 4. Before the next turn, Hermes prefetches relevant memory.
    print("\nprefetch:\n", provider.prefetch("what is tobie's role?"))

    # 5. Each completed turn is written back (async worker thread).
    provider.sync_turn("I moved to Lisbon", "Noted — Lisbon it is.")
    provider._write_q.join()  # example-only: wait for the async write to land

    print("\nrecall after turn →", provider.handle_tool_call(
        "spectron_recall", {"query": "tobie"}
    ))
    print("\ncontext →", provider.handle_tool_call(
        "spectron_context", {"query": "summarise what you know"}
    ))
    print("\nreflect →", provider.handle_tool_call(
        "spectron_reflect", {"query": "this session", "persist": True}
    ))

    # 6. Session end triggers background consolidation.
    provider.on_session_end([])
    provider.shutdown()
    print("\nsession ended cleanly.")


if __name__ == "__main__":
    main()
