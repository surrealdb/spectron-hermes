"""Unit tests for the Spectron memory provider using a mock Spectron client.

No network and no real ``surrealdb[spectron]`` install are required — the tests
inject a fake client by monkeypatching ``build_client``.
"""

from __future__ import annotations

import json

import pytest

from spectron_hermes import provider as provider_mod
from spectron_hermes.config import load_config, save_config_file
from spectron_hermes.provider import SpectronMemoryProvider
from spectron_hermes.tools import to_jsonable


class FakeResp:
    """Stands in for a pydantic-style SDK response object."""

    def __init__(self, **data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class FakeDocuments:
    def __init__(self, parent):
        self._parent = parent

    def upload(self, path, *, title=None, **kwargs):
        self._parent.calls.append(("upload", path, title))
        return FakeResp(document_id="doc:1", title=title or path)


class FakeSpectron:
    """Records calls and returns canned responses; can be told to fail."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.documents = FakeDocuments(self)

    def _maybe_fail(self, name):
        self.calls.append((name,))
        if self.fail:
            raise RuntimeError(f"boom in {name}")

    def recall(self, query, *, k=None, lens=None):
        self.calls.append(("recall", query, k, lens))
        if self.fail:
            raise RuntimeError("boom in recall")
        return FakeResp(memories=[{"text": "Tobie is CTO"}, {"text": "Tobie likes dark mode"}])

    def query_context(self, query, *, k=None, lens=None):
        self.calls.append(("query_context", query, k, lens))
        if self.fail:
            raise RuntimeError("boom in query_context")
        return FakeResp(answer="Tobie is the CTO and prefers dark mode.")

    def remember(self, text, *, scopes=None, **kwargs):
        # Param name mirrors the real SDK (scopes, plural). A wrong keyword from
        # the code under test lands in **kwargs, leaving scopes=None and failing
        # the scope assertions below.
        self.calls.append(("remember", text, scopes))
        return FakeResp(stored=True)

    def remember_many(self, items, *, session_id=None, scopes=None, **kwargs):
        self.calls.append(("remember_many", tuple(m["role"] for m in items), session_id, scopes))
        return FakeResp(count=len(items))

    def forget(self, query, *, purge=False, **kwargs):
        self.calls.append(("forget", query, purge))
        return FakeResp(forgotten=1)

    def reflect(self, query, *, persist=False, **kwargs):
        self.calls.append(("reflect", query, persist))
        return FakeResp(reflection="things changed")

    def consolidate(self, **kwargs):
        self.calls.append(("consolidate",))
        return FakeResp(ok=True)


@pytest.fixture
def hermes_home(tmp_path):
    save_config_file(
        {
            "endpoint": "https://example.spectron.dev",
            "context": "test-ctx",
            "top_k": 3,
        },
        str(tmp_path),
    )
    return str(tmp_path)


@pytest.fixture
def fake(monkeypatch):
    client = FakeSpectron()
    monkeypatch.setattr(provider_mod, "build_client", lambda cfg: client)
    monkeypatch.setattr(provider_mod, "spectron_installed", lambda: True)
    return client


def _make_provider(hermes_home, fake, monkeypatch, **init_kwargs):
    # API key is a secret sourced from the environment, not spectron.json.
    monkeypatch.setenv("SPECTRON_API_KEY", "sk-test")
    p = SpectronMemoryProvider()
    p.initialize("sess-1", hermes_home=hermes_home, **init_kwargs)
    return p


# -- config -----------------------------------------------------------------


def test_config_chain_file_then_env(hermes_home, monkeypatch):
    monkeypatch.setenv("SPECTRON_API_KEY", "sk-env")
    cfg = load_config(hermes_home)
    assert cfg.endpoint == "https://example.spectron.dev"  # from file
    assert cfg.context == "test-ctx"
    assert cfg.api_key == "sk-env"  # from env
    assert cfg.top_k == 3
    assert cfg.is_configured()


def test_is_available_requires_config_and_sdk(hermes_home, monkeypatch):
    monkeypatch.setattr(provider_mod, "spectron_installed", lambda: True)
    monkeypatch.setenv("SPECTRON_API_KEY", "sk")
    p = SpectronMemoryProvider()
    p._hermes_home = hermes_home
    assert p.is_available() is True

    # No SDK installed -> not available.
    monkeypatch.setattr(provider_mod, "spectron_installed", lambda: False)
    assert p.is_available() is False


# -- tools ------------------------------------------------------------------


def test_tool_schemas(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    schemas = p.get_tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {
        "spectron_recall",
        "spectron_remember",
        "spectron_context",
        "spectron_forget",
        "spectron_reflect",
        "spectron_upload",
    }
    for s in schemas:
        assert s["parameters"]["type"] == "object"


def test_dispatch_all_tools_return_json(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch, user_id="tobie")
    cases = [
        ("spectron_recall", {"query": "role?"}),
        ("spectron_remember", {"text": "Tobie is CTO"}),
        ("spectron_context", {"query": "summarise"}),
        ("spectron_forget", {"query": "old notes", "purge": True}),
        ("spectron_reflect", {"query": "this week", "persist": True}),
        ("spectron_upload", {"path": "/tmp/handbook.pdf", "title": "Handbook"}),
    ]
    for name, args in cases:
        out = json.loads(p.handle_tool_call(name, args))
        assert out.get("success") is True, out

    # Argument threading landed on the client.
    assert ("forget", "old notes", True) in fake.calls
    assert ("reflect", "this week", True) in fake.calls
    assert ("upload", "/tmp/handbook.pdf", "Handbook") in fake.calls
    # default scope derived from user_id was applied on remember.
    assert ("remember", "Tobie is CTO", "user/tobie") in fake.calls


def test_unknown_tool(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    out = json.loads(p.handle_tool_call("spectron_bogus", {}))
    assert "error" in out


# -- prefetch ---------------------------------------------------------------


def test_prefetch_hybrid_formats_hits(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    block = p.prefetch("what about tobie?")
    assert "Recalled from memory" in block
    assert "Tobie is CTO" in block
    assert ("recall", "what about tobie?", 3, None) in fake.calls


def test_prefetch_context_mode_uses_query_context(hermes_home, fake, monkeypatch):
    save_config_file({"recall_mode": "context"}, hermes_home)
    p = _make_provider(hermes_home, fake, monkeypatch)
    block = p.prefetch("summarise tobie")
    assert "prefers dark mode" in block
    assert any(c[0] == "query_context" for c in fake.calls)


def test_prefetch_tools_mode_is_silent(hermes_home, fake, monkeypatch):
    save_config_file({"recall_mode": "tools"}, hermes_home)
    p = _make_provider(hermes_home, fake, monkeypatch)
    assert p.prefetch("anything") == ""
    assert not any(c[0] in ("recall", "query_context") for c in fake.calls)


# -- writes -----------------------------------------------------------------


def test_sync_turn_writes_async(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch, user_id="tobie")
    p.sync_turn("hello", "hi there", session_id="sess-1")
    p._write_q.join()  # wait for the worker to drain
    writes = [c for c in fake.calls if c[0] == "remember_many"]
    assert writes, "expected a remember_many write"
    assert writes[0][1] == ("user", "assistant")
    assert writes[0][3] == "user/tobie"  # scope applied
    p.shutdown()


def test_session_write_mode_defers_to_session_end(hermes_home, fake, monkeypatch):
    save_config_file({"write_frequency": "session"}, hermes_home)
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.sync_turn("a", "b")  # should NOT write in session mode
    p._write_q.join()
    assert not any(c[0] == "remember_many" for c in fake.calls)

    p.on_session_end([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    p._write_q.join()
    assert any(c[0] == "remember_many" for c in fake.calls)
    p.shutdown()


def test_on_session_end_consolidates(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.on_session_end([])
    # consolidation runs on a daemon thread; give it a beat via queue join + retry
    import time

    for _ in range(50):
        if any(c[0] == "consolidate" for c in fake.calls):
            break
        time.sleep(0.01)
    assert any(c[0] == "consolidate" for c in fake.calls)
    p.shutdown()


# -- reliability ------------------------------------------------------------


def test_circuit_breaker_disables_after_failures(hermes_home, monkeypatch):
    client = FakeSpectron(fail=True)
    monkeypatch.setattr(provider_mod, "build_client", lambda cfg: client)
    monkeypatch.setattr(provider_mod, "spectron_installed", lambda: True)
    monkeypatch.setenv("SPECTRON_API_KEY", "sk")
    p = SpectronMemoryProvider()
    p.initialize("sess", hermes_home=hermes_home)

    for _ in range(provider_mod._FAILURE_THRESHOLD):
        assert p.prefetch("q") == ""  # fail open, never raises
    assert p._disabled is True
    # Once disabled, tool calls short-circuit with an error, no crash.
    out = json.loads(p.handle_tool_call("spectron_recall", {"query": "x"}))
    assert "error" in out
    p.shutdown()


def test_fail_open_never_raises(hermes_home, monkeypatch):
    client = FakeSpectron(fail=True)
    monkeypatch.setattr(provider_mod, "build_client", lambda cfg: client)
    monkeypatch.setattr(provider_mod, "spectron_installed", lambda: True)
    monkeypatch.setenv("SPECTRON_API_KEY", "sk")
    p = SpectronMemoryProvider()
    p.initialize("sess", hermes_home=hermes_home)
    # None of these should raise.
    assert p.prefetch("q") == ""
    out = json.loads(p.handle_tool_call("spectron_recall", {"query": "x"}))
    assert "error" in out
    p.shutdown()


# -- serialization ----------------------------------------------------------


def test_to_jsonable_variants():
    assert to_jsonable(FakeResp(a=1, b=[FakeResp(c=2)])) == {"a": 1, "b": [{"c": 2}]}
    assert to_jsonable({"x": (1, 2)}) == {"x": [1, 2]}
    assert to_jsonable("s") == "s"


def test_dispatch_kwargs_match_real_sdk():
    """Guard the keyword names we pass against the real Spectron SDK.

    Skipped when `surrealdb` isn't installed (e.g. CI runs --no-deps). Catches
    drift like remember(scope=...) vs the SDK's remember(scopes=...).
    """
    import inspect

    spectron = pytest.importorskip("surrealdb.spectron")

    def params(method_owner, method_name):
        return set(inspect.signature(getattr(method_owner, method_name)).parameters)

    Spectron = spectron.Spectron
    remember = params(Spectron, "remember")
    assert "scopes" in remember and "scope" not in remember
    assert "scopes" in params(Spectron, "remember_many")
    assert "lens" in params(Spectron, "recall")
    assert "lens" in params(Spectron, "query_context")
    assert "purge" in params(Spectron, "forget")
    assert "persist" in params(Spectron, "reflect")

    from surrealdb.spectron._namespaces.documents import BlockingDocuments

    upload = params(BlockingDocuments, "upload")
    assert "title" in upload and "scopes" in upload
