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

    endpoint = "https://example.spectron.dev"
    context_id = "test-ctx"

    def __init__(self, fail=False):
        self.calls = []
        # Full kwargs for each remember(), which `calls` deliberately flattens.
        self.remembers = []
        self.fail = fail
        self.documents = FakeDocuments(self)
        # Exceptions for the credential probes to raise, or None to succeed.
        self.health_error = None
        self.whoami_error = None

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
        self.remembers.append({"text": text, "scopes": scopes, **kwargs})
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

    def health(self):
        self.calls.append(("health",))
        if self.health_error:
            raise self.health_error
        return {"status": "ok"}

    def whoami(self, **kwargs):
        self.calls.append(("whoami",))
        if self.whoami_error:
            raise self.whoami_error
        return FakeResp(principal="agent:test")

    def close(self):
        self.calls.append(("close",))


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


# -- built-in memory mirror --------------------------------------------------


def _memory_calls(fake):
    """Mirror-relevant calls, in order, so ordering assertions are readable."""
    return [c for c in fake.calls if c[0] in ("remember", "forget")]


def test_on_memory_write_add_stores_a_fact(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch, user_id="tobie")
    p.on_memory_write(
        "add", "user", "Tobie prefers dark mode", metadata={"session_id": "sess-9"}
    )
    p._write_q.join()

    assert _memory_calls(fake) == [("remember", "Tobie prefers dark mode", "user/tobie")]
    stored = fake.remembers[0]
    assert stored["session_id"] == "sess-9"
    assert stored["labels"] == ["hermes", "memory:user"]
    p.shutdown()


def test_on_memory_write_replace_supersedes_then_stores(hermes_home, fake, monkeypatch):
    """Order matters: the old wording has to be superseded before the new one lands."""
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.on_memory_write(
        "replace", "memory", "Ships on Fridays", metadata={"old_text": "Ships on Mondays"}
    )
    p._write_q.join()

    assert _memory_calls(fake) == [
        ("forget", "Ships on Mondays", False),
        ("remember", "Ships on Fridays", None),
    ]
    p.shutdown()


def test_on_memory_write_replace_without_old_text_just_stores(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.on_memory_write("replace", "memory", "Ships on Fridays")
    p._write_q.join()

    assert _memory_calls(fake) == [("remember", "Ships on Fridays", None)]
    p.shutdown()


def test_on_memory_write_remove_supersedes_rather_than_purging(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.on_memory_write("remove", "memory", "Ships on Mondays")
    p._write_q.join()

    assert _memory_calls(fake) == [("forget", "Ships on Mondays", False)]
    p.shutdown()


def test_on_memory_write_ignores_write_frequency(hermes_home, fake, monkeypatch):
    """An explicit durable fact is not conversational turn traffic."""
    save_config_file({"write_frequency": "session"}, hermes_home)
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.on_memory_write("add", "memory", "Deploys are manual")
    p._write_q.join()

    assert _memory_calls(fake) == [("remember", "Deploys are manual", None)]
    p.shutdown()


@pytest.mark.parametrize(
    "action,target,content,metadata",
    [
        # Cron/subagent writes would corrupt the user's representation.
        ("add", "user", "from a subagent", {"execution_context": "subagent"}),
        ("add", "user", "from cron", {"execution_context": "cron"}),
        ("add", "memory", "   ", None),  # blank content
        ("noop", "memory", "something", None),  # action Hermes would never send
    ],
)
def test_on_memory_write_skips(hermes_home, fake, monkeypatch, action, target, content, metadata):
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.on_memory_write(action, target, content, metadata=metadata)
    p._write_q.join()

    assert _memory_calls(fake) == []
    p.shutdown()


def test_on_memory_write_is_silent_when_disabled(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    p._disabled = True
    p.on_memory_write("add", "memory", "should not be written")
    p._write_q.join()

    assert _memory_calls(fake) == []
    p.shutdown()


def test_on_memory_write_signature_selects_the_metadata_convention():
    """Hermes inspects our signature to decide how to pass provenance.

    `agent/memory_manager.py::_provider_memory_write_metadata_mode` falls back to
    a 3-arg legacy call unless it finds a `metadata` parameter — silently
    dropping `old_text`, and with it the supersede half of a replace.
    """
    import inspect

    params = inspect.signature(SpectronMemoryProvider.on_memory_write).parameters
    assert list(params) == ["self", "action", "target", "content", "metadata"]


# -- session lifecycle -------------------------------------------------------


def test_on_session_switch_retargets_writes(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.on_session_switch("sess-2", parent_session_id="sess-1", reset=True)
    p.sync_turn("hello", "hi")
    p._write_q.join()

    writes = [c for c in fake.calls if c[0] == "remember_many"]
    assert writes[0][2] == "sess-2"
    p.shutdown()


def test_on_delegation_records_task_and_result(hermes_home, fake, monkeypatch):
    p = _make_provider(hermes_home, fake, monkeypatch)
    p.on_delegation("summarise the docs", "done: 3 pages", child_session_id="sess-child")
    p._write_q.join()

    writes = [c for c in fake.calls if c[0] == "remember_many"]
    assert writes and writes[0][1] == ("user", "assistant")
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


# -- SDK diagnostics --------------------------------------------------------


def _fake_find_spec(monkeypatch, *, surrealdb: bool, spectron: bool):
    """Make surrealdb / surrealdb.spectron appear present or absent."""
    import importlib.util

    real = importlib.util.find_spec

    def fake(name, package=None):
        if name == "surrealdb":
            return object() if surrealdb else None
        if name.startswith("surrealdb."):
            return object() if spectron else None
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_spectron_status_ok(monkeypatch):
    from spectron_hermes.client import spectron_status

    _fake_find_spec(monkeypatch, surrealdb=True, spectron=True)
    assert spectron_status() == (True, "")


def test_spectron_status_sdk_missing(monkeypatch):
    from spectron_hermes.client import spectron_status

    _fake_find_spec(monkeypatch, surrealdb=False, spectron=False)
    ok, reason = spectron_status()
    assert ok is False
    assert "not installed" in reason
    assert "surrealdb>=3.0.0a4" in reason


def test_spectron_status_sdk_too_old(monkeypatch):
    """surrealdb 2.x satisfies Hermes' `__import__('surrealdb')` check but has no Spectron.

    Hermes then never upgrades the dependency, so this is the state a user with a
    pre-existing surrealdb silently lands in — the message has to name it.
    """
    from spectron_hermes.client import spectron_status

    _fake_find_spec(monkeypatch, surrealdb=True, spectron=False)
    ok, reason = spectron_status()
    assert ok is False
    assert "does not bundle" in reason
    assert "surrealdb>=3.0.0a4" in reason


# -- credential diagnostics -------------------------------------------------
#
# A rejected key reaches the user as one `[401] InvalidToken` warning and then
# silence, which doesn't say which of endpoint / context / key is wrong. These
# cover the split that tells them apart.


class _ApiError(Exception):
    """Stands in for SpectronAPIError without needing the SDK installed.

    `verify_credentials` classifies on the `status_code` attribute, exactly as
    the real error class exposes it.
    """

    def __init__(self, status_code, message="nope", trace_id=None):
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.trace_id = trace_id


def test_verify_credentials_ok(fake):
    from spectron_hermes.client import verify_credentials

    assert verify_credentials(fake) == (True, "")


def test_verify_credentials_blames_the_endpoint_when_unreachable(fake):
    from spectron_hermes.client import verify_credentials

    fake.health_error = ConnectionError("connection refused")
    ok, reason = verify_credentials(fake)
    assert ok is False
    assert "cannot reach https://example.spectron.dev" in reason
    assert "SPECTRON_ENDPOINT" in reason
    # whoami is pointless once the origin is wrong.
    assert not any(c[0] == "whoami" for c in fake.calls)


def test_verify_credentials_reports_a_rejected_key_with_the_trace(fake):
    from spectron_hermes.client import verify_credentials

    fake.whoami_error = _ApiError(401, "InvalidToken", trace_id="tr-42")
    ok, reason = verify_credentials(fake)
    assert ok is False
    assert "API key rejected" in reason
    assert "test-ctx" in reason
    assert "trace tr-42" in reason


def test_verify_credentials_distinguishes_scope_from_missing_context(fake):
    from spectron_hermes.client import verify_credentials

    fake.whoami_error = _ApiError(403)
    assert "does not authorize" in verify_credentials(fake)[1]

    fake.whoami_error = _ApiError(404)
    reason = verify_credentials(fake)[1]
    assert "does not exist" in reason and "SPECTRON_CONTEXT" in reason


def test_verify_credentials_survives_an_authenticated_health_endpoint(fake):
    """A 401 on the unauthenticated probe isn't evidence about the credentials."""
    from spectron_hermes.client import verify_credentials

    fake.health_error = _ApiError(401, "InvalidToken")
    assert verify_credentials(fake) == (True, "")


def test_status_reports_the_connection(hermes_home, fake, monkeypatch):
    monkeypatch.setattr(provider_mod, "spectron_status", lambda: (True, ""))
    monkeypatch.setenv("SPECTRON_API_KEY", "sk-test")
    fake.whoami_error = _ApiError(401, "InvalidToken")

    p = SpectronMemoryProvider()
    p._hermes_home = hermes_home

    assert "API key rejected" in p.get_status_config({})["connection"]


# -- interactive setup ------------------------------------------------------


@pytest.fixture
def answers(monkeypatch):
    """Script the setup prompts. Returns the dict of canned replies to mutate."""
    replies = {}

    def _ask(label, default=None):
        # Prefix match, not substring: "Recall mode (hybrid | context | tools)"
        # contains "context" and would otherwise steal that answer.
        lowered = label.lower()
        for key, value in replies.items():
            if lowered.startswith(key):
                return value
        return default or ""

    monkeypatch.setattr(provider_mod, "_ask", _ask)
    monkeypatch.setattr(provider_mod, "_ask_secret", _ask)
    # Hermes' config module isn't importable in tests; pretend the write worked.
    monkeypatch.setattr(
        provider_mod.SpectronMemoryProvider, "_save_hermes_config", staticmethod(lambda cfg: True)
    )
    # post_setup probes the server; stub it so these stay offline.
    # test_post_setup_warns_about_bad_credentials covers the failing path.
    monkeypatch.setattr(
        provider_mod.SpectronMemoryProvider, "_check_credentials", lambda self: (True, "")
    )
    return replies


def _setup(tmp_path):
    """Run post_setup against a throwaway HERMES_HOME; returns (provider, config)."""
    p = SpectronMemoryProvider()
    config = {}
    p.post_setup(str(tmp_path), config)
    return p, config


def test_post_setup_persists_key_to_env_and_activates(tmp_path, answers, monkeypatch):
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    answers.update(
        {
            "spectron api key": "sk-live",
            "spectron endpoint": "https://x.spectron.dev",
            "spectron context": "ctx",
        }
    )
    _, config = _setup(tmp_path)

    assert config["memory"]["provider"] == "spectron"

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "SPECTRON_API_KEY=sk-live" in env_text

    # The secret must never reach spectron.json.
    saved = json.loads((tmp_path / "spectron.json").read_text(encoding="utf-8"))
    assert saved["endpoint"] == "https://x.spectron.dev"
    assert saved["context"] == "ctx"
    assert "api_key" not in saved


def test_post_setup_warns_about_bad_credentials_but_still_activates(
    tmp_path, answers, monkeypatch, capsys
):
    """Setup is the last moment the user is watching — say the key is rejected.

    But don't refuse: all three settings are present, and the network merely
    being down at setup time is no reason to leave them unconfigured.
    """
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    monkeypatch.setattr(provider_mod, "spectron_status", lambda: (True, ""))
    monkeypatch.setattr(
        provider_mod.SpectronMemoryProvider,
        "_check_credentials",
        lambda self: (False, "API key rejected by https://x.spectron.dev"),
    )
    answers.update(
        {
            "spectron api key": "sk-wrong",
            "spectron endpoint": "https://x.spectron.dev",
            "spectron context": "ctx",
        }
    )
    _, config = _setup(tmp_path)

    assert "API key rejected" in capsys.readouterr().out
    assert config["memory"]["provider"] == "spectron"


def test_post_setup_refuses_to_activate_without_api_key(tmp_path, answers, monkeypatch):
    """The generic wizard prints success here; we must not."""
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    answers.update({"spectron endpoint": "https://x.spectron.dev", "spectron context": "ctx"})
    answers["spectron api key"] = ""

    _, config = _setup(tmp_path)

    assert not config.get("memory", {}).get("provider")
    assert not (tmp_path / ".env").exists()


def test_post_setup_persists_a_preexisting_exported_key(tmp_path, answers, monkeypatch):
    """This is the reported bug.

    Hermes' generic wizard turns an exported secret into a "blank to keep" prompt
    and writes nothing, so the key dies with the shell. Pressing enter here must
    still persist it.
    """
    monkeypatch.setenv("SPECTRON_API_KEY", "sk-exported")
    answers.update({"spectron endpoint": "https://x.spectron.dev", "spectron context": "ctx"})
    # No "api key" reply: the user just hits enter, taking the default.

    _, config = _setup(tmp_path)

    assert config["memory"]["provider"] == "spectron"
    assert "SPECTRON_API_KEY=sk-exported" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_post_setup_updates_an_existing_env_entry(tmp_path, answers, monkeypatch):
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER=keep\nSPECTRON_API_KEY=old\n", encoding="utf-8")

    answers.update(
        {"spectron api key": "new", "spectron endpoint": "https://x.spectron.dev", "spectron context": "ctx"}
    )
    _setup(tmp_path)

    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert "OTHER=keep" in lines
    assert "SPECTRON_API_KEY=new" in lines
    assert "SPECTRON_API_KEY=old" not in lines


def test_post_setup_keeps_top_k_numeric(tmp_path, answers, monkeypatch):
    """Prompts return strings; spectron.json should still hold an int."""
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    answers.update(
        {
            "spectron api key": "sk",
            "spectron endpoint": "https://x.spectron.dev",
            "spectron context": "ctx",
            "memories recalled": "9",
        }
    )
    _setup(tmp_path)

    saved = json.loads((tmp_path / "spectron.json").read_text(encoding="utf-8"))
    assert saved["top_k"] == 9
    assert isinstance(saved["top_k"], int)


def test_post_setup_survives_garbage_top_k(tmp_path, answers, monkeypatch):
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    answers.update(
        {
            "spectron api key": "sk",
            "spectron endpoint": "https://x.spectron.dev",
            "spectron context": "ctx",
            "memories recalled": "not-a-number",
        }
    )
    _setup(tmp_path)
    saved = json.loads((tmp_path / "spectron.json").read_text(encoding="utf-8"))
    assert saved["top_k"] == 5  # falls back to the current value


# -- .env fallback ----------------------------------------------------------


def test_config_reads_key_from_env_file(tmp_path, monkeypatch):
    """A key saved to .env must resolve even if nothing exported it.

    Whether `.env` reaches os.environ depends on the host process; without this
    fallback a correctly configured provider reports itself unavailable.
    """
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    save_config_file(
        {"endpoint": "https://x.spectron.dev", "context": "ctx"}, str(tmp_path)
    )
    (tmp_path / ".env").write_text("SPECTRON_API_KEY=sk-from-file\n", encoding="utf-8")

    cfg = load_config(str(tmp_path))
    assert cfg.api_key == "sk-from-file"
    assert cfg.is_configured()


def test_exported_env_var_outranks_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRON_API_KEY", "sk-exported")
    (tmp_path / ".env").write_text("SPECTRON_API_KEY=sk-from-file\n", encoding="utf-8")
    assert load_config(str(tmp_path)).api_key == "sk-exported"


def test_env_file_supplies_endpoint_and_context(tmp_path, monkeypatch):
    """`hermes plugins install` writes all requires_env to .env, not spectron.json."""
    for var in ("SPECTRON_API_KEY", "SPECTRON_ENDPOINT", "SPECTRON_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".env").write_text(
        "SPECTRON_API_KEY=sk\n"
        'SPECTRON_ENDPOINT="https://quoted.spectron.dev"\n'
        "# a comment\n"
        "SPECTRON_CONTEXT=ctx-from-env\n",
        encoding="utf-8",
    )
    cfg = load_config(str(tmp_path))
    assert cfg.endpoint == "https://quoted.spectron.dev"  # quotes stripped
    assert cfg.context == "ctx-from-env"
    assert cfg.is_configured()


def test_missing_env_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    assert load_config(str(tmp_path)).api_key is None


def test_ask_choice_rejects_invalid_values(monkeypatch, capsys):
    """load_config silently falls back on a bad value, so catch it at the prompt."""
    from spectron_hermes.config import RECALL_MODES

    answers = iter(["nonsense", "also-wrong", "tools"])
    monkeypatch.setattr(provider_mod, "_ask", lambda label, default=None: next(answers))

    assert provider_mod._ask_choice("Recall mode", RECALL_MODES, "hybrid") == "tools"
    assert "is not one of" in capsys.readouterr().out


def test_ask_choice_keeps_default_after_repeated_bad_input(monkeypatch):
    from spectron_hermes.config import RECALL_MODES

    monkeypatch.setattr(provider_mod, "_ask", lambda label, default=None: "bogus")
    assert provider_mod._ask_choice("Recall mode", RECALL_MODES, "context") == "context"


def test_env_write_neutralises_newlines(tmp_path):
    """A pasted secret containing a newline must not inject a second variable."""
    env_path = tmp_path / ".env"
    provider_mod._upsert_env_var(env_path, "SPECTRON_API_KEY", "abc\nEVIL=1")

    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["SPECTRON_API_KEY=abcEVIL=1"]


# -- status reporting -------------------------------------------------------


def test_get_status_config_redacts_the_key(hermes_home, fake, monkeypatch):
    # `fake` stands in for build_client so the connection probe stays local.
    monkeypatch.setenv("SPECTRON_API_KEY", "sk-supersecret")
    p = SpectronMemoryProvider()
    p._hermes_home = hermes_home

    status = p.get_status_config({})

    assert status["api_key"] == "…cret"
    assert "sk-supersecret" not in json.dumps(status)
    assert status["endpoint"] == "https://example.spectron.dev"


def test_get_status_config_flags_a_missing_key(hermes_home, fake, monkeypatch):
    monkeypatch.delenv("SPECTRON_API_KEY", raising=False)
    p = SpectronMemoryProvider()
    p._hermes_home = hermes_home
    assert p.get_status_config({})["api_key"] == "(not set)"


# -- schema fallback --------------------------------------------------------


def test_config_schema_shape_for_generic_wizard():
    """Hermes releases without post_setup fall back to this schema.

    The secret branch keys off `secret` + `env_var`, so both must stay put.
    """
    from spectron_hermes.config import config_schema

    schema = config_schema()
    assert schema[0]["key"] == "api_key"
    assert schema[0]["secret"] is True
    assert schema[0]["env_var"] == "SPECTRON_API_KEY"
    assert schema[0]["required"] is True

    by_key = {f["key"]: f for f in schema}
    for key in ("endpoint", "context"):
        assert by_key[key]["required"] is True


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

    # The built-in memory mirror writes single facts with provenance.
    assert {"session_id", "labels"} <= remember

    # Credential diagnostics: health() is unauthenticated, whoami() is
    # context-scoped. Losing either collapses the 401 triage back to a guess.
    for probe in ("health", "whoami"):
        assert callable(getattr(Spectron, probe, None)), probe

    from surrealdb.spectron._namespaces.documents import BlockingDocuments

    upload = params(BlockingDocuments, "upload")
    assert "title" in upload and "scopes" in upload
