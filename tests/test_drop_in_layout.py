"""Guards on the directory-plugin contract Hermes actually enforces.

Hermes discovers memory providers by scanning ``$HERMES_HOME/plugins/`` one level
deep (``plugins/memory/__init__.py::_iter_provider_dirs``). Every assertion here
mirrors a real condition in that loader — each one, if broken, makes the provider
vanish from ``hermes memory setup`` with no error and only a ``logger.debug``
trace. They are cheap to break and expensive to diagnose, hence the tests.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

import spectron_hermes
from spectron_hermes import _install

PACKAGE_DIR = Path(spectron_hermes.__file__).resolve().parent

# Hermes reads only the first 8 KiB of __init__.py for its discovery heuristic.
_HEURISTIC_WINDOW = 8192


# -- manifest ----------------------------------------------------------------


def test_manifest_name_is_spectron():
    """`hermes plugins install` names the installed directory from this key.

    And the directory name is in turn what the picker shows and what goes into
    config.yaml's `memory.provider`. `name: spectron_hermes` would install a
    provider that works but can never be selected as "spectron".
    """
    text = (PACKAGE_DIR / "plugin.yaml").read_text(encoding="utf-8")
    assert re.search(r"^name:\s*spectron\s*$", text, re.MULTILINE), text


def test_manifest_declares_sdk_and_credentials():
    yaml = pytest.importorskip("yaml")
    meta = yaml.safe_load((PACKAGE_DIR / "plugin.yaml").read_text(encoding="utf-8"))

    assert meta["name"] == "spectron"
    assert any("surrealdb" in dep for dep in meta["pip_dependencies"])

    # requires_env is what makes `hermes plugins install` prompt for the API key
    # and persist it to .env instead of relying on a shell export.
    names = {
        entry["name"] if isinstance(entry, dict) else entry
        for entry in meta["requires_env"]
    }
    assert {"SPECTRON_API_KEY", "SPECTRON_ENDPOINT", "SPECTRON_CONTEXT"} <= names

    api_key = next(
        e for e in meta["requires_env"] if isinstance(e, dict) and e["name"] == "SPECTRON_API_KEY"
    )
    assert api_key["secret"] is True


def test_after_install_notes_are_shipped():
    assert (PACKAGE_DIR / "after-install.md").is_file()


# -- discovery heuristic -----------------------------------------------------


def test_init_is_at_the_scanned_depth():
    """The loader requires `<plugins>/<name>/__init__.py`, not nested deeper."""
    assert (PACKAGE_DIR / "__init__.py").is_file()


def test_init_satisfies_the_text_heuristic():
    source = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")[:_HEURISTIC_WINDOW]
    assert "MemoryProvider" in source or "register_memory_provider" in source


# -- register(ctx) -----------------------------------------------------------


class _Collector:
    """Mirrors Hermes' `_ProviderCollector`: only the memory hook exists."""

    def __init__(self):
        self.provider = None

    def register_memory_provider(self, provider):
        self.provider = provider


def test_register_hands_back_a_provider():
    ctx = _Collector()
    spectron_hermes.register(ctx)
    assert isinstance(ctx.provider, spectron_hermes.SpectronMemoryProvider)
    assert ctx.provider.name == "spectron"


def test_register_does_not_raise_without_the_memory_hook():
    """The general PluginContext has no register_memory_provider.

    If this package is ever loaded through the entry-point/plugin path, it must
    degrade to a warning rather than throwing AttributeError into the CLI.
    """

    class BareCtx:
        def register_tool(self, *a, **k):
            pass

    spectron_hermes.register(BareCtx())  # must not raise


def test_provider_subclasses_the_real_abc_when_available():
    """The loader's fallback path does `issubclass(attr, agent.memory_provider.MemoryProvider)`.

    Inside Hermes the real ABC is importable and our provider must subclass it,
    otherwise only the register() path works.

    Resolve the base the same way `provider.py` does. Asserting against the
    mirror unconditionally passes only because Hermes is absent in CI — with it
    importable, the provider's base *is* the real ABC and the mirror is an
    unrelated class.
    """
    try:
        from agent.memory_provider import MemoryProvider
    except ImportError:
        from spectron_hermes._compat import MemoryProvider

    assert issubclass(spectron_hermes.SpectronMemoryProvider, MemoryProvider)


def test_compat_mirror_covers_the_real_abc():
    """`_compat.MemoryProvider` must not fall behind the upstream ABC.

    The mirror exists so this package imports outside Hermes, and its docstring
    promises it stays in sync. Nothing enforced that, which is how
    ``on_memory_write`` came to be missing from both the mirror and the
    provider. Skipped when Hermes isn't importable (the usual CI case).
    """
    real = pytest.importorskip("agent.memory_provider").MemoryProvider

    from spectron_hermes._compat import MemoryProvider as Mirror

    expected = {
        name
        for name in dir(real)
        if not name.startswith("_") and callable(getattr(real, name, None))
    }
    missing = sorted(expected - set(dir(Mirror)))
    assert not missing, f"_compat.MemoryProvider is missing: {missing}"


# -- eager submodule execution -----------------------------------------------


def test_every_submodule_imports_without_the_sdk(monkeypatch):
    """Hermes exec_modules *every* `*.py` in the plugin dir during discovery.

    A module-level `surrealdb` import in any of them would fail there and be
    swallowed at debug level, leaving a half-loaded package. Simulate a machine
    with no SDK by making the surrealdb spec lookup come back empty.
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "surrealdb" or name.startswith("surrealdb."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setitem(sys.modules, "surrealdb", None)  # import surrealdb -> ImportError

    pkg_name = "_spectron_layout_probe"
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, pkg_name, package)

    submodules = sorted(p for p in PACKAGE_DIR.glob("*.py") if p.name != "__init__.py")
    assert submodules, "expected submodules to probe"

    for path in submodules:
        sub_name = f"{pkg_name}.{path.stem}"
        sub_spec = importlib.util.spec_from_file_location(sub_name, path)
        sub_mod = importlib.util.module_from_spec(sub_spec)
        monkeypatch.setitem(sys.modules, sub_name, sub_mod)
        sub_spec.loader.exec_module(sub_mod)  # must not raise

    spec.loader.exec_module(package)  # must not raise


# -- installer ---------------------------------------------------------------


def test_install_creates_the_flat_spectron_layout(tmp_path):
    target = _install.install(tmp_path)

    assert target == tmp_path / "plugins" / "spectron"
    assert (target / "__init__.py").is_file()
    assert (target / "plugin.yaml").is_file()
    assert (target / "provider.py").is_file()
    # Nothing may sit between plugins/ and __init__.py.
    assert not (target / "src").exists()


def test_install_rejects_a_missing_hermes_home(tmp_path):
    with pytest.raises(FileNotFoundError):
        _install.install(tmp_path / "does-not-exist")


def test_install_refuses_to_clobber_without_force(tmp_path):
    _install.install(tmp_path)
    with pytest.raises(FileExistsError):
        _install.install(tmp_path)
    assert _install.install(tmp_path, force=True).is_dir()


def test_install_skips_pycache(tmp_path):
    cache = PACKAGE_DIR / "__pycache__"
    if not cache.exists():
        pytest.skip("no __pycache__ to exclude")
    assert not (_install.install(tmp_path) / "__pycache__").exists()


def test_hermes_home_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _install.default_hermes_home() == tmp_path

    monkeypatch.delenv("HERMES_HOME")
    assert _install.default_hermes_home().name in {"hermes", ".hermes"}


def test_main_reports_failure_for_bad_home(tmp_path, capsys):
    rc = _install.main(["--hermes-home", str(tmp_path / "nope")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_main_succeeds_and_prints_next_steps(tmp_path, capsys):
    assert _install.main(["--hermes-home", str(tmp_path)]) == 0
    assert "hermes memory setup" in capsys.readouterr().out
