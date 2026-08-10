"""Install this package as a Hermes directory plugin.

Hermes only discovers memory providers by scanning ``$HERMES_HOME/plugins/`` one
level deep, so a PyPI install alone is never enough — the package has to exist as
``$HERMES_HOME/plugins/spectron/``. This module is the ``spectron-hermes-install``
console script that puts it there.

Prefer ``hermes plugins install surrealdb/spectron-hermes/src/spectron_hermes`` when
Hermes' CLI is available; this script is the equivalent for people who arrived via
``pip install spectron-hermes``.

IMPORTANT: Hermes' directory loader ``exec_module``s *every* ``*.py`` in the plugin
directory while discovering providers, so this module must stay free of
import-time side effects — stdlib imports and function definitions only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

# The directory name is load-bearing: Hermes uses it as the provider name in the
# `hermes memory setup` picker and in config.yaml's `memory.provider` key.
PLUGIN_DIR_NAME = "spectron"

_EXCLUDE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def default_hermes_home() -> Path:
    """Resolve ``$HERMES_HOME`` the way Hermes itself does.

    Mirrors ``hermes_constants``: the ``HERMES_HOME`` environment variable first,
    then the platform-native default. Note that Hermes points this at
    ``<root>/profiles/<name>`` when a non-default profile is active, which is why
    the env var has to win.
    """
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def install(hermes_home: Path, *, force: bool = False) -> Path:
    """Copy this package to ``<hermes_home>/plugins/spectron/``.

    Raises ``FileNotFoundError`` if *hermes_home* does not exist and
    ``FileExistsError`` if the target is present and *force* is False.
    """
    if not hermes_home.is_dir():
        raise FileNotFoundError(
            f"{hermes_home} does not exist. Run Hermes once to create it, or pass "
            f"--hermes-home if your install lives elsewhere (a non-default profile "
            f"puts it under <root>/profiles/<name>)."
        )

    source = Path(__file__).resolve().parent
    target = hermes_home / "plugins" / PLUGIN_DIR_NAME

    if target.exists():
        if not force:
            raise FileExistsError(
                f"{target} already exists. Re-run with --force to overwrite it."
            )
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=_EXCLUDE)
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spectron-hermes-install",
        description=(
            "Install the Spectron memory provider into $HERMES_HOME/plugins/spectron/ "
            "so `hermes memory setup` can find it."
        ),
    )
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=None,
        help="Hermes home directory (default: $HERMES_HOME, else ~/.hermes).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing plugin directory.",
    )
    args = parser.parse_args(argv)

    hermes_home = args.hermes_home or default_hermes_home()

    try:
        target = install(hermes_home, force=args.force)
    except (FileNotFoundError, FileExistsError, OSError) as exc:
        # Fail loudly: silently installing where Hermes never scans is the whole
        # failure mode this script exists to prevent.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Installed Spectron memory provider to {target}")
    print("\nNext:")
    print("  hermes memory setup     # choose 'spectron'")
    print("  hermes memory status    # confirm it is active")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
