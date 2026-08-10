"""Spectron memory provider for the Hermes Agent runtime.

This package integrates `SurrealDB Spectron <https://surrealdb.com/platform/spectron>`_
as a Hermes ``MemoryProvider``.

Hermes discovers memory providers **by directory only**: it scans
``$HERMES_HOME/plugins/`` one level deep and imports ``<dir>/__init__.py``. The
directory name is what appears in ``hermes memory setup`` and what gets written to
``memory.provider`` in ``config.yaml``, so this package must be installed as
``$HERMES_HOME/plugins/spectron/``. Use ``hermes plugins install`` (which takes the
name from ``plugin.yaml``) or the bundled ``spectron-hermes-install`` script; see the
README.

Installing from PyPI alone does **not** register the provider — Hermes' plugin
entry-point system has no memory-provider registration hook.

The loader supports two conventions and this module exposes both:

* a module-level ``register(ctx)`` that calls ``ctx.register_memory_provider(...)``
* an exported :class:`SpectronMemoryProvider` subclass with a no-arg constructor

The literal string ``MemoryProvider`` must appear in this file — the directory loader
uses it as a discovery heuristic.
"""

import logging

from .provider import SpectronMemoryProvider

__all__ = ["SpectronMemoryProvider", "register"]

logger = logging.getLogger("spectron_hermes")


def register(ctx) -> None:
    """Register the Spectron memory provider with the Hermes plugin context.

    Only Hermes' memory loader passes a context that can accept a memory provider.
    The general plugin manager's ``PluginContext`` has no
    ``register_memory_provider``, so warn and return rather than raising an
    ``AttributeError`` up into the CLI if this is ever loaded that way.
    """
    if not hasattr(ctx, "register_memory_provider"):
        logger.warning(
            "This context cannot register a memory provider. Install Spectron as a "
            "directory plugin at $HERMES_HOME/plugins/spectron/ — e.g. "
            "`hermes plugins install surrealdb/spectron-hermes/src/spectron_hermes` "
            "or `spectron-hermes-install`."
        )
        return
    ctx.register_memory_provider(SpectronMemoryProvider())
