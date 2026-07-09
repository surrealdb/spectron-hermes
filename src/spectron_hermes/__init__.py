"""Spectron memory provider for the Hermes Agent runtime.

This package integrates `SurrealDB Spectron <https://surrealdb.com/platform/spectron>`_
as a Hermes ``MemoryProvider``. Hermes loads it either as a pip package (via the
``hermes_agent.plugins`` entry point) or as a drop-in directory under
``$HERMES_HOME/plugins/spectron/``.

Hermes' memory loader supports two conventions and this module exposes both:

* a module-level ``register(ctx)`` that calls ``ctx.register_memory_provider(...)``
* an exported :class:`SpectronMemoryProvider` subclass with a no-arg constructor

The literal string ``MemoryProvider`` must appear in this file — the drop-in
directory loader uses it as a discovery heuristic.
"""

from .provider import SpectronMemoryProvider

__all__ = ["SpectronMemoryProvider", "register"]


def register(ctx) -> None:
    """Register the Spectron memory provider with the Hermes plugin context."""
    ctx.register_memory_provider(SpectronMemoryProvider())
