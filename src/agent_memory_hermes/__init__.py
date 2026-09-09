"""Agent Memory provider for the Hermes Agent runtime.

This package integrates `SurrealDB Agent Memory <https://surrealdb.com/agent-memory>`_
as a Hermes ``MemoryProvider``. Hermes loads it either as a pip package (via the
``hermes_agent.plugins`` entry point) or as a drop-in directory under
``$HERMES_HOME/plugins/agent_memory/``.

Hermes' memory loader supports two conventions and this module exposes both:

* a module-level ``register(ctx)`` that calls ``ctx.register_memory_provider(...)``
* an exported :class:`AgentMemoryProvider` subclass with a no-arg constructor

The literal string ``MemoryProvider`` must appear in this file — the drop-in
directory loader uses it as a discovery heuristic.
"""

from .provider import AgentMemoryProvider

__all__ = ["AgentMemoryProvider", "register"]


def register(ctx) -> None:
    """Register the Agent Memory provider with the Hermes plugin context."""
    ctx.register_memory_provider(AgentMemoryProvider())
