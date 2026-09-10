"""Drive the Agent Memory provider against a LIVE Agent Memory instance.

Requires:
  * ``pip install "surrealdb[memory]>=3.0.0b8"`` (v3 bundles Agent Memory)
  * Agent Memory credentials in the environment:
        export AGENT_MEMORY_ENDPOINT="https://your-instance.agent-memory.dev"
        export AGENT_MEMORY_CONTEXT="my-context"
        export AGENT_MEMORY_API_KEY="..."

Run:

    python examples/live_session.py

This performs a real remember → recall round-trip, so it writes to your
Agent Memory context. Use a throwaway context if you don't want the data to stick.
"""

from __future__ import annotations

import sys
import time

from agent_memory_hermes.provider import AgentMemoryProvider


def main() -> int:
    provider = AgentMemoryProvider()

    if not provider.is_available():
        print(
            "Agent Memory is not configured. Set AGENT_MEMORY_ENDPOINT / AGENT_MEMORY_CONTEXT / "
            "AGENT_MEMORY_API_KEY and `pip install 'surrealdb[memory]>=3.0.0b8'`.",
            file=sys.stderr,
        )
        return 1

    # Hermes passes user_id; here it seeds the default scope (user/tobie).
    provider.initialize("live-example-session", user_id="tobie")

    print("remember →", provider.handle_tool_call(
        "agent_memory_remember", {"text": "Tobie was promoted to CTO"}
    ))

    # Give Agent Memory a moment to index the new memory before recalling.
    time.sleep(2)

    print("\nprefetch:\n", provider.prefetch("what is tobie's role?"))
    print("\nrecall →", provider.handle_tool_call(
        "agent_memory_recall", {"query": "tobie role"}
    ))
    print("\ncontext →", provider.handle_tool_call(
        "agent_memory_context", {"query": "what do you know about Tobie?"}
    ))

    # Persist the turn and close out the session (triggers consolidation).
    provider.sync_turn("Where do I work?", "You're the CTO.")
    provider.on_session_end([])
    provider.shutdown()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
