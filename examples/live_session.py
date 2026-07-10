"""Drive the Spectron provider against a LIVE Spectron instance.

Requires:
  * ``pip install "surrealdb[spectron]"``
  * Spectron credentials in the environment:
        export SPECTRON_ENDPOINT="https://your-instance.spectron.dev"
        export SPECTRON_CONTEXT="my-context"
        export SPECTRON_API_KEY="..."

Run:

    python examples/live_session.py

This performs a real remember → recall round-trip, so it writes to your
Spectron context. Use a throwaway context if you don't want the data to stick.
"""

from __future__ import annotations

import sys
import time

from spectron_hermes.provider import SpectronMemoryProvider


def main() -> int:
    provider = SpectronMemoryProvider()

    if not provider.is_available():
        print(
            "Spectron is not configured. Set SPECTRON_ENDPOINT / SPECTRON_CONTEXT / "
            "SPECTRON_API_KEY and `pip install 'surrealdb[spectron]'`.",
            file=sys.stderr,
        )
        return 1

    # Hermes passes user_id; here it seeds the default scope (user/tobie).
    provider.initialize("live-example-session", user_id="tobie")

    print("remember →", provider.handle_tool_call(
        "spectron_remember", {"text": "Tobie was promoted to CTO"}
    ))

    # Give Spectron a moment to index the new memory before recalling.
    time.sleep(2)

    print("\nprefetch:\n", provider.prefetch("what is tobie's role?"))
    print("\nrecall →", provider.handle_tool_call(
        "spectron_recall", {"query": "tobie role"}
    ))
    print("\ncontext →", provider.handle_tool_call(
        "spectron_context", {"query": "what do you know about Tobie?"}
    ))

    # Persist the turn and close out the session (triggers consolidation).
    provider.sync_turn("Where do I work?", "You're the CTO.")
    provider.on_session_end([])
    provider.shutdown()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
