# Examples

| File | What it shows | Needs |
|---|---|---|
| [`simulate_session.py`](simulate_session.py) | The full Hermes lifecycle (`is_available` → `initialize` → `prefetch` → tool calls → `sync_turn` → `on_session_end`) driven against a **fake in-memory Spectron client**. | Nothing but this package. |
| [`live_session.py`](live_session.py) | The same flow against a **real Spectron instance** — a genuine remember → recall round-trip. | `surrealdb>=3.0.0a4` + Spectron credentials. |
| [`spectron.json.example`](spectron.json.example) | Sample non-secret config (goes to `$HERMES_HOME/spectron.json`). | — |
| [`.env.example`](.env.example) | Sample environment / secrets. | — |

## Run the no-credentials demo

```bash
pip install -e .
python examples/simulate_session.py
```

This prints each step Hermes performs, so you can see exactly what the provider
does around a turn without touching a real Spectron backend.

## Run against real Spectron

```bash
pip install -e . "surrealdb>=3.0.0a4"
cp examples/.env.example .env    # fill in SPECTRON_API_KEY / ENDPOINT / CONTEXT
set -a; . ./.env; set +a
python examples/live_session.py
```

> `live_session.py` writes to your Spectron context — use a throwaway context if
> you don't want the demo data to persist.

## Using it inside Hermes

You normally don't call the provider directly — Hermes does. Install the package,
then:

```bash
hermes memory setup      # choose "spectron"
hermes memory status     # confirm active
hermes                   # chat with Spectron-backed memory
```

See the top-level [README](../README.md) for full configuration.
