# Spectron ⇄ Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) **memory provider**
backed by [SurrealDB Spectron](https://surrealdb.com/platform/spectron) —
provenance-first, tri-temporal agent memory with semantic, lexical, graph and
temporal recall.

Once installed and selected, the agent automatically:

- **recalls** relevant memories before every turn,
- **writes** each completed turn back to Spectron (asynchronously),
- **consolidates** memory when a session ends,

and gains six explicit memory tools it can call directly.

## Requirements

- Python 3.10+
- A running Hermes Agent install
- Spectron access (endpoint, context, API key). Spectron is in
  [invite-only preview](https://surrealdb.com/pricing/spectron).

## Install

```bash
pip install spectron-hermes-agent
```

This pulls in the SurrealDB SDK (`surrealdb` v3, which bundles Spectron) and
registers the plugin with Hermes via the `hermes_agent.plugins` entry point.

<details>
<summary>Alternative: drop-in directory</summary>

If your Hermes setup discovers memory providers by directory, copy the package
into `$HERMES_HOME/plugins/spectron/` (it ships a `plugin.yaml` manifest for
this) and `pip install "surrealdb>=3.0.0a1"`.
</details>

## Configure & activate

Provide credentials via the environment (the API key is a secret and belongs in
`.env`):

```bash
export SPECTRON_API_KEY="..."
export SPECTRON_ENDPOINT="https://your-instance.spectron.dev"
export SPECTRON_CONTEXT="my-context"
```

Then select the provider — Hermes will prompt for any missing settings:

```bash
hermes memory setup      # choose "spectron"
hermes memory status     # confirm it is active
hermes                   # run a session with Spectron-backed memory
```

Non-secret settings are written to `$HERMES_HOME/spectron.json`.

## Configuration

| Setting | Env var | Default | Notes |
|---|---|---|---|
| `api_key` | `SPECTRON_API_KEY` | — | **secret**, required (stored in `.env`) |
| `endpoint` | `SPECTRON_ENDPOINT` | — | required, origin with no trailing slash |
| `context` | `SPECTRON_CONTEXT` | — | required; Spectron pins a client to one context |
| `recall_mode` | `SPECTRON_RECALL_MODE` | `hybrid` | `hybrid` \| `context` \| `tools` |
| `write_frequency` | `SPECTRON_WRITE_FREQUENCY` | `turn` | `turn` \| `session` |
| `top_k` | `SPECTRON_TOP_K` | `5` | memories recalled per turn |
| `default_scope` | `SPECTRON_DEFAULT_SCOPE` | — | e.g. `user/tobie`; scope for writes / lens for reads |

**Recall modes:** `hybrid` injects raw recalled memories before each turn;
`context` injects a synthesised answer instead; `tools` injects nothing and lets
the model recall explicitly via `spectron_recall` / `spectron_context`.

## Tools exposed to the agent

| Tool | Spectron call | Purpose |
|---|---|---|
| `spectron_recall(query, k?)` | `recall` | Search memory (semantic/lexical/graph/temporal). |
| `spectron_remember(text, scope?)` | `remember` | Store a durable fact. |
| `spectron_context(query, k?)` | `query_context` | Synthesised answer from memory. |
| `spectron_forget(query, purge?)` | `forget` | Supersede (default) or hard-delete. |
| `spectron_reflect(query, persist?)` | `reflect` | Derive insights; optionally persist. |
| `spectron_upload(path, title?)` | `documents.upload` | Ingest a document into knowledge memory. |

## Reliability

The provider is built to never destabilise the agent:

- Writes run on a background daemon thread — turns never block on I/O.
- Every Spectron call is wrapped; failures are logged and degrade to empty
  results rather than raising into the agent loop (**fail open**).
- After repeated failures (or an auth error) a **circuit breaker** disables
  memory for the rest of the session.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests use a mock Spectron client and need neither a live server nor the
`surrealdb` SDK installed.

## License

Apache-2.0
