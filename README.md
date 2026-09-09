# AgentMemory ⇄ Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) **memory provider**
backed by [SurrealDB AgentMemory](https://surrealdb.com/agent-memory) —
provenance-first, tri-temporal agent memory with semantic, lexical, graph and
temporal recall.

Once installed and selected, the agent automatically:

- **recalls** relevant memories before every turn,
- **writes** each completed turn back to AgentMemory (asynchronously),
- **consolidates** memory when a session ends,

and gains six explicit memory tools it can call directly.

## Requirements

- Python 3.10+
- A running [Hermes Agent](https://github.com/NousResearch/hermes-agent) install
  (verified against Hermes Agent 0.18.2)
- AgentMemory access (endpoint, context, API key). AgentMemory is in
  [invite-only preview](https://surrealdb.com/pricing/agent_memory).

## Install

```bash
pip install agent-memory-hermes
```

This pulls in the SurrealDB SDK (`surrealdb` v3, which bundles AgentMemory) and
registers the plugin with Hermes via the `hermes_agent.plugins` entry point.

<details>
<summary>Alternative: drop-in directory</summary>

If your Hermes setup discovers memory providers by directory, copy the package
into `$HERMES_HOME/plugins/agent_memory/` (it ships a `plugin.yaml` manifest for
this) and `pip install "surrealdb[memory]>=3.0.0b8"`.
</details>

## Configure & activate

Provide credentials via the environment (the API key is a secret and belongs in
`.env`):

```bash
export AGENT_MEMORY_API_KEY="..."
export AGENT_MEMORY_ENDPOINT="https://your-instance.agent_memory.dev"
export AGENT_MEMORY_CONTEXT="my-context"
```

Then select the provider — Hermes will prompt for any missing settings:

```bash
hermes memory setup      # choose "agent_memory"
hermes memory status     # confirm it is active
hermes                   # run a session with AgentMemory-backed memory
```

Non-secret settings are written to `$HERMES_HOME/agent_memory.json`.

## Configuration

| Setting | Env var | Default | Notes |
|---|---|---|---|
| `api_key` | `AGENT_MEMORY_API_KEY` | — | **secret**, required (stored in `.env`) |
| `endpoint` | `AGENT_MEMORY_ENDPOINT` | — | required, origin with no trailing slash |
| `context` | `AGENT_MEMORY_CONTEXT` | — | required; AgentMemory pins a client to one context |
| `recall_mode` | `AGENT_MEMORY_RECALL_MODE` | `hybrid` | `hybrid` \| `context` \| `tools` |
| `write_frequency` | `AGENT_MEMORY_WRITE_FREQUENCY` | `turn` | `turn` \| `session` |
| `top_k` | `AGENT_MEMORY_TOP_K` | `5` | memories recalled per turn |
| `default_scope` | `AGENT_MEMORY_DEFAULT_SCOPE` | — | e.g. `user/tobie`; scope for writes / lens for reads |

**Recall modes:** `hybrid` injects raw recalled memories before each turn;
`context` injects a synthesised answer instead; `tools` injects nothing and lets
the model recall explicitly via `agent_memory_recall` / `agent_memory_context`.

## Tools exposed to the agent

| Tool | AgentMemory call | Purpose |
|---|---|---|
| `agent_memory_recall(query, k?)` | `recall` | Search memory (semantic/lexical/graph/temporal). |
| `agent_memory_remember(text, scope?)` | `remember` | Store a durable fact. |
| `agent_memory_context(query, k?)` | `query_context` | Synthesised answer from memory. |
| `agent_memory_forget(query, purge?)` | `forget` | Supersede (default) or hard-delete. |
| `agent_memory_reflect(query, persist?)` | `reflect` | Derive insights; optionally persist. |
| `agent_memory_upload(path, title?)` | `documents.upload` | Ingest a document into knowledge memory. |

## Reliability

The provider is built to never destabilise the agent:

- Writes run on a background daemon thread — turns never block on I/O.
- Every AgentMemory call is wrapped; failures are logged and degrade to empty
  results rather than raising into the agent loop (**fail open**).
- After repeated failures (or an auth error) a **circuit breaker** disables
  memory for the rest of the session.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests use a mock AgentMemory client and need neither a live server nor the
`surrealdb` SDK installed.

## License

Apache-2.0
