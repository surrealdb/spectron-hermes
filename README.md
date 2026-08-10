# Spectron ⇄ Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) **memory provider**
backed by [SurrealDB Spectron](https://surrealdb.com/platform/spectron) —
provenance-first, tri-temporal agent memory with semantic, lexical, graph and
temporal recall.

Once installed and selected, the agent automatically:

- **recalls** relevant memories before every turn,
- **writes** each completed turn back to Spectron (asynchronously),
- **mirrors** the built-in memory tool's `MEMORY.md` / `USER.md` writes into
  Spectron, so the two memories can't drift apart,
- **consolidates** memory when a session ends,

and gains six explicit memory tools it can call directly.

## Requirements

- Python 3.10+
- A running [Hermes Agent](https://github.com/NousResearch/hermes-agent) install
  (verified against Hermes Agent 0.20.0, git tag `v2026.8.3` — Hermes tags are
  CalVer while its release titles are semver)
- Spectron access (endpoint, context, API key). Spectron is in
  [invite-only preview](https://surrealdb.com/pricing/spectron).

## Install

Hermes discovers memory providers **by directory**: it scans `$HERMES_HOME/plugins/`
one level deep, and the directory name becomes the provider name in the picker and in
`config.yaml`'s `memory.provider` key. So the plugin has to land at
`$HERMES_HOME/plugins/spectron/` — the name matters.

Hermes' own installer reads the name from our `plugin.yaml`, so it puts it in the right
place automatically:

```bash
hermes plugins install surrealdb/spectron-hermes/src/spectron_hermes
```

<details>
<summary>Alternative: install from PyPI</summary>

`pip install spectron-hermes` installs the library and the SurrealDB SDK, but **does
not** register the provider — Hermes' plugin entry-point system has no
memory-provider registration hook. Run the bundled installer afterwards to place the
directory plugin:

```bash
pip install spectron-hermes
spectron-hermes-install          # → $HERMES_HOME/plugins/spectron/
```
</details>

<details>
<summary>Alternative: copy the directory manually</summary>

Copy `src/spectron_hermes/` to `$HERMES_HOME/plugins/spectron/` — flat, with
`__init__.py` directly inside:

```
$HERMES_HOME/plugins/spectron/
├── __init__.py
├── provider.py  client.py  config.py  tools.py  _compat.py
└── plugin.yaml
```

Then `uv pip install "surrealdb>=3.0.0a4"`.

Two things silently produce a provider that never appears in the picker: naming the
directory anything other than `spectron`, and nesting it (a `memory/` level, or
copying the repo so `__init__.py` ends up at `spectron/src/spectron_hermes/`). Note
also that `$HERMES_HOME` is **not** always `~/.hermes` — under a non-default profile
Hermes points it at `<root>/profiles/<name>`. Check with:

```bash
python -c "from hermes_constants import get_hermes_home as g; print(g())"
```
</details>

### A note on the pre-release dependency

Spectron ships only in `surrealdb` **v3**, which is currently alpha, so installing
`spectron-hermes` always pulls a pre-release. `pip` handles this per PEP 440 with no
flags. With uv you need **0.12.0 or newer**, which resolves pre-releases requested by
transitive dependencies; older uv reports `only surrealdb<3.0.0a4 is available`. On an
older uv, either pass `--prerelease=allow` or persist it in your own project:

```toml
[tool.uv]
prerelease = "allow"
```

The `hermes plugins install` path above sidesteps this entirely: Hermes installs
`surrealdb>=3.0.0a4` as a direct requirement, which resolves on any uv version.

## Configure & activate

```bash
hermes memory setup      # choose "spectron"; prompts for API key, endpoint, context
hermes memory status     # confirm it is active
hermes                   # run a session with Spectron-backed memory
```

Setup writes the API key to `$HERMES_HOME/.env` and non-secret settings to
`$HERMES_HOME/spectron.json`. It refuses to activate Spectron if the API key,
endpoint, or context is missing, rather than reporting success and going quietly
inert.

<details>
<summary>Headless / CI: configure via environment</summary>

Every setting has an env var (see the table below), so setup can be skipped
entirely:

```bash
export SPECTRON_API_KEY="..."
export SPECTRON_ENDPOINT="https://your-instance.spectron.dev"
export SPECTRON_CONTEXT="my-context"
```

For interactive installs, prefer letting `hermes memory setup` prompt. If
`SPECTRON_API_KEY` is already exported it is offered as the prompt default and saved
to `.env` on enter, so it survives the shell — but note that **Hermes' own generic
wizard** (used by other providers, and by this one on Hermes releases predating
`post_setup`) treats an exported secret as a reason to write nothing at all.
</details>

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

**Resolution order** for every setting: `$HERMES_HOME/spectron.json` → environment
variable → `$HERMES_HOME/.env` → default. Reading `.env` directly means a key saved by
`hermes memory setup` or `hermes plugins install` resolves even when nothing exported
it into the environment. The API key is never written to `spectron.json`.

## Tools exposed to the agent

| Tool | Spectron call | Purpose |
|---|---|---|
| `spectron_recall(query, k?)` | `recall` | Search memory (semantic/lexical/graph/temporal). |
| `spectron_remember(text, scope?)` | `remember` | Store a durable fact. |
| `spectron_context(query, k?)` | `query_context` | Synthesised answer from memory. |
| `spectron_forget(query, purge?)` | `forget` | Supersede (default) or hard-delete. |
| `spectron_reflect(query, persist?)` | `reflect` | Derive insights; optionally persist. |
| `spectron_upload(path, title?)` | `documents.upload` | Ingest a document into knowledge memory. |

## Mirroring the built-in memory tool

Hermes ships its own memory tool, which edits `MEMORY.md` and `USER.md` on disk.
Those writes are mirrored into Spectron via the `on_memory_write` hook, so a fact
the agent saves through the built-in tool is recallable from Spectron in later
sessions:

| Built-in action | Spectron call |
|---|---|
| `add` | `remember(content)` |
| `replace` | `forget(old_text)` then `remember(content)` |
| `remove` | `forget(content, purge=False)` — superseded, history kept |

Mirrored entries are labelled `hermes` and `memory:<target>`. Writes originating
from a subagent or cron run are skipped, and — unlike turn writes — mirroring is
not affected by `write_frequency`, since these are explicit durable facts rather
than conversational turns.

## Troubleshooting

**`[401] InvalidToken` / memory goes quiet after the first turn.** The provider
disables itself for the session on an auth error, so this shows up once in the
`hermes gateway` log and then nothing. Run:

```bash
hermes memory status
```

The `connection` row probes the endpoint (unauthenticated `health`) and then the
credentials (context-scoped `whoami`), and distinguishes:

| Reported | Meaning |
|---|---|
| cannot reach `<endpoint>` | `SPECTRON_ENDPOINT` should be the API **origin** — no path, no trailing slash |
| API key rejected … for context `'<ctx>'` | `SPECTRON_API_KEY` is wrong, or was issued for a different context |
| API key does not authorize context `'<ctx>'` | key is valid but lacks scope for this principal |
| context `'<ctx>'` does not exist | `SPECTRON_CONTEXT` is wrong for this endpoint |

Failures include a Spectron `trace` id when the server returns one — quote it
when reporting an issue. The same check runs at the end of `hermes memory setup`.

**Provider missing from `hermes memory setup`.** It must live at
`$HERMES_HOME/plugins/spectron/` — one level deep, named `spectron`, with
`__init__.py` directly inside. Note `$HERMES_HOME` is the profile directory under
a non-default profile, and is *not* `~/.hermes/hermes-agent`. See
[Install](#install).

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
