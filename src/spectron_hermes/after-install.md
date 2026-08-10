## Spectron memory installed

Activate it:

```
hermes memory setup      # choose "spectron"
hermes memory status     # confirm it is active
```

Setup prompts for your **API key**, **endpoint**, and **context**. The API key goes to
`$HERMES_HOME/.env`; everything else to `$HERMES_HOME/spectron.json`.

If you already exported `SPECTRON_API_KEY`, setup offers it as the default — press enter
to keep it and it will be saved properly.

Tunables (`recall_mode`, `write_frequency`, `top_k`, `default_scope`) are documented at
<https://github.com/surrealdb/spectron-hermes#configuration>.

Spectron is in invite-only preview — request access at
<https://surrealdb.com/pricing/spectron>.
