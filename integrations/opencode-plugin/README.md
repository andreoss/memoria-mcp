# @memoria/opencode-plugin

An [OpenCode](https://opencode.ai) plugin that adds automatic memory capture and recall on top of a self-hosted memoria `server`. See `ADR-42` (Increment 2a) for the real design decisions behind this package.

For on-demand memory tools alone, with no automatic behavior, point OpenCode at the `mcp` binary directly instead — see `docs/overview.md`'s OpenCode integration section (`ADR-42` Increment 1). This plugin is for the hooks that only a plugin can provide: session-idle auto-capture and pre-prompt recall.

## Prerequisites

- A running memoria `server` (`cargo run -p server`), reachable at the URL configured below.
- [Bun](https://bun.sh) — OpenCode's own plugin runtime. On NixOS, the generic install script at `bun.sh/install` produces a binary that will not run (NixOS's dynamic linker layout is non-standard); use `nix profile add nixpkgs#bun` instead.

## Installation

Build this package, then reference the built file directly in `opencode.json` (project-level) or `~/.config/opencode/opencode.jsonc` (global):

```bash
cd integrations/opencode-plugin
bun install --omit=peer
bun run build
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["/absolute/path/to/integrations/opencode-plugin/dist/index.js"]
}
```

## Configuration

Read from the environment:

| Variable | Default | Purpose |
|---|---|---|
| `MEMORIA_SERVER_URL` | `http://127.0.0.1:8080` | Base URL of the running `server` |
| `MEMORIA_API_KEY` | unset | Sent as `Authorization: Bearer <value>` on every request |
| `MEMORIA_OPENCODE_USER_ID` | the OS username | Overrides automatic user identity resolution |
| `MEMORIA_OPENCODE_MCP_NAME` | `memoria` | The key this plugin's own `mcp` server is registered under in `opencode.json`'s `mcp` block — used to build the real, prefixed tool names (`<name>_add_memory`, etc.) consolidation instructs the agent to call |
| `MEMORIA_DREAM` | unset | Set to `false`/`0`/`no`/`off` to disable consolidation regardless of `~/.memoria/opencode-plugin/dream-config.json` |

Project identity (memoria's `agent_id`) is derived automatically: the git remote's `owner-repo`, falling back to the repo root directory name, then the working directory name — the same fallback chain used by every real prior-art OpenCode memory plugin this design was compared against.

## What it does

- **`shell.env`** — exports the resolved `MEMORIA_OPENCODE_USER_ID`/`MEMORIA_OPENCODE_PROJECT_ID` into every shell command the agent runs.
- **`event`** (on a real `session.idle` event) — fetches the session's recent user messages via the OpenCode SDK client and stores them with `POST /memories` (`infer: true`, so whichever LLM provider `server` is configured with does the real fact extraction).
- **`chat.message`** — searches memoria (`POST /search`) for context relevant to the incoming message, and on the first message of a session, checks whether memory consolidation should run (see below).
- **`experimental.chat.messages.transform`** — injects search results (and, when triggered, the consolidation protocol) into the message actually sent to the model, marked `synthetic: true`.
- **`tool.execute.after`** — during a consolidation-triggered session, watches for the agent actually calling `<name>_add_memory`/`<name>_update_memory`/`<name>_delete_memory` to confirm real work happened before recording completion.
- **`dispose`** — if a consolidation was triggered and a real write was observed, records completion (resetting the gates); otherwise releases the lock so a future session can retry.

## Memory consolidation ("dream")

Requires Increment 1 (the `mcp` binary registered as a local MCP server, under the key named by `MEMORIA_OPENCODE_MCP_NAME`) also be configured — this plugin has no native tools of its own, so consolidation only works if the agent actually has memory-management tools available to call.

Gated by real thresholds (defaults: 24 hours since the last consolidation, 5 distinct sessions since, 20+ stored memories in scope), plus a filesystem lock so two concurrent sessions can't consolidate at once. When every gate passes, the plugin does not call an LLM itself — it injects a consolidation-protocol instruction into the agent's own next turn, asking it to review memories in scope, then delete/merge/rewrite as appropriate using its own already-available tools and credentials.

**State vs. config, a real distinction (Sprint 183):** OpenCode's own plugin registration is global to the machine, not per-project — every project's own sessions load the same plugin code. The consolidation *thresholds* (`dream-config.json`, below) are therefore intentionally one shared, global settings file, the same way `MEMORIA_DREAM` is one shared env var. The consolidation *state* — the session counter, the last-consolidated timestamp, and the mutex lock itself — is **not** shared: it's isolated per real scope (the same `{userId, agentId}` pair that already scopes the memory data itself), stored under `~/.memoria/opencode-plugin/state/<hash-of-scope>/`. This was a real bug fixed in Sprint 183, found via live multi-session QA on a genuinely shared, multi-tenant host: before the fix, one project's own session activity could silently starve or spuriously trigger a completely unrelated project's own consolidation cycle, because both read and wrote the exact same global state files.

Tune the thresholds with `~/.memoria/opencode-plugin/dream-config.json`:

```json
{ "enabled": true, "minHours": 24, "minSessions": 5, "minMemories": 20 }
```

Or disable entirely with `MEMORIA_DREAM=false`.

## Verification

Real, not simulated: verified against an actually-installed OpenCode instance (`opencode mcp list` / `opencode run`), a real local Ollama-backed provider, and a real running `server`. Confirmed `POST /search` and `POST /memories` both fire for real from the auto-capture and recall hooks; confirmed the consolidation gate's `GET /memories` call fires and session counting/lock acquire/release behaves correctly across real, separate `opencode run` invocations — including the lock being correctly released (not converted to a completion record) when no real write was observed. See `Backlog.adoc` Sprint 139 for the one real bug live verification caught and fixed there (`extractUserText`'s crash on non-array input); see Sprint 183 for a second, this time found by running genuinely concurrent `opencode run` sessions on a real shared host already running 13+ unrelated real projects through this same plugin — the global dream-state bug described above, confirmed live: a fresh per-scope session counter (`sessionsSince`) tracked only this project's own real session count after the fix, where before it jumped by amounts attributable to other, unrelated projects' concurrent activity.
