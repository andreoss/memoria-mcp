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

Project identity (memoria's `agent_id`) is derived automatically: the git remote's `owner-repo`, falling back to the repo root directory name, then the working directory name — the same fallback chain used by every real prior-art OpenCode memory plugin this design was compared against.

## What it does

- **`shell.env`** — exports the resolved `MEMORIA_OPENCODE_USER_ID`/`MEMORIA_OPENCODE_PROJECT_ID` into every shell command the agent runs.
- **`event`** (on a real `session.idle` event) — fetches the session's recent user messages via the OpenCode SDK client and stores them with `POST /memories` (`infer: true`, so whichever LLM provider `server` is configured with does the real fact extraction).
- **`chat.message`** — searches memoria (`POST /search`) for context relevant to the incoming message.
- **`experimental.chat.messages.transform`** — injects that search's results into the message actually sent to the model, wrapped in a `<memoria-context>` block and marked `synthetic: true`.

## What it deliberately does not do (yet)

Memory consolidation ("dream" — merging duplicates, dropping stale entries on a schedule) is `ADR-42` Increment 2b, a separate, not-yet-built piece with its own gating/state-file surface.

## Verification

Real, not simulated: verified against an actually-installed OpenCode instance (`opencode mcp list` / `opencode run`), a real local Ollama-backed provider, and a real running `server` — confirmed `POST /search` and `POST /memories` both fire for real, and the session-idle capture path actually stores a real record scoped to the resolved user/project identity. See `Backlog.adoc` Sprint 139 for the full account, including a real bug this verification caught and fixed (`extractUserText` crashed on a non-array `parts` value the type signature claimed could never occur).
