import type { PluginInput } from "@opencode-ai/plugin";
import { basename } from "node:path";

type BunShell = PluginInput["$"];

export function parseProjectFromRemote(remote: string): string | null {
  const trimmed = remote.trim();
  if (!trimmed) return null;
  const match = trimmed.match(/[:/]([^/:]+)\/([^/:]+?)(?:\.git)?\/?$/);
  if (!match) return null;
  return `${match[1]}-${match[2]}`;
}

export async function resolveProjectId($: BunShell, cwd: string): Promise<string> {
  const remote = await $`git remote get-url origin`.nothrow().quiet().text();
  const fromRemote = parseProjectFromRemote(remote);
  if (fromRemote) return fromRemote;

  const root = (await $`git rev-parse --show-toplevel`.nothrow().quiet().text()).trim();
  if (root) return basename(root);

  return basename(cwd);
}
