import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export type DreamConfig = {
  enabled: boolean;
  minHours: number;
  minSessions: number;
  minMemories: number;
};

export const DREAM_DEFAULTS: DreamConfig = {
  enabled: true,
  minHours: 24,
  minSessions: 5,
  minMemories: 20,
};

type DreamState = {
  lastConsolidatedAt: number;
  sessionsSince: number;
  lastSessionId: string | null;
};

type DreamLock = {
  pid: number;
  startedAt: number;
};

const LOCK_STALE_MS = 60 * 60 * 1000;

function ensureDir(dir: string): void {
  try {
    mkdirSync(dir, { recursive: true });
  } catch {
    return;
  }
}

function statePath(stateDir: string): string {
  return join(stateDir, "dream-state.json");
}

function lockPath(stateDir: string): string {
  return join(stateDir, "dream.lock");
}

function configPath(stateDir: string): string {
  return join(stateDir, "dream-config.json");
}

function readState(stateDir: string): DreamState {
  try {
    return JSON.parse(readFileSync(statePath(stateDir), "utf8")) as DreamState;
  } catch {
    return { lastConsolidatedAt: 0, sessionsSince: 0, lastSessionId: null };
  }
}

function writeState(stateDir: string, state: DreamState): void {
  ensureDir(stateDir);
  writeFileSync(statePath(stateDir), JSON.stringify(state, null, 2));
}

export function loadDreamConfig(stateDir: string, env: NodeJS.ProcessEnv): DreamConfig {
  let config: DreamConfig = { ...DREAM_DEFAULTS };
  try {
    if (existsSync(configPath(stateDir))) {
      const parsed = JSON.parse(readFileSync(configPath(stateDir), "utf8"));
      config = {
        enabled: typeof parsed.enabled === "boolean" ? parsed.enabled : config.enabled,
        minHours: typeof parsed.minHours === "number" ? parsed.minHours : config.minHours,
        minSessions: typeof parsed.minSessions === "number" ? parsed.minSessions : config.minSessions,
        minMemories: typeof parsed.minMemories === "number" ? parsed.minMemories : config.minMemories,
      };
    }
  } catch {
    config = { ...DREAM_DEFAULTS };
  }
  if (env.MEMORIA_DREAM !== undefined) {
    const value = env.MEMORIA_DREAM.toLowerCase();
    config.enabled = value !== "false" && value !== "0" && value !== "no" && value !== "off";
  }
  return config;
}

export function incrementSessionCount(stateDir: string, sessionId: string): void {
  const state = readState(stateDir);
  if (state.lastSessionId !== sessionId) {
    state.sessionsSince += 1;
    state.lastSessionId = sessionId;
    writeState(stateDir, state);
  }
}

export function checkCheapGates(stateDir: string, config: DreamConfig): { proceed: boolean; reason?: string } {
  const state = readState(stateDir);
  const hoursSince = (Date.now() - state.lastConsolidatedAt) / 3_600_000;
  if (hoursSince < config.minHours) {
    return { proceed: false, reason: `time: ${hoursSince.toFixed(1)}h < ${config.minHours}h` };
  }
  if (state.sessionsSince < config.minSessions) {
    return { proceed: false, reason: `sessions: ${state.sessionsSince} < ${config.minSessions}` };
  }
  return { proceed: true };
}

export function checkMemoryGate(memoryCount: number, config: DreamConfig): { pass: boolean; reason?: string } {
  if (memoryCount < config.minMemories) {
    return { pass: false, reason: `memories: ${memoryCount} < ${config.minMemories}` };
  }
  return { pass: true };
}

function createLockFile(stateDir: string): boolean {
  ensureDir(stateDir);
  const lock: DreamLock = { pid: process.pid, startedAt: Date.now() };
  try {
    writeFileSync(lockPath(stateDir), JSON.stringify(lock), { flag: "wx" });
    return true;
  } catch {
    return false;
  }
}

export function acquireDreamLock(stateDir: string): boolean {
  ensureDir(stateDir);
  const path = lockPath(stateDir);
  try {
    const existing = JSON.parse(readFileSync(path, "utf8")) as DreamLock;
    if (Date.now() - existing.startedAt < LOCK_STALE_MS) {
      return false;
    }
    try {
      unlinkSync(path);
    } catch {
      return false;
    }
  } catch {
    return createLockFile(stateDir);
  }
  return createLockFile(stateDir);
}

export function releaseDreamLock(stateDir: string): void {
  try {
    unlinkSync(lockPath(stateDir));
  } catch {
    return;
  }
}

export function recordDreamCompletion(stateDir: string): void {
  writeState(stateDir, { lastConsolidatedAt: Date.now(), sessionsSince: 0, lastSessionId: null });
}

export function buildDreamProtocol(mcpToolPrefix: string): string {
  const get = `${mcpToolPrefix}get_memories`;
  const add = `${mcpToolPrefix}add_memory`;
  const del = `${mcpToolPrefix}delete_memory`;
  return `<memoria-dream>
You are running memory consolidation. Complete these steps using the ${get}, ${add}, and ${del} tools:

1. ORIENT — Call ${get} to list stored memories in scope. Note the total count.

2. GATHER TARGETS — Review each memory. Classify as:
   - DELETE: sensitive information (API keys, passwords, tokens), duplicated or stale entries
   - MERGE: near-duplicates (same fact stated differently) — keep the better-worded one, delete the other
   - REWRITE: vague or poorly-worded entries — ${add} the improved text (infer: false), then ${del} the original
   - KEEP: everything else

3. CONSOLIDATE — Execute the changes using ${del}/${add} as classified above.

4. REPORT — Briefly summarize how many were reviewed, deleted, merged, and rewritten, then answer the user's actual message normally.
</memoria-dream>`;
}
