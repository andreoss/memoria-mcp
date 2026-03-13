import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  DREAM_DEFAULTS,
  acquireDreamLock,
  buildDreamProtocol,
  checkCheapGates,
  checkMemoryGate,
  incrementSessionCount,
  loadDreamConfig,
  recordDreamCompletion,
  releaseDreamLock,
} from "./dream";

let stateDir: string;

beforeEach(() => {
  stateDir = mkdtempSync(join(tmpdir(), "memoria-dream-test-"));
});

afterEach(() => {
  rmSync(stateDir, { recursive: true, force: true });
});

describe("loadDreamConfig", () => {
  test("returns real defaults when no config file exists", () => {
    expect(loadDreamConfig(stateDir, {} as NodeJS.ProcessEnv)).toEqual(DREAM_DEFAULTS);
  });

  test("MEMORIA_DREAM=false disables regardless of the config file", () => {
    const config = loadDreamConfig(stateDir, { MEMORIA_DREAM: "false" } as NodeJS.ProcessEnv);
    expect(config.enabled).toBe(false);
  });
});

describe("checkCheapGates", () => {
  test("a fresh state (never consolidated) still fails the sessions gate before any sessions are counted", () => {
    const result = checkCheapGates(stateDir, DREAM_DEFAULTS);
    expect(result.proceed).toBe(false);
    expect(result.reason).toContain("sessions");
  });

  test("passes once minSessions distinct sessions have been counted (time gate satisfied by a never-consolidated state)", () => {
    for (let i = 0; i < DREAM_DEFAULTS.minSessions; i++) {
      incrementSessionCount(stateDir, `session-${i}`);
    }
    expect(checkCheapGates(stateDir, DREAM_DEFAULTS).proceed).toBe(true);
  });

  test("the same sessionId repeated does not count twice", () => {
    incrementSessionCount(stateDir, "same-session");
    incrementSessionCount(stateDir, "same-session");
    incrementSessionCount(stateDir, "same-session");
    const result = checkCheapGates(stateDir, { ...DREAM_DEFAULTS, minSessions: 2 });
    expect(result.proceed).toBe(false);
  });

  test("recordDreamCompletion resets the gates", () => {
    for (let i = 0; i < DREAM_DEFAULTS.minSessions; i++) {
      incrementSessionCount(stateDir, `session-${i}`);
    }
    expect(checkCheapGates(stateDir, DREAM_DEFAULTS).proceed).toBe(true);
    recordDreamCompletion(stateDir);
    const result = checkCheapGates(stateDir, DREAM_DEFAULTS);
    expect(result.proceed).toBe(false);
    expect(result.reason).toContain("time");
  });
});

describe("checkMemoryGate", () => {
  test("fails below the threshold", () => {
    expect(checkMemoryGate(5, DREAM_DEFAULTS).pass).toBe(false);
  });

  test("passes at or above the threshold", () => {
    expect(checkMemoryGate(DREAM_DEFAULTS.minMemories, DREAM_DEFAULTS).pass).toBe(true);
  });
});

describe("acquireDreamLock / releaseDreamLock", () => {
  test("a second acquire fails while the first lock is held", () => {
    expect(acquireDreamLock(stateDir)).toBe(true);
    expect(acquireDreamLock(stateDir)).toBe(false);
  });

  test("release allows a subsequent acquire to succeed", () => {
    expect(acquireDreamLock(stateDir)).toBe(true);
    releaseDreamLock(stateDir);
    expect(acquireDreamLock(stateDir)).toBe(true);
  });
});

describe("buildDreamProtocol", () => {
  test("references the real, prefixed tool names", () => {
    const protocol = buildDreamProtocol("memoria_");
    expect(protocol).toContain("memoria_get_memories");
    expect(protocol).toContain("memoria_add_memory");
    expect(protocol).toContain("memoria_delete_memory");
  });
});
