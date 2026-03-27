import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { extractUserText, resolveServerUrl, resolveUserId, scopeStateKey } from "./index";
import { acquireDreamLock, incrementSessionCount, checkCheapGates, DREAM_DEFAULTS } from "./dream";

describe("resolveUserId", () => {
  test("prefers MEMORIA_OPENCODE_USER_ID when set", () => {
    expect(resolveUserId({ MEMORIA_OPENCODE_USER_ID: "alice", USER: "bob" } as NodeJS.ProcessEnv)).toBe("alice");
  });
});

describe("resolveServerUrl", () => {
  test("defaults to the local server", () => {
    expect(resolveServerUrl({} as NodeJS.ProcessEnv)).toBe("http://127.0.0.1:8080");
  });

  test("prefers MEMORIA_SERVER_URL when set", () => {
    expect(resolveServerUrl({ MEMORIA_SERVER_URL: "http://example.internal:9090" } as NodeJS.ProcessEnv)).toBe(
      "http://example.internal:9090",
    );
  });
});

describe("extractUserText", () => {
  test("joins text parts and ignores non-text parts", () => {
    const parts = [
      { type: "text", text: "first line" },
      { type: "tool" },
      { type: "text", text: "second line" },
    ];
    expect(extractUserText(parts)).toBe("first line\nsecond line");
  });

  test("returns an empty string when there are no text parts", () => {
    expect(extractUserText([{ type: "tool" }])).toBe("");
  });

  test("trims surrounding whitespace", () => {
    expect(extractUserText([{ type: "text", text: "  padded  " }])).toBe("padded");
  });

  test("returns an empty string instead of throwing when parts is not an array", () => {
    expect(extractUserText(undefined)).toBe("");
    expect(extractUserText(null)).toBe("");
  });
});

describe("scopeStateKey", () => {
  test("is deterministic for the same scope", () => {
    const a = scopeStateKey({ userId: "alice", agentId: "proj" });
    const b = scopeStateKey({ userId: "alice", agentId: "proj" });
    expect(a).toBe(b);
  });

  test("differs across distinct projects for the same user", () => {
    const a = scopeStateKey({ userId: "alice", agentId: "proj-one" });
    const b = scopeStateKey({ userId: "alice", agentId: "proj-two" });
    expect(a).not.toBe(b);
  });

  test("differs across distinct users for the same project", () => {
    const a = scopeStateKey({ userId: "alice", agentId: "proj" });
    const b = scopeStateKey({ userId: "bob", agentId: "proj" });
    expect(a).not.toBe(b);
  });
});

describe("dream state isolation across scopes (the real Sprint 183 regression: a shared host running many unrelated projects must not let one project's dream cycle contend with another's)", () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "memoria-scope-isolation-test-"));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function stateDirFor(scope: { userId: string; agentId: string }): string {
    return join(root, "state", scopeStateKey(scope));
  }

  test("a dream lock held for one project's scope does not block a different project's scope", () => {
    const projectA = stateDirFor({ userId: "a", agentId: "project-a" });
    const projectB = stateDirFor({ userId: "a", agentId: "project-b" });
    expect(acquireDreamLock(projectA)).toBe(true);
    expect(acquireDreamLock(projectB)).toBe(true);
  });

  test("session-count progress in one project's scope does not leak into a different project's scope", () => {
    const projectA = stateDirFor({ userId: "a", agentId: "project-a" });
    const projectB = stateDirFor({ userId: "a", agentId: "project-b" });
    for (let i = 0; i < DREAM_DEFAULTS.minSessions; i++) {
      incrementSessionCount(projectA, `session-${i}`);
    }
    expect(checkCheapGates(projectA, DREAM_DEFAULTS).proceed).toBe(true);
    expect(checkCheapGates(projectB, DREAM_DEFAULTS).proceed).toBe(false);
  });
});
