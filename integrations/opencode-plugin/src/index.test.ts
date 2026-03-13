import { describe, expect, test } from "bun:test";
import { extractUserText, resolveServerUrl, resolveUserId } from "./index";

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
