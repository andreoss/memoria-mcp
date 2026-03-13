import { afterEach, describe, expect, test } from "bun:test";
import { MemoriaClient } from "./client";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function fakeFetch(status: number, body: unknown): typeof fetch {
  return (async () =>
    new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })) as unknown as typeof fetch;
}

describe("MemoriaClient.addMemory", () => {
  test("posts to /memories with scope and infer, returns the real ids", async () => {
    let capturedUrl = "";
    let capturedBody: any;
    globalThis.fetch = (async (url: string, init: RequestInit) => {
      capturedUrl = url;
      capturedBody = JSON.parse(init.body as string);
      return new Response(JSON.stringify({ ids: ["rec-1"] }), { status: 201 });
    }) as unknown as typeof fetch;

    const client = new MemoriaClient({ baseUrl: "http://127.0.0.1:8080", apiKey: "test-key" });
    const ids = await client.addMemory("the sky is blue", { userId: "alice", agentId: "proj" }, false);

    expect(ids).toEqual(["rec-1"]);
    expect(capturedUrl).toBe("http://127.0.0.1:8080/memories");
    expect(capturedBody).toEqual({ content: "the sky is blue", infer: false, user_id: "alice", agent_id: "proj" });
  });

  test("throws with the real status and body on a non-ok response", async () => {
    globalThis.fetch = fakeFetch(400, { error: "validation error" });
    const client = new MemoriaClient({ baseUrl: "http://127.0.0.1:8080" });
    await expect(client.addMemory("x", { userId: "alice" }, true)).rejects.toThrow(/400/);
  });
});

describe("MemoriaClient.searchMemories", () => {
  test("posts to /search and returns the real results", async () => {
    globalThis.fetch = fakeFetch(200, { results: [{ id: "rec-1", score: 0.1, payload: { content: "hi" } }] });
    const client = new MemoriaClient({ baseUrl: "http://127.0.0.1:8080" });
    const results = await client.searchMemories("hi", { userId: "alice" }, 5);
    expect(results).toEqual([{ id: "rec-1", score: 0.1, payload: { content: "hi" } }]);
  });
});

describe("MemoriaClient.countMemories", () => {
  test("gets /memories with the real scope query params and returns the id count", async () => {
    let capturedUrl = "";
    globalThis.fetch = (async (url: string) => {
      capturedUrl = url;
      return new Response(JSON.stringify({ ids: ["rec-1", "rec-2"] }), { status: 200 });
    }) as unknown as typeof fetch;

    const client = new MemoriaClient({ baseUrl: "http://127.0.0.1:8080" });
    const count = await client.countMemories({ userId: "alice" });

    expect(count).toBe(2);
    expect(capturedUrl).toContain("user_id=alice");
    expect(capturedUrl).toContain("limit=10000");
  });
});
