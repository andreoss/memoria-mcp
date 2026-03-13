export type MemoriaScope = {
  userId?: string;
  agentId?: string;
  runId?: string;
};

export type MemoriaSearchResult = {
  id: string;
  score: number;
  payload: Record<string, string>;
};

export type MemoriaClientConfig = {
  baseUrl: string;
  apiKey?: string;
};

function scopeBody(scope: MemoriaScope): Record<string, string> {
  const body: Record<string, string> = {};
  if (scope.userId) body.user_id = scope.userId;
  if (scope.agentId) body.agent_id = scope.agentId;
  if (scope.runId) body.run_id = scope.runId;
  return body;
}

export class MemoriaClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;

  constructor(config: MemoriaClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.apiKey = config.apiKey;
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
    return headers;
  }

  async addMemory(content: string, scope: MemoriaScope, infer: boolean): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/memories`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ content, infer, ...scopeBody(scope) }),
    });
    if (!response.ok) {
      throw new Error(`memoria add_memory failed: ${response.status} ${await response.text()}`);
    }
    const data = (await response.json()) as { ids: string[] };
    return data.ids;
  }

  async searchMemories(query: string, scope: MemoriaScope, topK: number): Promise<MemoriaSearchResult[]> {
    const response = await fetch(`${this.baseUrl}/search`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ query, top_k: topK, ...scopeBody(scope) }),
    });
    if (!response.ok) {
      throw new Error(`memoria search_memories failed: ${response.status} ${await response.text()}`);
    }
    const data = (await response.json()) as { results: MemoriaSearchResult[] };
    return data.results;
  }

  async countMemories(scope: MemoriaScope): Promise<number> {
    const params = new URLSearchParams({ ...scopeBody(scope), limit: "10000" });
    const response = await fetch(`${this.baseUrl}/memories?${params.toString()}`, { headers: this.headers() });
    if (!response.ok) {
      throw new Error(`memoria get_memories failed: ${response.status} ${await response.text()}`);
    }
    const data = (await response.json()) as { ids: string[] };
    return data.ids.length;
  }
}
