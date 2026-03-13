import type { Plugin } from "@opencode-ai/plugin";
import { randomUUID } from "node:crypto";
import { homedir, userInfo } from "node:os";
import { join } from "node:path";
import { MemoriaClient, type MemoriaScope } from "./client";
import {
  acquireDreamLock,
  buildDreamProtocol,
  checkCheapGates,
  checkMemoryGate,
  incrementSessionCount,
  loadDreamConfig,
  recordDreamCompletion,
  releaseDreamLock,
} from "./dream";
import { resolveProjectId } from "./project";

export function resolveUserId(env: NodeJS.ProcessEnv): string {
  if (env.MEMORIA_OPENCODE_USER_ID) return env.MEMORIA_OPENCODE_USER_ID;
  try {
    return userInfo().username;
  } catch {
    return env.USER || env.USERNAME || "unknown";
  }
}

export function resolveServerUrl(env: NodeJS.ProcessEnv): string {
  return env.MEMORIA_SERVER_URL || "http://127.0.0.1:8080";
}

export function resolveMcpToolPrefix(env: NodeJS.ProcessEnv): string {
  return `${env.MEMORIA_OPENCODE_MCP_NAME || "memoria"}_`;
}

export function extractUserText(parts: Array<{ type: string; text?: string }> | undefined | null): string {
  if (!Array.isArray(parts)) return "";
  return parts
    .filter((part) => part.type === "text" && typeof part.text === "string")
    .map((part) => part.text as string)
    .join("\n")
    .trim();
}

const MemoriaPlugin: Plugin = async (ctx) => {
  const { $, client, directory } = ctx;
  const baseUrl = resolveServerUrl(process.env);
  const apiKey = process.env.MEMORIA_API_KEY;
  const memoria = new MemoriaClient({ baseUrl, apiKey });
  const userId = resolveUserId(process.env);
  const projectId = await resolveProjectId($, directory);
  const scope: MemoriaScope = { userId, agentId: projectId };
  const mcpToolPrefix = resolveMcpToolPrefix(process.env);

  const stateDir = join(homedir(), ".memoria", "opencode-plugin");
  const dreamConfig = loadDreamConfig(stateDir, process.env);

  const pendingContext = new Map<string, string>();
  const pendingDream = new Map<string, string>();
  const checkedSessions = new Set<string>();
  const dreamActiveSessions = new Set<string>();
  let dreamWriteSeen = false;

  const writeToolNames = new Set([`${mcpToolPrefix}add_memory`, `${mcpToolPrefix}update_memory`, `${mcpToolPrefix}delete_memory`]);

  return {
    "shell.env": async (_input, output) => {
      output.env.MEMORIA_OPENCODE_USER_ID = userId;
      output.env.MEMORIA_OPENCODE_PROJECT_ID = projectId;
    },

    event: async (input) => {
      if (input.event.type !== "session.idle") return;
      const sessionID = input.event.properties.sessionID;
      const result = await client.session.messages({ path: { id: sessionID }, query: { limit: 20 } });
      if (!result.data) return;
      const text = result.data
        .filter((entry) => entry.info.role === "user")
        .flatMap((entry) => extractUserText(entry.parts))
        .filter(Boolean)
        .join("\n")
        .trim();
      if (!text) return;
      try {
        await memoria.addMemory(text, scope, true);
      } catch {
        return;
      }
    },

    "chat.message": async (_input, output) => {
      const sessionID = output.message.sessionID;
      const query = extractUserText(output.parts);

      if (dreamConfig.enabled && !checkedSessions.has(sessionID)) {
        checkedSessions.add(sessionID);
        incrementSessionCount(stateDir, sessionID);
        try {
          const memoryCount = await memoria.countMemories(scope);
          const gates = checkCheapGates(stateDir, dreamConfig);
          const memGate = checkMemoryGate(memoryCount, dreamConfig);
          if (gates.proceed && memGate.pass && acquireDreamLock(stateDir)) {
            dreamActiveSessions.add(sessionID);
            pendingDream.set(sessionID, buildDreamProtocol(mcpToolPrefix));
          }
        } catch {}
      }

      if (!query) return;
      try {
        const results = await memoria.searchMemories(query, scope, 5);
        const context = results
          .map((result) => result.payload.content)
          .filter((content): content is string => Boolean(content))
          .join("\n");
        if (context) pendingContext.set(sessionID, context);
      } catch {
        return;
      }
    },

    "tool.execute.after": async (input) => {
      if (dreamActiveSessions.has(input.sessionID) && writeToolNames.has(input.tool)) {
        dreamWriteSeen = true;
      }
    },

    "experimental.chat.messages.transform": async (_input, output) => {
      const last = output.messages[output.messages.length - 1];
      if (!last) return;
      const sessionID = last.info.sessionID;
      const blocks: string[] = [];
      const context = pendingContext.get(sessionID);
      if (context) {
        pendingContext.delete(sessionID);
        blocks.push(`<memoria-context>\n${context}\n</memoria-context>`);
      }
      const dream = pendingDream.get(sessionID);
      if (dream) {
        pendingDream.delete(sessionID);
        blocks.push(dream);
      }
      if (blocks.length === 0) return;
      last.parts.push({
        id: randomUUID(),
        sessionID,
        messageID: last.info.id,
        type: "text",
        text: blocks.join("\n\n"),
        synthetic: true,
      });
    },

    dispose: async () => {
      if (dreamWriteSeen) {
        recordDreamCompletion(stateDir);
      } else {
        releaseDreamLock(stateDir);
      }
    },
  };
};

export default MemoriaPlugin;
