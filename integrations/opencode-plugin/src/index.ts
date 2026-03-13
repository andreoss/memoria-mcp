import type { Plugin } from "@opencode-ai/plugin";
import { randomUUID } from "node:crypto";
import { userInfo } from "node:os";
import { MemoriaClient, type MemoriaScope } from "./client";
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

  const pendingContext = new Map<string, string>();

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
      const query = extractUserText(output.parts);
      if (!query) return;
      try {
        const results = await memoria.searchMemories(query, scope, 5);
        const context = results
          .map((result) => result.payload.content)
          .filter((content): content is string => Boolean(content))
          .join("\n");
        if (context) pendingContext.set(output.message.sessionID, context);
      } catch {
        return;
      }
    },

    "experimental.chat.messages.transform": async (_input, output) => {
      const last = output.messages[output.messages.length - 1];
      if (!last) return;
      const context = pendingContext.get(last.info.sessionID);
      if (!context) return;
      pendingContext.delete(last.info.sessionID);
      last.parts.push({
        id: randomUUID(),
        sessionID: last.info.sessionID,
        messageID: last.info.id,
        type: "text",
        text: `<memoria-context>\n${context}\n</memoria-context>`,
        synthetic: true,
      });
    },
  };
};

export default MemoriaPlugin;
