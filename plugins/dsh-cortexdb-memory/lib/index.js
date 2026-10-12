// CortexDB memory for DeepSeek Harness.
//
// Three things, on dsh's own extension points:
//   - recall: before each prompt a person types, the brain is asked about it
//     and what it returns is attached for the model (agent/pre-step);
//   - capture: when a session has been quiet for a while, or ends, its durable
//     facts become memories (agent/turn-stopping, session/disposed);
//   - tools: cortexdb_memory_search / _store / _forget for the model to use
//     on purpose.
//
// The brain is a CortexDB gRPC sidecar — the shared one when `endpoint` names
// it, else one this plugin starts against the local database. Recall reads the
// whole brain unless scope/namespace/userId narrow it, so dsh sees what Claude
// Code, Codex and every other agent on the brain saved, and they see dsh's.

import z from "@deepseek-ai/schemastery";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";
import { createBackend, defaultDataDir, mcpBinary } from "./backend.js";
import { captureConfigured, createCapturer } from "./capture.js";
import { formatRecall, recallArgs, recallHits } from "./recall.js";

export const name = "cortexdb-memory";
export const inject = ["tools"];

export const Config = z.object({
  endpoint: z.string().default(process.env.CORTEXDB_REMOTE || process.env.CORTEXDB_GRPC_ENDPOINT || "127.0.0.1:47821")
    .description("CortexDB gRPC sidecar, host:port. A shared brain goes here."),
  token: z.string().default(process.env.CORTEXDB_GRPC_TOKEN || "").description("Bearer token for the sidecar, if it requires one."),
  autoStart: z.boolean().default(true).description("Start a sidecar on the local default endpoint when none answers."),
  dbPath: z.string().default(process.env.CORTEXDB_PATH || join(homedir(), ".cortexdb", "cortexdb.db")).description("Database a started sidecar opens."),
  binaryPath: z.string().default(process.env.CORTEXDB_GRPC_BIN || "").description("cortexdb-grpc to start instead of the release download."),
  mcpBinaryPath: z.string().default(process.env.CORTEXDB_MCP_BIN || "").description("cortexdb-mcp that runs captures instead of the release download."),
  dataDir: z.string().default(defaultDataDir()).description("Where downloaded binaries and capture exports live."),
  userId: z.string().default("").description("Narrow recall to one user. Empty reads the whole brain."),
  scope: z.string().default("").description("Narrow recall to one scope, and store memories in it (default global)."),
  namespace: z.string().default("").description("Narrow recall to one namespace, and store memories in it."),
  autoRecall: z.boolean().default(true).description("Recall before each prompt a person types."),
  recallTopK: z.natural().min(1).max(20).default(3).description("Memories and knowledge hits per automatic recall."),
  recallTimeoutMs: z.natural().min(100).default(4000).description("An automatic recall slower than this is skipped, never waited on."),
  callTimeoutMs: z.natural().min(100).default(30000).description("Timeout for the memory tools."),
  autoCapture: z.boolean().default(true).description("Distil durable facts from sessions into memories. Needs a model: CORTEXDB_LLM_* or captureLlm*."),
  captureIdleMinutes: z.natural().default(10).description("Capture a session this long after its last turn; 0 only captures when it ends."),
  captureLlmBaseUrl: z.string().default("").description("Chat endpoint the capture distils with (CORTEXDB_LLM_BASE_URL)."),
  captureLlmModel: z.string().default("").description("Model the capture distils with (CORTEXDB_LLM_MODEL)."),
  captureLlmApiKeyEnv: z.string().default("").description("Environment variable holding that endpoint's API key."),
});

const CONTEXT_SOURCE = { kind: "cortexdb-memory" };

function textOf(messages) {
  return messages
    .filter((message) => message?.source?.kind === "user")
    .flatMap((message) => message.content ?? [])
    .filter((block) => block?.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();
}

const hitSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    id: { type: "string", required: true },
    title: { type: "string", required: true },
    snippet: { type: "string", required: true },
    score: { type: "number", required: true },
    source: { type: "string", required: true },
  },
};

export function apply(ctx, config) {
  const backend = createBackend(config);
  let captureBinary = config.mcpBinaryPath || "";
  const capturer = createCapturer(config, { binary: () => captureBinary, logger: ctx.logger });
  const idle = new Map(); // session id -> pending idle capture
  const live = new Map(); // session id -> session, until it is captured at the end

  function captureAll() {
    for (const session of live.values()) {
      try {
        capturer.capture(session, { local: backend.sidecar.state.managed });
      } catch (error) {
        ctx.logger.warn(`cortexdb-memory: capture failed: ${error.message}`);
      }
    }
    live.clear();
  }

  // A headless run ends by disposing this plugin, not its session: what was
  // said in it is captured here, before the sidecar stops.
  ctx.effect(() => () => {
    for (const timer of idle.values()) clearTimeout(timer);
    idle.clear();
    if (config.autoCapture && captureConfigured(config)) captureAll();
    backend.close().catch(() => {});
  }, "cortexdb-memory: capture open sessions, stop timers and the sidecar");

  backend.ready().then(
    () => ctx.logger.info(`cortexdb-memory: connected to ${config.endpoint}${backend.sidecar.state.managed ? " (sidecar started by this plugin)" : ""}`),
    (error) => ctx.logger.warn(`cortexdb-memory: ${error.message}`),
  );
  if (config.autoCapture) capturer.prune();

  async function recall(prompt, overrides = {}) {
    const args = { ...recallArgs(prompt, config), ...overrides };
    return backend.call("knowledge_memory_recall", args, overrides.timeoutMs ?? config.callTimeoutMs);
  }

  if (config.autoRecall) {
    ctx.on("agent/pre-step", async ({ messages }, next) => {
      const prompt = textOf(messages ?? []);
      if (!prompt) return next();
      let block = "";
      try {
        const args = recallArgs(prompt, config);
        block = formatRecall(recallHits(await backend.call("knowledge_memory_recall", args, config.recallTimeoutMs)));
      } catch (error) {
        ctx.logger.warn(`cortexdb-memory: recall skipped: ${error.message}`);
      }
      const downstream = await next();
      if (!block || downstream.kind !== "enter") return downstream;
      return {
        ...downstream,
        messages: [...downstream.messages, createUserMessage({ content: [{ type: "text", text: block }], source: CONTEXT_SOURCE })],
      };
    });
  }

  if (config.autoCapture) {
    if (!captureConfigured(config)) {
      ctx.logger.info("cortexdb-memory: session capture is on but has no model; set CORTEXDB_LLM_BASE_URL or captureLlmBaseUrl to enable it");
    } else {
      if (!captureBinary) {
        mcpBinary(config).then(
          (path) => { captureBinary = path; },
          (error) => ctx.logger.warn(`cortexdb-memory: session capture unavailable: ${error.message}`),
        );
      }
      const run = (session, options) => {
        try {
          capturer.capture(session, options);
        } catch (error) {
          ctx.logger.warn(`cortexdb-memory: capture failed: ${error.message}`);
        }
      };

      ctx.on("agent/turn-stopping", ({ agent }) => {
        const session = agent?.session;
        if (!session) return;
        const id = String(session.header?.id ?? session.id);
        live.set(id, session);
        if (config.captureIdleMinutes <= 0) return;
        clearTimeout(idle.get(id));
        const timer = setTimeout(() => {
          idle.delete(id);
          run(session);
        }, config.captureIdleMinutes * 60_000);
        timer.unref?.();
        idle.set(id, timer);
      });

      ctx.on("session/disposed", (session) => {
        const id = String(session.header?.id ?? session.id);
        clearTimeout(idle.get(id));
        idle.delete(id);
        live.delete(id);
        // A sidecar this plugin started stops with it; the capture then
        // writes to the database file instead of a sidecar that is gone.
        run(session, { local: backend.sidecar.state.managed });
      });
    }
  }

  ctx.tools.register(defineTool({
    name: "cortexdb_memory_search",
    description: "Search CortexDB — the long-term memory and knowledge shared with your other agents — for what was decided, preferred, or learned before. Use it before answering questions about prior decisions, people, preferences or project history.",
    parameters: {
      query: { type: "string", required: true, description: "What to look for, in natural language." },
      limit: { type: "integer", description: "How many memories and knowledge hits to return (default 8)." },
    },
    output: {
      schema: {
        type: "object",
        additionalProperties: false,
        properties: {
          context: { type: "string", required: true },
          hits: { type: "array", required: true, items: hitSchema },
        },
      },
      render: (_args, value) => [{
        type: "text",
        text: value.context || formatRecall(value.hits) || "No CortexDB memories or knowledge matched.",
      }],
    },
    isConcurrencySafe: () => true,
    async execute(args) {
      const limit = Math.min(Math.max(Number(args.limit) || 8, 1), 50);
      const payload = await recall(args.query, { top_k_memories: limit, top_k_knowledge: limit, graph_light: false });
      return { context: String(payload?.context_pack?.text ?? ""), hits: recallHits(payload) };
    },
    presentCall: (args) => ({ card: "generic", title: `Search memory: ${args.query}`, kind: "search", rawInput: args }),
  }));

  ctx.tools.register(defineTool({
    name: "cortexdb_memory_store",
    description: "Save one durable fact, preference or decision to CortexDB so later sessions — yours and your other agents' — can recall it. One fact per call, stated so it can be understood without this conversation. When it replaces something an earlier memory said, name that memory in supersedes.",
    parameters: {
      content: { type: "string", required: true, description: "The fact, self-contained." },
      importance: { type: "number", description: "0 to 1; how much it matters later (default 0.6)." },
      supersedes: { type: "array", items: { type: "string" }, description: "IDs of memories this one makes untrue." },
    },
    output: {
      schema: {
        type: "object",
        additionalProperties: false,
        properties: { id: { type: "string", required: true } },
      },
      render: (_args, value) => [{ type: "text", text: `Saved to CortexDB as ${value.id}.` }],
    },
    async execute(args, exec) {
      const id = `dsh:${randomUUID()}`;
      const sessionId = exec?.agent?.session?.header?.id;
      await backend.call("memory_save", {
        memory_id: id,
        scope: config.scope || "global",
        ...(config.namespace ? { namespace: config.namespace } : {}),
        ...(config.userId ? { user_id: config.userId } : {}),
        content: args.content,
        importance: args.importance ?? 0.6,
        ...(args.supersedes?.length ? { supersedes: args.supersedes } : {}),
        metadata: { source: "dsh", agent: "dsh", ...(sessionId ? { session: String(sessionId) } : {}) },
      });
      return { id };
    },
    presentCall: (args) => ({ card: "generic", title: "Save to memory", kind: "other", rawInput: args }),
  }));

  ctx.tools.register(defineTool({
    name: "cortexdb_memory_forget",
    description: "Delete one CortexDB memory by its exact ID. Prefer cortexdb_memory_store with supersedes when a fact has changed: that keeps the history.",
    parameters: {
      id: { type: "string", required: true, description: "The memory ID, as search returned it." },
    },
    output: {
      schema: {
        type: "object",
        additionalProperties: false,
        properties: { id: { type: "string", required: true } },
      },
      render: (_args, value) => [{ type: "text", text: `Deleted ${value.id} from CortexDB.` }],
    },
    async execute(args) {
      await backend.call("memory_delete", { memory_id: args.id });
      return { id: args.id };
    },
    presentCall: (args) => ({ card: "generic", title: `Forget ${args.id}`, kind: "delete", rawInput: args }),
  }));
}
