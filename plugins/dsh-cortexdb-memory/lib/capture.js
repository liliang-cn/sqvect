// Session capture: when a session goes quiet or ends, the durable facts in it
// become memories — the same `cortexdb-mcp --capture-session` that distils a
// Claude Code or Codex session, writing the same records under auto:dsh-<id>:.
//
// The conversation is exported from the session in memory rather than read
// from dsh's own log: the log's layout, compression and flush timing belong to
// dsh's persistence plugin, and a capture that raced its final write would
// read a session without its last turn. The export is the log's shape with
// only the conversation in it, so the reader is the one `import_agent_memory`
// uses on dsh's own files.

import { spawn } from "node:child_process";
import { closeSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import { readdir, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { SessionSeq } from "@deepseek-ai/dsh-session";

// MIN_USER_TURNS matches the capture itself: a shorter exchange holds nothing
// worth a model call.
export const MIN_USER_TURNS = 3;

const KEEP_EXPORTS_MS = 7 * 24 * 60 * 60 * 1000;

function isPersonTyping(event) {
  return event?.type === "user/message" && event.data?.source?.kind === "user";
}

// exportConversation returns the session's conversation as JSONL — a header
// line, then its user/message and assistant/message events — and how many
// user turns it holds.
export function exportConversation(session) {
  const header = session.header ?? {};
  const lines = [JSON.stringify({ type: "session", version: 4, id: header.id ?? session.id, cwd: header.cwd })];
  let userTurns = 0;
  for (let seq = 0; seq < session.seq; seq += 1) {
    const event = session.eventAt(SessionSeq(seq));
    if (event?.type !== "user/message" && event?.type !== "assistant/message") continue;
    if (isPersonTyping(event)) userTurns += 1;
    lines.push(JSON.stringify({ type: event.type, data: event.data }));
  }
  return { jsonl: `${lines.join("\n")}\n`, userTurns };
}

// captureEnv is the environment the capture runs in: the brain to write to,
// and the model that distils the facts.
export function captureEnv(config, { local }, env = process.env) {
  const out = { ...env };
  if (local) {
    delete out.CORTEXDB_REMOTE;
    out.CORTEXDB_PATH = config.dbPath;
  } else {
    out.CORTEXDB_REMOTE = config.endpoint;
    if (config.token) out.CORTEXDB_GRPC_TOKEN = config.token;
  }
  if (config.captureLlmBaseUrl) out.CORTEXDB_LLM_BASE_URL = config.captureLlmBaseUrl;
  if (config.captureLlmModel) out.CORTEXDB_LLM_MODEL = config.captureLlmModel;
  if (config.captureLlmApiKeyEnv) out.CORTEXDB_LLM_API_KEY = env[config.captureLlmApiKeyEnv] ?? "";
  return out;
}

export function captureConfigured(config, env = process.env) {
  return Boolean(config.captureLlmBaseUrl || env.CORTEXDB_LLM_BASE_URL || env.CORTEXDB_LLM_URL);
}

// createCapturer runs captures for sessions, one at a time per session, and
// only when a session has said something since its last capture.
//
// capture() is synchronous on purpose. A headless run ends by disposing the
// plugin, not the session, and a dispose callback gets no second tick: the
// export is written and the capture process started before it returns, and
// the process, detached, outlives this one. `binary` therefore returns a path
// already resolved, or nothing yet.
export function createCapturer(config, { binary, logger, spawnProcess = spawn, now = Date.now }) {
  const root = join(config.dataDir, "capture");
  const captured = new Map(); // session id -> user turns at the last capture
  const running = new Set();

  function capture(session, { local = false } = {}) {
    const { jsonl, userTurns } = exportConversation(session);
    const id = String(session.header?.id ?? session.id);
    if (userTurns < MIN_USER_TURNS || captured.get(id) === userTurns || running.has(id)) return false;
    const bin = binary();
    if (!bin) {
      logger.warn(`cortexdb-memory: capture of ${id} skipped: cortexdb-mcp is not downloaded yet`);
      return false;
    }
    const dir = join(root, id.replace(/[^A-Za-z0-9._-]/g, "_"));
    mkdirSync(dir, { recursive: true });
    const file = join(dir, "session.jsonl");
    writeFileSync(file, jsonl, { mode: 0o600 });
    const log = openSync(join(root, "capture.log"), "a");
    let child;
    try {
      child = spawnProcess(bin, ["--capture-session", file], {
        detached: true,
        stdio: ["ignore", log, log],
        windowsHide: true,
        env: captureEnv(config, { local }),
      });
    } finally {
      closeSync(log);
    }
    running.add(id);
    captured.set(id, userTurns);
    child.once?.("exit", (code) => {
      running.delete(id);
      if (code) logger.warn(`cortexdb-memory: capture of ${id} exited with code ${code}; see ${join(root, "capture.log")}`);
    });
    child.once?.("error", (error) => {
      running.delete(id);
      captured.delete(id);
      logger.warn(`cortexdb-memory: capture of ${id} could not start: ${error.message}`);
    });
    child.unref?.();
    return true;
  }

  // prune removes exports no capture has touched in a week: a session that
  // quiet is done.
  async function prune() {
    let entries;
    try {
      entries = await readdir(root, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const dir = join(root, entry.name);
      try {
        if (now() - (await stat(dir)).mtimeMs > KEEP_EXPORTS_MS) await rm(dir, { recursive: true, force: true });
      } catch {}
    }
  }

  return { capture, prune, captured, running };
}
