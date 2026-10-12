import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, readdir, utimes } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { captureConfigured, captureEnv, createCapturer, exportConversation } from "../lib/capture.js";

function user(kind, text) {
  return { type: "user/message", data: { role: "user", source: { kind }, content: [{ type: "text", text }] } };
}

function assistant(text) {
  return { type: "assistant/message", data: { turn: 1, message: { role: "assistant", content: [{ type: "text", text }] } } };
}

function fakeSession(events, id = "session-3d09536c-89dd-4d52-863b-e218dc67efb9") {
  return { id, header: { id, cwd: "/src/app" }, seq: events.length, eventAt: (seq) => events[seq] };
}

const conversation = [
  { type: "turn/start", data: { turn: 1 } },
  user("user", "set up the venv"),
  user("runtime-context", "Current DSH file policy"),
  user("cortexdb-memory", "Relevant CortexDB memories..."),
  { type: "tool/result", data: { content: "x".repeat(1000) } },
  assistant("Using uv."),
  user("user", "why not pip?"),
  assistant("Debian splits venv out."),
  user("user", "ok, remember that"),
];

test("the export is the conversation, and only typed turns count", () => {
  const { jsonl, userTurns } = exportConversation(fakeSession(conversation));
  const lines = jsonl.trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(userTurns, 3);
  assert.deepEqual(lines[0], { type: "session", version: 4, id: "session-3d09536c-89dd-4d52-863b-e218dc67efb9", cwd: "/src/app" });
  assert.ok(lines.slice(1).every((l) => l.type === "user/message" || l.type === "assistant/message"));
  assert.equal(lines.length, 1 + 7);
});

test("capture writes to the sidecar, or to the file when the sidecar is going away", () => {
  const config = { endpoint: "127.0.0.1:47821", token: "t", dbPath: "/db/cortexdb.db" };
  const env = { CORTEXDB_REMOTE: "elsewhere:1", KEEP: "1" };
  const remote = captureEnv(config, { local: false }, env);
  assert.equal(remote.CORTEXDB_REMOTE, "127.0.0.1:47821");
  assert.equal(remote.CORTEXDB_GRPC_TOKEN, "t");
  const local = captureEnv(config, { local: true }, env);
  assert.equal(local.CORTEXDB_REMOTE, undefined);
  assert.equal(local.CORTEXDB_PATH, "/db/cortexdb.db");
  assert.equal(local.KEEP, "1");
});

test("the capture model comes from the config or the environment", () => {
  const config = { endpoint: "e", captureLlmBaseUrl: "https://llm/v1", captureLlmModel: "m", captureLlmApiKeyEnv: "MY_KEY" };
  const env = captureEnv(config, { local: false }, { MY_KEY: "k" });
  assert.equal(env.CORTEXDB_LLM_BASE_URL, "https://llm/v1");
  assert.equal(env.CORTEXDB_LLM_MODEL, "m");
  assert.equal(env.CORTEXDB_LLM_API_KEY, "k");
  assert.equal(captureConfigured({}, {}), false);
  assert.equal(captureConfigured({}, { CORTEXDB_LLM_BASE_URL: "x" }), true);
  assert.equal(captureConfigured({ captureLlmBaseUrl: "x" }, {}), true);
});

async function capturerFor(t) {
  const dataDir = await mkdtemp(join(tmpdir(), "cortexdb-dsh-"));
  const spawned = [];
  const children = [];
  const capturer = createCapturer({ dataDir, endpoint: "127.0.0.1:47821", dbPath: "/db" }, {
    binary: () => "/bin/cortexdb-mcp",
    logger: { warn() {} },
    spawnProcess(bin, args, options) {
      spawned.push({ bin, args, options });
      const handlers = {};
      const child = { once: (ev, fn) => { handlers[ev] = fn; }, unref() {}, exit: (code) => handlers.exit?.(code) };
      children.push(child);
      return child;
    },
  });
  return { dataDir, spawned, children, capturer };
}

test("a session is captured once it has three typed turns, and again only after more", async (t) => {
  const { dataDir, spawned, children, capturer } = await capturerFor(t);
  assert.equal(capturer.capture(fakeSession(conversation.slice(0, 7))), false, "two turns");
  assert.equal(spawned.length, 0);

  const session = fakeSession(conversation);
  assert.equal(capturer.capture(session), true);
  assert.equal(spawned[0].bin, "/bin/cortexdb-mcp");
  assert.equal(spawned[0].args[0], "--capture-session");
  assert.match(spawned[0].args[1], /session\.jsonl$/);
  assert.equal(spawned[0].options.detached, true);
  const exported = await readFile(spawned[0].args[1], "utf8");
  assert.match(exported, /ok, remember that/);

  assert.equal(capturer.capture(session), false, "still running");
  children[0].exit(0);
  assert.equal(capturer.capture(session), false, "nothing new said");

  const longer = fakeSession([...conversation, user("user", "and the cron")]);
  assert.equal(capturer.capture(longer), true);
  assert.equal(spawned.length, 2);
  assert.ok((await readdir(join(dataDir, "capture"))).includes("capture.log"));
});

test("no capture runs before its binary is there", async (t) => {
  const dataDir = await mkdtemp(join(tmpdir(), "cortexdb-dsh-"));
  const warnings = [];
  const capturer = createCapturer({ dataDir }, {
    binary: () => "",
    logger: { warn: (m) => warnings.push(m) },
    spawnProcess() { throw new Error("must not spawn"); },
  });
  assert.equal(capturer.capture(fakeSession(conversation)), false);
  assert.match(warnings[0], /not downloaded yet/);
});

test("exports a week old are pruned", async (t) => {
  const { dataDir, capturer } = await capturerFor(t);
  const old = join(dataDir, "capture", "session-old");
  const fresh = join(dataDir, "capture", "session-new");
  await mkdir(old, { recursive: true });
  await mkdir(fresh, { recursive: true });
  const weekAgo = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000);
  await utimes(old, weekAgo, weekAgo);
  await capturer.prune();
  const left = await readdir(join(dataDir, "capture"));
  assert.deepEqual(left, ["session-new"]);
});
