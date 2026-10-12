import assert from "node:assert/strict";
import http2 from "node:http2";
import test from "node:test";
import { createClient, decodeFields, encodeStrings } from "../lib/grpc.js";

function frame(message) {
  const header = Buffer.alloc(5);
  header.writeUInt32BE(message.length, 1);
  return Buffer.concat([header, message]);
}

// fakeSidecar answers gRPC the way the CortexDB sidecar does: one framed
// message, then grpc-status in the trailers.
async function fakeSidecar(t, handler) {
  const server = http2.createServer();
  const seen = [];
  server.on("stream", (stream, headers) => {
    const chunks = [];
    stream.on("data", (c) => chunks.push(c));
    stream.on("end", () => {
      const body = Buffer.concat(chunks);
      const request = decodeFields(body.subarray(5, 5 + body.readUInt32BE(1)));
      seen.push({ path: headers[":path"], headers, request });
      handler(stream, headers[":path"], request);
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());
  return { endpoint: `127.0.0.1:${server.address().port}`, seen };
}

function reply(stream, message, status = 0, detail = "") {
  stream.respond({ ":status": 200, "content-type": "application/grpc" }, { waitForTrailers: true });
  stream.on("wantTrailers", () => stream.sendTrailers({ "grpc-status": String(status), ...(detail ? { "grpc-message": encodeURIComponent(detail) } : {}) }));
  stream.end(frame(message));
}

test("protobuf strings round-trip, including CJK, and empty fields are left out", () => {
  const encoded = encodeStrings({ 1: "knowledge_memory_recall", 2: "{\"query\":\"用 uv\"}", 3: "" });
  const fields = decodeFields(encoded);
  assert.equal(fields[1].toString(), "knowledge_memory_recall");
  assert.equal(fields[2].toString(), "{\"query\":\"用 uv\"}");
  assert.equal(fields[3], undefined);
  const long = "x".repeat(300);
  assert.equal(decodeFields(encodeStrings({ 1: long }))[1].toString(), long);
});

test("CallTool sends name and arguments, with the token and deadline, and reads the result", async (t) => {
  const sidecar = await fakeSidecar(t, (stream) => reply(stream, encodeStrings({ 1: "{\"ok\":true}" })));
  const client = createClient(sidecar.endpoint, { token: "secret" });
  t.after(() => client.close());
  const res = await client.tools.CallTool({ name: "memory_get", argsJson: "{\"memory_id\":\"m\"}" }, 3000);
  assert.equal(res.resultJson, "{\"ok\":true}");
  const [call] = sidecar.seen;
  assert.equal(call.path, "/cortexdb.v1.ToolsService/CallTool");
  assert.equal(call.headers.authorization, "Bearer secret");
  assert.equal(call.headers["grpc-timeout"], "3000m");
  assert.equal(call.request[1].toString(), "memory_get");
  assert.equal(call.request[2].toString(), "{\"memory_id\":\"m\"}");
});

test("Health and Info decode the admin messages", async (t) => {
  const sidecar = await fakeSidecar(t, (stream, path) => {
    if (path.endsWith("/Health")) return reply(stream, Buffer.from([0x08, 0x01]));
    // version = "2.124.0", db_path = "/x.db", has_embedder = true, memories_without_vector = 300
    reply(stream, Buffer.concat([encodeStrings({ 1: "2.124.0", 2: "/x.db" }), Buffer.from([0x18, 0x01, 0x30, 0xac, 0x02])]));
  });
  const client = createClient(sidecar.endpoint);
  t.after(() => client.close());
  assert.deepEqual(await client.admin.Health({}), { ok: true });
  assert.deepEqual(await client.admin.Info({}), { version: "2.124.0", dbPath: "/x.db", hasEmbedder: true });
});

test("a failed call carries the server's status and message", async (t) => {
  const sidecar = await fakeSidecar(t, (stream) => reply(stream, Buffer.alloc(0), 16, "missing bearer token"));
  const client = createClient(sidecar.endpoint);
  t.after(() => client.close());
  await assert.rejects(client.tools.CallTool({ name: "x", argsJson: "{}" }), (error) => error.code === 16 && /missing bearer token/.test(error.message));
});

test("a call past its deadline is abandoned", async (t) => {
  const sidecar = await fakeSidecar(t, () => {});
  const client = createClient(sidecar.endpoint);
  t.after(() => client.close());
  await assert.rejects(client.tools.CallTool({ name: "slow", argsJson: "{}" }, 100), /DEADLINE_EXCEEDED/);
});

test("nothing listening is an error, not a hang", async () => {
  const client = createClient("127.0.0.1:1");
  await assert.rejects(client.admin.Health({}, 2000), /UNAVAILABLE/);
  client.close();
});
