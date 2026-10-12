import assert from "node:assert/strict";
import test from "node:test";
import { VERSION, createBackend, createSidecar, extractTarEntry } from "../lib/backend.js";

const healthy = {
  admin: {
    async Health() { return { ok: true }; },
    async Info() { return { version: VERSION, dbPath: "/tmp/cortexdb.db" }; },
  },
};

test("binaries are pinned to this package's version", () => {
  assert.match(VERSION, /^\d+\.\d+\.\d+/);
});

test("a sidecar that already answers is used, not replaced", async () => {
  const sidecar = createSidecar({ endpoint: "127.0.0.1:47821", autoStart: true }, { spawn() { throw new Error("must not spawn"); } });
  await sidecar.start(healthy);
  assert.equal(sidecar.state.connected, true);
  assert.equal(sidecar.state.managed, false);
});

test("without autoStart an unreachable brain is an error that says where", async () => {
  const down = { admin: { async Health() { throw new Error("connection refused"); } } };
  const sidecar = createSidecar({ endpoint: "127.0.0.1:47821", autoStart: false });
  await assert.rejects(sidecar.start(down), /unreachable at 127\.0\.0\.1:47821.*connection refused/);
});

test("a shared brain that is down is not replaced by a local one", async () => {
  const down = { admin: { async Health() { throw new Error("deadline exceeded"); } } };
  const sidecar = createSidecar({ endpoint: "192.168.123.252:47821", autoStart: true }, { spawn() { throw new Error("must not spawn"); } });
  await assert.rejects(sidecar.start(down), /autoStart only starts a sidecar on the local default/);
});

test("a started sidecar opens the configured database and is waited for", async () => {
  let attempts = 0;
  const client = {
    admin: {
      async Health() { attempts += 1; if (attempts < 3) throw new Error("refused"); return { ok: true }; },
      async Info() { return { version: VERSION }; },
    },
  };
  const spawned = [];
  let killed = false;
  const sidecar = createSidecar({ endpoint: "127.0.0.1:47821", autoStart: true, dbPath: "/tmp/cortexdb-dsh-test/x.db" }, {
    sleep: async () => {},
    binary: async () => "/opt/cortexdb-grpc",
    spawn(bin, args) { spawned.push({ bin, args }); return { once() {}, kill() { killed = true; } }; },
  });
  await sidecar.start(client);
  assert.deepEqual(spawned, [{ bin: "/opt/cortexdb-grpc", args: ["-db", "/tmp/cortexdb-dsh-test/x.db", "-addr", "127.0.0.1:47821"] }]);
  assert.equal(sidecar.state.managed, true);
  await sidecar.stop();
  assert.equal(killed, true);
});

test("call parses the tool's JSON, and a slow call is abandoned", async () => {
  const calls = [];
  const client = {
    tools: {
      async CallTool(req) {
        calls.push(req);
        if (req.name === "slow") return new Promise(() => {});
        return { resultJson: JSON.stringify({ ok: true }) };
      },
    },
    close() {},
  };
  const backend = createBackend({ endpoint: "127.0.0.1:47821", callTimeoutMs: 50 }, {
    connect: () => client,
    sidecar: () => ({ state: {}, start: async () => {}, stop: async () => {} }),
  });
  assert.deepEqual(await backend.call("memory_get", { memory_id: "m" }), { ok: true });
  assert.deepEqual(JSON.parse(calls[0].argsJson), { memory_id: "m" });
  await assert.rejects(backend.call("slow", {}), /slow timed out after 50 ms/);
});

test("the sidecar binary is found inside its release archive", () => {
  const header = Buffer.alloc(512);
  header.write("cortexdb-grpc_linux_arm64/cortexdb-grpc", 0);
  header.write("0000000005\0", 124);
  const tar = Buffer.concat([header, Buffer.from("hello"), Buffer.alloc(507), Buffer.alloc(1024)]);
  assert.equal(extractTarEntry(tar, "cortexdb-grpc").toString(), "hello");
  assert.throws(() => extractTarEntry(tar, "other"), /not found/);
});
