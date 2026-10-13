import assert from "node:assert/strict";
import http2 from "node:http2";
import test from "node:test";
import { createClient, decodeFields, encodeStrings } from "../lib/grpc.js";

test("protobuf decoding rejects truncated length-delimited and fixed-width fields", () => {
  for (const bytes of [[0x0a, 5, 0x61], [0x09, 1], [0x0d, 1]]) {
    assert.throws(() => decodeFields(Buffer.from(bytes)), /truncated/);
  }
  assert.equal(decodeFields(encodeStrings({ 1: "完整" }))[1].toString(), "完整");
});

test("a unary call rejects a truncated gRPC payload even with success trailers", async (t) => {
  const server = http2.createServer();
  server.on("stream", (stream) => {
    stream.on("data", () => {});
    stream.on("end", () => {
      stream.respond({ ":status": 200, "content-type": "application/grpc" }, { waitForTrailers: true });
      stream.on("wantTrailers", () => stream.sendTrailers({ "grpc-status": "0" }));
      // The frame claims a 20-byte message, but contains a valid 4-byte protobuf.
      stream.end(Buffer.concat([Buffer.from([0, 0, 0, 0, 20]), encodeStrings({ 1: "{}" })]));
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());
  const client = createClient(`127.0.0.1:${server.address().port}`);
  t.after(() => client.close());

  await assert.rejects(client.tools.CallTool({ name: "fixture", argsJson: "{}" }, 1000), /truncated/);
});
