import assert from "node:assert/strict";
import { test } from "node:test";
import { readBrain } from "../src/brain.js";

test("the brain is read page by page, entities apart from memory and document nodes", async () => {
  const call = async (name, args) => {
    if (name === "memory_list_all") return args.cursor ? { memories: [{ id: "m2" }] } : { memories: [{ id: "m1" }], next_cursor: "c1" };
    if (args.cursor) return { nodes: [{ id: "entity:e2" }, { id: "memory:m1" }, { id: "doc:d" }], edges: [{ from: "memory:m1", to: "entity:e1", type: "mentions" }] };
    return { nodes: [{ id: "entity:e1" }], edges: [{ from: "entity:e1", to: "entity:e2", type: "uses" }], next_cursor: "n1" };
  };
  const got = await readBrain(call);
  assert.deepEqual(got.memories.map((m) => m.id), ["m1", "m2"]);
  assert.deepEqual(got.nodes.map((n) => n.id), ["entity:e1", "entity:e2"]);
  assert.equal(got.edges.length, 2);
  assert.deepEqual(got.mentions, [["memory:m1", "entity:e1"]]);
});
