import assert from "node:assert/strict";
import { test } from "node:test";
import { entitiesOf, hitsOf, noteQuestion, proseOf, recallArgs, withoutTitle } from "../src/note.js";

test("a note's prose leaves out frontmatter, code, comments and link markup", () => {
  const text = [
    "---",
    "tags: [a, b]",
    "---",
    "# Release plan",
    "",
    "Ship [[CortexDB|the brain]] with [docs](https://x.y) and `uv`.",
    "```go",
    "func secret() {}",
    "```",
    "%% private aside %%",
    "- [ ] tell [[Hermes#Install]]",
    "![[diagram.png]]",
  ].join("\n");
  assert.equal(proseOf(text), "Release plan Ship the brain with docs and uv. tell Hermes");
});

test("the question is the title, then the start of the prose", () => {
  assert.equal(noteQuestion("Plan", "one two", 5), "Plan\none t");
  assert.equal(noteQuestion("", "body"), "body");
});

test("recall reads the whole brain unless scope or namespace is set", () => {
  const all = recallArgs("CortexDB release", { topK: 5 });
  assert.deepEqual(all.keywords, ["cortexdb", "release"]);
  assert.equal(all.top_k_memories, 5);
  assert.ok(!("scope" in all) && !("namespace" in all));
  const some = recallArgs("x y", { topK: 3, scope: "global", namespace: "work" });
  assert.equal(some.scope, "global");
  assert.equal(some.namespace, "work");
});

test("a note never comes back as an answer about itself", () => {
  const payload = {
    memories: [{ memory: { id: "m1", content: "## Uses uv\nmore", created_at: "2026-10-14T01:00:00Z", metadata: { superseded_by: "m2" } }, score: 0.9 }],
    knowledge: [
      { knowledge_id: "obsidian:v/a.md", title: "a", snippet: "self" },
      { knowledge_id: "doc", title: "", snippet: "other" },
    ],
  };
  const hits = hitsOf(payload, { knowledge: "obsidian:v/a.md" });
  assert.deepEqual(hits.map((h) => h.id), ["m1", "doc"]);
  assert.deepEqual(hitsOf(payload, { memory: "m1" }).map((h) => h.id), ["obsidian:v/a.md", "doc"]);
  assert.equal(hits[0].title, "Uses uv");
  assert.equal(hits[0].more, "more");
  assert.equal(hits[1].more, "other");
  assert.equal(hits[0].date, "2026-10-14");
  assert.equal(hits[0].superseded, true);
  assert.equal(hits[1].title, "doc");
});

test("entities come from graph facts, each once", () => {
  const names = entitiesOf({ graph_facts: [{ subject: "A", object: "B" }, { subject: "a", object: "C" }] });
  assert.deepEqual(names, ["A", "B", "C"]);
});

test("a knowledge snippet does not repeat the title the card already shows", () => {
  assert.equal(withoutTitle("# SDS 与 HA 市场调研 配合现代化控制面 LINSTOR", "SDS与HA市场调研"), "配合现代化控制面 LINSTOR");
  assert.equal(withoutTitle("## Setup Install Python with uv", "Setup"), "Install Python with uv");
  assert.equal(withoutTitle("# Other heading then text", "Setup"), "Other heading then text");
  assert.equal(withoutTitle("plain text", ""), "plain text");
});
