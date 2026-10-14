import assert from "node:assert/strict";
import { test } from "node:test";
import { buildVault, editedMemories, fileNameOf, parseMemoryNote } from "../src/export.js";

const brain = {
  memories: [
    { id: "auto:1:uv", content: "The user sets up Python with uv.\nNever system pip.", scope: "global", importance: 0.8, created_at: "2026-10-12T08:00:00Z", metadata: { source: "auto-capture" } },
    { id: "old:pip", content: "The user uses pip.", created_at: "2026-01-01T00:00:00Z", metadata: { superseded_by: "auto:1:uv" } },
  ],
  nodes: [
    { id: "entity:uv", label: "uv", type: "tool", degree: 3 },
    { id: "entity:python", label: "Python", type: "language", degree: 5 },
    { id: "entity:py2", label: "python", type: "concept", degree: 1 },
    { id: "entity:weird", label: 'a/b:c*?"<>|#^[x]', degree: 1 },
    { id: "entity:alone", label: "Setup Install Python", degree: 0 },
  ],
  edges: [
    { from: "entity:uv", to: "entity:python", type: "manages" },
    { from: "entity:uv", to: "entity:python", type: "manages" },
    { from: "entity:uv", to: "entity:python", type: "co_occurs" },
    { from: "entity:uv", to: "entity:missing", type: "uses" },
    { from: "entity:weird", to: "entity:py2", type: "names" },
    { from: "entity:alone", to: "entity:python", type: "co_occurs" },
  ],
  mentions: [
    ["memory:auto:1:uv", "entity:uv"],
    ["memory:auto:1:uv", "entity:python"],
    ["memory:gone", "entity:uv"],
  ],
};

const byPath = (files) => Object.fromEntries(files.map((f) => [f.path, f.content]));

test("names are safe file names, and unique without regard to case", () => {
  assert.equal(fileNameOf('a/b:c*?"<>|#^[x]'), "a b c x");
  assert.equal(fileNameOf("..hidden"), "hidden");
  const { files } = buildVault(brain, { folder: "CortexDB" });
  const paths = files.map((f) => f.path);
  assert.ok(paths.includes("CortexDB/Entities/Python.md"));
  assert.ok(paths.includes("CortexDB/Entities/python (2).md"));
  assert.ok(paths.includes("CortexDB/Entities/a b c x.md"));
  assert.ok(paths.includes("CortexDB/Memories/auto-1-uv.md"));
  assert.ok(paths.includes("CortexDB/Index.md"));
  assert.ok(!paths.some((p) => p.includes("Setup Install Python")), "an entity nothing connects gets no note");
});

test("a memory note's body is the memory's text, its links in properties", () => {
  const files = byPath(buildVault(brain, { folder: "CortexDB" }).files);
  const note = files["CortexDB/Memories/auto-1-uv.md"];
  assert.match(note, /^---\nid: "auto:1:uv"\nscope: "global"\nimportance: 0.8\ncreated: 2026-10-12T08:00:00Z\nsource: "auto-capture"\nmentions:\n  - "\[\[CortexDB\/Entities\/Python\|Python\]\]"\n  - "\[\[CortexDB\/Entities\/uv\|uv\]\]"\ntags:\n  - cortexdb\/memory\n---\n\nThe user sets up Python with uv.\nNever system pip.\n$/);
  assert.match(files["CortexDB/Memories/old-pip.md"], /superseded_by: "auto:1:uv"[\s\S]*cortexdb\/superseded/);
});

test("an entity note lists each relation once and its memories, newest first", () => {
  const files = byPath(buildVault(brain, { folder: "CortexDB" }).files);
  const uv = files["CortexDB/Entities/uv.md"];
  assert.equal(uv.match(/manages/g).length, 1);
  assert.ok(!uv.includes("co_occurs") && !uv.includes("missing"));
  assert.match(uv, /- manages → \[\[CortexDB\/Entities\/Python\|Python\]\]/);
  assert.ok(!uv.includes("# uv"), "the file name already titles the note");
  assert.match(files["CortexDB/Entities/a b c x.md"], /\n# a\/b:c\*\?"<> #\^ x\n/);
  assert.match(uv, /## Memories\n\n- \[\[CortexDB\/Memories\/auto-1-uv\|The user sets up Python with uv\.\]\] · 2026-10-12/);
  assert.match(files["CortexDB/Entities/Python.md"], /- \[\[CortexDB\/Entities\/uv\|uv\]\] → manages/);
  assert.match(files["CortexDB/Entities/Python.md"], /tags:\n  - cortexdb\/entity\n  - cortexdb\/type\/language/);
});

test("a memory note reads back the way --sync-memory reads it", () => {
  const { files, bodies } = buildVault(brain, { folder: "CortexDB" });
  const note = byPath(files)["CortexDB/Memories/auto-1-uv.md"];
  const { meta, body } = parseMemoryNote(note);
  assert.equal(meta.id, "auto:1:uv");
  assert.equal(meta.scope, "global");
  assert.equal(body, "The user sets up Python with uv.\nNever system pip.");
  assert.deepEqual(bodies["CortexDB/Memories/auto-1-uv.md"], { id: "auto:1:uv", body });
});

test("only edited bodies go back, under the id they were exported with", () => {
  const { files, bodies } = buildVault(brain, { folder: "CortexDB" });
  const current = byPath(files);
  const path = "CortexDB/Memories/auto-1-uv.md";
  assert.deepEqual(editedMemories(current, bodies), []);

  const edited = { ...current, [path]: current[path].replace("Never system pip.", "Never the system pip.") };
  assert.deepEqual(editedMemories(edited, bodies), [{ memory_id: "auto:1:uv", content: "The user sets up Python with uv.\nNever the system pip." }]);

  const renamed = { ...edited, [path]: edited[path].replace('id: "auto:1:uv"', 'id: "other"') };
  assert.deepEqual(editedMemories(renamed, bodies), []);

  const emptied = { ...current, [path]: current[path].replace(/---\n\n[\s\S]*$/, "---\n") };
  assert.deepEqual(editedMemories(emptied, bodies), []);
});
