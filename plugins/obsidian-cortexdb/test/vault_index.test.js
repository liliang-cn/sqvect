import assert from "node:assert/strict";
import { test } from "node:test";
import { folderList, indexable, knowledgeRequest, planIndex, sourceUrlOf } from "../src/vault_index.js";

test("folders are one per line, without slashes at the ends", () => {
  assert.deepEqual(folderList("Work/\n /Daily \n\nA,B"), ["Work", "Daily", "A", "B"]);
});

test("the export folder is never indexed, whatever else is set", () => {
  const scope = { include: [], exclude: ["Private"], exportFolder: "CortexDB" };
  assert.equal(indexable("Notes/a.md", scope), true);
  assert.equal(indexable("CortexDB/Memories/x.md", scope), false);
  assert.equal(indexable("CortexDBX/a.md", scope), true);
  assert.equal(indexable("Private/a.md", scope), false);
  assert.equal(indexable("img.png", scope), false);
  assert.equal(indexable("Notes/a.md", { ...scope, include: ["Work"] }), false);
  assert.equal(indexable("Work/a.md", { ...scope, include: ["Work"] }), true);
});

test("only changed notes are sent, and gone ones are taken out", () => {
  const plan = planIndex(
    [
      { path: "a.md", hash: "1" },
      { path: "b.md", hash: "2" },
      { path: "c.md", hash: "3" },
    ],
    { "a.md": "1", "b.md": "old", "gone.md": "x" },
  );
  assert.deepEqual(plan, { save: ["b.md", "c.md"], remove: ["gone.md"] });
});

test("a note's knowledge says which vault and path it is", () => {
  const req = knowledgeRequest("My Vault", "Work/Plan A.md", "text");
  assert.equal(req.knowledge_id, "obsidian:My Vault/Work/Plan A.md");
  assert.equal(req.title, "Plan A");
  assert.equal(req.source_url, sourceUrlOf("My Vault", "Work/Plan A.md"));
  assert.equal(req.source_url, "obsidian://open?vault=My%20Vault&file=Work%2FPlan%20A.md");
  assert.deepEqual(req.metadata, { source: "obsidian", vault: "My Vault", path: "Work/Plan A.md" });
});
