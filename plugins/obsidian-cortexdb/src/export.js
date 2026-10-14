// The brain as a folder of notes: one note per memory, one per graph entity,
// relations as links, so Obsidian's graph view and backlinks show what the
// brain holds and how it hangs together.
//
// A memory note's body is the memory's text and nothing else; everything else
// sits in its properties. That is what lets an edited body go back to the
// brain as the memory's new text — the same shape `cortexdb-mcp --sync-memory`
// reads — without a heading or a list of links being saved into it.

export const MEMORIES = "Memories";
export const ENTITIES = "Entities";
export const INDEX_NOTE = "Index.md";

// Graph edges that do not belong on an entity's page: `mentions` is the
// memory→entity link the memory notes already carry, and `co_occurs` only
// says two names appeared in one text.
const SKIPPED_EDGES = new Set(["mentions", "co_occurs"]);

// fileNameOf turns a label into a file name every platform Obsidian runs on
// accepts, and Obsidian can link to.
export function fileNameOf(label, max = 80) {
  let s = String(label ?? "")
    .replace(/[\\/:*?"<>|#^[\]\u0000-\u001f]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .join(" ")
    .replace(/^\.+/, "")
    .trim();
  if ([...s].length > max) s = [...s].slice(0, max).join("").trim();
  return s;
}

// uniqueName gives each note a name no other note in its folder has, ignoring
// case: macOS and Windows do not tell "Hive" from "hive", and a second note
// written over the first would silently drop it.
function uniqueName(name, used) {
  let candidate = name;
  for (let n = 2; used.has(candidate.toLowerCase()); n++) candidate = `${name} (${n})`;
  used.add(candidate.toLowerCase());
  return candidate;
}

function aliasOf(text) {
  return String(text ?? "").replace(/[[\]|]/g, " ").split(/\s+/).filter(Boolean).join(" ");
}

// link always carries an alias: the target is a path, and without one
// Obsidian shows the path.
function link(folder, sub, name, alias) {
  return `[[${folder}/${sub}/${name}|${aliasOf(alias) || name}]]`;
}

function yamlString(s) {
  return JSON.stringify(String(s ?? ""));
}

function tagOf(s) {
  return String(s ?? "").toLowerCase().replace(/[^\p{L}\p{N}_-]+/gu, "-").replace(/^-+|-+$/g, "") || "other";
}

function titleOf(memory) {
  for (const line of String(memory.content ?? "").split("\n")) {
    const t = line.replace(/^#+\s*/, "").trim();
    if (t) return [...t].length > 90 ? `${[...t].slice(0, 90).join("").trim()}…` : t;
  }
  return memory.id;
}

function dateOf(value) {
  const s = String(value ?? "");
  return /^\d{4}-\d{2}-\d{2}/.test(s) && !s.startsWith("0001-") ? s : "";
}

// buildVault renders the brain into notes under `folder`. It returns the
// notes to write, as [{path, content}], and for each memory note the body it
// was written with, so an edit can be told from the export.
export function buildVault({ memories = [], nodes = [], edges = [], mentions = [] }, { folder = "CortexDB", exportedAt = "" } = {}) {
  const files = [];
  const bodies = {};

  // An entity gets a note when there is something to put on it: a relation
  // to another entity, or a memory that mentions it. The rest are names an
  // extractor picked out of a text and nothing ever connected.
  const memoryIds = new Set(memories.map((m) => m?.id));
  const known = new Set(nodes.map((n) => n?.id));
  const connected = new Set();
  for (const e of edges) {
    if (!e || SKIPPED_EDGES.has(e.type) || e.from === e.to || !known.has(e.from) || !known.has(e.to)) continue;
    connected.add(e.from);
    connected.add(e.to);
  }
  for (const [memoryNode, entityId] of mentions) {
    if (known.has(entityId) && memoryIds.has(String(memoryNode ?? "").replace(/^memory:/, ""))) connected.add(entityId);
  }

  // Entities first: memory notes link to them by name.
  const entityName = new Map();
  const usedEntity = new Set();
  const byDegree = nodes.filter((n) => connected.has(n?.id)).sort((a, b) => (b.degree ?? 0) - (a.degree ?? 0) || String(a.id).localeCompare(String(b.id)));
  for (const node of byDegree) {
    if (!node?.id || entityName.has(node.id)) continue;
    const base = fileNameOf(node.label) || fileNameOf(String(node.id).replace(/^entity:/, "")) || "entity";
    entityName.set(node.id, uniqueName(base, usedEntity));
  }

  const memoryName = new Map();
  const usedMemory = new Set();
  const sortedMemories = [...memories]
    .filter((m) => m?.id)
    .sort((a, b) => String(b.created_at ?? "").localeCompare(String(a.created_at ?? "")) || String(a.id).localeCompare(String(b.id)));
  for (const m of sortedMemories) {
    memoryName.set(m.id, uniqueName(fileNameOf(m.id.replace(/:/g, "-")) || "memory", usedMemory));
  }

  // Which entities each memory mentions, and which memories each entity has.
  const entitiesOfMemory = new Map();
  const memoriesOfEntity = new Map();
  for (const [memoryNode, entityId] of mentions) {
    const memoryId = String(memoryNode ?? "").replace(/^memory:/, "");
    if (!memoryName.has(memoryId) || !entityName.has(entityId)) continue;
    if (!entitiesOfMemory.has(memoryId)) entitiesOfMemory.set(memoryId, new Set());
    entitiesOfMemory.get(memoryId).add(entityId);
    if (!memoriesOfEntity.has(entityId)) memoriesOfEntity.set(entityId, new Set());
    memoriesOfEntity.get(entityId).add(memoryId);
  }

  const memoryById = new Map(sortedMemories.map((m) => [m.id, m]));
  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  for (const m of sortedMemories) {
    const name = memoryName.get(m.id);
    const body = String(m.content ?? "").trim();
    const lines = ["---", `id: ${yamlString(m.id)}`];
    if (m.scope) lines.push(`scope: ${yamlString(m.scope)}`);
    if (m.namespace) lines.push(`namespace: ${yamlString(m.namespace)}`);
    if (m.importance) lines.push(`importance: ${Number(m.importance)}`);
    const created = dateOf(m.created_at);
    if (created) lines.push(`created: ${created}`);
    const source = String(m.metadata?.source ?? "").trim();
    if (source) lines.push(`source: ${yamlString(source)}`);
    const superseded = String(m.metadata?.superseded_by ?? "").trim();
    if (superseded) lines.push(`superseded_by: ${yamlString(superseded)}`);
    const mentioned = [...(entitiesOfMemory.get(m.id) ?? [])].map((id) => entityName.get(id)).sort();
    if (mentioned.length) {
      lines.push("mentions:");
      for (const e of mentioned) lines.push(`  - ${yamlString(`[[${folder}/${ENTITIES}/${e}|${e}]]`)}`);
    }
    lines.push("tags:", "  - cortexdb/memory");
    if (superseded) lines.push("  - cortexdb/superseded");
    lines.push("---", "", body, "");
    const path = `${folder}/${MEMORIES}/${name}.md`;
    files.push({ path, content: lines.join("\n") });
    bodies[path] = { id: m.id, body };
  }

  // Relations, each once: the graph keeps one edge per assertion, and the
  // same fact asserted forty times is still one line on a page.
  const out = new Map();
  const into = new Map();
  const seen = new Set();
  for (const e of edges) {
    if (!e?.from || !e?.to || e.from === e.to || SKIPPED_EDGES.has(e.type)) continue;
    if (!entityName.has(e.from) || !entityName.has(e.to)) continue;
    const key = `${e.from}\u0000${e.type}\u0000${e.to}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (!out.has(e.from)) out.set(e.from, []);
    out.get(e.from).push(e);
    if (!into.has(e.to)) into.set(e.to, []);
    into.get(e.to).push(e);
  }

  for (const [id, name] of entityName) {
    const node = nodeById.get(id) ?? {};
    const lines = ["---", `id: ${yamlString(id)}`];
    if (node.type) lines.push(`type: ${yamlString(node.type)}`);
    lines.push("tags:", "  - cortexdb/entity");
    if (node.type) lines.push(`  - cortexdb/type/${tagOf(node.type)}`);
    lines.push("---", "");
    // Obsidian titles a note with its file name; the label needs a heading
    // only when it had to change to become one.
    const label = aliasOf(node.label);
    if (label && label !== name) lines.push(`# ${label}`, "");
    const rel = [];
    const byRelation = (a, b) => String(a.type).localeCompare(String(b.type));
    for (const e of [...(out.get(id) ?? [])].sort(byRelation)) {
      rel.push(`- ${e.type || "related_to"} → ${link(folder, ENTITIES, entityName.get(e.to), nodeById.get(e.to)?.label)}`);
    }
    for (const e of [...(into.get(id) ?? [])].sort(byRelation)) {
      rel.push(`- ${link(folder, ENTITIES, entityName.get(e.from), nodeById.get(e.from)?.label)} → ${e.type || "related_to"}`);
    }
    if (rel.length) lines.push("## Relations", "", ...rel, "");
    const mems = [...(memoriesOfEntity.get(id) ?? [])]
      .map((mid) => memoryById.get(mid))
      .sort((a, b) => String(b.created_at ?? "").localeCompare(String(a.created_at ?? "")));
    if (mems.length) {
      lines.push("## Memories", "");
      for (const m of mems) {
        const date = dateOf(m.created_at).slice(0, 10);
        lines.push(`- ${link(folder, MEMORIES, memoryName.get(m.id), titleOf(m))}${date ? ` · ${date}` : ""}`);
      }
      lines.push("");
    }
    files.push({ path: `${folder}/${ENTITIES}/${name}.md`, content: lines.join("\n") });
  }

  const top = byDegree.filter((n) => entityName.has(n.id)).slice(0, 40);
  const recent = sortedMemories.slice(0, 40);
  const index = [
    "---",
    "tags:",
    "  - cortexdb/index",
    "---",
    "",
    "# CortexDB",
    "",
    `${sortedMemories.length} memories and ${entityName.size} entities${exportedAt ? `, exported ${exportedAt}` : ""}. This folder is rewritten on every export; edit a memory's text in its note and run **CortexDB: Save edited memories to the brain** to keep the change.`,
    "",
  ];
  if (top.length) {
    index.push("## Most connected", "");
    for (const n of top) index.push(`- ${link(folder, ENTITIES, entityName.get(n.id), n.label)}${n.type ? ` · ${n.type}` : ""}`);
    index.push("");
  }
  if (recent.length) {
    index.push("## Newest memories", "");
    for (const m of recent) {
      const date = dateOf(m.created_at).slice(0, 10);
      index.push(`- ${link(folder, MEMORIES, memoryName.get(m.id), titleOf(m))}${date ? ` · ${date}` : ""}`);
    }
    index.push("");
  }
  files.push({ path: `${folder}/${INDEX_NOTE}`, content: index.join("\n") });

  return { files, bodies };
}

// parseMemoryNote reads a memory note back: its properties as flat strings,
// the way `cortexdb-mcp --sync-memory` reads them, and its body.
export function parseMemoryNote(text) {
  const raw = String(text ?? "").replace(/^﻿/, "");
  const s = raw.replace(/^[ \t\r\n]+/, "");
  const meta = {};
  if (!s.startsWith("---")) return { meta, body: raw.trim() };
  const rest = s.slice(3);
  const end = rest.indexOf("\n---");
  if (end < 0) return { meta, body: raw.trim() };
  for (const line of rest.slice(0, end).split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    const i = t.indexOf(":");
    if (i <= 0) continue;
    const value = t.slice(i + 1).trim().replace(/^["']+|["']+$/g, "");
    if (value) meta[t.slice(0, i).trim()] = value;
  }
  let body = rest.slice(end + 4);
  const nl = body.indexOf("\n");
  body = nl >= 0 ? body.slice(nl + 1) : "";
  return { meta, body: body.trim() };
}

// editedMemories compares memory notes as they are now with the bodies they
// were exported with, and returns the memory_update requests for the changed
// ones. memory_update replaces the text and keeps the memory's metadata —
// where it came from, what superseded it — which a save would drop. A note
// whose id property was changed is skipped: the id says which memory a note
// is, and renaming one is not an edit. An emptied note is skipped too:
// deleting a memory is not something an export should do by accident.
export function editedMemories(current, exported) {
  const updates = [];
  for (const [path, text] of Object.entries(current)) {
    const was = exported[path];
    if (!was) continue;
    const { meta, body } = parseMemoryNote(text);
    if (!body || body === was.body) continue;
    if (meta.id && meta.id !== was.id) continue;
    updates.push({ memory_id: was.id, content: body });
  }
  return updates;
}
