// Notes into the brain's knowledge layer.
//
// Each note is one knowledge document, identified by its vault and path, so
// the agents that share the brain can find what was written here and say where
// it came from. A note is sent again only when its text changed, and a note
// that was deleted or moved out of the indexed folders is taken out again: the
// brain should not keep answering with a note that no longer exists.

// knowledgeIdOf is the knowledge id of a note.
export function knowledgeIdOf(vault, path) {
  return `obsidian:${vault}/${path}`;
}

// sourceUrlOf opens the note in Obsidian.
export function sourceUrlOf(vault, path) {
  return `obsidian://open?vault=${encodeURIComponent(vault)}&file=${encodeURIComponent(path)}`;
}

// folderList reads a settings field of one folder per line.
export function folderList(text) {
  return String(text ?? "")
    .split(/[\n,]/)
    .map((f) => f.trim().replace(/^\/+|\/+$/g, ""))
    .filter(Boolean);
}

function within(path, folder) {
  return path === folder || path.startsWith(`${folder}/`);
}

// indexable reports whether a note belongs in the index: inside an included
// folder (all of the vault when none is named), outside every excluded one,
// and never inside the export folder — those notes came from the brain.
export function indexable(path, { include = [], exclude = [], exportFolder = "" }) {
  if (!path.toLowerCase().endsWith(".md")) return false;
  if (exportFolder && within(path, exportFolder)) return false;
  if (exclude.some((f) => within(path, f))) return false;
  return include.length === 0 || include.some((f) => within(path, f));
}

// planIndex compares the notes now in the vault with what was last sent.
// `notes` is [{path, hash}]; `sent` maps path to the hash sent for it.
export function planIndex(notes, sent) {
  const save = [];
  const present = new Set();
  for (const note of notes) {
    present.add(note.path);
    if (sent[note.path] !== note.hash) save.push(note.path);
  }
  const remove = Object.keys(sent).filter((path) => !present.has(path));
  return { save: save.sort(), remove: remove.sort() };
}

// knowledgeRequest is the knowledge_save request for one note.
export function knowledgeRequest(vault, path, text) {
  const name = path.split("/").pop().replace(/\.md$/i, "");
  return {
    knowledge_id: knowledgeIdOf(vault, path),
    title: name,
    content: String(text ?? ""),
    source_url: sourceUrlOf(vault, path),
    metadata: { source: "obsidian", vault, path },
  };
}
