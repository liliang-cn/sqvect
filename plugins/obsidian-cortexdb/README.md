# CortexDB for Obsidian

An Obsidian plugin backed by a CortexDB gRPC server — your own sidecar, or the
shared brain your agents use:

- **A panel on the open note.** The right sidebar shows what the brain knows
  about the note in front of you: memories first, then knowledge documents,
  and the graph entities they touch as chips to ask about next. Open a note
  and it asks; type in the box to ask anything else. A hit can be inserted into
  the note as a quote callout, copied, or opened when it came from this vault.
- **Notes into the brain.** *Remember the selection* saves text as a memory.
  *Send this note to the brain* and *Send the vault to the brain* save notes as
  knowledge documents (`obsidian:<vault>/<path>`, with an `obsidian://` link
  back), so Claude Code, Codex, dsh, Hermes and every other agent sharing the
  brain can recall them. Only notes that changed are sent again; notes that
  were deleted are taken out. *Keep the brain current* does this as you write.
- **The brain as notes.** *Export the brain to the vault* writes every memory
  and graph entity into a folder (`CortexDB/` by default): one note per memory,
  one per entity, relations and mentions as links — so the graph view,
  backlinks and search work on what the brain holds. Edit a memory's text in
  its note and *Save edited memories to the brain* puts the edit back.

Desktop only: the plugin speaks gRPC over Node's `http2`, which Obsidian's
mobile app does not have.

## Install

Download the release archive into the vault's plugins folder:

```bash
VAULT=~/path/to/vault
curl -fsSL -o /tmp/cortexdb-obsidian.zip \
  https://github.com/liliang-cn/cortexdb/releases/latest/download/cortexdb-obsidian.zip
unzip -o /tmp/cortexdb-obsidian.zip -d "$VAULT/.obsidian/plugins/"
```

Then enable **CortexDB** under *Settings → Community plugins*. Each release
also carries `cortexdb-obsidian.zip.sha256`.

## Connect

*Settings → CortexDB*:

| Setting | Meaning |
| --- | --- |
| Endpoint | `host:port` of the CortexDB gRPC server (default `127.0.0.1:47821`), or an `https://` URL. **Test** shows its version. |
| Token | The server's bearer token, if it has one. Pick or create a secret: Obsidian keeps it in the system keychain, and the vault holds only the secret's name. |
| Results | Memories and knowledge documents shown for a note, each. |
| Scope and namespace | Narrow what the panel reads. Empty reads the whole brain. |
| Scope for remembered selections | Scope of memories saved with *Remember the selection* (default `global`). |
| Folders to send / to leave out | Which notes *Send the vault* covers. The export folder is never sent. |
| Keep the brain current | Send a note again a few seconds after it changes, and take it out when it is deleted. |
| Export folder | Where *Export the brain* writes. |

To run a brain of your own: `cortexdb-grpc -db ~/.cortexdb/cortexdb.db`
(release binaries are on the
[releases page](https://github.com/liliang-cn/cortexdb/releases)).

## The export

```
CortexDB/
  Index.md               counts, the most connected entities, the newest memories
  Memories/<id>.md       one per memory
  Entities/<name>.md     one per entity with a relation or a memory
```

A memory note's properties carry its id, scope, namespace, importance, date,
source, `superseded_by` when a newer memory replaced it, and `mentions` — links
to its entities. Its body is the memory's text and nothing else, which is why an
edit can go back: *Save edited memories to the brain* sends each body that
changed since the export through `memory_update`, keeping the memory's
metadata. The same folder also works with `cortexdb-mcp --sync-memory
<vault>/CortexDB/Memories` (without `--prune` unless the export is complete).

An entity note lists its relations, each once, and the memories that mention
it, newest first. Entities are tagged `cortexdb/type/<type>`, which the graph
view's groups can colour.

The folder is rewritten on every export. Notes an earlier export wrote whose
memory or entity is gone go to the trash. If a memory note holds an edit the
brain has not seen, the export says so before rewriting it.

## Develop

```bash
npm install
npm test           # node --test, no Obsidian needed
npm run build      # main.js
```

`main.js`, `manifest.json` and `styles.css` are the plugin; copy them into
`<vault>/.obsidian/plugins/cortexdb/`. The gRPC client is the DeepSeek Harness
plugin's (`../dsh-cortexdb-memory/lib/grpc.js`), bundled in.
