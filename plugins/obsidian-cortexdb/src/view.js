// The side panel: what the brain knows about the open note.

import { ItemView, MarkdownRenderer, MarkdownView, Notice, setIcon } from "obsidian";
import { entitiesOf, hitsOf, noteQuestion, recallArgs } from "./note.js";
import { knowledgeIdOf } from "./vault_index.js";

export const VIEW_TYPE = "cortexdb-panel";

export class CortexView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.request = 0;
    this.file = null;
    this.question = "";
  }

  getViewType() {
    return VIEW_TYPE;
  }

  getDisplayText() {
    return "CortexDB";
  }

  getIcon() {
    return "brain-circuit";
  }

  async onOpen() {
    const root = this.contentEl;
    root.empty();
    root.addClass("cortexdb-panel");

    const bar = root.createDiv({ cls: "cortexdb-bar" });
    this.input = bar.createEl("input", { type: "search", placeholder: "Ask the brain…", cls: "cortexdb-search" });
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.ask(this.input.value.trim());
    });
    const refresh = bar.createEl("button", { cls: "clickable-icon cortexdb-icon", attr: { "aria-label": "Ask about this note again" } });
    setIcon(refresh, "refresh-cw");
    refresh.addEventListener("click", () => this.showNote(this.app.workspace.getActiveFile(), true));

    this.status = root.createDiv({ cls: "cortexdb-status" });
    this.entities = root.createDiv({ cls: "cortexdb-entities" });
    this.list = root.createDiv({ cls: "cortexdb-list" });

    this.showNote(this.app.workspace.getActiveFile(), true);
  }

  async onClose() {
    this.request += 1;
  }

  // showNote asks about a note, unless it is the note already shown.
  showNote(file, force = false) {
    if (!file || file.extension !== "md") return;
    if (!force && this.file?.path === file.path) return;
    this.file = file;
    this.app.vault.cachedRead(file).then((text) => {
      if (this.file?.path !== file.path) return;
      this.input.value = "";
      this.run(noteQuestion(file.basename, text), `About ${file.basename}`);
    });
  }

  ask(question) {
    if (!question) return this.showNote(this.app.workspace.getActiveFile(), true);
    this.run(question, `“${question}”`);
  }

  async run(question, label) {
    const ticket = ++this.request;
    this.question = question;
    this.status.setText(`${label} — asking…`);
    this.status.removeClass("cortexdb-error");
    let payload;
    try {
      payload = await this.plugin.brain().call("knowledge_memory_recall", recallArgs(question, this.plugin.settings), 20000);
    } catch (error) {
      if (ticket !== this.request) return;
      this.status.setText(`Cannot reach the brain at ${this.plugin.settings.endpoint}: ${error.message}`);
      this.status.addClass("cortexdb-error");
      this.list.empty();
      this.entities.empty();
      return;
    }
    if (ticket !== this.request) return;
    const own = this.file
      ? { knowledge: knowledgeIdOf(this.app.vault.getName(), this.file.path), memory: this.plugin.exportedMemories[this.file.path]?.id ?? "" }
      : {};
    const hits = hitsOf(payload, own);
    this.status.setText(hits.length ? label : `${label} — the brain has nothing on this yet.`);
    this.renderEntities(entitiesOf(payload));
    this.renderHits(hits);
  }

  renderEntities(names) {
    this.entities.empty();
    for (const name of names) {
      const chip = this.entities.createEl("button", { text: name, cls: "cortexdb-chip" });
      chip.addEventListener("click", () => {
        this.input.value = name;
        this.ask(name);
      });
    }
  }

  renderHits(hits) {
    this.list.empty();
    for (const hit of hits) {
      const card = this.list.createDiv({ cls: `cortexdb-hit cortexdb-${hit.kind}` });
      const head = card.createDiv({ cls: "cortexdb-hit-head" });
      head.createSpan({ text: hit.kind === "memory" ? "Memory" : "Knowledge", cls: "cortexdb-badge" });
      if (hit.superseded) head.createSpan({ text: "superseded", cls: "cortexdb-badge cortexdb-old" });
      if (hit.date) head.createSpan({ text: hit.date, cls: "cortexdb-date" });
      card.createDiv({ text: hit.title, cls: "cortexdb-hit-title" });
      if (hit.more) {
        const body = card.createDiv({ cls: "cortexdb-hit-body is-clamped" });
        body.setText(hit.more);
        let rendered = false;
        body.addEventListener("click", async () => {
          if (!rendered) {
            rendered = true;
            body.empty();
            await MarkdownRenderer.render(this.app, hit.more, body, this.file?.path ?? "", this);
          }
          body.toggleClass("is-clamped", !body.hasClass("is-clamped"));
        });
      }

      const actions = card.createDiv({ cls: "cortexdb-actions" });
      this.action(actions, "text-cursor-input", "Insert into the note", () => this.insert(hit));
      this.action(actions, "copy", "Copy", async () => {
        await navigator.clipboard.writeText(hit.text);
        new Notice("Copied");
      });
      const note = this.plugin.noteFor(hit);
      if (note) this.action(actions, "file-text", "Open the note", () => this.app.workspace.getLeaf(false).openFile(note));
      actions.createSpan({ text: hit.id, cls: "cortexdb-id" });
    }
  }

  action(parent, icon, label, onClick) {
    const button = parent.createEl("button", { cls: "clickable-icon cortexdb-icon", attr: { "aria-label": label } });
    setIcon(button, icon);
    button.addEventListener("click", onClick);
  }

  // insert puts a hit into the note being edited, as a callout that says
  // where it came from.
  insert(hit) {
    const view = this.app.workspace.getMostRecentLeaf()?.view;
    const editor = this.app.workspace.activeEditor?.editor ?? (view instanceof MarkdownView ? view.editor : null);
    if (!editor) {
      new Notice("Open a note in editing mode to insert into it.");
      return;
    }
    const quoted = hit.text.split("\n").map((line) => `> ${line}`).join("\n");
    editor.replaceSelection(`> [!quote] CortexDB ${hit.kind} \`${hit.id}\`\n${quoted}\n`);
  }
}
