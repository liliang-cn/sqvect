// CortexDB for Obsidian.
//
// A panel that asks the brain about the open note; commands that send a
// selection, a note or the vault to it; and an export that writes the brain
// back as notes, whose edited memories can be saved to it again.

import { createHash } from "node:crypto";
import { debounce, Modal, normalizePath, Notice, Plugin, PluginSettingTab, SecretComponent, Setting, TFile } from "obsidian";
import { Brain, DEFAULT_ENDPOINT, readBrain } from "./brain.js";
import { buildVault, editedMemories, ENTITIES, INDEX_NOTE, MEMORIES } from "./export.js";
import { folderList, indexable, knowledgeIdOf, knowledgeRequest, planIndex } from "./vault_index.js";
import { CortexView, VIEW_TYPE } from "./view.js";

const DEFAULT_SETTINGS = {
  endpoint: DEFAULT_ENDPOINT,
  tokenSecret: "",
  scope: "",
  namespace: "",
  topK: 5,
  memoryScope: "global",
  indexInclude: "",
  indexExclude: "",
  indexOnSave: false,
  exportFolder: "CortexDB",
};

const hashOf = (text) => createHash("sha256").update(text).digest("hex");
const count = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export default class CortexDBPlugin extends Plugin {
  async onload() {
    const data = (await this.loadData()) ?? {};
    this.settings = { ...DEFAULT_SETTINGS, ...(data.settings ?? {}) };
    // What was last sent for each indexed note, and each exported memory
    // note's body as written: the two things that tell a change from none.
    this.indexed = data.indexed ?? {};
    this.exportedFiles = data.exportedFiles ?? [];
    this.exportedMemories = data.exportedMemories ?? {};

    this.registerView(VIEW_TYPE, (leaf) => new CortexView(leaf, this));
    this.addRibbonIcon("brain-circuit", "CortexDB", () => this.openPanel());
    this.addSettingTab(new CortexSettingTab(this.app, this));

    this.addCommand({ id: "open-panel", name: "Show what the brain knows about this note", callback: () => this.openPanel() });
    this.addCommand({
      id: "save-selection",
      name: "Remember the selection",
      editorCallback: (editor, ctx) => this.rememberSelection(editor.getSelection(), ctx.file),
    });
    this.addCommand({
      id: "save-note",
      name: "Send this note to the brain",
      checkCallback: (checking) => {
        // Any note can be sent on purpose, except the export's: those are the
        // brain's own memories already.
        const file = this.app.workspace.getActiveFile();
        if (!file || !indexable(file.path, { exportFolder: this.indexScope().exportFolder })) return false;
        if (!checking) this.sendNotes([file.path], []).catch((e) => new Notice(`CortexDB: ${e.message}`));
        return true;
      },
    });
    this.addCommand({ id: "index-vault", name: "Send the vault to the brain", callback: () => this.indexVault() });
    this.addCommand({ id: "export-brain", name: "Export the brain to the vault", callback: () => this.exportBrain() });
    this.addCommand({ id: "save-edited-memories", name: "Save edited memories to the brain", callback: () => this.saveEditedMemories() });

    this.registerEvent(
      this.app.workspace.on("file-open", (file) => {
        for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE)) leaf.view.showNote?.(file);
      }),
    );

    // Keeping the index current as notes change, when asked to.
    const pending = new Set();
    const flush = debounce(() => {
      const paths = [...pending];
      pending.clear();
      const gone = paths.filter((p) => !this.app.vault.getFileByPath(p));
      this.sendNotes(paths.filter((p) => !gone.includes(p)), gone.filter((p) => p in this.indexed)).catch(() => {});
    }, 5000, true);
    const touched = (path) => {
      if (!this.settings.indexOnSave) return;
      if (!indexable(path, this.indexScope()) && !(path in this.indexed)) return;
      pending.add(path);
      flush();
    };
    // After the layout is ready: while a vault loads, Obsidian reports every
    // file in it as created.
    this.app.workspace.onLayoutReady(() => {
      for (const event of ["create", "modify", "delete"]) {
        this.registerEvent(this.app.vault.on(event, (f) => f instanceof TFile && touched(f.path)));
      }
      this.registerEvent(
        this.app.vault.on("rename", (f, old) => {
          if (!(f instanceof TFile)) return;
          touched(old);
          touched(f.path);
        }),
      );
    });
  }

  onunload() {
    this.client?.close();
  }

  async save() {
    await this.saveData({
      settings: this.settings,
      indexed: this.indexed,
      exportedFiles: this.exportedFiles,
      exportedMemories: this.exportedMemories,
    });
  }

  token() {
    const id = this.settings.tokenSecret;
    return id ? (this.app.secretStorage?.getSecret(id) ?? "") : "";
  }

  // brain is the connection, made again whenever the endpoint or token changed.
  brain() {
    const token = this.token();
    if (!this.client || this.client.endpoint !== this.settings.endpoint || this.clientToken !== token) {
      this.client?.close();
      this.client = new Brain(this.settings.endpoint, token);
      this.clientToken = token;
    }
    return this.client;
  }

  async openPanel() {
    let leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE)[0];
    if (!leaf) {
      leaf = this.app.workspace.getRightLeaf(false);
      await leaf.setViewState({ type: VIEW_TYPE, active: true });
    }
    await this.app.workspace.revealLeaf(leaf);
  }

  // noteFor is the vault note a panel hit came from, if there is one: the
  // indexed note for knowledge, the exported note for a memory.
  noteFor(hit) {
    if (hit.kind === "knowledge") {
      const prefix = knowledgeIdOf(this.app.vault.getName(), "");
      return hit.id.startsWith(prefix) ? this.app.vault.getFileByPath(hit.id.slice(prefix.length)) : null;
    }
    for (const [path, was] of Object.entries(this.exportedMemories)) {
      if (was.id === hit.id) return this.app.vault.getFileByPath(path);
    }
    return null;
  }

  rememberSelection(selection, file) {
    const text = String(selection ?? "").trim();
    if (!text) {
      new Notice("Select the text to remember first.");
      return;
    }
    new RememberModal(this.app, text, async (content) => {
      const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
      const slug = content.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "").slice(0, 48) || "note";
      const metadata = { source: "obsidian", vault: this.app.vault.getName() };
      if (file) metadata.path = file.path;
      const req = { memory_id: `obsidian:${slug}-${stamp}`, content, metadata };
      if (this.settings.memoryScope) req.scope = this.settings.memoryScope;
      await this.brain().call("memory_save", req);
      new Notice("Remembered.");
    }).open();
  }

  indexScope() {
    return {
      include: folderList(this.settings.indexInclude),
      exclude: folderList(this.settings.indexExclude),
      exportFolder: normalizePath(this.settings.exportFolder || "CortexDB"),
    };
  }

  // sendNotes saves notes as knowledge and removes the ones that are gone.
  async sendNotes(paths, removed, progress = () => {}) {
    const vault = this.app.vault.getName();
    const brain = this.brain();
    let done = 0;
    for (const path of paths) {
      const file = this.app.vault.getFileByPath(path);
      if (!file) continue;
      const text = await this.app.vault.cachedRead(file);
      await brain.call("knowledge_save", knowledgeRequest(vault, path, text), 120000);
      this.indexed[path] = hashOf(text);
      if (++done % 10 === 0) {
        progress(done);
        await this.save();
      }
    }
    for (const path of removed) {
      await brain.call("knowledge_delete", { knowledge_id: knowledgeIdOf(vault, path) });
      delete this.indexed[path];
    }
    await this.save();
    return done;
  }

  async indexVault() {
    const scope = this.indexScope();
    const notes = [];
    for (const file of this.app.vault.getMarkdownFiles()) {
      if (!indexable(file.path, scope)) continue;
      notes.push({ path: file.path, hash: hashOf(await this.app.vault.cachedRead(file)) });
    }
    const plan = planIndex(notes, this.indexed);
    if (plan.save.length === 0 && plan.remove.length === 0) {
      new Notice(`CortexDB: ${notes.length === 1 ? "the one note is" : `all ${notes.length} notes are`} already in the brain.`);
      return;
    }
    const message =
      `Send ${plan.save.length} of ${count(notes.length, "note")} to the brain at ${this.settings.endpoint}` +
      (plan.remove.length ? ` and take out ${plan.remove.length} that are no longer in the vault` : "") +
      "? Every agent that shares this brain will be able to read them.";
    new ConfirmModal(this.app, "Send the vault to the brain", message, async () => {
      const notice = new Notice("CortexDB: sending notes…", 0);
      try {
        const sent = await this.sendNotes(plan.save, plan.remove, (n) => notice.setMessage(`CortexDB: sent ${n} of ${plan.save.length} notes…`));
        notice.setMessage(`CortexDB: sent ${count(sent, "note")}${plan.remove.length ? `, took out ${plan.remove.length}` : ""}.`);
      } catch (error) {
        notice.setMessage(`CortexDB: stopped — ${error.message}. Running it again carries on where it stopped.`);
      }
      setTimeout(() => notice.hide(), 8000);
    }).open();
  }

  // exportBrain checks first that no memory note holds an edit the brain has
  // not seen: the export rewrites those notes, and the edit would be gone.
  async exportBrain() {
    const unsaved = editedMemories(await this.readExportedMemories(), this.exportedMemories);
    if (unsaved.length === 0) return this.writeExport();
    new ConfirmModal(
      this.app,
      "Export the brain",
      `${count(unsaved.length, "memory note")} ${unsaved.length === 1 ? "has" : "have"} edits not saved to the brain yet. Exporting rewrites them and the edits are lost. Run “Save edited memories to the brain” first to keep them, or go ahead to discard them.`,
      () => this.writeExport(),
    ).open();
  }

  async readExportedMemories() {
    const current = {};
    for (const path of Object.keys(this.exportedMemories)) {
      const file = this.app.vault.getFileByPath(path);
      if (file) current[path] = await this.app.vault.read(file);
    }
    return current;
  }

  async writeExport() {
    const folder = normalizePath(this.settings.exportFolder || "CortexDB");
    const notice = new Notice("CortexDB: reading the brain…", 0);
    try {
      const brain = await readBrain((name, args, ms) => this.brain().call(name, args, ms), (m) => notice.setMessage(`CortexDB: ${m}`));
      const { files, bodies } = buildVault(brain, { folder, exportedAt: new Date().toISOString().slice(0, 16).replace("T", " ") });

      for (const dir of [folder, `${folder}/${MEMORIES}`, `${folder}/${ENTITIES}`]) {
        if (!this.app.vault.getFolderByPath(dir)) await this.app.vault.createFolder(dir);
      }
      let written = 0;
      for (const { path, content } of files) {
        const existing = this.app.vault.getFileByPath(path);
        if (!existing) await this.app.vault.create(path, content);
        else if ((await this.app.vault.read(existing)) !== content) await this.app.vault.modify(existing, content);
        if (++written % 200 === 0) notice.setMessage(`CortexDB: wrote ${written} of ${files.length} notes…`);
      }

      // Notes an earlier export wrote and this one did not: their memory or
      // entity is gone from the brain. They go to the trash, not away.
      const now = new Set(files.map((f) => f.path));
      const stale = this.exportedFiles.filter((p) => !now.has(p));
      for (const path of stale) {
        const file = this.app.vault.getFileByPath(path);
        if (file) await this.app.fileManager.trashFile(file);
      }

      this.exportedFiles = [...now];
      this.exportedMemories = bodies;
      await this.save();
      const memories = Object.keys(bodies).length;
      notice.setMessage(`CortexDB: exported ${count(memories, "memory", "memories")} and ${count(files.length - memories - 1, "entity", "entities")} to ${folder}${stale.length ? `; ${count(stale.length, "note")} no longer in the brain went to the trash` : ""}.`);
      const index = this.app.vault.getFileByPath(`${folder}/${INDEX_NOTE}`);
      if (index) await this.app.workspace.getLeaf(false).openFile(index);
    } catch (error) {
      notice.setMessage(`CortexDB: export failed — ${error.message}`);
    }
    setTimeout(() => notice.hide(), 10000);
  }

  async saveEditedMemories() {
    const updates = editedMemories(await this.readExportedMemories(), this.exportedMemories);
    if (updates.length === 0) {
      new Notice("CortexDB: no memory note was edited since the export.");
      return;
    }
    const names = updates.slice(0, 8).map((u) => `• ${u.memory_id}`).join("\n");
    const more = updates.length > 8 ? `\n…and ${updates.length - 8} more` : "";
    new ConfirmModal(this.app, "Save edited memories", `Replace the text of ${count(updates.length, "memory", "memories")} in the brain with your edits?\n\n${names}${more}`, async () => {
      let saved = 0;
      try {
        for (const req of updates) {
          await this.brain().call("memory_update", req);
          const path = Object.keys(this.exportedMemories).find((p) => this.exportedMemories[p].id === req.memory_id);
          if (path) this.exportedMemories[path] = { id: req.memory_id, body: req.content };
          saved++;
        }
        new Notice(`CortexDB: saved ${count(saved, "edited memory", "edited memories")}.`);
      } catch (error) {
        new Notice(`CortexDB: saved ${saved}, then failed — ${error.message}`);
      }
      await this.save();
    }).open();
  }
}

class RememberModal extends Modal {
  constructor(app, text, onSave) {
    super(app);
    this.text = text;
    this.onSave = onSave;
  }

  onOpen() {
    this.setTitle("Remember");
    this.contentEl.createEl("p", { text: "Saved as a memory every agent sharing the brain can recall. Edit it into one plain fact first if it helps.", cls: "setting-item-description" });
    const area = this.contentEl.createEl("textarea", { cls: "cortexdb-remember" });
    area.value = this.text;
    new Setting(this.contentEl).addButton((b) =>
      b
        .setButtonText("Remember")
        .setCta()
        .onClick(async () => {
          const content = area.value.trim();
          if (!content) return;
          b.setDisabled(true);
          try {
            await this.onSave(content);
            this.close();
          } catch (error) {
            new Notice(`CortexDB: ${error.message}`);
            b.setDisabled(false);
          }
        }),
    );
  }

  onClose() {
    this.contentEl.empty();
  }
}

class ConfirmModal extends Modal {
  constructor(app, title, message, onConfirm) {
    super(app);
    this.title = title;
    this.message = message;
    this.onConfirm = onConfirm;
  }

  onOpen() {
    this.setTitle(this.title);
    this.contentEl.createEl("p", { text: this.message, cls: "cortexdb-confirm" });
    new Setting(this.contentEl)
      .addButton((b) => b.setButtonText("Cancel").onClick(() => this.close()))
      .addButton((b) =>
        b
          .setButtonText("Go ahead")
          .setCta()
          .onClick(() => {
            this.close();
            this.onConfirm();
          }),
      );
  }

  onClose() {
    this.contentEl.empty();
  }
}

class CortexSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    const s = this.plugin.settings;
    const save = () => this.plugin.save();
    containerEl.empty();

    new Setting(containerEl).setName("Brain").setHeading();
    const status = new Setting(containerEl)
      .setName("Endpoint")
      .setDesc("The CortexDB gRPC server: host:port, or an https:// URL.")
      .addText((t) =>
        t
          .setPlaceholder(DEFAULT_ENDPOINT)
          .setValue(s.endpoint)
          .onChange(async (v) => {
            s.endpoint = v.trim() || DEFAULT_ENDPOINT;
            await save();
          }),
      );
    status.addButton((b) =>
      b.setButtonText("Test").onClick(async () => {
        try {
          const info = await this.plugin.brain().info();
          new Notice(`CortexDB ${info.version} at ${s.endpoint}${info.hasEmbedder ? ", with an embedder" : ""}`);
        } catch (error) {
          new Notice(`Cannot reach ${s.endpoint}: ${error.message}`);
        }
      }),
    );
    new Setting(containerEl)
      .setName("Token")
      .setDesc("The secret holding the server's bearer token, if it needs one. Secrets live in Obsidian's keychain, not in this vault.")
      .addComponent((el) =>
        new SecretComponent(this.app, el).setValue(s.tokenSecret).onChange(async (v) => {
          s.tokenSecret = v;
          await save();
        }),
      );

    new Setting(containerEl).setName("Panel").setHeading();
    new Setting(containerEl)
      .setName("Results")
      .setDesc("Memories and knowledge documents shown for a note, each.")
      .addSlider((sl) =>
        sl
          .setLimits(1, 20, 1)
          .setValue(s.topK)
          .setDynamicTooltip()
          .onChange(async (v) => {
            s.topK = v;
            await save();
          }),
      );
    new Setting(containerEl)
      .setName("Scope and namespace")
      .setDesc("Leave empty to read the whole brain, including what other agents saved.")
      .addText((t) =>
        t
          .setPlaceholder("scope")
          .setValue(s.scope)
          .onChange(async (v) => {
            s.scope = v.trim();
            await save();
          }),
      )
      .addText((t) =>
        t
          .setPlaceholder("namespace")
          .setValue(s.namespace)
          .onChange(async (v) => {
            s.namespace = v.trim();
            await save();
          }),
      );
    new Setting(containerEl)
      .setName("Scope for remembered selections")
      .addText((t) =>
        t.setValue(s.memoryScope).onChange(async (v) => {
          s.memoryScope = v.trim();
          await save();
        }),
      );

    new Setting(containerEl).setName("Sending notes").setHeading();
    new Setting(containerEl)
      .setName("Folders to send")
      .setDesc("One per line. Empty sends the whole vault. The export folder is never sent.")
      .addTextArea((t) =>
        t.setValue(s.indexInclude).onChange(async (v) => {
          s.indexInclude = v;
          await save();
        }),
      );
    new Setting(containerEl)
      .setName("Folders to leave out")
      .addTextArea((t) =>
        t.setValue(s.indexExclude).onChange(async (v) => {
          s.indexExclude = v;
          await save();
        }),
      );
    new Setting(containerEl)
      .setName("Keep the brain current")
      .setDesc("Send a note again a few seconds after it changes, and take it out when it is deleted.")
      .addToggle((t) =>
        t.setValue(s.indexOnSave).onChange(async (v) => {
          s.indexOnSave = v;
          await save();
        }),
      );

    new Setting(containerEl).setName("Export").setHeading();
    new Setting(containerEl)
      .setName("Export folder")
      .setDesc("Where Export the brain writes its notes. It is rewritten on every export.")
      .addText((t) =>
        t.setValue(s.exportFolder).onChange(async (v) => {
          s.exportFolder = v.trim() || "CortexDB";
          await save();
        }),
      );
  }
}
