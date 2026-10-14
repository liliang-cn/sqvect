// What the brain is asked about an open note, and what comes back.
//
// A note is not a prompt: it has frontmatter, code, link syntax and can run to
// thousands of words. The question is its title and the prose it opens with —
// what a person would say the note is about — and the keywords are that prose's
// words, deduplicated, the way the recall hooks build them.

import { keywordsFromPrompt } from "../../dsh-cortexdb-memory/lib/recall.js";

// QUERY_CHARS is how much of a note's prose goes into the question. Recall
// ranks on what a note is about; the tail of a long note only dilutes that.
export const QUERY_CHARS = 1200;
const MAX_KEYWORDS = 40;

// proseOf is a note's text without frontmatter, code, comments or link markup.
export function proseOf(text) {
  let s = String(text ?? "").replace(/^﻿/, "");
  s = s.replace(/^---\r?\n[\s\S]*?\r?\n---[^\n]*(\n|$)/, "");
  s = s.replace(/^(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1[^\n]*$/gm, " ");
  s = s.replace(/%%[\s\S]*?%%/g, " ").replace(/<!--[\s\S]*?-->/g, " ");
  s = s.replace(/!\[\[[^\]]*\]\]/g, " ").replace(/!\[[^\]]*\]\([^)]*\)/g, " ");
  s = s.replace(/\[\[([^\]|#]*)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]/g, (_, target, alias) => alias || target);
  s = s.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");
  s = s.replace(/`([^`]*)`/g, "$1");
  s = s.replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+(\[.\]\s+)?/gm, "");
  return s.split(/\s+/).filter(Boolean).join(" ");
}

// noteQuestion is the question asked for a note: its title, then the start of
// its prose.
export function noteQuestion(title, text, maxChars = QUERY_CHARS) {
  const prose = [...proseOf(text)].slice(0, maxChars).join("");
  return [String(title ?? "").trim(), prose].filter(Boolean).join("\n");
}

// recallArgs is the knowledge_memory_recall request for a question. Scope and
// namespace are only sent when configured: left out, recall reads the whole
// brain, which is where every agent's memories live.
export function recallArgs(question, config) {
  const args = {
    query: question,
    keywords: keywordsFromPrompt(question).slice(0, MAX_KEYWORDS),
    top_k_memories: config.topK,
    top_k_knowledge: config.topK,
    graph_light: true,
  };
  if (config.scope) args.scope = config.scope;
  if (config.namespace) args.namespace = config.namespace;
  return args;
}

// hitsOf reads a recall answer into rows for the panel, memories first.
// `skip` names the note's own knowledge id or memory id: an indexed note, or
// an exported memory, is always its own best match, and telling a person what
// they are reading says nothing.
export function hitsOf(payload, { knowledge: skipKnowledge = "", memory: skipMemory = "" } = {}) {
  const hits = [];
  for (const hit of payload?.memories ?? []) {
    const m = hit?.memory ?? {};
    if (!m.id || m.id === skipMemory) continue;
    hits.push({
      kind: "memory",
      id: m.id,
      title: firstLine(m.content) || m.id,
      text: String(m.content ?? "").trim(),
      more: afterFirstLine(m.content),
      date: dateOf(m.created_at),
      superseded: Boolean(String(m.metadata?.superseded_by ?? "").trim()),
      score: Number(hit?.score ?? 0),
    });
  }
  for (const hit of payload?.knowledge ?? []) {
    if (!hit?.knowledge_id || hit.knowledge_id === skipKnowledge) continue;
    hits.push({
      kind: "knowledge",
      id: hit.knowledge_id,
      title: String(hit.title ?? "").trim() || hit.knowledge_id,
      text: String(hit.snippet ?? "").trim(),
      more: String(hit.snippet ?? "").trim(),
      source: String(hit.source_url ?? ""),
      score: Number(hit?.score ?? 0),
    });
  }
  return hits;
}

// entitiesOf is the graph entities recall touched, for the panel to offer as
// next questions.
export function entitiesOf(payload, max = 12) {
  const seen = new Set();
  const out = [];
  for (const fact of payload?.graph_facts ?? []) {
    for (const name of [fact?.subject, fact?.object]) {
      const key = String(name ?? "").toLowerCase();
      if (!key || seen.has(key)) continue;
      seen.add(key);
      out.push(String(name));
      if (out.length >= max) return out;
    }
  }
  return out;
}

export function firstLine(text) {
  for (const line of String(text ?? "").split("\n")) {
    const t = line.replace(/^#+\s*/, "").trim();
    if (t) return [...t].length > 120 ? `${[...t].slice(0, 120).join("").trim()}…` : t;
  }
  return "";
}

// afterFirstLine is what a memory says beyond the line used as its title.
function afterFirstLine(text) {
  const lines = String(text ?? "").trim().split("\n");
  const first = lines.findIndex((l) => l.trim());
  if (first < 0) return "";
  const rest = lines.slice(first + 1).join("\n").trim();
  const title = lines[first].replace(/^#+\s*/, "").trim();
  // A first line too long for a title is cut there; the rest of it is still
  // worth showing.
  return [...title].length > 120 ? String(text).trim() : rest;
}

function dateOf(value) {
  const s = String(value ?? "");
  return /^\d{4}-\d{2}-\d{2}/.test(s) && !s.startsWith("0001-") ? s.slice(0, 10) : "";
}
