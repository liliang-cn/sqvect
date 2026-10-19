package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAPromptIsAskedWithTheProjectAndTheLastReply(t *testing.T) {
	transcript := filepath.Join(t.TempDir(), "s.jsonl")
	lines := []string{
		`{"type":"user","message":{"content":"linstor-server: resize leaves the old size"}}`,
		`{"type":"assistant","message":{"content":[{"type":"text","text":"Root cause: VolumeDefinition resize skips the satellite update.\nI suggest a fix plus a regression test."}]}}`,
		`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{"file_path":"a.go"}}]}}`,
		`{"type":"user","message":{"content":"按建议 · 这个修复前后可以验证吗"}}`,
	}
	if err := os.WriteFile(transcript, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := situationFor(hookInput{Cwd: "/Users/a/Works/linstor-server", TranscriptPath: transcript})

	got := s.query("按建议 · 这个修复前后可以验证吗")
	want := "按建议 · 这个修复前后可以验证吗\n\nProject: linstor-server\nContext: Root cause: VolumeDefinition resize skips the satellite update. I suggest a fix plus a regression test."
	if got != want {
		t.Fatalf("query =\n%q\nwant\n%q", got, want)
	}
	kw := s.keywords("按建议 · 这个修复前后可以验证吗")
	if kw[len(kw)-1] != "linstor-server" {
		t.Fatalf("keywords %v do not name the project", kw)
	}
	if e := s.entities(); len(e) != 1 || e[0] != "linstor-server" {
		t.Fatalf("entities = %v", e)
	}
}

func TestASessionInTheHomeDirectoryNamesNoProject(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	s := situationFor(hookInput{Cwd: home})
	if got := s.query("what port?"); got != "what port?" {
		t.Fatalf("query = %q, want the prompt as written", got)
	}
}

func TestEachHitCarriesWhereItCameFromAndWhyItCameBack(t *testing.T) {
	hits, err := parseRecallPayload(`{
		"memory_decision": {"effective_mode": "auto"},
		"knowledge_decision": {"effective_mode": "hybrid"},
		"memories": [{"score": 0.9312, "similarity": 0.8037, "via": ["semantic", "keyword"], "memory": {"id": "auto:d0b123ed:verify-fix", "content": "verify by reverting the fix",
			"created_at": "2026-10-09T02:25:42Z",
			"metadata": {"date": "2026-10-09", "source": "auto-capture", "model": "haiku", "agent": "claude-code",
				"host": "mac", "project": "linstor-server", "session": "d0b123ed-e6d6", "written_by": "legacy-token"}}}],
		"knowledge": [{"knowledge_id": "noted:1", "title": "Resize runbook", "snippet": "Resize, then verify.",
			"author": "noted", "source_url": "noted://note/1", "vector_score": 0.3026, "lexical_score": 0.7809,
			"metadata": {"updated": "2026-10-18T09:00:28Z", "written_by": "hermes"}}]
	}`)
	if err != nil {
		t.Fatal(err)
	}
	got := formatRecallHits(hits)
	for _, want := range []string{
		"- auto:d0b123ed:verify-fix [2026-10-09 · claude-code auto-capture (haiku) · host mac · project linstor-server · session d0b123ed · semantic 0.80 + keyword]: verify by reverting the fix",
		"- Resize runbook [2026-10-18 · noted · key hermes · noted://note/1 · semantic 0.30 + keyword 0.78]: Resize, then verify.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if strings.Contains(got, "legacy-token") {
		t.Errorf("the shared token was shown as a writer:\n%s", got)
	}
}

func TestAMemoryFoundByKeywordsAloneSaysSo(t *testing.T) {
	hits, err := parseRecallPayload(`{"memory_decision": {"effective_mode": "lexical"},
		"memories": [{"score": 0.42, "memory": {"id": "m1", "content": "x"}}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if got := hits[0].Trail; got != "keyword" {
		t.Fatalf("trail = %q, want the keyword retriever named", got)
	}
}
