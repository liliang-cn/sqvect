package cortexdb

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// An index recall finds what a full one finds, each hit cut to its first
// line, and memory_get memory_ids fetches the ones picked from it whole.
func TestRecallIndexThenFetch(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "index.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx := context.Background()
	long := "Deploys go through the staging cluster first.\n" + strings.Repeat("Then the canary runs for an hour and is watched. ", 20)
	for _, m := range []MemorySaveRequest{
		{MemoryID: "deploy-staging", Scope: MemoryScopeGlobal, Content: long},
		{MemoryID: "deploy-window", Scope: MemoryScopeGlobal, Content: "Deploys happen on Tuesdays, never on Fridays."},
	} {
		if _, err := db.SaveMemory(ctx, m); err != nil {
			t.Fatal(err)
		}
	}

	full, err := db.KnowledgeMemory().Recall(ctx, KnowledgeMemoryRecallRequest{Query: "deploys", Keywords: []string{"deploys"}, DisableKnowledge: true})
	if err != nil {
		t.Fatal(err)
	}
	index, err := db.KnowledgeMemory().Recall(ctx, KnowledgeMemoryRecallRequest{Query: "deploys", Keywords: []string{"deploys"}, DisableKnowledge: true, Detail: RecallDetailIndex})
	if err != nil {
		t.Fatal(err)
	}
	if index.Detail != RecallDetailIndex || len(index.Memories) != len(full.Memories) || len(index.Memories) == 0 {
		t.Fatalf("index found %d memories (detail %q), full found %d", len(index.Memories), index.Detail, len(full.Memories))
	}
	var ids []string
	for i, hit := range index.Memories {
		if hit.Memory.ID != full.Memories[i].Memory.ID {
			t.Fatalf("index reordered hits: %s vs %s", hit.Memory.ID, full.Memories[i].Memory.ID)
		}
		if strings.Contains(hit.Memory.Content, "\n") || len([]rune(hit.Memory.Content)) > indexLineRunes+1 {
			t.Fatalf("index line not cut: %q", hit.Memory.Content)
		}
		if !strings.Contains(index.ContextPack.Text, hit.Memory.ID) {
			t.Fatalf("context pack does not list %s:\n%s", hit.Memory.ID, index.ContextPack.Text)
		}
		ids = append(ids, hit.Memory.ID)
	}
	if len(index.ContextPack.Text) >= len(full.ContextPack.Text) {
		t.Fatalf("index pack (%d chars) is not smaller than the full one (%d)", len(index.ContextPack.Text), len(full.ContextPack.Text))
	}

	got, err := db.GetMemory(ctx, MemoryGetRequest{MemoryIDs: append(ids, "no-such-memory")})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Memories) != len(ids) || len(got.Missing) != 1 || got.Missing[0] != "no-such-memory" {
		t.Fatalf("memory_ids: %d memories, missing %v", len(got.Memories), got.Missing)
	}
	for i, m := range got.Memories {
		if m.ID != ids[i] {
			t.Fatalf("order: %s at %d, want %s", m.ID, i, ids[i])
		}
		if m.ID == "deploy-staging" && m.Content != long {
			t.Fatal("memory_get returned a cut memory")
		}
	}
}

func TestFirstLine(t *testing.T) {
	if got := firstLine("\n  two  words \nsecond", 120); got != "two words" {
		t.Fatalf("%q", got)
	}
	if got := firstLine(strings.Repeat("记", 130), 120); len([]rune(got)) != 121 || !strings.HasSuffix(got, "…") {
		t.Fatalf("%q", got)
	}
}
