package liveview

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// Clicking a memory node shows what was saved, not the clipped summary the
// node is labelled with.
func TestARecordForAMemoryNodeCarriesTheWholeMemory(t *testing.T) {
	ctx := context.Background()
	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(t.TempDir(), "brain.db")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	long := "The brain runs as its own HA resource. " + strings.Repeat("Every node in the chain follows the Primary. ", 20)
	if _, err := db.SaveMemory(ctx, cortexdb.MemorySaveRequest{
		MemoryID: "brain-ha", Scope: "global", Content: long, Importance: 0.8,
		Metadata: map[string]any{"type": "project", "project": "cortexdb", "agentgo": "{\"blob\":1}"},
		Entities: []cortexdb.ToolEntityInput{{Name: "CortexDB", Type: "project"}},
	}); err != nil {
		t.Fatalf("save: %v", err)
	}

	rec, err := SourceFor(db, "test").Record(ctx, "memory:brain-ha")
	if err != nil {
		t.Fatalf("record: %v", err)
	}
	if !rec.Found || rec.Memory == nil {
		t.Fatalf("record found %v, memory %+v; want the memory", rec.Found, rec.Memory)
	}
	if rec.Memory.Content != long {
		t.Errorf("memory content is %d chars, want the whole %d", len(rec.Memory.Content), len(long))
	}
	if rec.Memory.Scope != "global" || rec.Memory.Importance != 0.8 || rec.Memory.CreatedAt == "" {
		t.Errorf("memory = %+v, want scope global, importance 0.8 and a save time", rec.Memory)
	}
	facts := map[string]string{}
	for _, f := range rec.Memory.Facts {
		facts[f.Key] = f.Value
	}
	if facts["type"] != "project" || facts["project"] != "cortexdb" {
		t.Errorf("facts = %v, want type and project", facts)
	}
	if _, ok := facts["agentgo"]; ok {
		t.Error("the agentgo blob is shown as a fact")
	}
}

func TestMemoryNodeID(t *testing.T) {
	for _, c := range []struct {
		typ, id, want string
		ok            bool
	}{
		{"memory", "memory:abc", "abc", true},
		{"", "memory:abc", "abc", true},
		{"entity", "memory:abc", "", false},
		{"memory", "entity:abc", "", false},
	} {
		got, ok := memoryNodeID(c.typ, c.id)
		if got != c.want || ok != c.ok {
			t.Errorf("memoryNodeID(%q, %q) = %q, %v; want %q, %v", c.typ, c.id, got, ok, c.want, c.ok)
		}
	}
}
