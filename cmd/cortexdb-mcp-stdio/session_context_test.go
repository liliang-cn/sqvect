package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	cortexdb "github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

func TestSessionIndexListsTheProjectsCurrentMemories(t *testing.T) {
	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(t.TempDir(), "brain.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx := context.Background()
	project := []cortexdb.ToolEntityInput{{Name: "CortexDB", Type: "project"}}
	for _, m := range []cortexdb.MemorySaveRequest{
		{MemoryID: "port-old", Scope: "global", Content: "The sidecar listens on 8080.", Entities: project},
		{MemoryID: "port-new", Scope: "global", Content: "The sidecar listens on 47821.\nIt was moved off 8080.", Entities: project, Supersedes: []string{"port-old"}},
		{MemoryID: "release", Scope: "global", Content: "Releases bump five manifests with version.go.", Entities: project},
		{MemoryID: "elsewhere", Scope: "global", Content: "Athanor deploys to the apps VM.", Entities: []cortexdb.ToolEntityInput{{Name: "Athanor", Type: "project"}}},
	} {
		if _, err := db.SaveMemory(ctx, m); err != nil {
			t.Fatal(err)
		}
	}
	call := localToolCaller(db)

	index := sessionIndex(ctx, call, []string{"cortex-db"}, 30, hookContextBudget)
	for _, want := range []string{"CortexDB memories about CortexDB", "- port-new ", "The sidecar listens on 47821.", "- release "} {
		if !strings.Contains(index, want) {
			t.Fatalf("index lacks %q:\n%s", want, index)
		}
	}
	for _, unwanted := range []string{"port-old", "elsewhere", "It was moved off"} {
		if strings.Contains(index, unwanted) {
			t.Fatalf("index has %q:\n%s", unwanted, index)
		}
	}
	if got := sessionIndex(ctx, call, []string{"no-such-project"}, 30, hookContextBudget); got != "" {
		t.Fatalf("unknown project: %q", got)
	}

	// A tight budget drops whole lines and says how many.
	for i := 0; i < 10; i++ {
		if _, err := db.SaveMemory(ctx, cortexdb.MemorySaveRequest{MemoryID: fmt.Sprintf("note-%d", i), Scope: "global", Content: fmt.Sprintf("Note %d about how the brain is run day to day.", i), Entities: project}); err != nil {
			t.Fatal(err)
		}
	}
	small := sessionIndex(ctx, call, []string{"CortexDB"}, 30, 400)
	if !strings.Contains(small, "more not shown") || len([]rune(small)) > 400 {
		t.Fatalf("budget not kept (%d runes):\n%s", len([]rune(small)), small)
	}
}

func TestProjectNamesForAGitRepository(t *testing.T) {
	root := filepath.Join(t.TempDir(), "CortexDB")
	sub := filepath.Join(root, "cmd", "tool")
	if err := os.MkdirAll(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	got := projectNamesFor(sub)
	if strings.Join(got, ",") != "tool,CortexDB" {
		t.Fatalf("%v", got)
	}
	if projectKey("cortex-db") != projectKey("CortexDB") || projectKey("Athanor") == projectKey("CortexDB") {
		t.Fatal("projectKey")
	}
}

func TestFitContextKeepsTheBudget(t *testing.T) {
	lines := make([]string, 50)
	for i := range lines {
		lines[i] = "- " + strings.Repeat("记忆", 60)
	}
	out := fitContext("header", lines, "footer", 2000)
	if n := len([]rune(out)); n > 2000 || !strings.HasSuffix(out, "footer") || !strings.Contains(out, "more not shown") {
		t.Fatalf("%d runes", n)
	}
	if out := fitContext("h", []string{"- a", "- b"}, "f", 2000); out != "h\n- a\n- b\nf" {
		t.Fatalf("%q", out)
	}
	if out := fitContext(strings.Repeat("h", 100), []string{"- a"}, "", 50); out != "" {
		t.Fatalf("nothing fits, got %q", out)
	}
}
