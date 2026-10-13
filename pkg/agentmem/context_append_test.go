package agentmem_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/agentmem"
)

func TestAppendContextKeepsConcurrentSections(t *testing.T) {
	store, cleanup := newStore(t)
	defer cleanup()
	ctx := context.Background()
	scope := agentmem.Scope{Type: agentmem.ScopeUser, ID: "writer"}
	slot := agentmem.ContextSlot("notes")
	if err := store.SetContext(ctx, scope, slot, "seed"); err != nil {
		t.Fatal(err)
	}
	const writers = 32
	start := make(chan struct{})
	done := make(chan error, writers)
	for i := 0; i < writers; i++ {
		go func(i int) {
			<-start
			done <- store.AppendContext(ctx, scope, slot, fmt.Sprintf("section-%02d", i))
		}(i)
	}
	close(start)
	for i := 0; i < writers; i++ {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
	got, err := store.GetContext(ctx, scope, slot)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(got, "\n")
	if len(lines) != writers+1 {
		t.Fatalf("got %d sections, want %d; concurrent appends lost data: %q", len(lines), writers+1, got)
	}
	for i := 0; i < writers; i++ {
		if strings.Count(got, fmt.Sprintf("section-%02d", i)) != 1 {
			t.Fatalf("section %d missing or duplicated: %q", i, got)
		}
	}
}

func TestAppendContextKeepsNewlineAndCreationBehavior(t *testing.T) {
	store, cleanup := newStore(t)
	defer cleanup()
	ctx := context.Background()
	scope := agentmem.Scope{Type: agentmem.ScopeUser, ID: "writer"}
	slot := agentmem.ContextSlot("notes")
	for _, section := range []string{"first\n", "second", "third"} {
		if err := store.AppendContext(ctx, scope, slot, section); err != nil {
			t.Fatal(err)
		}
	}
	got, err := store.GetContext(ctx, scope, slot)
	if err != nil || got != "first\nsecond\nthird" {
		t.Fatalf("content = %q, error = %v", got, err)
	}
	if err := store.AppendContext(ctx, scope, "", "invalid"); err == nil {
		t.Fatal("empty slot accepted")
	}
}
