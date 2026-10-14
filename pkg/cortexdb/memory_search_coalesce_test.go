package cortexdb

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
)

// Identical searches arriving together get the same answer, each its own
// copy, and every one of them counts as a recall.
func TestConcurrentIdenticalSearchesShareARunAndCountEveryRecall(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "flights.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx := context.Background()
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "m1", Scope: "global", Namespace: "hive", Content: "The hive recalls memory through the shared brain at the floating IP."}); err != nil {
		t.Fatal(err)
	}
	req := MemorySearchRequest{Query: "shared brain floating IP", Scope: "global", Namespace: "hive", TopK: 3}

	const callers = 8
	var wg sync.WaitGroup
	responses := make([]*MemorySearchResponse, callers)
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			resp, err := db.SearchMemory(ctx, req)
			if err != nil {
				t.Error(err)
				return
			}
			responses[i] = resp
		}(i)
	}
	wg.Wait()
	for i, resp := range responses {
		if resp == nil || len(resp.Results) != 1 || resp.Results[0].Memory.ID != "m1" {
			t.Fatalf("caller %d: %+v", i, resp)
		}
	}
	responses[0].Results[0].Memory.Metadata["poisoned"] = true
	if _, ok := responses[1].Results[0].Memory.Metadata["poisoned"]; ok {
		t.Fatal("callers share one metadata map")
	}

	got, err := db.GetMemory(ctx, MemoryGetRequest{MemoryID: "m1"})
	if err != nil {
		t.Fatal(err)
	}
	if n := memoryRecallCount(got.Memory.Metadata); n != callers {
		t.Fatalf("recall_count = %d after %d recalls", n, callers)
	}

	// A search on its own still counts once.
	if _, err := db.SearchMemory(ctx, req); err != nil {
		t.Fatal(err)
	}
	got, _ = db.GetMemory(ctx, MemoryGetRequest{MemoryID: "m1"})
	if n := memoryRecallCount(got.Memory.Metadata); n != callers+1 {
		t.Fatalf("recall_count = %d, want %d", n, callers+1)
	}
}

// A caller that gives up gets its own context's error, not a stale answer.
func TestASearchItsCallerLeftReturnsTheCallersError(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "left.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := db.SearchMemory(ctx, MemorySearchRequest{Query: "anything", Scope: "global"}); err == nil {
		t.Fatal("want the cancelled caller's error")
	}
}
