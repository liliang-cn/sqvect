package cortexdb

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func onlyUser(user string) CallerScope {
	return CallerScope{
		Key:    user,
		Memory: func(b MemoryBucket) bool { return b.UserID == user },
	}
}

// Identical searches that arrive together share one run, and the run is
// checked against whoever started it: a caller allowed less must not be
// handed the answer of a caller allowed more.
func TestCallersWithDifferentScopesDoNotShareASearch(t *testing.T) {
	emb := &gateEmbedder{entered: make(chan struct{}), release: make(chan struct{})}
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "share.db")), WithEmbedder(emb))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.SaveMemory(context.Background(), MemorySaveRequest{
		MemoryID: "z1", UserID: "zeus", SessionID: "zeus-sess", Content: "Zeus keeps combination 9931 in the drawer.",
	}); err != nil {
		t.Fatal(err)
	}

	wide := WithCallerScope(context.Background(), CallerScope{Key: "pair", Memory: func(MemoryBucket) bool { return true }})
	narrow := WithCallerScope(context.Background(), onlyUser("hermes"))
	req := MemorySearchRequest{Query: "combination drawer", UserID: "hermes", SessionID: "zeus-sess", TopK: 5}

	emb.armed.Store(true)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); _, _ = db.SearchMemory(wide, req) }()
	<-emb.entered
	var got *MemorySearchResponse
	go func() { defer wg.Done(); got, err = db.SearchMemory(narrow, req) }()
	time.Sleep(100 * time.Millisecond) // long enough to join a shared run
	close(emb.release)
	wg.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Results) != 0 {
		t.Fatalf("hermes got the answer of a search a wider caller started: %+v", got.Results)
	}
}

// The global bucket is the shared pool and belongs to nobody, whoever writes
// to it first; a session belongs to its first user, and a save for another
// user does not land in it.
func TestABucketCannotBeTakenBeforeItsUser(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "squat.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	hermes := WithCallerScope(context.Background(), onlyUser("hermes"))

	if _, err := db.SaveMemory(hermes, MemorySaveRequest{
		MemoryID: "planted", UserID: "hermes", Scope: "global", Namespace: "fresh", Content: "planted",
	}); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("a confined caller created and owned a global bucket: %v", err)
	}
	if _, err := db.SaveMemory(context.Background(), MemorySaveRequest{
		MemoryID: "g1", UserID: "zeus", Scope: "global", Namespace: "fresh", Content: "shared note",
	}); err != nil {
		t.Fatalf("operator global save: %v", err)
	}
	if resp, err := db.SearchMemory(hermes, MemorySearchRequest{Query: "shared note", UserID: "hermes", Scope: "global", Namespace: "fresh"}); err != nil || len(resp.Results) != 0 {
		t.Fatalf("the global bucket answered to a user-confined caller: %v %+v", err, resp)
	}

	if _, err := db.SaveMemory(hermes, MemorySaveRequest{
		MemoryID: "h1", UserID: "hermes", SessionID: "zeus-future", Content: "hermes got here first",
	}); err != nil {
		t.Fatalf("hermes could not open a session of its own: %v", err)
	}
	if _, err := db.SaveMemory(context.Background(), MemorySaveRequest{
		MemoryID: "z1", UserID: "zeus", SessionID: "zeus-future", Content: "Zeus's secret",
	}); err == nil || !strings.Contains(err.Error(), "belongs to another user") {
		t.Fatalf("a save for zeus landed in a session hermes owns: %v", err)
	}
}

// gateEmbedder holds the first embedding after it is armed until released, so
// a second search can arrive while the first is still running.
type gateEmbedder struct {
	armed   atomic.Bool
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (*gateEmbedder) Dim() int { return 4 }

func (e *gateEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if e.armed.Load() {
		first := false
		e.once.Do(func() { first = true; close(e.entered) })
		if first {
			<-e.release
		}
	}
	v := []float32{1, 1, 1, 1}
	if strings.Contains(text, "drawer") {
		v[0] = 4
	}
	return v, nil
}

func (e *gateEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		out[i], _ = e.Embed(ctx, text)
	}
	return out, nil
}
