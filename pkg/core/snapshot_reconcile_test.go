package core

import (
	"context"
	"database/sql"
	"fmt"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/internal/encoding"
)

const reconcileTestDim = 16

func openSnapshotStore(t *testing.T, path string, hnsw bool) *SQLiteStore {
	t.Helper()
	cfg := DefaultConfig()
	cfg.Path = path
	cfg.VectorDim = reconcileTestDim
	cfg.HNSW.Enabled = hnsw
	cfg.AutoSave.Enabled = false
	s, err := NewWithConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	return s
}

func reconcileTestVectors(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		out[i] = v
	}
	return out
}

func upsertAll(t *testing.T, s *SQLiteStore, prefix string, vecs [][]float32) {
	t.Helper()
	for i, v := range vecs {
		if err := s.Upsert(context.Background(), &Embedding{ID: fmt.Sprintf("%s%d", prefix, i), Vector: v, Content: "c"}); err != nil {
			t.Fatal(err)
		}
	}
}

func topID(t *testing.T, s *SQLiteStore, q []float32) string {
	t.Helper()
	res, err := s.Search(context.Background(), q, SearchOptions{TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		return ""
	}
	return res[0].ID
}

func closeStore(t *testing.T, s *SQLiteStore) {
	t.Helper()
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

// Two processes on one file, the way a server and a command-line tool share
// a brain: each saves the index it holds on Close, and the last one to close
// wins. The row only the first one wrote was then absent from every snapshot
// after, and a search for its own vector returned something else.
func TestReopenFindsRowAStaleSnapshotLacks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.db")
	vecs := reconcileTestVectors(40, reconcileTestDim, 1)

	seed := openSnapshotStore(t, path, true)
	upsertAll(t, seed, "base", vecs[:30])
	closeStore(t, seed)

	server := openSnapshotStore(t, path, true)
	tool := openSnapshotStore(t, path, true)
	if err := tool.Upsert(context.Background(), &Embedding{ID: "late", Vector: vecs[30], Content: "late"}); err != nil {
		t.Fatal(err)
	}
	closeStore(t, tool)                         // snapshot: base + late
	upsertAll(t, server, "server", vecs[31:33]) // the server changes too...
	closeStore(t, server)                       // ...so its snapshot, without late, replaces that one

	reopened := openSnapshotStore(t, path, true)
	defer closeStore(t, reopened)
	if got := topID(t, reopened, vecs[30]); got != "late" {
		t.Fatalf("searching for late's own vector returned %q: the loaded snapshot was not reconciled with the table", got)
	}
	if _, ok := liveNodes(reopened)["late"]; !ok {
		t.Fatal("late is stored but not in the index")
	}
}

// A row deleted by a process without the index leaves its id in the snapshot.
func TestReconcileRemovesIDNoLongerStored(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.db")
	vecs := reconcileTestVectors(20, reconcileTestDim, 2)

	s := openSnapshotStore(t, path, true)
	upsertAll(t, s, "v", vecs)
	closeStore(t, s)

	other := openSnapshotStore(t, path, false)
	if err := other.Delete(context.Background(), "v3"); err != nil {
		t.Fatal(err)
	}
	closeStore(t, other)

	reopened := openSnapshotStore(t, path, true)
	defer closeStore(t, reopened)
	live := liveNodes(reopened)
	if _, ok := live["v3"]; ok {
		t.Fatal("v3 was deleted from the table but is still live in the index")
	}
	if len(live) != 19 {
		t.Fatalf("expected 19 live nodes, got %d", len(live))
	}
}

// A row rewritten by another process after the snapshot keeps its id and
// changes its vector; the reload must index the new one.
func TestReconcileRefreshesVectorRewrittenAfterSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.db")
	vecs := reconcileTestVectors(21, reconcileTestDim, 3)

	s := openSnapshotStore(t, path, true)
	upsertAll(t, s, "v", vecs[:20])
	closeStore(t, s)

	other := openSnapshotStore(t, path, false)
	if err := other.Upsert(context.Background(), &Embedding{ID: "v5", Vector: vecs[20], Content: "rewritten"}); err != nil {
		t.Fatal(err)
	}
	closeStore(t, other)

	reopened := openSnapshotStore(t, path, true)
	defer closeStore(t, reopened)
	if got := reopened.hnswIndex.Nodes["v5"].Vector; !equalVectors(got, vecs[20]) {
		t.Fatal("the index kept v5's old vector")
	}
}

// A snapshot whose vectors are of a dimension the table no longer holds — a
// lexical index whose rows were all re-embedded — is rebuilt, not patched.
func TestReconcileRebuildsSnapshotOfAnotherDimension(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.db")
	vecs := reconcileTestVectors(20, reconcileTestDim, 4)

	s := openSnapshotStore(t, path, true)
	upsertAll(t, s, "v", vecs)
	closeStore(t, s)

	wide := reconcileTestVectors(20, 2*reconcileTestDim, 5)
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range wide {
		blob, _ := encoding.EncodeVector(v)
		if _, err := db.Exec(`UPDATE embeddings SET vector = ? WHERE id = ?`, blob, fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	_ = db.Close()

	cfg := DefaultConfig()
	cfg.Path = path
	cfg.HNSW.Enabled = true
	cfg.AutoSave.Enabled = false
	reopened, err := NewWithConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := reopened.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer closeStore(t, reopened)
	for id, dim := range liveNodes(reopened) {
		if dim != 2*reconcileTestDim {
			t.Fatalf("node %s holds a %d-d vector after reload", id, dim)
		}
	}
	if got := topID(t, reopened, wide[7]); got != "v7" {
		t.Fatalf("searching for v7's vector returned %q", got)
	}
}

// Upserting an id the index holds replaces its vector. Insert refused a live
// id, so every re-embedding left the old vector searchable.
func TestUpsertReplacesIndexedVector(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.db")
	vecs := reconcileTestVectors(21, reconcileTestDim, 6)

	s := openSnapshotStore(t, path, true)
	upsertAll(t, s, "v", vecs[:20])
	if err := s.UpsertBatch(context.Background(), []*Embedding{{ID: "v4", Vector: vecs[20], Content: "re-embedded"}}); err != nil {
		t.Fatal(err)
	}
	if got := s.hnswIndex.Nodes["v4"].Vector; !equalVectors(got, vecs[20]) {
		t.Fatal("UpsertBatch left v4's old vector in the index")
	}
	if got := topID(t, s, vecs[20]); got != "v4" {
		t.Fatalf("searching for v4's new vector returned %q", got)
	}
	closeStore(t, s)

	reopened := openSnapshotStore(t, path, true)
	defer closeStore(t, reopened)
	if got := reopened.hnswIndex.Nodes["v4"].Vector; !equalVectors(got, vecs[20]) {
		t.Fatal("the snapshot kept v4's old vector")
	}
}

// Deleting by document removes the ids from the index too.
func TestDeleteByDocIDRemovesFromIndex(t *testing.T) {
	s := openSnapshotStore(t, filepath.Join(t.TempDir(), "brain.db"), true)
	defer closeStore(t, s)
	ctx := context.Background()
	if err := s.CreateDocument(ctx, &Document{ID: "doc"}); err != nil {
		t.Fatal(err)
	}
	vecs := reconcileTestVectors(3, reconcileTestDim, 7)
	for i, v := range vecs {
		if err := s.Upsert(ctx, &Embedding{ID: fmt.Sprintf("c%d", i), Vector: v, Content: "c", DocID: "doc"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.DeleteByDocID(ctx, "doc"); err != nil {
		t.Fatal(err)
	}
	if live := liveNodes(s); len(live) != 0 {
		t.Fatalf("expected no live nodes after deleting the document, got %v", live)
	}
}

func equalVectors(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// liveNodes is the index's live ids with their vectors' dimensions.
func liveNodes(s *SQLiteStore) map[string]int {
	out := map[string]int{}
	for id, n := range s.hnswIndex.Nodes {
		if !n.Deleted {
			out[id] = len(n.Vector)
		}
	}
	return out
}
