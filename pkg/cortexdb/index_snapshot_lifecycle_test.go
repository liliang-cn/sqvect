package cortexdb

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

var lifecycleWords = []string{
	"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
	"india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
	"quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
}

// openLikeServer opens the DB the way cmd/cortexdb-grpc does: DefaultConfig
// and an embedder.
func openLikeServer(t *testing.T, path string) *DB {
	t.Helper()
	db, err := Open(DefaultConfig(path), WithEmbedder(newKeywordEmbedder(lifecycleWords...)))
	if err != nil {
		t.Fatal(err)
	}
	return db
}

func saveWord(t *testing.T, db *DB, word string) {
	t.Helper()
	if _, err := db.SaveKnowledge(context.Background(), KnowledgeSaveRequest{
		KnowledgeID: "k-" + word,
		Content:     fmt.Sprintf("%s %s %s", word, word, word),
	}); err != nil {
		t.Fatal(err)
	}
}

// nearestDoc is the document of the chunk nearest to word's vector.
func nearestDoc(t *testing.T, db *DB, word string) string {
	t.Helper()
	vec, err := newKeywordEmbedder(lifecycleWords...).Embed(context.Background(), word)
	if err != nil {
		t.Fatal(err)
	}
	res, err := db.Vector().Search(context.Background(), vec, core.SearchOptions{TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		return ""
	}
	return res[0].DocID
}

func closeDB(t *testing.T, db *DB) {
	t.Helper()
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

// The server's lifecycle: knowledge saved in one run is found by vector
// search in the next, across clean Close and reopen. This held before the
// fix too; it pins that the facade's Close does save the snapshot.
func TestKnowledgeVectorSurvivesServerRestarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.db")

	db := openLikeServer(t, path)
	for _, w := range lifecycleWords[:12] {
		saveWord(t, db, w)
	}
	closeDB(t, db)

	db = openLikeServer(t, path)
	saveWord(t, db, "xray")
	closeDB(t, db)

	db = openLikeServer(t, path)
	defer closeDB(t, db)
	if got := nearestDoc(t, db, "xray"); got != "k-xray" {
		t.Fatalf("knowledge saved in the previous run is not found by vector search: nearest is %q", got)
	}
}

// The same brain opened by a second process — a command-line tool beside the
// server — while the server runs. Each saves the index it holds on Close, so
// the server's, which never saw the tool's write, replaced the tool's
// snapshot, and the row was missing from the index on every open after.
func TestKnowledgeVectorSurvivesSecondProcessOnSameFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.db")

	db := openLikeServer(t, path)
	for _, w := range lifecycleWords[:12] {
		saveWord(t, db, w)
	}
	closeDB(t, db)

	server := openLikeServer(t, path)
	tool := openLikeServer(t, path)
	saveWord(t, tool, "xray")
	closeDB(t, tool)
	saveWord(t, server, "whiskey")
	closeDB(t, server)

	db = openLikeServer(t, path)
	defer closeDB(t, db)
	if got := nearestDoc(t, db, "xray"); got != "k-xray" {
		t.Fatalf("knowledge written by the second process is not found by vector search: nearest is %q", got)
	}
	if got := nearestDoc(t, db, "whiskey"); got != "k-whiskey" {
		t.Fatalf("knowledge written by the server is not found by vector search: nearest is %q", got)
	}
}
