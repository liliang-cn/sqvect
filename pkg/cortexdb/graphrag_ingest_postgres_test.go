package cortexdb

import (
	"context"
	"database/sql"
	"os"
	"strings"
	"sync"
	"testing"
)

// The ingest paths on PostgreSQL, where the dedupe lookup locks the hash
// for the transaction: two racing ingests store one copy, a re-ingest over
// other text replaces the chunks, and the same text again is unchanged.
func TestTheIngestPathsHoldOnPostgres(t *testing.T) {
	db := openPostgresBrain(t, 4)
	tools := db.GraphRAGTools()
	ctx := context.Background()
	content := "The same page saved from two browsers at once, on PostgreSQL."
	var wg sync.WaitGroup
	results := make([]*ToolIngestDocumentResponse, 2)
	errs := make([]error, 2)
	for i, id := range []string{"pg-a", "pg-b"} {
		wg.Add(1)
		go func(i int, id string) {
			defer wg.Done()
			results[i], errs[i] = tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: id, Content: content, Dedupe: true})
		}(i, id)
	}
	wg.Wait()
	duplicates := 0
	for i := range results {
		if errs[i] != nil {
			t.Fatalf("ingest %d: %v", i, errs[i])
		}
		if results[i].DuplicateOf != "" {
			duplicates++
		}
	}
	if duplicates != 1 {
		t.Fatalf("%d duplicates, want 1: %+v %+v", duplicates, results[0], results[1])
	}
	// duplicate_of names the copy that was stored; the other answer is empty.
	stored := results[0].DuplicateOf + results[1].DuplicateOf
	again, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: stored, Content: content})
	if err != nil || !again.Unchanged {
		t.Fatalf("the same text again: %+v %v", again, err)
	}
	other, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: stored, Content: "Other text, one chunk."})
	if err != nil || other.Unchanged || len(other.ChunkNodeIDs) != 1 {
		t.Fatalf("other text: %+v %v", other, err)
	}
	chunks, err := db.store.GetByDocID(ctx, stored)
	if err != nil || len(chunks) != 1 || !strings.Contains(chunks[0].Content, "Other text") {
		t.Fatalf("chunk rows after other text: %d %v", len(chunks), err)
	}
}

// A dedupe ingest that cannot take the hash lock reports it and writes
// nothing. Another session holds the lock, and this brain's sessions give up
// a lock wait after 200 ms (lock_timeout on their connection string).
func TestADedupeIngestThatCannotLockTheHashIsReported(t *testing.T) {
	dsn := os.Getenv("CORTEXDB_TEST_POSTGRES")
	if dsn == "" {
		t.Skip("CORTEXDB_TEST_POSTGRES unset — PostgreSQL is NOT covered by this run")
	}
	sep := "?"
	if strings.Contains(dsn, "?") {
		sep = "&"
	}
	t.Setenv("CORTEXDB_TEST_POSTGRES", dsn+sep+"options=-c%20lock_timeout%3D200ms")
	db := openPostgresBrain(t, 4)
	ctx := context.Background()
	content := "A page whose hash another session holds."

	holder, err := sql.Open("pgx", dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Close()
	lock, err := holder.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = lock.Rollback() }()
	if _, err := lock.ExecContext(ctx, `SELECT pg_advisory_xact_lock(hashtext($1))`, contentSHA256(content)); err != nil {
		t.Fatal(err)
	}

	_, err = db.GraphRAGTools().IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "locked", Content: content, Dedupe: true})
	if err == nil || !strings.Contains(err.Error(), "lock content hash") {
		t.Fatalf("ingest under a held lock: %v", err)
	}
	if _, err := db.store.GetDocument(ctx, "locked"); err == nil {
		t.Fatal("the record was written although the lock was not taken")
	}
}
