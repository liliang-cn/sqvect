package cortexdb

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

func openIngestBrain(t *testing.T, name string) (*DB, *GraphRAGToolbox, context.Context) {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), name+".db")))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db, db.GraphRAGTools(), context.Background()
}

// The content's hash is kept with the document, so a sync job can tell a
// changed source from an unchanged one without re-reading the brain.
func TestIngestDocumentKeepsTheContentHash(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "hash")
	resp, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{
		DocumentID: "d1", Title: "One", Content: "The ledger moves to the new cluster on Sunday.",
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if len(resp.ContentSHA256) != 64 || strings.Trim(resp.ContentSHA256, "0123456789abcdef") != "" {
		t.Fatalf("content hash is not hex sha-256: %q", resp.ContentSHA256)
	}
	doc, err := db.store.GetDocument(ctx, "d1")
	if err != nil {
		t.Fatalf("get document: %v", err)
	}
	if doc.Metadata[contentHashKey] != resp.ContentSHA256 {
		t.Fatalf("record metadata hash %v, want %s", doc.Metadata[contentHashKey], resp.ContentSHA256)
	}
	node, err := db.graph.GetNode(ctx, resp.DocumentNodeID)
	if err != nil {
		t.Fatalf("get node: %v", err)
	}
	if node.Properties[contentHashKey] != resp.ContentSHA256 {
		t.Fatalf("node property hash %v, want %s", node.Properties[contentHashKey], resp.ContentSHA256)
	}
}

// A re-ingest identical to what is stored writes nothing and says so: the
// sync job that re-ingests every followed source on every pass must not
// rewrite every unchanged document.
func TestAnIdenticalReingestWritesNothing(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "unchanged")
	req := ToolIngestDocumentRequest{
		DocumentID: "d1", Title: "One", Content: "The ledger moves to the new cluster on Sunday.",
		SourceURL: "https://wiki/d1", Metadata: map[string]string{"classification": "internal"},
	}
	first, err := tools.IngestDocument(ctx, req)
	if err != nil {
		t.Fatalf("first ingest: %v", err)
	}
	again, err := tools.IngestDocument(ctx, req)
	if err != nil {
		t.Fatalf("second ingest: %v", err)
	}
	if !again.Unchanged {
		t.Fatal("an identical re-ingest was not reported unchanged")
	}
	if again.DocumentNodeID != first.DocumentNodeID || strings.Join(again.ChunkNodeIDs, ",") != strings.Join(first.ChunkNodeIDs, ",") {
		t.Fatalf("unchanged re-ingest returned other ids: %+v vs %+v", again, first)
	}
	doc, err := db.store.GetDocument(ctx, "d1")
	if err != nil {
		t.Fatalf("get document: %v", err)
	}
	if doc.Version != 1 {
		t.Fatalf("version advanced on an unchanged re-ingest: %d", doc.Version)
	}

	// Any difference is a real re-ingest: a changed classification first,
	// then a dropped metadata key, where the request says less than the
	// record.
	req.Metadata = map[string]string{"classification": "confidential"}
	changed, err := tools.IngestDocument(ctx, req)
	if err != nil {
		t.Fatalf("changed ingest: %v", err)
	}
	if changed.Unchanged {
		t.Fatal("a changed classification was reported unchanged")
	}
	req.Metadata = nil
	narrower, err := tools.IngestDocument(ctx, req)
	if err != nil {
		t.Fatalf("re-ingest without the metadata: %v", err)
	}
	if narrower.Unchanged {
		t.Fatal("a dropped metadata key was reported unchanged")
	}
	if doc, _ = db.store.GetDocument(ctx, "d1"); doc.Version != 3 {
		t.Fatalf("version after two real changes: %d", doc.Version)
	}
}

// With dedupe, the same content under a second id is not stored twice; the
// response names the document that already holds it. Without dedupe both are
// stored, as before, and the hashes say they are the same.
func TestDedupeRefusesTheSameContentUnderAnotherID(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "dedupe")
	content := "The same page saved twice from two browsers."
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "web-1", Content: content}); err != nil {
		t.Fatalf("first ingest: %v", err)
	}

	dup, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "web-2", Content: content, Dedupe: true})
	if err != nil {
		t.Fatalf("dedupe ingest: %v", err)
	}
	if dup.DuplicateOf != "web-1" || dup.DocumentNodeID != "" || len(dup.ChunkNodeIDs) != 0 {
		t.Fatalf("duplicate was not refused: %+v", dup)
	}
	if _, err := db.store.GetDocument(ctx, "web-2"); !errors.Is(err, core.ErrNotFound) {
		t.Fatalf("the duplicate was stored: %v", err)
	}

	// Without dedupe the caller gets both, and can see they match.
	both, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "web-3", Content: content})
	if err != nil {
		t.Fatalf("plain ingest: %v", err)
	}
	if both.DuplicateOf != "" || both.ContentSHA256 != dup.ContentSHA256 {
		t.Fatalf("plain ingest of the same content: %+v", both)
	}

	// Different content under a new id is never a duplicate.
	fresh, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "web-4", Content: "Something else entirely.", Dedupe: true})
	if err != nil {
		t.Fatalf("fresh ingest: %v", err)
	}
	if fresh.DuplicateOf != "" || len(fresh.ChunkNodeIDs) == 0 {
		t.Fatalf("fresh content was treated as a duplicate: %+v", fresh)
	}
}

// A brain that cannot be read reports the failure: a dedupe ingest does not
// pretend the content is new, and an ordinary re-ingest does not pretend it
// is unchanged.
func TestAnUnreadableBrainReportsTheFailureOnIngest(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "unreadable")
	req := ToolIngestDocumentRequest{DocumentID: "d1", Content: "The zebra ledger reconciles the quarterly accounts."}
	if _, err := tools.IngestDocument(ctx, req); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := tools.IngestDocument(ctx, req); err == nil {
		t.Fatal("a re-ingest on a closed brain reported success")
	}
	req.DocumentID, req.Dedupe = "d2", true
	if _, err := tools.IngestDocument(ctx, req); err == nil {
		t.Fatal("a dedupe ingest on a closed brain reported success")
	}
}

// A brain whose chunk table cannot be read reports the failure on an
// unchanged re-ingest, rather than answering with an empty chunk list.
func TestAnUnreadableChunkTableFailsTheUnchangedPath(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "chunks-unreadable")
	req := ToolIngestDocumentRequest{DocumentID: "d1", Content: "The zebra ledger reconciles the quarterly accounts."}
	if _, err := tools.IngestDocument(ctx, req); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SQL().Exec(`ALTER TABLE embeddings RENAME COLUMN doc_id TO doc_id_hidden`); err != nil {
		t.Fatal(err)
	}
	if _, err := tools.IngestDocument(ctx, req); err == nil || !strings.Contains(err.Error(), "list chunks of d1") {
		t.Fatalf("an unchanged re-ingest with an unreadable chunk table: %v", err)
	}
}

// A brain whose collections table cannot be read reports the failure, on an
// unchanged re-ingest and on a dedupe check alike, rather than answering
// from a collection it could not find.
func TestAnUnreadableCollectionTableFailsTheUnchangedPathAndDedupe(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "collections-unreadable")
	req := ToolIngestDocumentRequest{DocumentID: "d1", Content: "The zebra ledger reconciles the quarterly accounts."}
	if _, err := tools.IngestDocument(ctx, req); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SQL().Exec(`ALTER TABLE collections RENAME COLUMN name TO name_hidden`); err != nil {
		t.Fatal(err)
	}
	if _, err := tools.IngestDocument(ctx, req); err == nil || !strings.Contains(err.Error(), "find collection of d1") {
		t.Fatalf("an unchanged re-ingest with an unreadable collections table: %v", err)
	}
	req.DocumentID, req.Dedupe = "d2", true
	if _, err := tools.IngestDocument(ctx, req); err == nil || !strings.Contains(err.Error(), "collection") {
		t.Fatalf("a dedupe ingest with an unreadable collections table: %v", err)
	}
}
