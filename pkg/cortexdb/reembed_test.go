package cortexdb

import (
	"context"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/internal/encoding"
	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

// fixedDimEmbedder produces vectors of one size, standing in for a concrete model.
type fixedDimEmbedder struct {
	dim   int
	calls int
}

func (e *fixedDimEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.calls++
	vector := make([]float32, e.dim)
	for i := range vector {
		vector[i] = float32(len(text)%7+1) / float32(i+1)
	}
	return vector, nil
}

func (e *fixedDimEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for _, text := range texts {
		vector, err := e.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		out = append(out, vector)
	}
	return out, nil
}

func (e *fixedDimEmbedder) Dim() int { return e.dim }

// A store that outlived an embedding-model change holds vectors of the old size. They
// cannot enter the vector index, so they stop being retrievable by similarity while
// lexical search still finds them — the loss is invisible without a report.
func newReembedDB(t *testing.T, embedder Embedder) *DB {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "reembed.db")), WithEmbedder(embedder))
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// seedRow writes a row straight into SQLite, bypassing Upsert's dimension adaptation.
// That is how legacy rows actually came to exist: they were written while the store's
// dimension *was* the old one, and Upsert would now quietly adapt them instead.
func seedRow(t *testing.T, db *DB, id, collection, content string, dim int) {
	t.Helper()
	ctx := context.Background()
	if _, err := db.store.GetCollection(ctx, collection); err != nil {
		if _, err := db.store.CreateCollection(ctx, collection, dim); err != nil {
			t.Fatalf("failed to create collection %s: %v", collection, err)
		}
	}
	created, err := db.store.GetCollection(ctx, collection)
	if err != nil {
		t.Fatalf("failed to read collection %s: %v", collection, err)
	}
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = 0.25
	}
	blob, err := encoding.EncodeVector(vector)
	if err != nil {
		t.Fatalf("failed to encode vector: %v", err)
	}
	if _, err := db.store.GetDB().ExecContext(ctx,
		`INSERT INTO embeddings (id, collection_id, vector, content, metadata) VALUES (?, ?, ?, ?, '{}')`,
		id, created.ID, blob, content); err != nil {
		t.Fatalf("failed to seed %s: %v", id, err)
	}
}

func TestDimensionReportSurfacesDrift(t *testing.T) {
	db := newReembedDB(t, &fixedDimEmbedder{dim: 8})
	ctx := context.Background()

	seedRow(t, db, "new-1", "current", "现代分块", 8)
	seedRow(t, db, "old-1", "legacy", "旧模型分块一", 32)
	seedRow(t, db, "old-2", "legacy", "旧模型分块二", 32)

	report, err := db.DimensionReport(ctx)
	if err != nil {
		t.Fatalf("dimension report failed: %v", err)
	}
	byName := map[string]core.CollectionDimensions{}
	for _, entry := range report.Collections {
		byName[entry.Collection] = entry
	}
	if got := byName["current"].RowsWithDim(8); got != 1 {
		t.Errorf("collection current: 8-dim rows = %d, want 1 (%+v)", got, byName["current"].Dimensions)
	}
	if got := byName["legacy"].RowsWithDim(32); got != 2 {
		t.Errorf("collection legacy: 32-dim rows = %d, want 2 (%+v)", got, byName["legacy"].Dimensions)
	}
	if byName["legacy"].Mismatched != 0 {
		t.Errorf("legacy declared 32 dims, so its rows are not mismatched: %+v", byName["legacy"])
	}
	if !report.NeedsRepair() && report.Mismatched != 0 {
		t.Error("NeedsRepair disagrees with the mismatch count")
	}
}

func TestReembedMismatchedVectorsRepairsOldModelRows(t *testing.T) {
	embedder := &fixedDimEmbedder{dim: 8}
	db := newReembedDB(t, embedder)
	ctx := context.Background()

	seedRow(t, db, "new-1", "current", "现代分块", 8)
	seedRow(t, db, "old-1", "legacy", "旧模型分块一", 32)
	seedRow(t, db, "old-2", "legacy", "旧模型分块二", 32)

	// Dry run reports the candidates and writes nothing.
	dry, err := db.ReembedMismatchedVectors(ctx, ReembedOptions{DryRun: true})
	if err != nil {
		t.Fatalf("dry run failed: %v", err)
	}
	if dry.Candidates != 2 {
		t.Errorf("dry run found %d candidates, want 2", dry.Candidates)
	}
	if dry.Reembedded != 0 {
		t.Errorf("dry run rewrote %d rows, want 0", dry.Reembedded)
	}
	if embedder.calls != 0 {
		t.Errorf("dry run called the embedder %d times, want 0", embedder.calls)
	}

	got, err := db.ReembedMismatchedVectors(ctx, ReembedOptions{BatchSize: 1})
	if err != nil {
		t.Fatalf("re-embed failed: %v", err)
	}
	if got.Reembedded != 2 || got.Failed != 0 {
		t.Fatalf("re-embedded %d rows with %d failures, want 2/0 (%v)", got.Reembedded, got.Failed, got.Errors)
	}

	// Every row now shares the embedder's dimensionality, so nothing is left adrift.
	after, err := db.ReembedMismatchedVectors(ctx, ReembedOptions{DryRun: true})
	if err != nil {
		t.Fatalf("post-repair dry run failed: %v", err)
	}
	if after.Candidates != 0 {
		t.Errorf("%d rows still mismatched after repair", after.Candidates)
	}
	report, err := db.DimensionReport(ctx)
	if err != nil {
		t.Fatalf("post-repair report failed: %v", err)
	}
	for _, entry := range report.Collections {
		if entry.RowsWithDim(32) != 0 {
			t.Errorf("collection %s still holds 32-dim vectors: %+v", entry.Collection, entry.Dimensions)
		}
	}

	// The collection's declared dimension is brought in line, otherwise the drift report
	// would keep flagging rows that are now correct.
	if got.Reconciled == 0 {
		t.Error("no collection had its declared dimension reconciled")
	}
	if report.Mismatched != 0 {
		t.Errorf("%d rows still reported as mismatched after repair", report.Mismatched)
	}
	for _, entry := range report.Collections {
		if entry.Declared != 8 {
			t.Errorf("collection %s still declares %d dimensions", entry.Collection, entry.Declared)
		}
	}

	// Content is untouched — only the vector is recomputed.
	row, err := db.store.GetByID(ctx, "old-1")
	if err != nil {
		t.Fatalf("failed to read repaired row: %v", err)
	}
	if row.Content != "旧模型分块一" {
		t.Errorf("content changed to %q", row.Content)
	}
	if len(row.Vector) != 8 {
		t.Errorf("repaired vector has %d dimensions, want 8", len(row.Vector))
	}
}

// Vectors repaired by an earlier pass leave correct numbers behind stale metadata; a
// second real pass has nothing to embed but must still reconcile the collection.
func TestRepairReconcilesMetadataWithNothingToReembed(t *testing.T) {
	embedder := &fixedDimEmbedder{dim: 8}
	db := newReembedDB(t, embedder)
	ctx := context.Background()

	// A collection declaring 32 whose rows are already the embedder's size.
	seedRow(t, db, "row-1", "legacy", "已经修好的分块", 32)
	if _, err := db.store.GetDB().ExecContext(ctx,
		`UPDATE embeddings SET vector = ? WHERE id = 'row-1'`, mustEncode(t, 8)); err != nil {
		t.Fatalf("failed to stage repaired vector: %v", err)
	}

	got, err := db.ReembedMismatchedVectors(ctx, ReembedOptions{})
	if err != nil {
		t.Fatalf("repair failed: %v", err)
	}
	if got.Candidates != 0 || got.Reembedded != 0 {
		t.Errorf("expected nothing to re-embed, got candidates=%d reembedded=%d", got.Candidates, got.Reembedded)
	}
	if got.Reconciled != 1 {
		t.Errorf("reconciled %d collections, want 1", got.Reconciled)
	}
	report, err := db.DimensionReport(ctx)
	if err != nil {
		t.Fatalf("report failed: %v", err)
	}
	if report.Mismatched != 0 {
		t.Errorf("report still flags %d rows after reconciliation", report.Mismatched)
	}
}

func mustEncode(t *testing.T, dim int) []byte {
	t.Helper()
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = 0.5
	}
	blob, err := encoding.EncodeVector(vector)
	if err != nil {
		t.Fatalf("failed to encode vector: %v", err)
	}
	return blob
}

func TestReembedWithoutEmbedderIsRefused(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "no-embedder.db")))
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	defer func() { _ = db.Close() }()

	if _, err := db.ReembedMismatchedVectors(context.Background(), ReembedOptions{}); err == nil {
		t.Error("re-embedding without an embedder was accepted")
	}
}

// A repair changes a row's vector and nothing else: the document it belongs
// to, its metadata and its ACL stay. It used to write back a row read with its
// id and content alone, so every repaired chunk lost its document, its chunk
// metadata and who may read it (#84).
func TestARepairChangesOnlyTheVector(t *testing.T) {
	db := newReembedDB(t, &fixedDimEmbedder{dim: 8})
	ctx := context.Background()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{
		KnowledgeID: "doc-1", Collection: "kb", Content: "The ledger moves on Sunday.",
		Metadata: map[string]string{"source": "wiki"},
	}); err != nil {
		t.Fatal(err)
	}
	// The chunk as an older model left it: a vector of another width, and an
	// ACL that says who may read it.
	old := make([]float32, 64)
	old[0] = 1
	blob, err := encoding.EncodeVector(old)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.store.GetDB().ExecContext(ctx,
		`UPDATE embeddings SET vector = ?, acl = '["team"]' WHERE doc_id = 'doc-1'`, blob); err != nil {
		t.Fatal(err)
	}
	var id string
	if err := db.store.GetDB().QueryRowContext(ctx, `SELECT id FROM embeddings WHERE doc_id = 'doc-1' LIMIT 1`).Scan(&id); err != nil {
		t.Fatal(err)
	}
	before, err := db.store.GetByID(ctx, id)
	if err != nil {
		t.Fatal(err)
	}

	report, err := db.ReembedMismatchedVectors(ctx, ReembedOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Reembedded != 1 {
		t.Fatalf("re-embedded %d rows, want 1: %+v", report.Reembedded, report)
	}
	after, err := db.store.GetByID(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if len(after.Vector) != 8 {
		t.Fatalf("vector has %d dimensions after the repair, want 8", len(after.Vector))
	}
	if after.DocID != "doc-1" || after.Metadata["source"] != "wiki" ||
		!reflect.DeepEqual(after.Metadata, before.Metadata) || !reflect.DeepEqual(after.ACL, []string{"team"}) {
		t.Fatalf("the repair changed the row: doc_id=%q metadata=%v acl=%v, was metadata=%v", after.DocID, after.Metadata, after.ACL, before.Metadata)
	}
}

// With an embedder, ingest_document embeds its chunks as knowledge_save does.
// It used to write lexical vectors of another width, which the vector index
// refused, so every chunk it stored was invisible to semantic search (#84).
func TestIngestDocumentEmbedsWithTheEmbedder(t *testing.T) {
	emb := &fixedDimEmbedder{dim: 8}
	db := newReembedDB(t, emb)
	ctx := context.Background()
	if _, err := db.GraphRAGTools().IngestDocument(ctx, ToolIngestDocumentRequest{
		DocumentID: "ingested", Content: "The ledger moves on Sunday.", Collection: "kb",
	}); err != nil {
		t.Fatal(err)
	}
	if emb.calls == 0 {
		t.Fatal("ingest_document did not call the embedder")
	}
	var id string
	if err := db.store.GetDB().QueryRowContext(ctx, `SELECT id FROM embeddings WHERE doc_id = 'ingested' LIMIT 1`).Scan(&id); err != nil {
		t.Fatal(err)
	}
	chunk, err := db.store.GetByID(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunk.Vector) != 8 {
		t.Fatalf("chunk vector has %d dimensions, want the embedder's 8", len(chunk.Vector))
	}
	report, err := db.DimensionReport(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if report.NeedsRepair() {
		t.Fatalf("the dimension report flags rows ingest_document just wrote: %+v", report)
	}
}

// A document stored before the embedder was configured is not "unchanged" to
// an ingest that would embed it: the same text is stored again with the
// embedder's vectors, instead of keeping vectors the index refuses.
func TestALexicallyStoredDocumentIsReembeddedOnItsNextIngest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lexical-first.db")
	lexical, err := Open(DefaultConfig(path))
	if err != nil {
		t.Fatal(err)
	}
	req := ToolIngestDocumentRequest{DocumentID: "doc", Content: "The ledger moves on Sunday.", Collection: "notes"}
	ctx := context.Background()
	if _, err := lexical.GraphRAGTools().IngestDocument(ctx, req); err != nil {
		t.Fatal(err)
	}
	if err := lexical.Close(); err != nil {
		t.Fatal(err)
	}

	emb := &fixedDimEmbedder{dim: 8}
	embedded, err := Open(DefaultConfig(path), WithEmbedder(emb))
	if err != nil {
		t.Fatal(err)
	}
	defer embedded.Close()
	resp, err := embedded.GraphRAGTools().IngestDocument(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Unchanged || emb.calls == 0 {
		t.Fatalf("the re-ingest kept the lexical vectors: unchanged=%v embedder calls=%d", resp.Unchanged, emb.calls)
	}
	// And now that it is embedded, the same ingest is unchanged.
	again, err := embedded.GraphRAGTools().IngestDocument(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if !again.Unchanged {
		t.Fatal("an identical ingest after the re-embedding was not unchanged")
	}
}

// A repair before this fix left chunk rows without their document and their
// metadata. The repair now gives both back, from the chunk id and the
// document record, so search reports the chunk's source again and the expiry
// and review gates, which find a chunk's document by doc_id, see it again.
func TestARepairRestoresChunksAnEarlierRepairStripped(t *testing.T) {
	db := newReembedDB(t, &fixedDimEmbedder{dim: 8})
	ctx := context.Background()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{
		KnowledgeID: "doc:a", Title: "Ledger", Collection: "kb", Content: "The ledger moves on Sunday.",
		Metadata: map[string]string{"source": "wiki"},
	}); err != nil {
		t.Fatal(err)
	}
	var id string
	if err := db.store.GetDB().QueryRowContext(ctx, `SELECT id FROM embeddings WHERE doc_id = 'doc:a' LIMIT 1`).Scan(&id); err != nil {
		t.Fatal(err)
	}
	// What the old repair wrote back: the vector, and nothing else.
	if _, err := db.store.GetDB().ExecContext(ctx, `UPDATE embeddings SET doc_id = NULL, metadata = '{}' WHERE id = ?`, id); err != nil {
		t.Fatal(err)
	}

	report, err := db.ReembedMismatchedVectors(ctx, ReembedOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Restored != 1 {
		t.Fatalf("restored %d rows, want 1: %+v", report.Restored, report)
	}
	got, err := db.store.GetByID(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"graph_kind": "chunk", "document_id": "doc:a", "chunk_index": "0", "title": "Ledger", "source": "wiki"}
	if got.DocID != "doc:a" {
		t.Fatalf("doc_id = %q, want doc:a", got.DocID)
	}
	for k, v := range want {
		if got.Metadata[k] != v {
			t.Fatalf("metadata[%s] = %q, want %q (all: %v)", k, got.Metadata[k], v, got.Metadata)
		}
	}
}

func TestParseChunkIDTakesTheIndexAfterTheLastColon(t *testing.T) {
	for id, want := range map[string]struct {
		doc   string
		index int
		ok    bool
	}{
		"chunk:doc-1:000":       {"doc-1", 0, true},
		"chunk:noted:a:b:012":   {"noted:a:b", 12, true},
		"chunk:doc-1:1001":      {"doc-1", 1001, true},
		"chunk::000":            {"", 0, false},
		"node:doc-1:000":        {"", 0, false},
		"chunk:doc-1:not-index": {"", 0, false},
	} {
		doc, index, ok := parseChunkID(id)
		if doc != want.doc || index != want.index || ok != want.ok {
			t.Errorf("parseChunkID(%q) = %q, %d, %v; want %q, %d, %v", id, doc, index, ok, want.doc, want.index, want.ok)
		}
	}
}
