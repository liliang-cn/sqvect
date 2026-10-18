package cortexdb

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

func strPtr(s string) *string { return &s }

// Dedupe is decided within a collection: the same page may live in two
// collections, and duplicate_of never names a document of another one.
func TestDedupeIsDecidedWithinTheCollection(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "dedupe-collection")
	content := "The same page saved from two browsers into two collections."
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "a-1", Content: content, Collection: "a"}); err != nil {
		t.Fatal(err)
	}
	inB, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "b-1", Content: content, Collection: "b", Dedupe: true})
	if err != nil {
		t.Fatal(err)
	}
	if inB.DuplicateOf != "" || len(inB.ChunkNodeIDs) == 0 {
		t.Fatalf("the same content in another collection was refused: %+v", inB)
	}
	inA, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "a-2", Content: content, Collection: "a", Dedupe: true})
	if err != nil {
		t.Fatal(err)
	}
	if inA.DuplicateOf != "a-1" {
		t.Fatalf("the same content in the same collection was not refused, or named another collection's document: %+v", inA)
	}
	// A document is never a duplicate of itself: a dedupe re-ingest that
	// changes the title is a plain re-ingest.
	self, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "a-1", Title: "Retitled", Content: content, Collection: "a", Dedupe: true})
	if err != nil || self.DuplicateOf != "" || self.Unchanged || len(self.ChunkNodeIDs) == 0 {
		t.Fatalf("a document was a duplicate of itself: %+v %v", self, err)
	}
	// A document with no chunks sits in no collection, and says so without
	// an error.
	if ids, collection, err := db.storedChunks(ctx, "never-ingested"); err != nil || len(ids) != 0 || collection != "" {
		t.Fatalf("stored chunks of a document that does not exist: %v %q %v", ids, collection, err)
	}
	// With the chunk table unreadable, a dedupe check reports the failure
	// rather than treating the content as new.
	if _, err := db.SQL().Exec(`ALTER TABLE embeddings RENAME COLUMN collection_id TO collection_hidden`); err != nil {
		t.Fatal(err)
	}
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "a-3", Content: content, Collection: "a", Dedupe: true}); err == nil || !strings.Contains(err.Error(), "look up content hash") {
		t.Fatalf("a dedupe check with the chunk table unreadable: %v", err)
	}
}

// "Unchanged" means the stored document would come out identical: same
// collection and same chunking, not only the same text and provenance.
func TestAnUnchangedReingestMeansTheSameCollectionAndTheSameChunking(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "unchanged-exact")
	req := ToolIngestDocumentRequest{DocumentID: "d1", Content: "one two three four five six seven eight nine ten eleven twelve", Collection: "a", ChunkSize: 5}
	if _, err := tools.IngestDocument(ctx, req); err != nil {
		t.Fatal(err)
	}
	doc, err := db.store.GetDocument(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Metadata[chunkSizeKey] != "5" || doc.Metadata[chunkOverlapKey] != "0" {
		t.Fatalf("the chunking is not kept on the record: %v", doc.Metadata)
	}

	req.ChunkSize = 6
	if resp, err := tools.IngestDocument(ctx, req); err != nil || resp.Unchanged {
		t.Fatalf("another chunk size was reported unchanged: %+v %v", resp, err)
	}
	if resp, err := tools.IngestDocument(ctx, req); err != nil || !resp.Unchanged {
		t.Fatalf("the same chunk size again was not unchanged: %+v %v", resp, err)
	}
	req.Collection = "b"
	resp, err := tools.IngestDocument(ctx, req)
	if err != nil || resp.Unchanged || resp.Collection != "b" {
		t.Fatalf("another collection was reported unchanged: %+v %v", resp, err)
	}
	if _, c, _ := db.storedChunks(ctx, "d1"); c != "b" {
		t.Fatalf("the document did not move to b: %s", c)
	}
}

// The hash follows the content through every write path: a knowledge update
// recomputes it, a metadata-only update keeps it, and knowledge_save sets it
// so dedupe sees documents saved either way.
func TestTheContentHashFollowsTheContentThroughKnowledgeWrites(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "hash-follows")
	original := "The ledger moves to the new cluster on Sunday."
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: original}); err != nil {
		t.Fatal(err)
	}
	updated := "The ledger stays on the old cluster until Monday."
	if _, err := db.UpdateKnowledge(ctx, KnowledgeUpdateRequest{KnowledgeID: "d1", Content: &updated}); err != nil {
		t.Fatal(err)
	}
	doc, err := db.store.GetDocument(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Metadata[contentHashKey] != contentSHA256(updated) {
		t.Fatalf("hash after a content update: %v, want that of the new content", doc.Metadata[contentHashKey])
	}
	if resp, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: original}); err != nil || resp.Unchanged {
		t.Fatalf("re-ingest of the original text after an update was unchanged: %+v %v", resp, err)
	}
	if _, err := db.UpdateKnowledge(ctx, KnowledgeUpdateRequest{KnowledgeID: "d1", Metadata: map[string]string{"owner": "ops"}}); err != nil {
		t.Fatal(err)
	}
	if doc, _ = db.store.GetDocument(ctx, "d1"); doc.Metadata[contentHashKey] != contentSHA256(original) || doc.Metadata["owner"] != "ops" {
		t.Fatalf("a metadata-only update lost the hash: %v", doc.Metadata)
	}

	// A record written before the marks existed has no metadata at all; an
	// update of it gets the marks like any other.
	if _, err := db.SQL().Exec(`UPDATE documents SET metadata = NULL WHERE id = 'd1'`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.UpdateKnowledge(ctx, KnowledgeUpdateRequest{KnowledgeID: "d1", Title: strPtr("Ledger")}); err != nil {
		t.Fatal(err)
	}
	if doc, _ = db.store.GetDocument(ctx, "d1"); doc.Metadata[contentHashKey] != contentSHA256(original) {
		t.Fatalf("an update of a record without metadata did not stamp the hash: %v", doc.Metadata)
	}

	saved, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "k1", Content: "Saved through knowledge_save, not ingested."})
	if err != nil {
		t.Fatal(err)
	}
	if saved.Knowledge.Metadata[contentHashKey] != contentSHA256("Saved through knowledge_save, not ingested.") {
		t.Fatalf("knowledge_save did not hash its content: %v", saved.Knowledge.Metadata)
	}
	dup, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "k2", Content: "Saved through knowledge_save, not ingested.", Dedupe: true})
	if err != nil || dup.DuplicateOf != "k1" {
		t.Fatalf("dedupe did not see the document saved through knowledge_save: %+v %v", dup, err)
	}
}

// An ingest writes everything or nothing: when the chunks cannot be written,
// no record and no node are left behind, and a shorter re-ingest leaves no
// chunk of the longer text behind.
func TestAnIngestWritesEverythingOrNothing(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "atomic")
	if _, err := db.SQL().Exec(`CREATE TRIGGER no_chunks BEFORE INSERT ON embeddings BEGIN SELECT RAISE(ABORT, 'no chunks tonight'); END`); err != nil {
		t.Fatal(err)
	}
	long := strings.Repeat("word ", 400)
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: long, ChunkSize: 10}); err == nil || !strings.Contains(err.Error(), "no chunks tonight") {
		t.Fatalf("ingest with unwritable chunks: %v", err)
	}
	if _, err := db.store.GetDocument(ctx, "d1"); !errors.Is(err, core.ErrNotFound) {
		t.Fatalf("a record was left behind by a failed ingest: %v", err)
	}
	if _, err := db.graph.GetNode(ctx, graphDocumentNodeID("d1")); err == nil {
		t.Fatal("a document node was left behind by a failed ingest")
	}
	if _, err := db.SQL().Exec(`DROP TRIGGER no_chunks`); err != nil {
		t.Fatal(err)
	}
	first, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: long, ChunkSize: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(first.ChunkNodeIDs) < 40 {
		t.Fatalf("the long text made %d chunks", len(first.ChunkNodeIDs))
	}
	short, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: "two words", ChunkSize: 10})
	if err != nil {
		t.Fatal(err)
	}
	stored, _, err := db.storedChunks(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	if len(short.ChunkNodeIDs) != 1 || len(stored) != 1 {
		t.Fatalf("chunks of the longer text were left behind: %d returned, %d stored", len(short.ChunkNodeIDs), len(stored))
	}
	if _, err := db.graph.GetNode(ctx, first.ChunkNodeIDs[len(first.ChunkNodeIDs)-1]); err == nil {
		t.Fatal("a chunk node of the longer text was left behind")
	}
}

// The chunk ids an unchanged re-ingest returns come in chunk order, past the
// thousandth chunk where the ids' text order and their index order part.
func TestTheChunkIDsOfAnUnchangedReingestComeInChunkOrder(t *testing.T) {
	_, tools, ctx := openIngestBrain(t, "chunk-order")
	words := make([]string, 1001)
	for i := range words {
		words[i] = fmt.Sprintf("w%d", i)
	}
	req := ToolIngestDocumentRequest{DocumentID: "d1", Content: strings.Join(words, " "), ChunkSize: 1}
	if _, err := tools.IngestDocument(ctx, req); err != nil {
		t.Fatal(err)
	}
	again, err := tools.IngestDocument(ctx, req)
	if err != nil || !again.Unchanged {
		t.Fatalf("re-ingest: %+v %v", again, err)
	}
	if len(again.ChunkNodeIDs) != 1001 {
		t.Fatalf("%d chunk ids", len(again.ChunkNodeIDs))
	}
	for i, id := range again.ChunkNodeIDs {
		if id != graphChunkNodeID("d1", i) {
			t.Fatalf("chunk %d is %s", i, id)
		}
	}
}
