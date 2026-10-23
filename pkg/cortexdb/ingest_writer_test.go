package cortexdb

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
)

// ingest_document signs its document as knowledge_save does: under a writer,
// written_by is the writer whatever the caller sent, and updated_by, which
// only an update sets, goes. author is the caller's data and is kept.
func TestAnIngestRecordsItsWriterAndAClientCannotClaimAnother(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "ingest-writer.db")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	hermes := WithWriter(context.Background(), "hermes")

	if _, err := db.GraphRAGTools().IngestDocument(hermes, ToolIngestDocumentRequest{
		DocumentID: "d1",
		Content:    "The gateway fronts the store.",
		Author:     "zeus",
		Metadata:   map[string]string{metadataWrittenBy: "zeus", metadataUpdatedBy: "zeus", "team": "a"},
	}); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	got, err := db.GetKnowledge(context.Background(), KnowledgeGetRequest{KnowledgeID: "d1"})
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	meta := got.Knowledge.Metadata
	if meta[metadataWrittenBy] != "hermes" {
		t.Errorf("written_by = %v, want hermes", meta[metadataWrittenBy])
	}
	if _, ok := meta[metadataUpdatedBy]; ok {
		t.Errorf("an ingest kept the updated_by its caller sent: %v", meta)
	}
	if got.Knowledge.Author != "zeus" || meta["team"] != "a" {
		t.Errorf("author %q, team %v: want the caller's zeus and a kept", got.Knowledge.Author, meta["team"])
	}
}

func TestWithoutAWriterAnIngestStoresMetadataAsGiven(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "ingest-nowriter.db")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()

	if _, err := db.GraphRAGTools().IngestDocument(context.Background(), ToolIngestDocumentRequest{
		DocumentID: "d1",
		Content:    "Imported from an export.",
		Metadata:   map[string]string{metadataWrittenBy: "hermes", metadataUpdatedBy: "zeus"},
	}); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	got, err := db.GetKnowledge(context.Background(), KnowledgeGetRequest{KnowledgeID: "d1"})
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if w, u := got.Knowledge.Metadata[metadataWrittenBy], got.Knowledge.Metadata[metadataUpdatedBy]; w != "hermes" || u != "zeus" {
		t.Errorf("written_by=%v updated_by=%v, want the given hermes and zeus kept", w, u)
	}
}

// The signature is part of what an ingest stores, so it takes part in
// "unchanged": the same text again by the same key changes nothing, and by
// another key it is written again and signed by that key, as knowledge_save
// signs every save.
func TestAReIngestByAnotherKeyIsSignedByIt(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "ingest-resign.db")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	req := ToolIngestDocumentRequest{DocumentID: "d1", Content: "The gateway fronts the store."}
	ingest := func(writer string) *ToolIngestDocumentResponse {
		t.Helper()
		resp, err := db.GraphRAGTools().IngestDocument(WithWriter(context.Background(), writer), req)
		if err != nil {
			t.Fatalf("ingest as %s: %v", writer, err)
		}
		return resp
	}
	ingest("hermes")
	if !ingest("hermes").Unchanged {
		t.Errorf("the same ingest by the same key was not unchanged")
	}
	if ingest("zeus").Unchanged {
		t.Errorf("the same ingest by another key was unchanged, so it keeps the first key's signature")
	}
	got, err := db.GetKnowledge(context.Background(), KnowledgeGetRequest{KnowledgeID: "d1"})
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if w := got.Knowledge.Metadata[metadataWrittenBy]; w != "zeus" {
		t.Errorf("written_by = %v, want zeus, who ingested last", w)
	}
}

// WriterKeys names the keys stampWriter and stampUpdater write, for the
// allowlist to take back on the tools whose facade signs.
func TestWriterKeysAreTheKeysAWriterIsRecordedUnder(t *testing.T) {
	if got := WriterKeys(); len(got) != 2 || got[0] != metadataWrittenBy || got[1] != metadataUpdatedBy {
		t.Fatalf("WriterKeys() = %v, want [written_by updated_by]", got)
	}
}

// InsertGraphDocument, the typed gRPC write of a document, signs as
// ingest_document does: under a writer, written_by is the writer and the
// updated_by sent goes, also on a call that sends no metadata. The caller's
// map is not changed, and without a writer the metadata is stored as given.
// It stores the metadata on the chunks, which are read here.
func TestAGraphInsertIsSignedByItsWriter(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "insert-writer.db")), WithEmbedder(newKeywordEmbedder("lyre", "gate")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()
	hermes := WithWriter(ctx, "hermes")
	chunkMetadata := func(id string) map[string]string {
		var raw string
		if err := db.SQL().QueryRow(`SELECT metadata FROM embeddings WHERE doc_id = ?`, id).Scan(&raw); err != nil {
			t.Fatalf("read %s: %v", id, err)
		}
		meta := map[string]string{}
		if err := json.Unmarshal([]byte(raw), &meta); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		return meta
	}
	sent := map[string]string{metadataWrittenBy: "zeus", metadataUpdatedBy: "zeus", "team": "a"}
	for _, c := range []struct {
		ctx      context.Context
		id       string
		metadata map[string]string
		want     string
	}{
		{hermes, "g1", sent, "hermes"},
		{hermes, "g2", nil, "hermes"},
		{ctx, "g3", map[string]string{metadataWrittenBy: "zeus"}, "zeus"},
	} {
		if _, err := db.InsertGraphDocument(c.ctx, GraphRAGDocument{ID: c.id, Content: "The lyre at the gate.", Metadata: c.metadata}, GraphRAGIngestOptions{}); err != nil {
			t.Fatalf("insert %s: %v", c.id, err)
		}
		if meta := chunkMetadata(c.id); meta[metadataWrittenBy] != c.want {
			t.Errorf("%s: written_by = %v, want %s", c.id, meta[metadataWrittenBy], c.want)
		}
	}
	if meta := chunkMetadata("g1"); meta[metadataUpdatedBy] != "" || meta["team"] != "a" {
		t.Errorf("g1 metadata %v: want no updated_by and team a", meta)
	}
	if sent[metadataWrittenBy] != "zeus" || sent[metadataUpdatedBy] != "zeus" {
		t.Errorf("the caller's map was changed: %v", sent)
	}
}

// InsertText and InsertTextBatch, the typed gRPC writes of plain text, sign
// as InsertGraphDocument does: under a writer the row records the writer and
// no updated_by, whatever the caller sent; without one it stores what it was
// sent. A call with no metadata is signed too, on both. The caller's map
// is not changed.
func TestATypedTextInsertIsSignedByItsWriter(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "text-writer.db")), WithEmbedder(newKeywordEmbedder("lyre", "gate")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()
	hermes := WithWriter(ctx, "hermes")
	rowMetadata := func(id string) map[string]string {
		var raw string
		if err := db.SQL().QueryRow(`SELECT metadata FROM embeddings WHERE id = ?`, id).Scan(&raw); err != nil {
			t.Fatalf("read %s: %v", id, err)
		}
		meta := map[string]string{}
		if raw == "" {
			return meta
		}
		if err := json.Unmarshal([]byte(raw), &meta); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		return meta
	}
	sent := map[string]string{metadataWrittenBy: "zeus", metadataUpdatedBy: "zeus", "team": "a"}
	if err := db.InsertText(hermes, "t1", "The lyre.", sent); err != nil {
		t.Fatal(err)
	}
	if err := db.InsertTextBatch(hermes, map[string]string{"t2": "The gate.", "t3": "The lyre at the gate."}, sent); err != nil {
		t.Fatal(err)
	}
	if err := db.InsertText(hermes, "t4", "The lyre again.", nil); err != nil {
		t.Fatal(err)
	}
	if err := db.InsertTextBatch(hermes, map[string]string{"t7": "The gate at dusk."}, nil); err != nil {
		t.Fatal(err)
	}
	if err := db.InsertText(ctx, "t5", "The gate again.", map[string]string{metadataWrittenBy: "zeus"}); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ id, want string }{{"t1", "hermes"}, {"t2", "hermes"}, {"t3", "hermes"}, {"t4", "hermes"}, {"t7", "hermes"}, {"t5", "zeus"}} {
		meta := rowMetadata(c.id)
		if meta[metadataWrittenBy] != c.want || meta[metadataUpdatedBy] != "" {
			t.Errorf("%s: metadata %v, want written_by %s and no updated_by", c.id, meta, c.want)
		}
	}
	if meta := rowMetadata("t2"); meta["team"] != "a" {
		t.Errorf("t2 lost the caller's other keys: %v", meta)
	}
	if sent[metadataWrittenBy] != "zeus" || sent[metadataUpdatedBy] != "zeus" {
		t.Errorf("the caller's map was changed: %v", sent)
	}
	// Without a writer, no metadata stays no metadata, as on main.
	if err := db.InsertText(ctx, "t6", "The lyre once more.", nil); err != nil {
		t.Fatal(err)
	}
	var raw string
	if err := db.SQL().QueryRow(`SELECT metadata FROM embeddings WHERE id = 't6'`).Scan(&raw); err != nil || raw != "" {
		t.Errorf("t6 stored metadata %q (%v), want none, as without the signature", raw, err)
	}
}
