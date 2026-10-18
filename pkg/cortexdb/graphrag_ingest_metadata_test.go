package cortexdb

import (
	"context"
	"path/filepath"
	"testing"
)

// ingest_document used to keep the caller's metadata on the chunks only. The
// document record held id, title, content and version, and the document node
// held id and title — so a search hit, which reads provenance off the record,
// came back with no source, no author and no metadata for anything ingested
// this way, while the same document saved through knowledge_save kept all
// three. Classification, source and date of a document are exactly what a
// reader needs next to a hit.
func TestIngestDocumentKeepsProvenanceOnTheRecordAndTheNode(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "ingest_metadata.db")))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	defer func() { _ = db.Close() }()
	ctx := context.Background()
	tools := db.GraphRAGTools()

	resp, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{
		DocumentID: "spec-1",
		Title:      "Migration spec",
		Content:    "The migration moves the ledger to the new cluster. The cutover happens on a Sunday.",
		SourceURL:  "smb://files/acme/spec.docx",
		Author:     "mdupont",
		Metadata:   map[string]string{"classification": "internal", "consulted_at": "2026-10-18"},
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}

	doc, err := db.store.GetDocument(ctx, "spec-1")
	if err != nil {
		t.Fatalf("get document: %v", err)
	}
	if doc.SourceURL != "smb://files/acme/spec.docx" || doc.Author != "mdupont" {
		t.Fatalf("record lost source or author: %+v", doc)
	}
	if doc.Metadata["classification"] != "internal" || doc.Metadata["consulted_at"] != "2026-10-18" {
		t.Fatalf("record lost metadata: %v", doc.Metadata)
	}

	node, err := db.graph.GetNode(ctx, resp.DocumentNodeID)
	if err != nil {
		t.Fatalf("get document node: %v", err)
	}
	for k, want := range map[string]string{
		"document_id":    "spec-1",
		"title":          "Migration spec",
		"source_url":     "smb://files/acme/spec.docx",
		"author":         "mdupont",
		"classification": "internal",
	} {
		if got, _ := node.Properties[k].(string); got != want {
			t.Fatalf("document node property %s = %q, want %q (all: %v)", k, got, want, node.Properties)
		}
	}

	hits, err := db.SearchKnowledge(ctx, KnowledgeSearchRequest{Query: "ledger cluster cutover", TopK: 3})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits.Results) == 0 {
		t.Fatal("no hit for the ingested document")
	}
	hit := hits.Results[0]
	if hit.SourceURL != "smb://files/acme/spec.docx" || hit.Author != "mdupont" || hit.Metadata["classification"] != "internal" {
		t.Fatalf("search hit carries no provenance: %+v", hit)
	}
}

// A caller's metadata key cannot displace the two properties the graph
// relies on to find a document's node and its title.
func TestIngestDocumentMetadataCannotOverrideTheGraphKeys(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "ingest_metadata_keys.db")))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	defer func() { _ = db.Close() }()
	ctx := context.Background()

	resp, err := db.GraphRAGTools().IngestDocument(ctx, ToolIngestDocumentRequest{
		DocumentID: "doc-k",
		Title:      "Real title",
		Content:    "Some content worth a chunk or two of text.",
		Metadata:   map[string]string{"document_id": "forged", "title": "forged"},
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	node, err := db.graph.GetNode(ctx, resp.DocumentNodeID)
	if err != nil {
		t.Fatalf("get document node: %v", err)
	}
	if node.Properties["document_id"] != "doc-k" || node.Properties["title"] != "Real title" {
		t.Fatalf("metadata displaced the graph keys: %v", node.Properties)
	}
}

// A second ingest of the same document is the current statement about it:
// its provenance replaces the previous one, the way knowledge_save behaves,
// and the version still advances.
func TestReingestReplacesTheDocumentProvenance(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "ingest_metadata_again.db")))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	defer func() { _ = db.Close() }()
	ctx := context.Background()
	tools := db.GraphRAGTools()

	first := ToolIngestDocumentRequest{
		DocumentID: "doc-v",
		Title:      "v1",
		Content:    "First version of the text, long enough to chunk.",
		SourceURL:  "https://wiki/doc-v",
		Metadata:   map[string]string{"classification": "public"},
	}
	if _, err := tools.IngestDocument(ctx, first); err != nil {
		t.Fatalf("first ingest: %v", err)
	}
	second := first
	second.Title = "v2"
	second.Content = "Second version of the text, also long enough to chunk."
	second.Metadata = map[string]string{"classification": "confidential", "retention_until": "2031-10-18"}
	if _, err := tools.IngestDocument(ctx, second); err != nil {
		t.Fatalf("second ingest: %v", err)
	}

	doc, err := db.store.GetDocument(ctx, "doc-v")
	if err != nil {
		t.Fatalf("get document: %v", err)
	}
	if doc.Version != 2 || doc.Title != "v2" || doc.SourceURL != "https://wiki/doc-v" {
		t.Fatalf("record after re-ingest: %+v", doc)
	}
	if doc.Metadata["classification"] != "confidential" || doc.Metadata["retention_until"] != "2031-10-18" {
		t.Fatalf("metadata not replaced: %v", doc.Metadata)
	}
}
