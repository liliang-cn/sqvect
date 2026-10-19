package cortexdb

import (
	"context"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

// Under the review gate an edge does not name the chunk of a proposed
// document either, and a brain whose chunk nodes cannot be read hides every
// chunk an edge names, so an edge that only chunks back is left out: the
// review gate fails closed, where the expiry alone is advisory.
func TestUnderTheReviewGateAnEdgeDoesNotNameAProposedChunk(t *testing.T) {
	db := openReviewBrain(t, "review-provenance", WithReviewGate())
	ctx := context.Background()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "p", Content: "The heron ledger, proposed.", ReviewStatus: ReviewProposed}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "a", Content: "The heron ledger, served."}); err != nil {
		t.Fatal(err)
	}
	proposed, served := graphChunkNodeID("p", 0), graphChunkNodeID("a", 0)
	edges := []*graph.GraphEdge{{ID: "e", Properties: map[string]any{"chunk_ids": []any{proposed, served}}}}
	got := db.withoutExpiredProvenance(ctx, edges)
	if ids := stringsFromAny(got[0].Properties["chunk_ids"]); len(ids) != 1 || ids[0] != served {
		t.Errorf("chunk_ids %v, want the served chunk only", ids)
	}
	if _, err := db.SQL().Exec(`ALTER TABLE graph_nodes RENAME COLUMN content TO content_hidden`); err != nil {
		t.Fatal(err)
	}
	if got = db.withoutExpiredProvenance(ctx, edges); len(got) != 0 {
		t.Errorf("with the chunk nodes unreadable, %d edges, want none under the review gate", len(got))
	}
}

// InsertGraphDocument takes a document read back with its review decision,
// and keeps the stored decision, not the one sent: the decision is the
// review fields' alone, and a caller's copy lands neither on the record nor
// on the chunks.
func TestInsertGraphDocumentKeepsTheStoredReviewDecision(t *testing.T) {
	db := openReviewBrain(t, "review-insert", WithReviewGate(), WithEmbedder(newKeywordEmbedder("heron", "ledger")))
	ctx := context.Background()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "p", Content: "The heron ledger, proposed.", ReviewStatus: ReviewProposed}); err != nil {
		t.Fatal(err)
	}
	sent := map[string]string{reviewStatusKey: ReviewAccepted, reviewedByKey: "me", "team": "a"}
	if _, err := db.InsertGraphDocument(ctx, GraphRAGDocument{ID: "p", Content: "The heron ledger, again.", Metadata: sent}, GraphRAGIngestOptions{}); err != nil {
		t.Fatal(err)
	}
	doc, err := db.store.GetDocument(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Metadata[reviewStatusKey] != ReviewProposed || doc.Metadata[reviewedByKey] != nil {
		t.Errorf("the record says %v, want the stored proposal", doc.Metadata)
	}
	chunk, err := db.store.GetByID(ctx, graphChunkNodeID("p", 0))
	if err != nil {
		t.Fatal(err)
	}
	if chunk.Metadata[reviewStatusKey] == ReviewAccepted || chunk.Metadata["team"] != "a" {
		t.Errorf("the chunk says %v, want the caller's team and not its decision", chunk.Metadata)
	}
}

// The review gate hides a proposed document through the same gate as the
// expiry, so it holds on the same class: every gated read tool of
// readToolsAndTheExpiry leaves the proposed document out and serves the
// accepted one.
func TestEveryGatedReadToolHidesAProposedDocument(t *testing.T) {
	db := openReviewBrain(t, "review-class", WithReviewGate())
	everyGatedToolHides(t, db, db.GraphRAGTools(), context.Background(), func(r *KnowledgeSaveRequest) { r.ReviewStatus = ReviewProposed })
}
