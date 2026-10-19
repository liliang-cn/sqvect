package cortexdb

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

func pastInstant() string { return time.Now().UTC().Add(-time.Hour).Format(time.RFC3339) }

// With an embedder, the seeds of a hybrid search come from the vector index,
// a path the lexical filter never saw: an expired document must be left out
// there too, and out of search_text.
func TestAnExpiredDocumentIsLeftOutOfTheVectorPathToo(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "vector.db")), WithEmbedder(NewMockEmbedder(8)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	content := "The zebra ledger reconciles the quarterly accounts."
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "old", Content: content, ExpiresAt: strPtr(pastInstant())}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "live", Content: content}); err != nil {
		t.Fatal(err)
	}
	resp, err := db.SearchKnowledge(ctx, KnowledgeSearchRequest{Query: "zebra ledger", TopK: 10, RetrievalMode: RetrievalModeHybrid})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(hitIDs(resp), ","); strings.Contains(got, "old") || !strings.Contains(got, "live") {
		t.Fatalf("hybrid search returned %q", got)
	}
	text, err := db.GraphRAGTools().SearchText(ctx, ToolSearchTextRequest{Query: "zebra ledger", TopK: 10})
	if err != nil {
		t.Fatal(err)
	}
	var docs []string
	for _, c := range text.Chunks {
		docs = append(docs, c.DocumentID)
	}
	if got := strings.Join(docs, ","); strings.Contains(got, "old") || !strings.Contains(got, "live") {
		t.Fatalf("search_text returned %q", got)
	}
}

// cortex_query runs its own lanes (vector, lexical, hybrid, graph) straight
// against the store: an expired document must be left out of each of them.
func TestAnExpiredDocumentIsLeftOutOfCortexQuery(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "query.db")), WithEmbedder(NewMockEmbedder(8)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	tools := db.GraphRAGTools()
	content := "The zebra ledger reconciles the quarterly accounts."
	for id, until := range map[string]string{"old": pastInstant(), "live": ""} {
		ingested, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: id, Content: content, ExpiresAt: strPtr(until)})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tools.UpsertEntities(ctx, ToolUpsertEntitiesRequest{DocumentID: id, Entities: []ToolEntityInput{{Name: "Zebra", ChunkIDs: ingested.ChunkNodeIDs}}}); err != nil {
			t.Fatal(err)
		}
	}
	for _, lane := range []QueryPrefetch{
		{Kind: QueryPrefetchVector, Query: "zebra ledger"},
		{Kind: QueryPrefetchLexical, Query: "zebra ledger"},
		{Kind: QueryPrefetchHybrid, Query: "zebra ledger"},
		{Kind: QueryPrefetchGraph, EntityNames: []string{"Zebra"}},
	} {
		resp, err := db.Query(ctx, QueryRequest{Limit: 10, Prefetch: []QueryPrefetch{lane}})
		if err != nil {
			t.Fatalf("%s lane: %v", lane.Kind, err)
		}
		var docs []string
		for _, r := range resp.Results {
			docs = append(docs, r.DocID)
		}
		if got := strings.Join(docs, ","); strings.Contains(got, "old") || !strings.Contains(got, "live") {
			t.Fatalf("cortex_query %s lane returned %q", lane.Kind, got)
		}
	}
}

// The tools that read chunks by id or through the graph leave an expired
// document's chunks out: get_chunks, build_context, search_chunks_by_entities
// and expand_graph, whose subgraph loses the document's nodes and edges.
func TestTheChunkToolsLeaveAnExpiredDocumentOut(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "chunk-tools")
	oldChunk, liveChunk := graphChunkNodeID("old", 0), graphChunkNodeID("live", 0)
	mentions := func(chunk string) []ToolEntityInput {
		return []ToolEntityInput{{Name: "Arvid Lund", ChunkIDs: []string{chunk}}}
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "old", Content: "Arvid Lund was raised in Lund.", Entities: mentions(oldChunk), ExpiresAt: strPtr(pastInstant())}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "live", Content: "Arvid Lund teaches in Uppsala.", Entities: mentions(liveChunk)}); err != nil {
		t.Fatal(err)
	}
	ids := []string{oldChunk, liveChunk}

	got, err := tools.GetChunks(ctx, ToolGetChunksRequest{ChunkIDs: ids})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Chunks) != 1 || got.Chunks[0].DocumentID != "live" {
		t.Fatalf("get_chunks returned %+v", got.Chunks)
	}
	packed, err := tools.BuildContext(ctx, ToolBuildContextRequest{ChunkIDs: ids})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(packed.Context, "raised in Lund") || !strings.Contains(packed.Context, "Uppsala") {
		t.Fatalf("build_context packed %q", packed.Context)
	}
	byEntity, err := tools.SearchChunksByEntities(ctx, ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Arvid Lund"}, TopK: 10})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range byEntity.Chunks {
		if c.DocumentID == "old" {
			t.Fatalf("search_chunks_by_entities returned the expired chunk: %+v", byEntity.Chunks)
		}
	}
	if len(byEntity.Chunks) == 0 {
		t.Fatal("search_chunks_by_entities lost the live chunk")
	}

	// An edge from a live node into the expired document goes with it.
	if err := db.graph.UpsertEdge(ctx, &graph.GraphEdge{ID: "edge:cites:live:old", FromNodeID: EntityNodeID("Arvid Lund"), ToNodeID: oldChunk, EdgeType: "cites", Weight: 1}); err != nil {
		t.Fatal(err)
	}
	sub, err := tools.ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: []string{EntityNodeID("Arvid Lund")}, MaxHops: 2})
	if err != nil {
		t.Fatal(err)
	}
	var seen []string
	for _, n := range sub.Nodes {
		seen = append(seen, n.ID)
	}
	nodes := strings.Join(seen, ",")
	if strings.Contains(nodes, oldChunk) || strings.Contains(nodes, graphDocumentNodeID("old")) {
		t.Fatalf("expand_graph kept the expired document's nodes: %s", nodes)
	}
	if !strings.Contains(nodes, liveChunk) || !strings.Contains(nodes, graphDocumentNodeID("live")) {
		t.Fatalf("expand_graph lost the live document's nodes: %s", nodes)
	}
	for _, e := range sub.Edges {
		if e.FromNodeID == oldChunk || e.ToNodeID == oldChunk || e.FromNodeID == graphDocumentNodeID("old") || e.ToNodeID == graphDocumentNodeID("old") {
			t.Fatalf("expand_graph kept an edge of the expired document: %+v", e)
		}
	}
}

// The read-time filter has an index to walk: the expiry query reads the
// documents table through idx_documents_expires_at, not row by row.
func TestTheDocumentExpiryIsIndexed(t *testing.T) {
	db, _, ctx := openIngestBrain(t, "indexed")
	rows, err := db.SQL().QueryContext(ctx, `EXPLAIN QUERY PLAN SELECT id FROM documents WHERE `+db.expiredPredicate(), "2031-01-01T00:00:00Z")
	if err != nil {
		t.Fatal(err)
	}
	var plan []string
	err = eachRow(rows, func() error {
		var id, parent, notUsed int
		var detail string
		if err := rows.Scan(&id, &parent, &notUsed, &detail); err != nil {
			return err
		}
		plan = append(plan, detail)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(plan, "; "); !strings.Contains(got, "idx_documents_expires_at") {
		t.Fatalf("the expiry query does not use the index: %s", got)
	}
}
