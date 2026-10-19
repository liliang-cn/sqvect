package cortexdb

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

// InsertGraphDocument over an existing id keeps the stored expiry: it has no
// expires_at field, so like a re-ingest that leaves the field out it does
// not lift a retention rule. The marks of an earlier ingest are not kept.
func TestInsertGraphDocumentKeepsTheStoredExpiry(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "insert.db")), WithEmbedder(newKeywordEmbedder("arvid", "lund", "raised")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	past := pastInstant()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "old", Content: "Arvid Lund was raised in Lund.", ExpiresAt: &past}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.InsertGraphDocument(ctx, GraphRAGDocument{ID: "old", Content: "Arvid Lund was raised in Lund."}, GraphRAGIngestOptions{}); err != nil {
		t.Fatal(err)
	}
	doc, err := db.store.GetDocument(ctx, "old")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Metadata[expiresAtKey] != past {
		t.Errorf("the expiry after InsertGraphDocument is %v, want %s", doc.Metadata[expiresAtKey], past)
	}
	if doc.Metadata[contentHashKey] != nil {
		t.Errorf("InsertGraphDocument kept the marks of the earlier write: %v", doc.Metadata)
	}
	hits, err := db.SearchText(ctx, "Arvid Lund raised", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Errorf("the expired document is served after InsertGraphDocument: %d hits", len(hits))
	}
}

// With nothing expired, a pin to documents filters the seeds the search
// found, as on main: the search does not go on past top_k to look for the
// pinned document, which on a large brain would read the whole store.
func TestAPinWithNothingExpiredFiltersTheSeedsAsOnMain(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "pin.db")), WithEmbedder(newKeywordEmbedder("heron", "ledger")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	for i := 0; i < 11; i++ {
		if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: fmt.Sprintf("d%02d", i), Content: "The heron ledger, copy " + fmt.Sprint(i) + "."}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "d11", Content: "A note on something else."}); err != nil {
		t.Fatal(err)
	}
	res, err := db.SearchGraphRAG(ctx, "heron ledger", GraphRAGQueryOptions{TopK: 2, DisableGraph: true, Plan: &RetrievalPlan{Filters: &RetrievalFilters{DocumentIDs: []string{"d11"}}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Chunks) != 0 {
		t.Errorf("the pinned search went past its two seeds and found %d chunks of d11", len(res.Chunks))
	}
}

// The stored instant is UTC with a four-digit year, so text order is time
// order. An instant whose year leaves 0001-9999 once in UTC is refused: as
// text, "10000-01-01" sorts before "2026", and the document would be expired
// at once.
func TestAnExpiryOutsideFourDigitYearsIsRefused(t *testing.T) {
	for _, c := range []struct {
		value string
		ok    bool
	}{
		{"9999-12-31T23:30:00-01:00", false},
		{"0000-06-01T00:00:00Z", false},
		{"0001-01-01T00:30:00+01:00", false},
		{"9999-12-31T23:59:59Z", true},
		{"0001-01-01T00:00:00Z", true},
	} {
		got, err := normalizeExpiresAt(c.value)
		if (err == nil) != c.ok {
			t.Errorf("%s: %q %v, want accepted %v", c.value, got, err, c.ok)
		}
	}
}

// A source lane names rows by id, and the brain reads them by id; the gate
// after each prefetch leaves out a row of an expired document, which the
// source cannot know about.
func TestASourceLaneDoesNotServeAnExpiredRow(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "source.db")), WithEmbedder(newKeywordEmbedder("heron", "ledger")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	past := pastInstant()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "old", Content: "The heron ledger, old copy.", ExpiresAt: &past}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "live", Content: "The heron ledger, live copy."}); err != nil {
		t.Fatal(err)
	}
	rowOf := func(doc string) string {
		var id string
		if err := db.SQL().QueryRow(`SELECT id FROM embeddings WHERE doc_id = ?`, doc).Scan(&id); err != nil {
			t.Fatal(err)
		}
		return id
	}
	src := &fakeSource{name: "meili", hits: []QuerySourceHit{{ID: rowOf("old"), Score: 9}, {ID: rowOf("live"), Score: 3}}}
	WithQuerySource(src)(db)
	resp, err := db.Query(ctx, QueryRequest{Query: "heron", Prefetch: []QueryPrefetch{{Kind: QueryPrefetchSource, Source: "meili"}}, Limit: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 1 || resp.Results[0].ID != rowOf("live") {
		t.Fatalf("results %+v, want the live row only", resp.Results)
	}
}

// expand_graph with relation_policies scores the nodes it reaches; the node
// of an expired chunk leaves the scores with the node, so its id is not
// named there either.
func TestAScoredExpansionDoesNotScoreAnExpiredChunk(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "scored")
	oldChunk, liveChunk := graphChunkNodeID("old", 0), graphChunkNodeID("live", 0)
	mention := func(chunk string) []ToolEntityInput {
		return []ToolEntityInput{{Name: "Arvid Lund", ChunkIDs: []string{chunk}}}
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "old", Content: "Arvid Lund grew up by the harbour.", Entities: mention(oldChunk), ExpiresAt: strPtr(pastInstant())}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "live", Content: "Arvid Lund teaches in Uppsala.", Entities: mention(liveChunk)}); err != nil {
		t.Fatal(err)
	}
	out, err := tools.Call(ctx, "expand_graph", json.RawMessage(`{"node_ids":["`+oldChunk+`","`+liveChunk+`"],"max_hops":2,"relation_policies":{"mentions":{"weight":1}}}`))
	if err != nil {
		t.Fatal(err)
	}
	resp, ok := out.(*ToolExpandGraphResponse)
	if !ok {
		t.Fatalf("expand_graph returned %T", out)
	}
	if _, scored := resp.Scores[oldChunk]; scored {
		t.Errorf("the expired chunk is scored: %v", resp.Scores)
	}
	if _, scored := resp.Scores[liveChunk]; !scored {
		t.Errorf("the live chunk is not scored, so the check proves nothing: %v", resp.Scores)
	}
}

// The engine never stores an empty expiry, but a row written by hand can
// carry one. The gate reads it as no expiry, and so does the dedupe lookup:
// the document stays a holder of its content.
func TestAnEmptyExpiryWrittenByHandIsNoExpiry(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "empty-expiry")
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "a", Content: "The heron ledger."}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SQL().Exec(`UPDATE documents SET metadata = json_set(metadata, '$.` + expiresAtKey + `', '') WHERE id = 'a'`); err != nil {
		t.Fatal(err)
	}
	resp, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "b", Content: "The heron ledger.", Dedupe: true})
	if err != nil {
		t.Fatal(err)
	}
	if resp.DuplicateOf != "a" {
		t.Fatalf("duplicate_of %q, want a: an empty expiry is no expiry", resp.DuplicateOf)
	}
	hits, err := tools.SearchText(ctx, ToolSearchTextRequest{Query: "heron ledger", TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits.Chunks) != 1 || hits.Chunks[0].DocumentID != "a" {
		t.Errorf("search_text served %+v, want a: an empty expiry is no expiry", hits.Chunks)
	}
	report, err := db.SweepExpired(ctx, time.Now(), true)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Documents) != 0 {
		t.Errorf("the sweep lists %v, want nothing: an empty expiry is no expiry", report.Documents)
	}
}

// knowledge_delete has no condition: a record already gone when its delete
// runs is no error, as on main. Only the sweep's conditional delete answers
// errKept.
func TestAnUnconditionalCleanupOfAGoneRecordIsNoError(t *testing.T) {
	db, _, ctx := openIngestBrain(t, "gone")
	if err := db.cleanupKnowledgeArtifacts(ctx, "never-there"); err != nil {
		t.Fatalf("cleanup of a gone record: %v", err)
	}
}

// withoutExpiredProvenance on edges written by hand, one per place an edge
// names a chunk or a document: what names the expired document is taken
// out of a copy, what names the live one stays, an edge only the expired
// document backs is left out, and the graph's own edge is left as it was. A brain whose chunk nodes cannot be read hides nothing,
// as the gate does everywhere: the sweep is the guarantee.
func TestAnEdgeDoesNotNameAnExpiredChunkOrDocument(t *testing.T) {
	db, _, ctx := openIngestBrain(t, "provenance")
	past := pastInstant()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "old", Content: "Arvid Lund grew up by the harbour.", ExpiresAt: &past}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "live", Content: "Arvid Lund teaches in Uppsala."}); err != nil {
		t.Fatal(err)
	}
	oldChunk, liveChunk := graphChunkNodeID("old", 0), graphChunkNodeID("live", 0)
	edges := []*graph.GraphEdge{
		{ID: "chunks", Properties: map[string]any{"chunk_ids": []any{oldChunk, liveChunk}}},
		{ID: "mixed", Properties: map[string]any{"source_chunk_id": oldChunk, "document_id": "old", "chunk_ids": []any{liveChunk}}},
		{ID: "source-live", Properties: map[string]any{"source_chunk_id": liveChunk, "document_id": "live"}},
		{ID: "bare"},
		{ID: "plain", Properties: map[string]any{"note": "no provenance"}},
		{ID: "source-old", Properties: map[string]any{"source_chunk_id": oldChunk, "document_id": "old"}},
	}
	got := db.withoutExpiredProvenance(ctx, edges)
	if len(got) != 5 || got[3].ID != "bare" || got[4].ID != "plain" {
		t.Fatalf("%d edges, want the 5 a live document backs or that name nothing, without source-old", len(got))
	}
	if ids := stringsFromAny(got[0].Properties["chunk_ids"]); len(ids) != 1 || ids[0] != liveChunk {
		t.Errorf("chunk_ids %v, want the live chunk only", ids)
	}
	if _, ok := got[1].Properties["source_chunk_id"]; ok {
		t.Errorf("the edge still names the expired chunk: %v", got[1].Properties)
	}
	if _, ok := got[1].Properties["document_id"]; ok {
		t.Errorf("the edge still names the expired document: %v", got[1].Properties)
	}
	if got[2] != edges[2] || got[3] != edges[3] {
		t.Error("an edge that names nothing expired was copied")
	}
	if edges[1].Properties["source_chunk_id"] != oldChunk || len(stringsFromAny(edges[0].Properties["chunk_ids"])) != 2 {
		t.Error("the graph's own edges were changed")
	}
	// An edge whose only provenance is the expired chunk it came from, or
	// the expired document alone, is left out.
	for _, props := range []map[string]any{{"source_chunk_id": oldChunk}, {"document_id": "old"}, {"chunk_ids": []any{oldChunk}}} {
		if alone := db.withoutExpiredProvenance(ctx, []*graph.GraphEdge{{ID: "only", Properties: props}}); len(alone) != 0 {
			t.Errorf("an edge only the expired document backs (%v) was kept", props)
		}
	}
	// An edge that one live mark still backs stays, whatever else it names:
	// a live document with expired chunks, or a live source chunk alone.
	for _, c := range []struct {
		props      map[string]any
		keep, gone string
	}{
		{map[string]any{"document_id": "live", "chunk_ids": []any{oldChunk}}, "document_id", "chunk_ids"},
		{map[string]any{"source_chunk_id": liveChunk, "chunk_ids": []any{oldChunk}}, "source_chunk_id", "chunk_ids"},
	} {
		kept := db.withoutExpiredProvenance(ctx, []*graph.GraphEdge{{ID: "backed", Properties: c.props}})
		if len(kept) != 1 {
			t.Errorf("an edge a live mark backs (%v) was left out", c.props)
			continue
		}
		if _, ok := kept[0].Properties[c.keep]; !ok || len(stringsFromAny(kept[0].Properties[c.gone])) != 0 {
			t.Errorf("edge %v, want %s kept and %s emptied", kept[0].Properties, c.keep, c.gone)
		}
	}
	if _, err := db.SQL().Exec(`ALTER TABLE graph_nodes RENAME COLUMN content TO content_hidden`); err != nil {
		t.Fatal(err)
	}
	if got := db.withoutExpiredProvenance(ctx, edges); got[0] != edges[0] || got[1] != edges[1] {
		t.Error("with the chunk nodes unreadable, the edges were changed")
	}
}

// An edge that only an expired document backs is left out of the edges the
// tools return: its id can name the document (upsert_relations with a
// document_id), so keeping it would name what the search hides. An edge that
// a live document also backs stays, with the expired provenance taken out. A
// path through a node or an edge left out is not returned, in search_paths as
// in shortest_path.
func TestAnEdgeKnownOnlyFromAnExpiredDocumentIsLeftOut(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "expired-edges")
	oldChunk, liveChunk := graphChunkNodeID("old", 0), graphChunkNodeID("live", 0)
	mentions := func(chunk string, names ...string) []ToolEntityInput {
		var out []ToolEntityInput
		for _, n := range names {
			out = append(out, ToolEntityInput{Name: n, ChunkIDs: []string{chunk}})
		}
		return out
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "old", Content: "Arvid Lund grew up by the harbour of Lund, born in Malmo.", Entities: mentions(oldChunk, "Arvid Lund", "Lund", "Malmo"), ExpiresAt: strPtr(pastInstant())}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "live", Content: "Arvid Lund teaches in Uppsala.", Entities: mentions(liveChunk, "Arvid Lund", "Uppsala")}); err != nil {
		t.Fatal(err)
	}
	call := func(name, args string) string {
		t.Helper()
		out, err := tools.Call(ctx, name, json.RawMessage(args))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		body, _ := json.Marshal(out)
		return string(body)
	}
	call("upsert_relations", `{"relations":[{"from":"Arvid Lund","to":"Lund","type":"visits","chunk_ids":["`+oldChunk+`","`+liveChunk+`"]}]}`)
	call("upsert_relations", `{"document_id":"old","relations":[{"from":"Arvid Lund","to":"Malmo","type":"born_in"}]}`)
	arvid := EntityNodeID("Arvid Lund")

	path := call("knowledge_memory_shortest_path", `{"from_node_id":"`+arvid+`","to_node_id":"`+EntityNodeID("Lund")+`"}`)
	if !strings.Contains(path, liveChunk) || strings.Contains(path, oldChunk) {
		t.Errorf("the path over the shared edge, want the live chunk only: %s", path)
	}
	if _, err := tools.Call(ctx, "knowledge_memory_shortest_path", json.RawMessage(`{"from_node_id":"`+arvid+`","to_node_id":"`+EntityNodeID("Malmo")+`"}`)); err == nil || !strings.Contains(err.Error(), "no path") {
		t.Errorf("a path over an edge only the expired document backs: %v, want no path", err)
	}
	for name, args := range map[string]string{
		"expand_graph": `{"node_ids":["` + arvid + `"],"max_hops":2}`,
		"search_paths": `{"entity_names":["Arvid Lund"],"include_bookkeeping":true,"direction":"both","max_depth":2}`,
	} {
		body := call(name, args)
		for _, leak := range []string{"harbour", oldChunk, "doc:old", "born_in"} {
			if strings.Contains(body, leak) {
				t.Errorf("%s returned what only the expired document backs (%s): %.300s", name, leak, body)
			}
		}
		if !strings.Contains(body, liveChunk) {
			t.Errorf("%s did not return the live chunk either, so its check proves nothing: %.300s", name, body)
		}
	}
}
