package cortexdb

import (
	"context"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

// openLaneScopeBrain writes two collections that share entities. Texts are
// lower case, so the graph holds only the entities and relations named here.
//
// alpha: a1 Kestrel Vale, a2 Osprey, a3 Heron Mill, a4 Lark Weir, a5 Owl
// Pass, a6 Raven Hill, a9 Swift Brook, a10 Crane Lake, a11 Finch Dale; a1
// states Kestrel Vale -drains-> Lark Weir (its document and chunk); the
// relation Lark Weir -fills-> Crane Lake names document a6 alone, and
// Kestrel Vale -guards-> Finch Dale names chunk a5 alone, so that the walk
// reads the collection of each from the provenance, not from a node it
// meets.
//
// beta: b1 mentions Kestrel Vale, Project Nightjar (only beta knows it),
// Heron Mill and Owl Pass, and states Kestrel Vale -funds-> Project
// Nightjar and Kestrel Vale -supplies-> Heron Mill.
//
// With no provenance: Kestrel Vale -knows-> Project Nightjar, Project
// Nightjar -watches-> Osprey, Lark Weir -feeds-> Swift Brook. A memory
// mentions Kestrel Vale and Raven Hill.
//
// From Kestrel Vale, within three hops, each alpha chunk but a1, a4, a9,
// a10 and a11 is reached only through something alpha may not see: a2 through Project
// Nightjar, a3 over beta's relation, a5 through beta's chunk, a6 through the
// memory.
func openLaneScopeBrain(t *testing.T) (*DB, context.Context) {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "lane.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	save := func(id, collection, content string, names []string, relations ...ToolRelationInput) {
		t.Helper()
		chunk := graphChunkNodeID(id, 0)
		entities := make([]ToolEntityInput, 0, len(names))
		for _, name := range names {
			entities = append(entities, ToolEntityInput{Name: name, ChunkIDs: []string{chunk}})
		}
		for i := range relations {
			relations[i].ChunkIDs = []string{chunk}
		}
		if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: id, Collection: collection, Content: content, Entities: entities, Relations: relations}); err != nil {
			t.Fatal(err)
		}
	}
	save("a2", "alpha", "the osprey nests near the alpha dam.", []string{"Osprey"})
	save("a3", "alpha", "the heron mill grinds alpha grain.", []string{"Heron Mill"})
	save("a4", "alpha", "the lark weir floods alpha fields.", []string{"Lark Weir"})
	save("a5", "alpha", "the owl pass guards the alpha road.", []string{"Owl Pass"})
	save("a6", "alpha", "the raven hill overlooks the alpha mill.", []string{"Raven Hill"})
	save("a9", "alpha", "the swift brook runs past alpha.", []string{"Swift Brook"})
	save("a10", "alpha", "the crane lake lies below alpha.", []string{"Crane Lake"})
	save("a11", "alpha", "the finch dale opens north of alpha.", []string{"Finch Dale"})
	save("a1", "alpha", "kestrel vale runs the alpha mill.", []string{"Kestrel Vale"},
		ToolRelationInput{From: "Kestrel Vale", To: "Lark Weir", Type: "drains"})
	save("b1", "beta", "kestrel vale secretly funds project nightjar with betasecret money.",
		[]string{"Kestrel Vale", "Project Nightjar", "Heron Mill", "Owl Pass"},
		ToolRelationInput{From: "Kestrel Vale", To: "Project Nightjar", Type: "funds"},
		ToolRelationInput{From: "Kestrel Vale", To: "Heron Mill", Type: "supplies"})
	if _, err := db.GraphRAGTools().UpsertRelations(ctx, ToolUpsertRelationsRequest{Relations: []ToolRelationInput{
		{From: "Kestrel Vale", To: "Project Nightjar", Type: "knows"},
		{From: "Project Nightjar", To: "Osprey", Type: "watches"},
		{From: "Lark Weir", To: "Swift Brook", Type: "feeds"},
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.GraphRAGTools().UpsertRelations(ctx, ToolUpsertRelationsRequest{DocumentID: "a6", Relations: []ToolRelationInput{
		{From: "Lark Weir", To: "Crane Lake", Type: "fills"},
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.GraphRAGTools().UpsertRelations(ctx, ToolUpsertRelationsRequest{Relations: []ToolRelationInput{
		{From: "Kestrel Vale", To: "Finch Dale", Type: "guards", ChunkIDs: []string{graphChunkNodeID("a5", 0)}},
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "m1", UserID: "u", Content: "kestrel vale walks up raven hill.",
		Entities: []ToolEntityInput{{Name: "Kestrel Vale"}, {Name: "Raven Hill"}}}); err != nil {
		t.Fatal(err)
	}
	return db, ctx
}

// graphLaneDocs is the documents of the chunks the graph lane alone returns
// for a seed, sorted.
func graphLaneDocs(t *testing.T, db *DB, ctx context.Context, collection, seed string, hops int) []string {
	t.Helper()
	resp, err := db.Query(ctx, QueryRequest{Collection: collection, Limit: 20,
		Prefetch: []QueryPrefetch{{Kind: QueryPrefetchGraph, EntityNames: []string{seed}, MaxHops: hops}}})
	if err != nil {
		t.Fatal(err)
	}
	docs := make([]string, 0, len(resp.Results))
	for _, r := range resp.Results {
		docs = append(docs, r.DocID)
	}
	sort.Strings(docs)
	return docs
}

// With a collection, the graph lane walks only what the collection may see:
// it starts from no entity only another collection knows, and crosses no
// such entity, no relation only another collection asserts, no chunk of
// another collection and no memory. A relation the collection states, by
// its document or by its chunk, and one with no provenance between entities
// it knows, are still followed.
func TestTheGraphLaneOfAScopedQueryDoesNotCrossAnotherCollectionsRelation(t *testing.T) {
	db, ctx := openLaneScopeBrain(t)
	for _, c := range []struct {
		seed string
		hops int
		want string
	}{
		// The case of the report: a seed only beta knows, two hops, used
		// to return a1 and so tell alpha that beta relates the two.
		{"Project Nightjar", 2, ""},
		{"Project Nightjar", 1, ""},
		{"Kestrel Vale", 1, "a1"},
		{"Kestrel Vale", 3, "a1,a10,a11,a4,a9"},
	} {
		if got := strings.Join(graphLaneDocs(t, db, ctx, "alpha", c.seed, c.hops), ","); got != c.want {
			t.Errorf("alpha, seed %q, %d hops: %q, want %q", c.seed, c.hops, got, c.want)
		}
	}
	// beta walks its own relation.
	if got := strings.Join(graphLaneDocs(t, db, ctx, "beta", "Project Nightjar", 1), ","); got != "b1" {
		t.Errorf("beta, seed Project Nightjar: %q, want b1", got)
	}
}

// Without a collection the lane walks the whole graph, as on main: every
// path the scoped lane refuses is there to be refused.
func TestTheGraphLaneWithoutACollectionKeepsItsHops(t *testing.T) {
	db, ctx := openLaneScopeBrain(t)
	got := graphLaneDocs(t, db, ctx, "", "Kestrel Vale", 3)
	for _, want := range []string{"a1", "a2", "a3", "a4", "a5", "a6", "a9", "a10", "a11", "b1"} {
		if !slices.Contains(got, want) {
			t.Errorf("unscoped lane from Kestrel Vale, 3 hops: %v, want %s among them", got, want)
		}
	}
	if got := strings.Join(graphLaneDocs(t, db, ctx, "", "Project Nightjar", 2), ","); !strings.Contains(got, "a1") {
		t.Errorf("unscoped lane from Project Nightjar, 2 hops: %q, want a1 among them", got)
	}
}

// Under a collection, the lane enters no node that no collection holds: a
// decision, and a node written with an id of its own, carry text another
// user wrote and name no collection. The decision links Kestrel Vale to
// Heron Mill, the note links Kestrel Vale to Raven Hill; neither link may
// bring a3 or a6 into alpha's lane. Without a collection, both are walked.
func TestTheGraphLaneOfAScopedQueryEntersNoNodeNoCollectionHolds(t *testing.T) {
	db, ctx := openLaneScopeBrain(t)
	if _, err := db.RecordDecision(ctx, DecisionRecordRequest{Actor: "beta-op", Note: "kestrel vale buys from heron mill",
		Subject: EntityNodeID("Kestrel Vale"), Premises: []string{EntityNodeID("Heron Mill")}}); err != nil {
		t.Fatal(err)
	}
	if err := db.graph.UpsertNode(ctx, &graph.GraphNode{ID: "note:n1", NodeType: "Note", Content: "kestrel vale and raven hill share a road"}); err != nil {
		t.Fatal(err)
	}
	for i, to := range []string{EntityNodeID("Kestrel Vale"), EntityNodeID("Raven Hill")} {
		if err := db.graph.UpsertEdge(ctx, &graph.GraphEdge{ID: "note-edge-" + strconv.Itoa(i), FromNodeID: "note:n1", ToNodeID: to, EdgeType: "about", Weight: 1}); err != nil {
			t.Fatal(err)
		}
	}
	if got := strings.Join(graphLaneDocs(t, db, ctx, "alpha", "Kestrel Vale", 3), ","); got != "a1,a10,a11,a4,a9" {
		t.Errorf("alpha, seed Kestrel Vale, 3 hops, past a decision and a note: %q, want a1,a10,a11,a4,a9", got)
	}
	// Remove beta's relation to Heron Mill and the memory: without a
	// collection, only the decision and the note still reach a3 and a6.
	if _, err := db.SQL().Exec(`DELETE FROM graph_edges WHERE edge_type = 'supplies' OR from_node_id = 'memory:m1' OR to_node_id = 'memory:m1'`); err != nil {
		t.Fatal(err)
	}
	got := graphLaneDocs(t, db, ctx, "", "Kestrel Vale", 3)
	for _, want := range []string{"a3", "a6"} {
		if !slices.Contains(got, want) {
			t.Errorf("unscoped lane from Kestrel Vale, 3 hops: %v, want %s among them", got, want)
		}
	}
}

// A document may have chunks in two collections. The walk takes a chunk
// of such a document, as the document has a chunk in the collection; the
// lane keeps only the chunks whose own row is in the collection
// (chunksInScope).
func TestTheGraphLaneKeepsOnlyTheCollectionsChunksOfASplitDocument(t *testing.T) {
	db, ctx := openLaneScopeBrain(t)
	var size int
	if err := db.SQL().QueryRow(`SELECT length(vector) FROM embeddings WHERE id = ?`, graphChunkNodeID("a1", 0)).Scan(&size); err != nil {
		t.Fatal(err)
	}
	vector := make([]float32, size/4)
	vector[0] = 1
	split := graphChunkNodeID("a1", 1)
	if err := db.Vector().Upsert(ctx, &core.Embedding{ID: split, Collection: "beta", DocID: "a1", Content: "the beta half of a1, betasecret", Vector: vector}); err != nil {
		t.Fatal(err)
	}
	if err := db.graph.UpsertNode(ctx, &graph.GraphNode{ID: split, NodeType: "chunk", Content: "the beta half of a1, betasecret", Properties: map[string]any{"document_id": "a1"}}); err != nil {
		t.Fatal(err)
	}
	if err := db.graph.UpsertEdge(ctx, &graph.GraphEdge{ID: "split-mention", FromNodeID: split, ToNodeID: EntityNodeID("Kestrel Vale"), EdgeType: "mentions", Weight: 1}); err != nil {
		t.Fatal(err)
	}
	resp, err := db.Query(ctx, QueryRequest{Collection: "alpha", Limit: 20,
		Prefetch: []QueryPrefetch{{Kind: QueryPrefetchGraph, EntityNames: []string{"Kestrel Vale"}, MaxHops: 1}}})
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, r := range resp.Results {
		ids = append(ids, r.ID)
	}
	if strings.Join(ids, ",") != graphChunkNodeID("a1", 0) {
		t.Errorf("alpha, seed Kestrel Vale, 1 hop: %v, want only %s", ids, graphChunkNodeID("a1", 0))
	}
}

// A relation that the collection's own document states is the collection's,
// whatever it cites as support (knowledge_save and upsert_relations take
// support_edge_ids from the writer): alpha's a6 states Kestrel Vale -backs->
// Heron Mill and cites beta's relation, and alpha's lane follows it to a3.
func TestTheGraphLaneFollowsARelationTheCollectionStatesWhateverItCites(t *testing.T) {
	db, ctx := openLaneScopeBrain(t)
	beta, err := db.graph.GetEdges(ctx, EntityNodeID("Heron Mill"), "both")
	if err != nil {
		t.Fatal(err)
	}
	var supplies string
	for _, edge := range beta {
		if edge.EdgeType == "supplies" {
			supplies = edge.ID
		}
	}
	if supplies == "" {
		t.Fatal("beta's relation to Heron Mill is missing")
	}
	chunk := graphChunkNodeID("a12", 0)
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "a12", Collection: "alpha", Content: "kestrel vale backs the heron mill.",
		Entities:  []ToolEntityInput{{Name: "Kestrel Vale", ChunkIDs: []string{chunk}}, {Name: "Heron Mill", ChunkIDs: []string{chunk}}},
		Relations: []ToolRelationInput{{From: "Kestrel Vale", To: "Heron Mill", Type: "backs", ChunkIDs: []string{chunk}, SupportEdgeIDs: []string{supplies, "edge:gone"}}}}); err != nil {
		t.Fatal(err)
	}
	if got := graphLaneDocs(t, db, ctx, "alpha", "Kestrel Vale", 2); !slices.Contains(got, "a3") {
		t.Errorf("alpha, seed Kestrel Vale, 2 hops, over a12's relation: %v, want a3 among them", got)
	}
}
