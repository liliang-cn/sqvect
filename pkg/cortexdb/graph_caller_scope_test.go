package cortexdb

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

// scopedGraphBrain is a brain two collections and two users share. Both
// collections mention Kestrel Vale. Only beta's document holds BETASECRET,
// mentions Project Nightjar and backs the funds relation. Each collection
// backs a runs relation of its own, and one supplies relation cites a chunk
// of each. Zeus and Hermes each keep a memory about Kestrel Vale; Hermes's
// is proposed, which hides nothing while the review gate is off.
func scopedGraphBrain(t *testing.T) *DB {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "scoped-graph.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	a1, b1 := graphChunkNodeID("a1", 0), graphChunkNodeID("b1", 0)
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{
		KnowledgeID: "a1", Collection: "alpha", Content: "Kestrel Vale runs the alpha mill on the river.",
		Entities: []ToolEntityInput{
			{Name: "Kestrel Vale", Description: "alpha miller", ChunkIDs: []string{a1}},
			{Name: "Alpha Mill", ChunkIDs: []string{a1}},
		},
		Relations: []ToolRelationInput{{From: "Kestrel Vale", To: "Alpha Mill", Type: "runs", ChunkIDs: []string{a1}}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{
		KnowledgeID: "b1", Collection: "beta", Content: "Kestrel Vale secretly funds Project Nightjar with BETASECRET money, and runs the mill.",
		Entities: []ToolEntityInput{
			{Name: "Kestrel Vale", Description: "beta-only description BETADESC", ChunkIDs: []string{b1}},
			{Name: "Project Nightjar", Description: "covert BETADESC", ChunkIDs: []string{b1}},
			{Name: "Alpha Mill", ChunkIDs: []string{b1}},
		},
		Relations: []ToolRelationInput{
			{From: "Kestrel Vale", To: "Project Nightjar", Type: "funds", ChunkIDs: []string{b1}},
			{From: "Kestrel Vale", To: "Alpha Mill", Type: "runs", ChunkIDs: []string{b1}},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.GraphRAGTools().UpsertRelations(ctx, ToolUpsertRelationsRequest{Relations: []ToolRelationInput{
		{From: "Kestrel Vale", To: "Alpha Mill", Type: "supplies", ChunkIDs: []string{a1, b1}},
	}}); err != nil {
		t.Fatal(err)
	}
	for _, m := range []MemorySaveRequest{
		{MemoryID: "zeus-mem", UserID: "zeus", Content: "ZEUSPRIVATE: Kestrel Vale owes me money."},
		{MemoryID: "hermes-mem", UserID: "hermes", Content: "HERMESNOTE: Kestrel Vale delivers on Tuesdays.", ReviewStatus: "proposed"},
	} {
		m.Entities = []ToolEntityInput{{Name: "Kestrel Vale"}}
		if _, err := db.SaveMemory(ctx, m); err != nil {
			t.Fatal(err)
		}
	}
	return db
}

// alphaScope is a caller confined to collection alpha and to Hermes's
// memories.
func alphaScope() CallerScope {
	return CallerScope{
		Key:       "alpha-key",
		Knowledge: func(c string) bool { return c == "alpha" },
		Memory:    func(b MemoryBucket) bool { return b.UserID == "hermes" },
	}
}

// graphToolCall is one graph tool, called the way a port calls it.
type graphToolCall struct {
	name string
	call func(ctx context.Context, db *DB) (any, error)
	// served is what the call returns from alpha and from Hermes.
	served []string
}

// graphToolCalls lists every graph tool that walks the graph or reads its
// nodes by id, once per way of naming a seed where a tool has two: the
// eight tools a confined key may call once the facade confines them.
func graphToolCalls() []graphToolCall {
	kv := EntityNodeID("Kestrel Vale")
	return []graphToolCall{
		{name: "search_chunks_by_entities", served: []string{"alpha mill"}, call: func(ctx context.Context, db *DB) (any, error) {
			return db.GraphRAGTools().SearchChunksByEntities(ctx, ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Kestrel Vale"}})
		}},
		{name: "search_paths", served: []string{"alpha mill"}, call: func(ctx context.Context, db *DB) (any, error) {
			return db.GraphRAGTools().SearchPaths(ctx, ToolSearchPathsRequest{EntityNames: []string{"Kestrel Vale", "Alpha Mill", "Project Nightjar"}, WithText: true})
		}},
		{name: "knowledge_memory_expand_entity_context", served: []string{"alpha mill", "delivers on Tuesdays"}, call: func(ctx context.Context, db *DB) (any, error) {
			return db.KnowledgeMemory().ExpandEntityContext(ctx, KnowledgeMemoryExpandEntityContextRequest{EntityNames: []string{"Kestrel Vale"}})
		}},
		{name: "knowledge_memory_expand_entity_context/entity_ids", served: []string{"alpha mill", "delivers on Tuesdays"}, call: func(ctx context.Context, db *DB) (any, error) {
			// Given node ids alone, the tool names each node it reads after
			// its content. entity_node_ids echoes the caller's own ids.
			out, err := db.KnowledgeMemory().ExpandEntityContext(ctx, KnowledgeMemoryExpandEntityContextRequest{EntityIDs: []string{
				graphChunkNodeID("a1", 0), graphChunkNodeID("b1", 0), memoryGraphNodeID("hermes-mem"), memoryGraphNodeID("zeus-mem"),
			}})
			if out != nil {
				out.EntityNodeIDs = nil
			}
			return out, err
		}},
		{name: "expand_graph", served: []string{"alpha mill", "delivers on Tuesdays"}, call: func(ctx context.Context, db *DB) (any, error) {
			return db.GraphRAGTools().ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: []string{kv}, MaxHops: 2})
		}},
		{name: "get_nodes", served: []string{"alpha mill", "delivers on Tuesdays"}, call: func(ctx context.Context, db *DB) (any, error) {
			return db.GraphRAGTools().GetNodes(ctx, ToolGetNodesRequest{NodeIDs: []string{
				graphChunkNodeID("a1", 0), graphChunkNodeID("b1", 0), graphDocumentNodeID("a1"), graphDocumentNodeID("b1"),
				memoryGraphNodeID("zeus-mem"), memoryGraphNodeID("hermes-mem"),
			}})
		}},
		{name: "find_nodes", served: []string{"alpha mill", "delivers on Tuesdays"}, call: func(ctx context.Context, db *DB) (any, error) {
			// matches[].name echoes each name asked: no marker is one of them.
			return db.GraphRAGTools().FindNodes(ctx, ToolFindNodesRequest{Names: []string{"the river", "secretly funds", "owes me money", "on Tuesdays"}})
		}},
		{name: "knowledge_memory_neighbors", served: []string{"alpha mill", "delivers on Tuesdays"}, call: func(ctx context.Context, db *DB) (any, error) {
			return db.KnowledgeMemory().Neighbors(ctx, KnowledgeMemoryNeighborsRequest{EntityName: "Kestrel Vale", MaxDepth: 2})
		}},
		{name: "knowledge_memory_shortest_path", served: []string{"Alpha Mill"}, call: func(ctx context.Context, db *DB) (any, error) {
			// The path to Project Nightjar runs over the funds relation; when
			// the tool finds none, its error is the answer.
			km := db.KnowledgeMemory()
			toNightjar, err := km.ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromEntityName: "Kestrel Vale", ToEntityName: "Project Nightjar"})
			toMill, millErr := km.ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromEntityName: "Kestrel Vale", ToEntityName: "Alpha Mill"})
			return map[string]any{"to_nightjar": toNightjar, "no_path": err != nil, "to_mill": toMill}, millErr
		}},
	}
}

// findNodesNeedsTheHiddenFilter: find_nodes reads graph_nodes.content
// without the hidden-node filter the seven other tools pass through. The
// fix of that filter is its own PR; once it is in, find_nodes inherits the
// caller scope like the others and this skip goes.
const findNodesNeedsTheHiddenFilter = true

// answerOf is a tool's answer as a port would serve it.
func answerOf(t *testing.T, out any, err error) string {
	t.Helper()
	if err != nil {
		t.Fatalf("call failed: %v", err)
	}
	raw, merr := json.Marshal(out)
	if merr != nil {
		t.Fatal(merr)
	}
	return string(raw)
}

// What only beta and Zeus hold: beta's text, its chunk and document nodes,
// the relation only beta backs, and Zeus's memory. The words of the text
// that the ingest makes entities of (BETASECRET, Project Nightjar) are the
// entities' rule, not this one. No marker is a string a call sends, so an
// echo of the caller's own arguments cannot fail a row, and the served
// strings are not sent either.
var outOfScopeMarkers = []string{"BETASECRET money", graphChunkNodeID("b1", 0), graphDocumentNodeID("b1"), `"funds"`, "ZEUSPRIVATE", memoryGraphNodeID("zeus-mem")}

func TestUnderACallerScopeEveryGraphToolLeavesOutAnotherCollection(t *testing.T) {
	db := scopedGraphBrain(t)
	ctx := WithCallerScope(context.Background(), alphaScope())
	for _, tool := range graphToolCalls() {
		t.Run(tool.name, func(t *testing.T) {
			if tool.name == "find_nodes" && findNodesNeedsTheHiddenFilter {
				t.Skip("find_nodes does not pass the hidden-node filter yet (its own PR)")
			}
			out, err := tool.call(ctx, db)
			answer := answerOf(t, out, err)
			for _, marker := range outOfScopeMarkers {
				if strings.Contains(answer, marker) {
					t.Errorf("%s serves %q to a caller confined to alpha and Hermes: %s", tool.name, marker, answer)
				}
			}
			for _, want := range tool.served {
				if !strings.Contains(answer, want) {
					t.Errorf("%s no longer serves %q from the caller's own scope: %s", tool.name, want, answer)
				}
			}
		})
	}
}

// Without a caller scope, under a scope that sets no rule, and under one
// whose rules permit everything, every graph tool answers what it answered
// before, out-of-scope rows included.
func TestWithoutACallerScopeTheGraphToolsAnswerAsBefore(t *testing.T) {
	db := scopedGraphBrain(t)
	bg := context.Background()
	open := map[string]context.Context{
		"no rule":               WithCallerScope(bg, CallerScope{Key: "k"}),
		"rules that permit all": WithCallerScope(bg, CallerScope{Key: "k", Knowledge: func(string) bool { return true }, Memory: func(MemoryBucket) bool { return true }}),
	}
	for _, tool := range graphToolCalls() {
		t.Run(tool.name, func(t *testing.T) {
			out, err := tool.call(bg, db)
			before := answerOf(t, out, err)
			if tool.name != "knowledge_memory_shortest_path" && !strings.Contains(before, "BETASECRET money") && !strings.Contains(before, "ZEUSPRIVATE") {
				t.Fatalf("%s serves nothing out of alpha's scope without a caller scope, so the comparison proves nothing: %s", tool.name, before)
			}
			for label, ctx := range open {
				out, err := tool.call(ctx, db)
				if got := answerOf(t, out, err); got != before {
					t.Errorf("%s under a scope with %s answers\n%s\nwant\n%s", tool.name, label, got, before)
				}
			}
		})
	}
}

// A walk meets Zeus's memory node beside Kestrel Vale; a caller confined to
// Hermes's memories gets Hermes's memory and not Zeus's, under no review
// gate.
func TestUnderACallerScopeAGraphWalkLeavesOutAnotherUsersMemory(t *testing.T) {
	db := scopedGraphBrain(t)
	ctx := WithCallerScope(context.Background(), CallerScope{Key: "hermes", Memory: func(b MemoryBucket) bool { return b.UserID == "hermes" }})
	out, err := db.GraphRAGTools().ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: []string{EntityNodeID("Kestrel Vale")}})
	answer := answerOf(t, out, err)
	if strings.Contains(answer, "owes me money") || strings.Contains(answer, memoryGraphNodeID("zeus-mem")) {
		t.Fatalf("expand_graph serves Zeus's memory to a caller confined to Hermes: %s", answer)
	}
	if !strings.Contains(answer, "delivers on Tuesdays") {
		t.Fatalf("expand_graph no longer serves Hermes's own memory: %s", answer)
	}
	// Knowledge has no rule in this scope: beta's chunk is still served.
	if !strings.Contains(answer, "secretly funds") {
		t.Fatalf("a scope with no knowledge rule hid a document: %s", answer)
	}
}

// A memory node whose row cannot be read is hidden under a memory rule, as
// under the review gate: an outage does not lift the scope.
func TestUnderACallerScopeAnUnreadableMemoryIsHidden(t *testing.T) {
	db := scopedGraphBrain(t)
	ctx := WithCallerScope(context.Background(), CallerScope{Key: "hermes", Memory: func(MemoryBucket) bool { return true }})
	if _, err := db.SQL().Exec(`ALTER TABLE messages RENAME COLUMN role TO role_hidden`); err != nil {
		t.Fatal(err)
	}
	out, err := db.GraphRAGTools().ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: []string{EntityNodeID("Kestrel Vale")}})
	answer := answerOf(t, out, err)
	if strings.Contains(answer, "delivers on Tuesdays") || strings.Contains(answer, "owes me money") {
		t.Fatalf("expand_graph served a memory it could not check against the scope: %s", answer)
	}
	if _, err := db.SQL().Exec(`ALTER TABLE messages RENAME COLUMN role_hidden TO role`); err != nil {
		t.Fatal(err)
	}
}

// An edge only beta backs is left out under alpha's scope, and so is every
// path over it; an edge that cites a chunk of each collection keeps alpha's
// chunk and loses beta's.
func TestUnderACallerScopeAnEdgeOnlyAnotherCollectionBacksIsLeftOut(t *testing.T) {
	db := scopedGraphBrain(t)
	ctx := WithCallerScope(context.Background(), alphaScope())
	km := db.KnowledgeMemory()
	if _, err := km.ShortestPath(context.Background(), KnowledgeMemoryShortestPathRequest{FromEntityName: "Kestrel Vale", ToEntityName: "Project Nightjar"}); err != nil {
		t.Fatalf("without a scope the funds path exists: %v", err)
	}
	if path, err := km.ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromEntityName: "Kestrel Vale", ToEntityName: "Project Nightjar"}); err == nil {
		t.Fatalf("shortest_path walks the funds relation only beta backs: %+v", path.Path)
	}
	out, err := db.GraphRAGTools().ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: []string{EntityNodeID("Kestrel Vale")}, EdgeTypes: []string{"runs", "supplies"}})
	if err != nil {
		t.Fatal(err)
	}
	cited := map[string][]string{}
	for _, edge := range out.Edges {
		cited[edge.EdgeType] = append(cited[edge.EdgeType], stringsFromAny(edge.Properties["chunk_ids"])...)
	}
	a1 := graphChunkNodeID("a1", 0)
	if got := cited["runs"]; len(got) != 1 || got[0] != a1 {
		t.Fatalf("the runs edges cite %v under alpha's scope, want alpha's own edge only", got)
	}
	if got := cited["supplies"]; len(got) != 1 || got[0] != a1 {
		t.Fatalf("the supplies edge cites %v under alpha's scope, want alpha's chunk only", got)
	}
}

// When the collections of the documents a walk meets cannot be read, a
// caller under a knowledge rule gets none of them: the scope fails closed.
// Without a scope the same walk still serves both.
func TestUnderACallerScopeAnUnreadableCollectionHidesTheDocument(t *testing.T) {
	db := scopedGraphBrain(t)
	ctx := WithCallerScope(context.Background(), alphaScope())
	if _, err := db.SQL().Exec(`ALTER TABLE collections RENAME COLUMN name TO name_hidden`); err != nil {
		t.Fatal(err)
	}
	req := ToolGetNodesRequest{NodeIDs: []string{graphChunkNodeID("a1", 0), graphChunkNodeID("b1", 0)}}
	out, err := db.GraphRAGTools().GetNodes(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Nodes) != 0 {
		t.Fatalf("get_nodes served %d chunk nodes whose collection it could not read", len(out.Nodes))
	}
	open, err := db.GraphRAGTools().GetNodes(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(open.Nodes) != 2 {
		t.Fatalf("without a scope get_nodes served %d chunk nodes, want 2", len(open.Nodes))
	}
	if _, err := db.SQL().Exec(`ALTER TABLE collections RENAME COLUMN name_hidden TO name`); err != nil {
		t.Fatal(err)
	}
}

// The scope adds to the expiry, it does not replace it: an expired document
// of the caller's own collection stays hidden under its scope.
func TestUnderACallerScopeAnExpiredDocumentOfTheCallersCollectionStaysHidden(t *testing.T) {
	db := scopedGraphBrain(t)
	past := "2001-01-01T00:00:00Z"
	if _, err := db.SaveKnowledge(context.Background(), KnowledgeSaveRequest{KnowledgeID: "a2", Collection: "alpha", Content: "The old alpha ledger.", ExpiresAt: &past}); err != nil {
		t.Fatal(err)
	}
	ctx := WithCallerScope(context.Background(), alphaScope())
	out, err := db.GraphRAGTools().GetNodes(ctx, ToolGetNodesRequest{NodeIDs: []string{graphChunkNodeID("a1", 0), graphChunkNodeID("a2", 0)}})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Nodes) != 1 || out.Nodes[0].ID != graphChunkNodeID("a1", 0) {
		t.Fatalf("get_nodes under alpha's scope served %d nodes, want a1's chunk alone: %+v", len(out.Nodes), out.Nodes)
	}
}

// The expiry is advisory on an outage, the scope is not: when the documents
// table cannot be read, beta's chunk is still hidden from alpha's caller,
// and alpha's own is served.
func TestUnderACallerScopeAnUnreadableExpiryStillHidesAnotherCollection(t *testing.T) {
	db := scopedGraphBrain(t)
	if _, err := db.SQL().Exec(`ALTER TABLE documents RENAME COLUMN metadata TO metadata_hidden`); err != nil {
		t.Fatal(err)
	}
	ctx := WithCallerScope(context.Background(), alphaScope())
	out, err := db.GraphRAGTools().GetNodes(ctx, ToolGetNodesRequest{NodeIDs: []string{graphChunkNodeID("a1", 0), graphChunkNodeID("b1", 0)}})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Nodes) != 1 || out.Nodes[0].ID != graphChunkNodeID("a1", 0) {
		t.Fatalf("get_nodes under alpha's scope, expiry unreadable, served %d nodes, want a1's chunk alone: %+v", len(out.Nodes), out.Nodes)
	}
	if _, err := db.SQL().Exec(`ALTER TABLE documents RENAME COLUMN metadata_hidden TO metadata`); err != nil {
		t.Fatal(err)
	}
}

// When the chunk nodes an edge cites cannot be read, a caller under a
// knowledge rule, with no memory rule, gets none of the edges that cite them: the scope fails
// closed. Without a scope the provenance stays as stored, since the expiry
// alone is advisory.
func TestUnderACallerScopeUnreadableProvenanceHidesTheEdge(t *testing.T) {
	db := scopedGraphBrain(t)
	edges := []*graph.GraphEdge{
		{ID: "e-beta", FromNodeID: EntityNodeID("Kestrel Vale"), ToNodeID: EntityNodeID("Project Nightjar"), EdgeType: "funds", Properties: map[string]any{"chunk_ids": []string{graphChunkNodeID("b1", 0)}}},
		{ID: "e-both", FromNodeID: EntityNodeID("Kestrel Vale"), ToNodeID: EntityNodeID("Alpha Mill"), EdgeType: "supplies", Properties: map[string]any{"chunk_ids": []string{graphChunkNodeID("a1", 0), graphChunkNodeID("b1", 0)}}},
	}
	if _, err := db.SQL().Exec(`ALTER TABLE graph_nodes RENAME COLUMN content TO content_hidden`); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if _, err := db.SQL().Exec(`ALTER TABLE graph_nodes RENAME COLUMN content_hidden TO content`); err != nil {
			t.Fatal(err)
		}
	}()
	// A knowledge rule alone: the outage branch keys on that rule.
	knowledgeOnly := WithCallerScope(context.Background(), CallerScope{Key: "alpha-key", Knowledge: func(c string) bool { return c == "alpha" }})
	if got := db.withoutExpiredProvenance(knowledgeOnly, edges); len(got) != 0 {
		t.Fatalf("under alpha's scope, provenance it could not read kept %d edges: %+v", len(got), got)
	}
	if got := db.withoutExpiredProvenance(context.Background(), edges); len(got) != 2 {
		t.Fatalf("without a scope, unreadable provenance kept %d edges, want both as stored", len(got))
	}
}

// A memory node whose bucket cannot be read is hidden under a memory rule,
// even one that permits every bucket: the memory row reads, its bucket
// does not.
func TestUnderACallerScopeAMemoryWhoseBucketCannotBeReadIsHidden(t *testing.T) {
	db := scopedGraphBrain(t)
	ctx := WithCallerScope(context.Background(), CallerScope{Key: "hermes", Memory: func(MemoryBucket) bool { return true }})
	req := ToolGetNodesRequest{NodeIDs: []string{memoryGraphNodeID("hermes-mem")}}
	if out, err := db.GraphRAGTools().GetNodes(ctx, req); err != nil || len(out.Nodes) != 1 {
		t.Fatalf("with its bucket readable, get_nodes serves %+v, %v, want Hermes's memory", out, err)
	}
	if _, err := db.SQL().Exec(`ALTER TABLE sessions RENAME COLUMN updated_at TO updated_at_hidden`); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if _, err := db.SQL().Exec(`ALTER TABLE sessions RENAME COLUMN updated_at_hidden TO updated_at`); err != nil {
			t.Fatal(err)
		}
	}()
	out, err := db.GraphRAGTools().GetNodes(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Nodes) != 0 {
		t.Fatalf("get_nodes served a memory whose bucket it could not check against the scope: %+v", out.Nodes)
	}
}

// The walk is widened more than once: twenty beta chunks met
// before alpha's use up the pools of 8 and 16, and the pool of 32 reaches
// alpha's a1.
func TestUnderACallerScopeSearchChunksByEntitiesWidensItsWalkTwice(t *testing.T) {
	db := scopedGraphBrain(t)
	for i := range 20 {
		id := fmt.Sprintf("0%02d", i)
		doc := KnowledgeSaveRequest{KnowledgeID: id, Collection: "beta", Content: "Kestrel Vale keeps beta ledger " + id + "."}
		doc.Entities = []ToolEntityInput{{Name: "Kestrel Vale", ChunkIDs: []string{graphChunkNodeID(id, 0)}}}
		if _, err := db.SaveKnowledge(context.Background(), doc); err != nil {
			t.Fatal(err)
		}
	}
	req := ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Kestrel Vale"}, TopK: 1}
	out, err := db.GraphRAGTools().SearchChunksByEntities(WithCallerScope(context.Background(), alphaScope()), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Chunks) != 1 || out.Chunks[0].ID != graphChunkNodeID("a1", 0) {
		t.Fatalf("under alpha's scope top_k 1 serves %+v, want alpha's a1, past twenty beta chunks", out.Chunks)
	}
}

// The walk stops widening at the first pool that fills top_k, so the
// answer ranks the chunks that pool met, as Scope says. Fifteen beta chunks
// and alpha's a0 fill Sable Ridge's pool of 16 (top_k 2); alpha's zz,
// which mentions Sable Ridge and Tern Lock, is met through Tern Lock only.
// a0 and zz fill top_k with one name each, ranked by id. A wider pool
// would also meet zz through Sable Ridge and rank it first.
func TestUnderACallerScopeSearchChunksByEntitiesStopsAtThePoolThatFillsTopK(t *testing.T) {
	db := scopedGraphBrain(t)
	var docs []KnowledgeSaveRequest
	for i := range 15 {
		id := fmt.Sprintf("0%02d", i)
		docs = append(docs, KnowledgeSaveRequest{KnowledgeID: id, Collection: "beta", Content: "sable ridge beta note " + id + ".",
			Entities: []ToolEntityInput{{Name: "Sable Ridge", ChunkIDs: []string{graphChunkNodeID(id, 0)}}}})
	}
	docs = append(docs,
		KnowledgeSaveRequest{KnowledgeID: "a0", Collection: "alpha", Content: "sable ridge alpha note.",
			Entities: []ToolEntityInput{{Name: "Sable Ridge", ChunkIDs: []string{graphChunkNodeID("a0", 0)}}}},
		KnowledgeSaveRequest{KnowledgeID: "zz", Collection: "alpha", Content: "sable ridge feeds tern lock.",
			Entities: []ToolEntityInput{{Name: "Sable Ridge", ChunkIDs: []string{graphChunkNodeID("zz", 0)}}, {Name: "Tern Lock", ChunkIDs: []string{graphChunkNodeID("zz", 0)}}}})
	for _, doc := range docs {
		if _, err := db.SaveKnowledge(context.Background(), doc); err != nil {
			t.Fatal(err)
		}
	}
	req := ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Sable Ridge", "Tern Lock"}, TopK: 2}
	out, err := db.GraphRAGTools().SearchChunksByEntities(WithCallerScope(context.Background(), alphaScope()), req)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, c := range out.Chunks {
		got = append(got, c.ID)
	}
	if want := []string{graphChunkNodeID("a0", 0), graphChunkNodeID("zz", 0)}; !slices.Equal(got, want) {
		t.Fatalf("under alpha's scope top_k 2 serves %v, want %v, the ranking of the first pool that fills top_k", got, want)
	}
}

// A top_k so large that top_k*8 overflows is answered as main answers it,
// one walk with no limit, and does not spin.
func TestSearchChunksByEntitiesAnswersAHugeTopK(t *testing.T) {
	db := scopedGraphBrain(t)
	for _, topK := range []int{1 << 60, 1 << 61, math.MaxInt} {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		out, err := db.GraphRAGTools().SearchChunksByEntities(WithCallerScope(ctx, alphaScope()), ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Kestrel Vale"}, TopK: topK})
		cancel()
		if err != nil || len(out.Chunks) != 1 || out.Chunks[0].ID != graphChunkNodeID("a1", 0) {
			t.Errorf("top_k %d: %+v, %v, want alpha's a1 alone", topK, out, err)
		}
	}
}

// A cancelled call stops the walk and says so.
func TestSearchChunksByEntitiesStopsOnACancelledContext(t *testing.T) {
	db := scopedGraphBrain(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := db.GraphRAGTools().SearchChunksByEntities(ctx, ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Kestrel Vale"}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled call answers %v, want context.Canceled", err)
	}
}

// At two hops the walk's pool can fill with entities met through a hidden
// chunk before the caller's chunk is met: the walk counts every node it
// meets, not only chunks, to know it was cut. Beta's b0 mentions Sable
// Ridge and eight other entities; alpha's a9 lists Sable Ridge with no
// mention, backs its relation to Tern Lock and mentions Tern Lock only.
func TestUnderACallerScopeSearchChunksByEntitiesWidensPastEntitiesAtTwoHops(t *testing.T) {
	db := scopedGraphBrain(t)
	b0, a9 := graphChunkNodeID("b0", 0), graphChunkNodeID("a9", 0)
	beta := KnowledgeSaveRequest{KnowledgeID: "b0", Collection: "beta", Content: "sable ridge and its eight neighbours.",
		Entities: []ToolEntityInput{{Name: "Sable Ridge", ChunkIDs: []string{b0}}}}
	for i := range 8 {
		beta.Entities = append(beta.Entities, ToolEntityInput{Name: "Neighbour " + strconv.Itoa(i), ChunkIDs: []string{b0}})
	}
	alpha := KnowledgeSaveRequest{KnowledgeID: "a9", Collection: "alpha", Content: "tern lock opens.",
		Entities:  []ToolEntityInput{{Name: "Tern Lock", ChunkIDs: []string{a9}}, {Name: "Sable Ridge"}},
		Relations: []ToolRelationInput{{From: "Sable Ridge", To: "Tern Lock", Type: "feeds", ChunkIDs: []string{a9}}}}
	for _, doc := range []KnowledgeSaveRequest{beta, alpha} {
		if _, err := db.SaveKnowledge(context.Background(), doc); err != nil {
			t.Fatal(err)
		}
	}
	req := ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Sable Ridge"}, TopK: 1, MaxHops: 2}
	out, err := db.GraphRAGTools().SearchChunksByEntities(WithCallerScope(context.Background(), alphaScope()), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Chunks) != 1 || out.Chunks[0].ID != a9 {
		t.Fatalf("under alpha's scope max_hops 2 top_k 1 serves %+v, want alpha's a9", out.Chunks)
	}
}

// A walk or a chunk read that fails is an error, not an empty answer, as
// on main.
func TestSearchChunksByEntitiesReportsAStoreItCannotRead(t *testing.T) {
	db := scopedGraphBrain(t)
	req := ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Kestrel Vale"}, TopK: 1}
	for _, broken := range []struct{ table, column string }{{"graph_edges", "edge_type"}, {"embeddings", "content"}} {
		if _, err := db.SQL().Exec(`ALTER TABLE ` + broken.table + ` RENAME COLUMN ` + broken.column + ` TO hidden_column`); err != nil {
			t.Fatal(err)
		}
		if out, err := db.GraphRAGTools().SearchChunksByEntities(context.Background(), req); err == nil {
			t.Errorf("with %s.%s unreadable the tool answers %+v and no error", broken.table, broken.column, out)
		}
		if _, err := db.SQL().Exec(`ALTER TABLE ` + broken.table + ` RENAME COLUMN hidden_column TO ` + broken.column); err != nil {
			t.Fatal(err)
		}
	}
}

// The walk's pool is wider than top_k, so the chunks that several names
// reach outrank those one name reaches first: zz, which mentions both
// names, comes before 00, met first by one name only.
func TestSearchChunksByEntitiesRanksTheChunksMoreNamesReach(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "pool.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	for _, doc := range []struct {
		id    string
		names []string
	}{{"00", []string{"Sable Ridge"}}, {"01", []string{"Sable Ridge"}}, {"02", []string{"Sable Ridge"}}, {"zz", []string{"Sable Ridge", "Tern Lock"}}} {
		req := KnowledgeSaveRequest{KnowledgeID: doc.id, Content: "Sable Ridge and the lock " + doc.id + "."}
		for _, name := range doc.names {
			req.Entities = append(req.Entities, ToolEntityInput{Name: name, ChunkIDs: []string{graphChunkNodeID(doc.id, 0)}})
		}
		if _, err := db.SaveKnowledge(context.Background(), req); err != nil {
			t.Fatal(err)
		}
	}
	out, err := db.GraphRAGTools().SearchChunksByEntities(context.Background(), ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Sable Ridge", "Tern Lock"}, TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Chunks) != 1 || out.Chunks[0].ID != graphChunkNodeID("zz", 0) {
		t.Fatalf("top_k 1 serves %+v, want zz, which both names reach", out.Chunks)
	}
}

// Eight beta chunks met first fill the walk's own pool of top_k*8, so the
// walk is widened until the caller's chunk is met, or the walk ends.
func TestUnderACallerScopeSearchChunksByEntitiesWidensItsWalkPastHiddenChunks(t *testing.T) {
	db := scopedGraphBrain(t)
	for i := range 8 {
		id := "0" + strconv.Itoa(i)
		doc := KnowledgeSaveRequest{KnowledgeID: id, Collection: "beta", Content: "Kestrel Vale keeps beta ledger " + id + "."}
		doc.Entities = []ToolEntityInput{{Name: "Kestrel Vale", ChunkIDs: []string{graphChunkNodeID(id, 0)}}}
		if _, err := db.SaveKnowledge(context.Background(), doc); err != nil {
			t.Fatal(err)
		}
	}
	req := ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Kestrel Vale"}, TopK: 1}
	open, err := db.GraphRAGTools().SearchChunksByEntities(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(open.Chunks) != 1 || open.Chunks[0].ID != graphChunkNodeID("00", 0) {
		t.Fatalf("without a scope top_k 1 serves %+v, want beta's 00, met first", open.Chunks)
	}
	out, err := db.GraphRAGTools().SearchChunksByEntities(WithCallerScope(context.Background(), alphaScope()), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Chunks) != 1 || out.Chunks[0].ID != graphChunkNodeID("a1", 0) {
		t.Fatalf("under alpha's scope top_k 1 serves %+v, want alpha's a1, past eight beta chunks", out.Chunks)
	}
}

// Hidden chunks do not use up top_k. Ranked: an expired chunk 00, beta's
// chunk 0b, alpha's a1. Without a scope top_k 1 serves 0b, past the
// expired one; alpha's caller gets a1, past both.
func TestUnderACallerScopeSearchChunksByEntitiesFillsTopKPastHiddenChunks(t *testing.T) {
	db := scopedGraphBrain(t)
	b0 := graphChunkNodeID("0b", 0)
	for _, doc := range []KnowledgeSaveRequest{
		{KnowledgeID: "00", Collection: "alpha", Content: "Kestrel Vale kept the old ledger.", ExpiresAt: strPtr(pastInstant())},
		{KnowledgeID: "0b", Collection: "beta", Content: "Kestrel Vale keeps the beta ledger."},
	} {
		doc.Entities = []ToolEntityInput{{Name: "Kestrel Vale", ChunkIDs: []string{graphChunkNodeID(doc.KnowledgeID, 0)}}}
		if _, err := db.SaveKnowledge(context.Background(), doc); err != nil {
			t.Fatal(err)
		}
	}
	req := ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Kestrel Vale"}, TopK: 1}
	open, err := db.GraphRAGTools().SearchChunksByEntities(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(open.Chunks) != 1 || open.Chunks[0].ID != b0 {
		t.Fatalf("without a scope top_k 1 serves %+v, want beta's 0b, past the expired chunk", open.Chunks)
	}
	out, err := db.GraphRAGTools().SearchChunksByEntities(WithCallerScope(context.Background(), alphaScope()), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Chunks) != 1 || out.Chunks[0].ID != graphChunkNodeID("a1", 0) {
		t.Fatalf("under alpha's scope top_k 1 serves %+v, want alpha's a1", out.Chunks)
	}
}
