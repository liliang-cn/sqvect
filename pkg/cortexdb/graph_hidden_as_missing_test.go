package cortexdb

import (
	"context"
	"encoding/json"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

// seedCall is one graph tool called with a seed the caller names, once per
// argument that names a seed. Where the argument is a list, the seed sits
// between two seeds that do not exist, so the order of the answer counts
// too.
type seedCall struct {
	name   string
	byName bool
	call   func(ctx context.Context, db *DB, seed string) (any, error)
}

func seedCalls() []seedCall {
	kv := EntityNodeID("Kestrel Vale")
	ids := func(seed string) []string { return []string{"entity:aa_nope", seed, "entity:zz_nope"} }
	names := func(seed string) []string { return []string{"Aa Nope", seed, "Zz Nope"} }
	tools := func(db *DB) *GraphRAGToolbox { return db.GraphRAGTools() }
	km := func(db *DB) *KnowledgeMemory { return db.KnowledgeMemory() }
	return []seedCall{
		{"search_chunks_by_entities entity_names", true, func(ctx context.Context, db *DB, s string) (any, error) {
			return tools(db).SearchChunksByEntities(ctx, ToolSearchChunksByEntitiesRequest{EntityNames: names(s), MaxHops: 2})
		}},
		{"search_paths node_ids", false, func(ctx context.Context, db *DB, s string) (any, error) {
			return tools(db).SearchPaths(ctx, ToolSearchPathsRequest{NodeIDs: ids(s), WithText: true})
		}},
		{"search_paths entity_names", true, func(ctx context.Context, db *DB, s string) (any, error) {
			return tools(db).SearchPaths(ctx, ToolSearchPathsRequest{EntityNames: names(s), WithText: true})
		}},
		{"knowledge_memory_expand_entity_context entity_ids", false, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).ExpandEntityContext(ctx, KnowledgeMemoryExpandEntityContextRequest{EntityIDs: ids(s)})
		}},
		{"knowledge_memory_expand_entity_context entity_names", true, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).ExpandEntityContext(ctx, KnowledgeMemoryExpandEntityContextRequest{EntityNames: names(s)})
		}},
		{"expand_graph node_ids", false, func(ctx context.Context, db *DB, s string) (any, error) {
			return tools(db).ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: ids(s), MaxHops: 2})
		}},
		{"get_nodes node_ids", false, func(ctx context.Context, db *DB, s string) (any, error) {
			return tools(db).GetNodes(ctx, ToolGetNodesRequest{NodeIDs: ids(s)})
		}},
		{"knowledge_memory_neighbors node_id", false, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).Neighbors(ctx, KnowledgeMemoryNeighborsRequest{NodeID: s, MaxDepth: 2})
		}},
		{"knowledge_memory_neighbors entity_name", true, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).Neighbors(ctx, KnowledgeMemoryNeighborsRequest{EntityName: s, MaxDepth: 2})
		}},
		{"knowledge_memory_shortest_path from_node_id", false, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromNodeID: s, ToNodeID: kv})
		}},
		{"knowledge_memory_shortest_path to_node_id", false, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromNodeID: kv, ToNodeID: s})
		}},
		{"knowledge_memory_shortest_path from_node_id = to_node_id", false, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromNodeID: s, ToNodeID: s})
		}},
		{"knowledge_memory_shortest_path from_entity_name", true, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromEntityName: s, ToEntityName: "Kestrel Vale"})
		}},
		{"knowledge_memory_shortest_path to_entity_name", true, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromEntityName: "Kestrel Vale", ToEntityName: s})
		}},
		{"knowledge_memory_shortest_path from_entity_name = to_entity_name", true, func(ctx context.Context, db *DB, s string) (any, error) {
			return km(db).ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromEntityName: s, ToEntityName: s})
		}},
	}
}

// hiddenSeedIDs and hiddenSeedNames are what a caller confined to alpha and
// Hermes may name and must be answered as if they did not exist: beta's
// chunk, Zeus's memory, and the entity only beta knows.
var (
	hiddenSeedIDs   = []string{graphChunkNodeID("b1", 0), memoryGraphNodeID("zeus-mem"), EntityNodeID("Project Nightjar")}
	hiddenSeedNames = []string{"Project Nightjar"}
)

// seedAnswer is a call's answer, its error included, with the seed and the
// id a name derives masked: what the caller sent is not what the tool
// serves.
func seedAnswer(t *testing.T, out any, err error, seed string) string {
	t.Helper()
	raw, merr := json.Marshal(out)
	if merr != nil {
		t.Fatal(merr)
	}
	answer := string(raw)
	if err != nil {
		answer += " error: " + err.Error()
	}
	for _, s := range []string{EntityNodeID(seed), seed} {
		answer = strings.ReplaceAll(answer, s, "<seed>")
	}
	return answer
}

// The class of a seed: for each of the seven graph tools a confined key may
// call, and each argument that names a seed, a seed the scope hides answers
// byte for byte as a seed that does not exist, error included, its own
// name masked, and a seed that does not exist answers as it does without a
// scope: the graph's own answer. Without a scope the hidden seeds answer
// otherwise.
func TestEveryGraphToolAnswersAHiddenSeedAsAMissingOne(t *testing.T) {
	db := scopedGraphBrain(t)
	scoped := WithCallerScope(context.Background(), alphaScope())
	for _, tool := range seedCalls() {
		t.Run(tool.name, func(t *testing.T) {
			hidden, missing := hiddenSeedIDs, EntityNodeID("No Such")
			if tool.byName {
				hidden, missing = hiddenSeedNames, "No Such"
			}
			out, err := tool.call(scoped, db, missing)
			want := seedAnswer(t, out, err, missing)
			out, err = tool.call(context.Background(), db, missing)
			openMissing := seedAnswer(t, out, err, missing)
			if want != openMissing {
				t.Errorf("%s with a missing seed answers\n%s\nunder the scope, and\n%s\nwithout it", tool.name, want, openMissing)
			}
			differs := false
			for _, seed := range hidden {
				out, err := tool.call(scoped, db, seed)
				if got := seedAnswer(t, out, err, seed); got != want {
					t.Errorf("%s with hidden %s answers\n%s\nwant, as for a missing seed,\n%s", tool.name, seed, got, want)
				}
				out, err = tool.call(context.Background(), db, seed)
				differs = differs || seedAnswer(t, out, err, seed) != openMissing
			}
			if !differs && !strings.HasPrefix(tool.name, "knowledge_memory_shortest_path from_") {
				t.Errorf("%s: without a scope no hidden seed answers otherwise than a missing one, so the check proves nothing", tool.name)
			}
		})
	}
}

// heronBrain is scopedGraphBrain where alpha writes HeronGate, which folds
// to the same name as Heron Gate, and, when twin is set, beta writes Heron
// Gate itself.
func heronBrain(t *testing.T, twin bool) *DB {
	t.Helper()
	db := scopedGraphBrain(t)
	docs := []KnowledgeSaveRequest{{KnowledgeID: "h1", Collection: "alpha", Content: "HeronGate opens at dawn for Kestrel Vale.",
		Entities:  []ToolEntityInput{{Name: "HeronGate"}, {Name: "Kestrel Vale"}},
		Relations: []ToolRelationInput{{From: "HeronGate", To: "Kestrel Vale", Type: "serves"}}}}
	if twin {
		docs = append(docs, KnowledgeSaveRequest{KnowledgeID: "h2", Collection: "beta", Content: "Heron Gate hides BETASECRET for Kestrel Vale.",
			Entities:  []ToolEntityInput{{Name: "Heron Gate"}, {Name: "Kestrel Vale"}},
			Relations: []ToolRelationInput{{From: "Heron Gate", To: "Kestrel Vale", Type: "hides"}}})
	}
	for _, doc := range docs {
		for i := range doc.Entities {
			doc.Entities[i].ChunkIDs = []string{graphChunkNodeID(doc.KnowledgeID, 0)}
		}
		doc.Relations[0].ChunkIDs = []string{graphChunkNodeID(doc.KnowledgeID, 0)}
		if _, err := db.SaveKnowledge(context.Background(), doc); err != nil {
			t.Fatal(err)
		}
	}
	return db
}

// walkCounters matches the counters of search_paths, which count the stored
// graph, hidden nodes included, as the PR says.
var walkCounters = regexp.MustCompile(`"expansions":[0-9]+`)

// The class of a name whose match the scope hides, met beside a match it
// serves: alpha owns HeronGate, a fold match of Heron Gate. Whether or not
// beta wrote Heron Gate, an exact match, every graph tool that takes a name
// answers alpha's key the same, the walk's counters aside. It takes two
// brains, since in one brain the hidden match cannot be told from the
// masked seed.
func TestEveryGraphToolAnswersANameTheSameWhetherOrNotAHiddenTwinExists(t *testing.T) {
	alone, twinned := heronBrain(t, false), heronBrain(t, true)
	scoped := WithCallerScope(context.Background(), alphaScope())
	const seed = "Heron Gate"
	for _, tool := range seedCalls() {
		if !tool.byName {
			continue
		}
		t.Run(tool.name, func(t *testing.T) {
			out, err := tool.call(scoped, alone, seed)
			want := walkCounters.ReplaceAllString(seedAnswer(t, out, err, seed), "")
			out, err = tool.call(scoped, twinned, seed)
			if got := walkCounters.ReplaceAllString(seedAnswer(t, out, err, seed), ""); got != want {
				t.Errorf("%s with %q answers\n%s\nwhen beta wrote it, and\n%s\nwhen nobody did", tool.name, seed, got, want)
			}
			out, err = tool.call(context.Background(), alone, seed)
			openAlone := seedAnswer(t, out, err, seed)
			out, err = tool.call(context.Background(), twinned, seed)
			// Beta's edge leaves Heron Gate, and a path follows edges out of
			// its start, so a path to Heron Gate is no path in either brain.
			if seedAnswer(t, out, err, seed) == openAlone && tool.name != "knowledge_memory_shortest_path to_entity_name" {
				t.Errorf("%s: without a scope beta's Heron Gate changes nothing, so the check proves nothing", tool.name)
			}
		})
	}
}

// A name search_paths looks up resolves to an exact or a fold match, and
// not to a match that merely contains it, with or without a scope.
func TestASearchPathsNameResolvesToAnExactOrAFoldMatchOnly(t *testing.T) {
	db := heronBrain(t, false)
	if _, err := db.SaveKnowledge(context.Background(), KnowledgeSaveRequest{KnowledgeID: "wh", Collection: "alpha", Content: "wren hollow"}); err != nil {
		t.Fatal(err)
	}
	for label, ctx := range map[string]context.Context{"no scope": context.Background(), "alpha's scope": WithCallerScope(context.Background(), alphaScope())} {
		out, err := db.GraphRAGTools().SearchPaths(ctx, ToolSearchPathsRequest{EntityNames: []string{"wren hollow", "Heron Gate", "Heron"}})
		if err != nil {
			t.Fatal(err)
		}
		want := []PathSeed{{Name: "wren hollow", NodeID: graphChunkNodeID("wh", 0), Match: "exact"}, {Name: "Heron Gate", NodeID: EntityNodeID("HeronGate"), Match: "fold"}}
		if !slices.Equal(out.Seeds, want) || !slices.Equal(out.Unresolved, []string{"Heron"}) {
			t.Errorf("under %s seeds %+v unresolved %v, want %+v and [Heron]", label, out.Seeds, out.Unresolved, want)
		}
	}
}

// ospreyBrain is scopedGraphBrain where alpha writes Osprey Bay as a place
// and, when retyped is set, beta then writes it as an acquisition target.
func ospreyBrain(t *testing.T, retyped bool) *DB {
	t.Helper()
	db := scopedGraphBrain(t)
	docs := []KnowledgeSaveRequest{{KnowledgeID: "t1", Collection: "alpha", Content: "osprey bay holds the alpha store.",
		Entities: []ToolEntityInput{{Name: "Osprey Bay", Type: "place", ChunkIDs: []string{graphChunkNodeID("t1", 0)}}}}}
	if retyped {
		docs = append(docs, KnowledgeSaveRequest{KnowledgeID: "t2", Collection: "beta", Content: "osprey bay is for sale.",
			Entities: []ToolEntityInput{{Name: "Osprey Bay", Type: "acquisition_target", ChunkIDs: []string{graphChunkNodeID("t2", 0)}}}})
	}
	for _, doc := range docs {
		if _, err := db.SaveKnowledge(context.Background(), doc); err != nil {
			t.Fatal(err)
		}
	}
	return db
}

// writeTimes matches the times of a node's writes, which differ between two
// brains written a moment apart.
var writeTimes = regexp.MustCompile(`"(created_at|updated_at|valid_from|recorded_at)":"[^"]*"`)

// The class of a type filter: a reduced entity has no type, so under a
// knowledge rule it matches no node_types filter, and a walk filtered by
// a type answers alpha's key the same whether or not beta wrote that type.
// Without a scope the filter selects on the stored type, so beta's write
// changes the answer.
func TestUnderACallerScopeATypeFilterDoesNotSelectOnAnotherCollectionsType(t *testing.T) {
	plain, retyped := ospreyBrain(t, false), ospreyBrain(t, true)
	scoped := WithCallerScope(context.Background(), alphaScope())
	seed := graphChunkNodeID("t1", 0)
	calls := map[string]func(ctx context.Context, db *DB, nodeType string) (any, error){
		"expand_graph": func(ctx context.Context, db *DB, nodeType string) (any, error) {
			return db.GraphRAGTools().ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: []string{seed}, NodeTypes: []string{nodeType}})
		},
		// With relation policies the walk scores each node it keeps, and
		// the scores name them too.
		"expand_graph with relation_policies": func(ctx context.Context, db *DB, nodeType string) (any, error) {
			return db.GraphRAGTools().ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: []string{seed}, NodeTypes: []string{nodeType},
				RelationPolicies: graph.RelationPolicies{graph.RelationPolicyDefaultKey: {Weight: 1}}})
		},
		"knowledge_memory_neighbors": func(ctx context.Context, db *DB, nodeType string) (any, error) {
			return db.KnowledgeMemory().Neighbors(ctx, KnowledgeMemoryNeighborsRequest{NodeID: seed, NodeTypes: []string{nodeType}})
		},
	}
	for name, call := range calls {
		for _, nodeType := range []string{"place", "acquisition_target"} {
			out, err := call(scoped, plain, nodeType)
			want := writeTimes.ReplaceAllString(answerOf(t, out, err), "")
			out, err = call(scoped, retyped, nodeType)
			if got := writeTimes.ReplaceAllString(answerOf(t, out, err), ""); got != want {
				t.Errorf("%s node_types [%s] answers alpha\n%s\nwhen beta retyped Osprey Bay, and\n%s\nwhen it did not", name, nodeType, got, want)
			}
			// expand_graph serves its seed whatever its type, as on main.
			if strings.HasPrefix(name, "expand_graph") && !strings.Contains(want, `"id":"`+seed+`"`) {
				t.Errorf("expand_graph node_types [%s] no longer serves its seed: %s", nodeType, want)
			}
			if strings.Contains(want, EntityNodeID("Osprey Bay")) {
				t.Errorf("%s node_types [%s] serves alpha the reduced entity by its stored type: %s", name, nodeType, want)
			}
			out, err = call(context.Background(), plain, nodeType)
			openPlain := answerOf(t, out, err)
			out, err = call(context.Background(), retyped, nodeType)
			if answerOf(t, out, err) == openPlain {
				t.Errorf("%s node_types [%s]: without a scope beta's type changes nothing, so the check proves nothing", name, nodeType)
			}
		}
	}
}

// Under a scope the seeds are read to know which may be walked from. A
// cancelled call or a node store it cannot read is an error, as it is
// without a scope, not an answer with nothing in it.
func TestUnderACallerScopeASeedReadThatFailsIsAnError(t *testing.T) {
	calls := map[string]func(ctx context.Context, db *DB) (any, error){
		"search_chunks_by_entities": func(ctx context.Context, db *DB) (any, error) {
			return db.GraphRAGTools().SearchChunksByEntities(ctx, ToolSearchChunksByEntitiesRequest{EntityNames: []string{"Kestrel Vale"}})
		},
		"expand_graph": func(ctx context.Context, db *DB) (any, error) {
			return db.GraphRAGTools().ExpandGraph(ctx, ToolExpandGraphRequest{NodeIDs: []string{EntityNodeID("Kestrel Vale")}})
		},
		"knowledge_memory_neighbors": func(ctx context.Context, db *DB) (any, error) {
			return db.KnowledgeMemory().Neighbors(ctx, KnowledgeMemoryNeighborsRequest{EntityName: "Kestrel Vale"})
		},
		"knowledge_memory_shortest_path": func(ctx context.Context, db *DB) (any, error) {
			return db.KnowledgeMemory().ShortestPath(ctx, KnowledgeMemoryShortestPathRequest{FromEntityName: "Kestrel Vale", ToEntityName: "Alpha Mill"})
		},
	}
	for name, call := range calls {
		db := scopedGraphBrain(t)
		cancelled, cancel := context.WithCancel(WithCallerScope(context.Background(), alphaScope()))
		cancel()
		if out, err := call(cancelled, db); !failedOnTheStore(err) {
			t.Errorf("%s, cancelled under alpha's scope, answers %s, %v, not the store's error", name, answerOf(t, out, nil), err)
		}
		if _, err := db.SQL().Exec(`ALTER TABLE graph_nodes RENAME COLUMN content TO hidden_column`); err != nil {
			t.Fatal(err)
		}
		if out, err := call(WithCallerScope(context.Background(), alphaScope()), db); !failedOnTheStore(err) {
			t.Errorf("%s, with graph_nodes.content unreadable under alpha's scope, answers %s, %v, not the store's error", name, answerOf(t, out, nil), err)
		}
	}
}

// failedOnTheStore tells the store's error from the answers a missing or
// hidden node gets ("no path found", "node not found").
func failedOnTheStore(err error) bool {
	return err != nil && !strings.Contains(err.Error(), "no path found") && !strings.Contains(err.Error(), "node not found")
}
