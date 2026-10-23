package httpapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// The REST twin of the gRPC test, on the same graph: alpha and beta both
// mention Kestrel Vale; only beta's text holds the secret, mentions Project
// Nightjar and backs the funds relation; Zeus keeps a memory about Kestrel
// Vale.
var twoCollectionGraph = []struct{ tool, args string }{
	{"knowledge_save", `{"knowledge_id":"a1","collection":"alpha","content":"Kestrel Vale runs the alpha mill on the river.",
		"entities":[{"name":"Kestrel Vale","description":"alpha miller","chunk_ids":["chunk:a1:000"]},{"name":"Alpha Mill","chunk_ids":["chunk:a1:000"]}],
		"relations":[{"from":"Kestrel Vale","to":"Alpha Mill","type":"runs","chunk_ids":["chunk:a1:000"]}]}`},
	{"knowledge_save", `{"knowledge_id":"b1","collection":"beta","content":"Kestrel Vale secretly funds Project Nightjar with BETASECRET money.",
		"entities":[{"name":"Kestrel Vale","description":"beta-only description BETADESC","chunk_ids":["chunk:b1:000"]},{"name":"Project Nightjar","chunk_ids":["chunk:b1:000"]}],
		"relations":[{"from":"Kestrel Vale","to":"Project Nightjar","type":"funds","chunk_ids":["chunk:b1:000"]}]}`},
	{"memory_save", `{"memory_id":"zeus-mem","user_id":"zeus","content":"ZEUSPRIVATE: Kestrel Vale owes me money.","entities":[{"name":"Kestrel Vale"}]}`},
	// A write that names a document records it on the entity.
	{"upsert_entities", `{"document_id":"b1","entities":[{"name":"Kestrel Vale"}]}`},
}

var graphWalks = []struct{ tool, args string }{
	{"search_chunks_by_entities", `{"entity_names":["Kestrel Vale","Project Nightjar"],"max_hops":2}`},
	{"search_paths", `{"entity_names":["Kestrel Vale","Alpha Mill","Project Nightjar"],"with_text":true}`},
	{"knowledge_memory_expand_entity_context", `{"entity_names":["Kestrel Vale"]}`},
	{"expand_graph", `{"node_ids":["entity:kestrel_vale"],"max_hops":2}`},
	{"get_nodes", `{"node_ids":["entity:kestrel_vale","entity:project_nightjar","chunk:a1:000","chunk:b1:000","doc:b1","memory:zeus-mem"]}`},
	{"knowledge_memory_neighbors", `{"entity_name":"Kestrel Vale","max_depth":2}`},
	{"knowledge_memory_shortest_path", `{"from_entity_name":"Kestrel Vale","to_entity_name":"Alpha Mill"}`},
}

var outsideAlpha = []string{"secretly funds", "chunk:b1:000", "doc:b1", "entity:project_nightjar", "BETADESC", `"beta"`, `"source_document_ids"`, "owes me money", "memory:zeus-mem", "0001-01-01"}

// seedWalks is each graph tool the allowlist opens, once per argument that
// names a seed, %s standing for the seed. Where the argument is a list, the
// seed sits between two seeds that do not exist, so the order of the answer
// counts too.
var seedWalks = []struct {
	tool, args string
	byName     bool
}{
	{"search_chunks_by_entities", `{"entity_names":["Aa Nope",%s,"Zz Nope"],"max_hops":2}`, true},
	{"search_paths", `{"node_ids":["entity:aa_nope",%s,"entity:zz_nope"],"with_text":true}`, false},
	{"search_paths", `{"entity_names":["Aa Nope",%s,"Zz Nope"],"with_text":true}`, true},
	{"knowledge_memory_expand_entity_context", `{"entity_ids":["entity:aa_nope",%s,"entity:zz_nope"]}`, false},
	{"knowledge_memory_expand_entity_context", `{"entity_names":["Aa Nope",%s,"Zz Nope"]}`, true},
	{"expand_graph", `{"node_ids":["entity:aa_nope",%s,"entity:zz_nope"],"max_hops":2}`, false},
	{"get_nodes", `{"node_ids":["entity:aa_nope",%s,"entity:zz_nope"]}`, false},
	{"knowledge_memory_neighbors", `{"node_id":%s,"max_depth":2}`, false},
	{"knowledge_memory_neighbors", `{"entity_name":%s,"max_depth":2}`, true},
	{"knowledge_memory_shortest_path", `{"from_node_id":%s,"to_node_id":"entity:kestrel_vale"}`, false},
	{"knowledge_memory_shortest_path", `{"from_node_id":"entity:kestrel_vale","to_node_id":%s}`, false},
	{"knowledge_memory_shortest_path", `{"from_node_id":%[1]s,"to_node_id":%[1]s}`, false},
	{"knowledge_memory_shortest_path", `{"from_entity_name":%s,"to_entity_name":"Kestrel Vale"}`, true},
	{"knowledge_memory_shortest_path", `{"from_entity_name":"Kestrel Vale","to_entity_name":%s}`, true},
	{"knowledge_memory_shortest_path", `{"from_entity_name":%[1]s,"to_entity_name":%[1]s}`, true},
}

// hiddenSeeds are the seeds a key confined to alpha may name and must be
// answered as if they did not exist: beta's chunk, Zeus's memory, and the
// entity only beta knows, by id and by name.
var hiddenSeeds = map[bool][]string{false: {"chunk:b1:000", "memory:zeus-mem", "entity:project_nightjar"}, true: {"Project Nightjar"}}

// missingSeed is a seed that does not exist, by id and by name.
var missingSeed = map[bool]string{false: "entity:no_such", true: "No Such"}

// everySeedAnswersAsAMissingOne calls each row of seedWalks through answer,
// which returns a port's answer (result and error code) and, apart, the
// error message. For the alpha key a hidden seed answers byte for byte as
// the missing one, message included, the seed and the id a name derives
// masked, and the missing one answers as it does for the operator, whose
// error messages a confined key does not get; for the operator some hidden
// seed of each row answers otherwise, except where a path from it does not
// exist either way.
func everySeedAnswersAsAMissingOne(t *testing.T, answer func(secret, tool, args string) (string, string)) {
	t.Helper()
	mask := func(got, seed string) string {
		for _, s := range []string{cortexdb.EntityNodeID(seed), seed} {
			got = strings.ReplaceAll(got, s, "<seed>")
		}
		return got
	}
	both := func(secret string, row int, seed string) (string, string) {
		w := seedWalks[row]
		quoted, _ := json.Marshal(seed)
		body, message := answer(secret, w.tool, fmt.Sprintf(w.args, quoted))
		return mask(body, seed), mask(message, seed)
	}
	call := func(secret string, row int, seed string) string {
		body, message := both(secret, row, seed)
		return body + " | " + message
	}
	for row, w := range seedWalks {
		missing := missingSeed[w.byName]
		want := call("alpha-secret", row, missing)
		alphaMissing, _ := both("alpha-secret", row, missing)
		openMissing, _ := both("op-secret", row, missing)
		if alphaMissing != openMissing {
			t.Errorf("%s %s with a missing seed answers\n%s\nto alpha, and\n%s\nto the operator", w.tool, w.args, alphaMissing, openMissing)
		}
		openMissingWhole := call("op-secret", row, missing)
		differs := false
		for _, seed := range hiddenSeeds[w.byName] {
			if got := call("alpha-secret", row, seed); got != want {
				t.Errorf("alpha %s %s with hidden %s answers\n%s\nwant, as for a missing seed,\n%s", w.tool, w.args, seed, got, want)
			}
			differs = differs || call("op-secret", row, seed) != openMissingWhole
		}
		if !differs && !strings.HasPrefix(w.args, `{"from_`) {
			t.Errorf("operator %s %s: no hidden seed answers otherwise than a missing one, so the check proves nothing", w.tool, w.args)
		}
	}
}

// heronGraph is what the operator writes on top of twoCollectionGraph:
// alpha's HeronGate, a fold match of Heron Gate, and, in the twinned brain,
// beta's Heron Gate itself.
func heronGraph(twin bool) []struct{ tool, args string } {
	writes := []struct{ tool, args string }{{"knowledge_save", `{"knowledge_id":"h1","collection":"alpha","content":"HeronGate opens at dawn for Kestrel Vale.",
		"entities":[{"name":"HeronGate","chunk_ids":["chunk:h1:000"]},{"name":"Kestrel Vale","chunk_ids":["chunk:h1:000"]}],
		"relations":[{"from":"HeronGate","to":"Kestrel Vale","type":"serves","chunk_ids":["chunk:h1:000"]}]}`}}
	if twin {
		writes = append(writes, struct{ tool, args string }{"knowledge_save", `{"knowledge_id":"h2","collection":"beta","content":"Heron Gate hides BETASECRET for Kestrel Vale.",
		"entities":[{"name":"Heron Gate","chunk_ids":["chunk:h2:000"]},{"name":"Kestrel Vale","chunk_ids":["chunk:h2:000"]}],
		"relations":[{"from":"Heron Gate","to":"Kestrel Vale","type":"hides","chunk_ids":["chunk:h2:000"]}]}`})
	}
	return append(slices.Clone(twoCollectionGraph), writes...)
}

// walkCounters matches the counters of search_paths, which count the
// stored graph, hidden nodes included, as the PR says.
var walkCounters = regexp.MustCompile(`"expansions":[0-9]+`)

// everyNameAnswersAlikeBesideAHiddenTwin calls each row of seedWalks that
// names a seed with Heron Gate, in a brain where only alpha's fold match
// HeronGate exists (alone) and in one where beta also wrote Heron Gate
// (twinned). The alpha key is answered the same in both, the walk's
// counters aside; the operator is answered otherwise, except where no path
// leaves Heron Gate's way in either brain.
func everyNameAnswersAlikeBesideAHiddenTwin(t *testing.T, alone, twinned func(secret, tool, args string) (string, string)) {
	t.Helper()
	const seed = "Heron Gate"
	quoted, _ := json.Marshal(seed)
	call := func(answer func(secret, tool, args string) (string, string), secret, tool, args string) string {
		body, message := answer(secret, tool, fmt.Sprintf(args, quoted))
		got := walkCounters.ReplaceAllString(body+" | "+message, "")
		for _, s := range []string{cortexdb.EntityNodeID(seed), seed} {
			got = strings.ReplaceAll(got, s, "<seed>")
		}
		return got
	}
	for _, w := range seedWalks {
		if !w.byName {
			continue
		}
		want := call(alone, "alpha-secret", w.tool, w.args)
		if got := call(twinned, "alpha-secret", w.tool, w.args); got != want {
			t.Errorf("alpha %s %s with %q answers\n%s\nwhen beta wrote it, and\n%s\nwhen nobody did", w.tool, w.args, seed, got, want)
		}
		// Beta's edge leaves Heron Gate, and a path follows edges out of its
		// start, so a path to Heron Gate is no path in either brain.
		if call(alone, "op-secret", w.tool, w.args) == call(twinned, "op-secret", w.tool, w.args) && w.args != `{"from_entity_name":"Kestrel Vale","to_entity_name":%s}` {
			t.Errorf("operator %s %s: beta's Heron Gate changes nothing, so the check proves nothing", w.tool, w.args)
		}
	}
}

// A key confined to collection alpha calls each graph walk on
// /v1/tools/<name> and is served alpha's chunks and the entity both
// collections know, by name, and nothing of beta's or of Zeus's.
// The tools that read the whole graph stay refused.
func TestAConfinedKeyWalksTheGraphOfItsCollectionOnlyOverHTTP(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "alpha", Secret: "alpha-secret", Clearance: authz.ReadOnly, Scope: authz.Scope{Collection: "alpha"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	for _, w := range twoCollectionGraph {
		wantStatus(t, srv, http.MethodPost, "/v1/tools/"+w.tool, "op-secret", w.args, http.StatusOK)
	}
	for _, walk := range graphWalks {
		status, payload := do(t, srv, http.MethodPost, "/v1/tools/"+walk.tool, "alpha-secret", walk.args)
		if status != http.StatusOK {
			t.Errorf("alpha %s = %d: %s", walk.tool, status, payload)
			continue
		}
		answer := string(payload)
		for _, marker := range outsideAlpha {
			if strings.Contains(answer, marker) {
				t.Errorf("alpha %s is served %s: %s", walk.tool, marker, answer)
			}
		}
		if !strings.Contains(answer, "Kestrel Vale") {
			t.Errorf("alpha %s is not served its own collection: %s", walk.tool, answer)
		}
	}
	if status, payload := do(t, srv, http.MethodPost, "/v1/tools/expand_graph", "op-secret", `{"node_ids":["entity:kestrel_vale"],"max_hops":2}`); status != http.StatusOK ||
		!strings.Contains(string(payload), "secretly funds") || !strings.Contains(string(payload), "BETADESC") || !strings.Contains(string(payload), `"source_document_ids"`) {
		t.Fatalf("operator expand_graph = %d, want beta's chunk, its description and the documents named: %s", status, payload)
	}
	for _, tool := range []string{"graph_list_all", "graph_statistics", "find_nodes"} {
		wantStatus(t, srv, http.MethodPost, "/v1/tools/"+tool, "alpha-secret", `{}`, http.StatusForbidden)
	}
}

// Over /v1/tools/<name>, each graph tool answers a seed the alpha key may
// not see as one that does not exist, for every argument that names a seed.
func TestAConfinedKeyIsAnsweredAHiddenSeedAsAMissingOneOverHTTP(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "alpha", Secret: "alpha-secret", Clearance: authz.ReadOnly, Scope: authz.Scope{Collection: "alpha"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	for _, w := range twoCollectionGraph {
		wantStatus(t, srv, http.MethodPost, "/v1/tools/"+w.tool, "op-secret", w.args, http.StatusOK)
	}
	everySeedAnswersAsAMissingOne(t, func(secret, tool, args string) (string, string) {
		status, payload := do(t, srv, http.MethodPost, "/v1/tools/"+tool, secret, args)
		if status != http.StatusOK {
			return fmt.Sprint(status), string(payload)
		}
		return fmt.Sprintf("%d %s", status, payload), ""
	})
}

// Over POST /v1/tools, a name answers the alpha key the same whether or not
// another collection wrote a node that matches it better than alpha's own.
func TestAConfinedKeyIsAnsweredANameAlikeBesideAHiddenTwinOverHTTP(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "alpha", Secret: "alpha-secret", Clearance: authz.ReadOnly, Scope: authz.Scope{Collection: "alpha"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	brain := func(twin bool) func(secret, tool, args string) (string, string) {
		srv := newPolicyServer(t, Options{Keys: ks})
		for _, w := range heronGraph(twin) {
			wantStatus(t, srv, http.MethodPost, "/v1/tools/"+w.tool, "op-secret", w.args, http.StatusOK)
		}
		return func(secret, tool, args string) (string, string) {
			status, payload := do(t, srv, http.MethodPost, "/v1/tools/"+tool, secret, args)
			if status != http.StatusOK {
				return fmt.Sprint(status), string(payload)
			}
			return fmt.Sprintf("%d %s", status, payload), ""
		}
	}
	everyNameAnswersAlikeBesideAHiddenTwin(t, brain(false), brain(true))
}
