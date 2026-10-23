package httpapi

import (
	"net/http"
	"strconv"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// POST /v1/query runs db.Query for a key confined to a collection. Its graph
// lane must not walk what another collection wrote: from an entity only beta
// knows, two hops away over beta's relation, the alpha key used to get
// alpha's chunk, and so learn that beta relates the two.
func TestAConfinedQueryDoesNotWalkAnotherCollectionsRelationOverHTTP(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op", Clearance: authz.ReadWrite},
		{ID: "alpha", Secret: "alpha-secret", Clearance: authz.ReadOnly, Scope: authz.Scope{Collection: "alpha"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "op",
		`{"knowledge_id":"a1","collection":"alpha","content":"kestrel vale runs the alpha mill.","entities":[{"name":"Kestrel Vale","chunk_ids":["chunk:a1:000"]}]}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "op",
		`{"knowledge_id":"b1","collection":"beta","content":"kestrel vale secretly funds project nightjar with betasecret money.","entities":[{"name":"Kestrel Vale","chunk_ids":["chunk:b1:000"]},{"name":"Project Nightjar","chunk_ids":["chunk:b1:000"]}],"relations":[{"from":"Kestrel Vale","to":"Project Nightjar","type":"funds","chunk_ids":["chunk:b1:000"]}]}`, http.StatusOK)

	lane := func(seed string, hops int) string {
		t.Helper()
		body := `{"collection":"alpha","limit":10,"prefetch":[{"kind":"graph","entity_names":["` + seed + `"],"max_hops":` + strconv.Itoa(hops) + `}]}`
		status, payload := do(t, srv, http.MethodPost, "/v1/query", "alpha-secret", body)
		if status != http.StatusOK {
			t.Fatalf("POST /v1/query %s = %d: %s", body, status, payload)
		}
		return string(payload)
	}
	if got := lane("Project Nightjar", 2); strings.Contains(got, "chunk:a1:000") || strings.Contains(got, "betasecret") {
		t.Errorf("alpha key, seed only beta knows, 2 hops: %s", got)
	}
	// The lane still serves alpha's own graph.
	if got := lane("Kestrel Vale", 2); !strings.Contains(got, "chunk:a1:000") || strings.Contains(got, "chunk:b1:000") {
		t.Errorf("alpha key, seed Kestrel Vale, 2 hops: %s", got)
	}
}
