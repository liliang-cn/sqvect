package httpapi

import (
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// The REST twin: POST /v1/tools/knowledge_get with knowledge_ids, from a key
// confined to a collection, lists another collection's document under
// missing, in the order asked, as an id that names nothing, with 200. Both
// forms at once are refused.
func TestAConfinedKeyListsOnlyItsCollectionWithKnowledgeGetOverREST(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "op-secret", `{"knowledge_id":"hr-1","content":"The salaries of 2026.","collection":"secrets"}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "op-secret", `{"knowledge_id":"n1","content":"Orpheus plays the lyre.","collection":"notes"}`, http.StatusOK)

	status, payload := do(t, srv, http.MethodPost, "/v1/tools/knowledge_get", "notes-secret", `{"knowledge_ids":["absent","hr-1","n1"]}`)
	if status != http.StatusOK {
		t.Fatalf("notes lists: %d %s, want 200", status, payload)
	}
	var got struct {
		Knowledges []struct {
			ID string `json:"id"`
		} `json:"knowledges"`
		Missing []string `json:"missing"`
	}
	if err := json.Unmarshal(payload, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Knowledges) != 1 || got.Knowledges[0].ID != "n1" || !reflect.DeepEqual(got.Missing, []string{"absent", "hr-1"}) {
		t.Errorf("served %+v, missing %v; want [n1], [absent hr-1]", got.Knowledges, got.Missing)
	}
	if strings.Contains(string(payload), "salaries") {
		t.Errorf("the other collection's text was served: %s", payload)
	}
	status, payload = do(t, srv, http.MethodPost, "/v1/tools/knowledge_get", "notes-secret", `{"knowledge_id":"hr-1","knowledge_ids":["n1"]}`)
	if status != http.StatusForbidden || !strings.Contains(errorMessage(t, payload), "knowledge_ids") {
		t.Errorf("both forms: %d %s, want 403 naming knowledge_ids", status, payload)
	}
}
