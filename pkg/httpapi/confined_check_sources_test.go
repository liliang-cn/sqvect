package httpapi

import (
	"net/http"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// TestAConfinedKeyChecksAgainstItsOwnSourcesOverGRPC, over REST.
func TestAConfinedKeyChecksAgainstItsOwnSourcesOverREST(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadOnly, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	for _, body := range []string{
		`{"knowledge_id":"hr-salaries-2026","content":"Orpheus earns 9000 in Thrace.","collection":"secrets"}`,
		`{"knowledge_id":"n1","content":"Orpheus plays the lyre in Thrace.","collection":"notes"}`,
	} {
		if status, payload := do(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "op-secret", body); status != http.StatusOK {
			t.Fatalf("knowledge_save = %d: %s", status, payload)
		}
	}
	status, payload := do(t, srv, http.MethodPost, "/v1/tools/check_against_sources", "notes-secret", `{"collection":"notes","sentences":[`+
		`{"text":"Orpheus earns 9000 in Thrace.","sources":["hr-salaries-2026","chunk:hr-salaries-2026:000"]},`+
		`{"text":"Orpheus plays in Thrace.","sources":["n1"]}]}`)
	if status != http.StatusOK {
		t.Fatalf("check = %d: %s", status, payload)
	}
	got := string(payload)
	for _, want := range []string{
		`{"holds":false,"missing":["9000","Thrace"],"sources_missing":["hr-salaries-2026","chunk:hr-salaries-2026:000"]}`,
		`{"holds":true,"missing":[],"sources_missing":[]}`, `"counts":{"sentences":2,"hold":1}`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("got %s, want %s", got, want)
		}
	}
	if msg := wantStatus(t, srv, http.MethodPost, "/v1/tools/check_against_sources", "notes-secret",
		`{"collection":"secrets","sentences":[{"text":"x","sources":["hr-salaries-2026"]}]}`, http.StatusForbidden); !strings.Contains(msg, `asks for "secrets"`) {
		t.Errorf("another collection: %q, want a denial", msg)
	}
}
