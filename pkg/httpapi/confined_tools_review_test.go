package httpapi

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// The ways out of a confinement that review found on the first wiring, each
// closed by the allowlist and held here over REST, as TestTheWaysOutOf…
// holds them over gRPC.
func TestTheWaysOutOfAConfinementFoundInReviewAreClosedOverREST(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	tool := func(secret, name, body string, want int) string {
		t.Helper()
		if want < 400 {
			status, payload := do(t, srv, http.MethodPost, "/v1/tools/"+name, secret, body)
			if status != want {
				t.Fatalf("%s as %q = %d, want %d: %s", name, secret, status, want, payload)
			}
			return string(payload)
		}
		return wantStatus(t, srv, http.MethodPost, "/v1/tools/"+name, secret, body, want)
	}
	denied := func(what, secret, name, body, want string) {
		t.Helper()
		if msg := tool(secret, name, body, http.StatusForbidden); !strings.Contains(msg, want) {
			t.Errorf("%s: %q, want a denial saying %q", what, msg, want)
		}
	}
	tool("op-secret", "knowledge_save", `{"knowledge_id":"hr-salaries-2026","content":"Orpheus earns a secret salary.","collection":"secrets"}`, http.StatusOK)
	tool("op-secret", "knowledge_save", `{"knowledge_id":"n1","content":"Orpheus plays the lyre on Fridays.","collection":"notes"}`, http.StatusOK)
	tool("op-secret", "memory_save", `{"memory_id":"zeus-1","content":"Zeus keeps the thunderbolt in the drawer.","user_id":"zeus","scope":"user"}`, http.StatusOK)
	tool("op-secret", "memory_save", `{"memory_id":"hermes-1","content":"Hermes tunes the lyre on Fridays.","user_id":"hermes","scope":"user"}`, http.StatusOK)

	// 1. consolidate's promotion.
	denied("consolidate", "hermes-secret", "knowledge_memory_consolidate",
		`{"user_id":"hermes","promote_to_knowledge":true,"promotion":{"memory_ids":["zeus-1"],"knowledge_id":"hr-salaries-2026","collection":"secrets"}}`,
		"not a tool a confined key may call")

	// 2. Recall across stores: the other store is switched off for the key.
	got := tool("hermes-secret", "knowledge_memory_recall", `{"query":"lyre Fridays","user_id":"hermes","collection":"notes"}`, http.StatusOK)
	if strings.Contains(got, "plays the lyre") || !strings.Contains(got, "tunes the lyre") {
		t.Errorf("hermes's recall reached knowledge, or lost its memory: %s", got)
	}
	got = tool("notes-secret", "knowledge_memory_recall", `{"query":"lyre Fridays","collection":"notes","user_id":"hermes","DISABLE_MEMORY":false}`, http.StatusOK)
	if strings.Contains(got, "tunes the lyre") || !strings.Contains(got, "plays the lyre") {
		t.Errorf("notes's recall reached memory, or lost its note: %s", got)
	}

	// 3. cortex_query's graph prefetch.
	denied("cortex_query", "notes-secret", "cortex_query", `{"query":"Orpheus","collection":"notes","entity_names":["Orpheus"]}`, "not a tool a confined key may call")

	// 4. extract_conversation by raw bucket id, or persisting.
	denied("extract_conversation by session", "notes-secret", "extract_conversation",
		`{"collection":"notes","session_id":"memory:user:zeus:default"}`, "takes no session_id from a confined key")
	denied("extract_conversation persisting", "notes-secret", "extract_conversation",
		`{"collection":"notes","text":"Orpheus met Eurydice.","persist":true}`, "takes no persist from a confined key")
	tool("notes-secret", "extract_conversation", `{"collection":"notes","text":"Orpheus met Eurydice in Thrace."}`, http.StatusOK)

	// 5. supersedes: a foreign id and a missing id are both not found.
	for _, target := range []string{"zeus-1", "no-such-memory"} {
		if msg := tool("hermes-secret", "memory_save", `{"memory_id":"h-2","content":"newer","user_id":"hermes","scope":"user","supersedes":["`+target+`"]}`, http.StatusNotFound); !strings.Contains(msg, "not found") {
			t.Errorf("supersedes %s: %q", target, msg)
		}
	}
	if got := tool("op-secret", "memory_get", `{"memory_id":"zeus-1"}`, http.StatusOK); strings.Contains(got, "superseded_by") {
		t.Errorf("zeus-1 was retired: %s", got)
	}
	tool("hermes-secret", "memory_save", `{"memory_id":"h-2","content":"newer","user_id":"hermes","scope":"user","supersedes":["hermes-1"]}`, http.StatusOK)

	// Also held: declared entities, which write the shared graph (chunk ids
	// on them linked it to chunks of any collection), and a search pinned to
	// another collection's document.
	denied("declared entities", "notes-secret", "knowledge_save",
		`{"knowledge_id":"n2","collection":"notes","content":"x","entities":[{"name":"Orpheus","chunk_ids":["chunk:hr-salaries-2026:000"]}]}`,
		"entities")
	if got := tool("notes-secret", "knowledge_search", `{"query":"salary","collection":"notes","plan":{"filters":{"document_ids":["hr-salaries-2026"]}}}`, http.StatusOK); strings.Contains(got, "secret salary") {
		t.Errorf("notes pinned another collection's document and read it: %s", got)
	}
}

// With no database to read rows from, the tool route refuses a confined key
// rather than skipping the row guard.
func TestTheToolRouteRefusesAConfinedKeyWhenItHasNoDatabaseToGuardWith(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	served := false
	handler := authenticate(ks, authorize(route{method: http.MethodPost, path: toolRoutePath}, func(http.ResponseWriter, *http.Request) {
		served = true
	}))
	req := httptest.NewRequest(http.MethodPost, "/v1/tools/knowledge_search", strings.NewReader(`{"query":"q","collection":"notes"}`))
	req.Header.Set("Authorization", "Bearer notes-secret")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if served || rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "no database to guard rows against") {
		t.Fatalf("the tool route without a database answered %d (served: %v): %s", rec.Code, served, rec.Body.String())
	}
}
