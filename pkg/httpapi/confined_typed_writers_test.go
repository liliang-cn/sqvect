package httpapi

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// The facade signs the two typed writes: written_by is replaced by the key
// that writes, and updated_by goes. So a confined key that sends them back,
// as a row read back carries them, is not refused for that, and the row
// keeps the facade's signature, not the one sent. Another spelling, which
// the facade does not replace, keeps the allowlist's refusal, and so does a
// memory written through the tool route.
func TestATypedWriteTakesTheSignatureTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	for _, c := range []struct{ secret, path, body, read, record, writer string }{
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"k1","collection":"notes","content":"Orpheus plays.","metadata":{"written_by":"zeus","updated_by":"zeus","color":"blue"}}`,
			"/v1/knowledge?knowledge_id=k1", "knowledge", "notes"},
		{"hermes-secret", "/v1/memory", `{"memory_id":"h1","user_id":"hermes","scope":"user","content":"Hermes tunes the lyre.","metadata":{"written_by":"zeus","updated_by":"zeus"}}`,
			"/v1/memory?memory_id=h1", "memory", "hermes"},
	} {
		if status, payload := do(t, srv, http.MethodPost, c.path, c.secret, c.body); status != http.StatusOK {
			t.Errorf("%s with the facade's signature keys: %d %s, want 200", c.path, status, payload)
			continue
		}
		status, payload := do(t, srv, http.MethodGet, c.read, "op-secret", "")
		if status != http.StatusOK {
			t.Fatalf("GET %s: %d %s", c.read, status, payload)
		}
		var got map[string]struct {
			Metadata map[string]any `json:"metadata"`
		}
		if err := json.Unmarshal(payload, &got); err != nil {
			t.Fatal(err)
		}
		meta := got[c.record].Metadata
		if meta["written_by"] != c.writer {
			t.Errorf("%s: written_by is %v, want the facade's %q", c.path, meta["written_by"], c.writer)
		}
		if _, ok := meta["updated_by"]; ok {
			t.Errorf("%s: updated_by is %v, want none on a save", c.path, meta["updated_by"])
		}
	}
	for _, c := range []struct{ path, body string }{
		{"/v1/tools/knowledge_save", `{"knowledge_id":"k2","collection":"notes","content":"x","metadata":{"Written_By":"zeus"}}`},
		{"/v1/knowledge", `{"knowledge_id":"k3","collection":"notes","content":"x","metadata":{"Written_By":"zeus"}}`},
	} {
		status, payload := do(t, srv, http.MethodPost, c.path, "notes-secret", c.body)
		if status != http.StatusForbidden || !strings.Contains(errorMessage(t, payload), "metadata.written_by") {
			t.Errorf("%s %s: %d %s, want 403 naming metadata.written_by", c.path, c.body, status, payload)
		}
	}
	status, payload := do(t, srv, http.MethodPost, "/v1/tools/memory_save", "hermes-secret",
		`{"memory_id":"h2","user_id":"hermes","scope":"user","content":"x","metadata":{"written_by":"zeus"}}`)
	if status != http.StatusForbidden || !strings.Contains(errorMessage(t, payload), "metadata.written_by") {
		t.Errorf("memory_save with a signature: %d %s, want 403 naming metadata.written_by", status, payload)
	}
}

// A key that reaches no row of a typed write's store (one confined to a
// collection on POST /v1/memory, one confined to a memory field on POST
// /v1/knowledge) is left to the row checks and the handler, which answer it
// as main does: the scope field the route does not have is the handler's
// 400, and a body without it is the row checks' 403.
func TestATypedWriteFromAKeyOfTheOtherStoreKeepsMainsAnswers(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
		{ID: "lab", Secret: "lab-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Namespace: "lab"}},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "mine", Secret: "mine-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{MemoryScope: "user"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	for _, c := range []struct {
		secret, path, body string
		status             int
		want               string
	}{
		{"notes-secret", "/v1/memory", `{"memory_id":"m1","collection":"notes","content":"x"}`, http.StatusBadRequest, `unknown field "collection"`},
		{"notes-secret", "/v1/memory", `{"memory_id":"m1","content":"x"}`, http.StatusForbidden, "leaves collection unset"},
		{"lab-secret", "/v1/knowledge", `{"knowledge_id":"l1","namespace":"lab","content":"x"}`, http.StatusBadRequest, `unknown field "namespace"`},
		{"lab-secret", "/v1/knowledge", `{"knowledge_id":"l1","content":"x"}`, http.StatusForbidden, "leaves namespace unset"},
		{"hermes-secret", "/v1/knowledge", `{"knowledge_id":"h1","user_id":"hermes","content":"x"}`, http.StatusBadRequest, `unknown field "user_id"`},
		{"mine-secret", "/v1/knowledge", `{"knowledge_id":"u1","scope":"user","content":"x"}`, http.StatusBadRequest, `unknown field "scope"`},
		// A body the handler's decoder refuses is the handler's 400 for a
		// key of either store, as on main, not an allowlist refusal.
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n9","collection":"notes","content":"x","topK":1}`, http.StatusBadRequest, `unknown field "topK"`},
		{"hermes-secret", "/v1/memory", `{"memory_id":"h9","user_id":"hermes","scope":"user","content":"x","topK":1}`, http.StatusBadRequest, `unknown field "topK"`},
	} {
		status, payload := do(t, srv, http.MethodPost, c.path, c.secret, c.body)
		if status != c.status || !strings.Contains(errorMessage(t, payload), c.want) {
			t.Errorf("%s %s: %d %s, want %d naming %s", c.path, c.body, status, payload, c.status, c.want)
		}
	}
}

// Every 404 a tool returns to a confined key is the bare "not found", the
// toolbox's "unknown tool" answer included: its text can carry an id the
// caller sent, so a missing id would otherwise read differently from a
// foreign one. The policy refuses a tool name the toolbox does not define
// before the tool runs. The operator keeps the tool's wording.
func TestAToolNotFoundThatNamesAnUnknownToolIsWithheldLikeAnyOther(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	body := `{"memory_id":"h1","user_id":"hermes","scope":"user","content":"Hermes runs.","supersedes":["unknown tool q"]}`
	status, payload := do(t, srv, http.MethodPost, "/v1/tools/memory_save", "hermes-secret", body)
	if status != http.StatusNotFound || errorMessage(t, payload) != "not found" {
		t.Errorf("a missing supersedes id that reads \"unknown tool\": %d %s, want 404 \"not found\"", status, payload)
	}
	status, payload = do(t, srv, http.MethodPost, "/v1/tools/memory_save", "op-secret", body)
	if status != http.StatusNotFound || !strings.Contains(errorMessage(t, payload), "unknown tool q") {
		t.Errorf("the operator's answer: %d %s, want 404 with the tool's wording", status, payload)
	}
}
