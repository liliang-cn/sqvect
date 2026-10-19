package httpapi

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// The REST twin of the gRPC test: POST /v1/knowledge takes back the marks of
// a document a confined key wrote, and the document keeps the engine's. A
// memory and another spelling keep the allowlist's refusal.
func TestATypedKnowledgeWriteTakesTheMarksTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret", `{"knowledge_id":"k1","collection":"notes","content":"Orpheus plays."}`, http.StatusOK)
	read := func() map[string]any {
		t.Helper()
		status, payload := do(t, srv, http.MethodGet, "/v1/knowledge?knowledge_id=k1", "op-secret", "")
		if status != http.StatusOK {
			t.Fatalf("GET: %d %s", status, payload)
		}
		var got struct {
			Knowledge struct {
				Metadata map[string]any `json:"metadata"`
			} `json:"knowledge"`
		}
		if err := json.Unmarshal(payload, &got); err != nil {
			t.Fatal(err)
		}
		return got.Knowledge.Metadata
	}
	stored := read()
	engine := map[string]any{}
	for _, k := range cortexdb.EngineDocumentKeys() {
		if k == "_expires_at" || strings.HasPrefix(k, "_review") {
			continue // a document with no expiry or decision has none: confined_expiry_roundtrip_test.go, confined_review_roundtrip_test.go
		}
		if stored[k] == nil {
			t.Fatalf("the document carries no %s: %v", k, stored)
		}
		engine[k], stored[k] = stored[k], "forged"
	}
	body, _ := json.Marshal(map[string]any{"knowledge_id": "k1", "collection": "notes", "content": "Orpheus plays.", "metadata": stored})
	if status, payload := do(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret", string(body)); status != http.StatusOK {
		t.Fatalf("POST /v1/knowledge with the metadata it read: %d %s, want 200", status, payload)
	}
	after := read()
	for k, v := range engine {
		if after[k] != v {
			t.Errorf("%s is %v, want the engine's %v", k, after[k], v)
		}
	}
	for _, c := range []struct{ secret, path, body string }{
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"k3","collection":"notes","content":"x","metadata":{"_Content_SHA256":"forged"}}`},
		{"hermes-secret", "/v1/memory", `{"memory_id":"h1","user_id":"hermes","scope":"user","content":"x","metadata":{"_content_sha256":"forged"}}`},
	} {
		status, payload := do(t, srv, http.MethodPost, c.path, c.secret, c.body)
		if status != http.StatusForbidden || !strings.Contains(errorMessage(t, payload), "metadata._content_sha256") {
			t.Errorf("%s %s: %d %s, want 403 naming metadata._content_sha256", c.path, c.body, status, payload)
		}
	}
}

// The tool route takes back the marks too. A document written by
// ingest_document carries them and no signature, and main lets a confined
// key send its metadata back to ingest_document, knowledge_save and
// knowledge_update. The facade writes its own marks over the sent ones on
// all three, so they are not refused, and the forged values are not kept.
func TestAToolWriteTakesTheMarksTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/tools/ingest_document", "notes-secret",
		`{"document_id":"d1","collection":"notes","content":"Orpheus plays.","metadata":{"team":"a"}}`, http.StatusOK)
	read := func() map[string]any {
		t.Helper()
		status, payload := do(t, srv, http.MethodPost, "/v1/tools/knowledge_get", "notes-secret", `{"knowledge_id":"d1"}`)
		if status != http.StatusOK {
			t.Fatalf("knowledge_get: %d %s", status, payload)
		}
		var got struct {
			Knowledge struct {
				Metadata map[string]any `json:"metadata"`
			} `json:"knowledge"`
		}
		if err := json.Unmarshal(payload, &got); err != nil {
			t.Fatal(err)
		}
		return got.Knowledge.Metadata
	}
	stored := read()
	engine := map[string]any{}
	for _, k := range cortexdb.EngineDocumentKeys() {
		if k == "_expires_at" || strings.HasPrefix(k, "_review") {
			continue // a document with no expiry or decision has none: confined_expiry_roundtrip_test.go, confined_review_roundtrip_test.go
		}
		if stored[k] == nil {
			t.Fatalf("the document carries no %s: %v", k, stored)
		}
		engine[k], stored[k] = stored[k], "forged"
	}
	for _, c := range []struct {
		tool, id string
		content  bool
	}{{"ingest_document", "document_id", true}, {"knowledge_save", "knowledge_id", true}, {"knowledge_update", "knowledge_id", true}, {"knowledge_update", "knowledge_id", false}} {
		args := map[string]any{c.id: "d1", "collection": "notes", "metadata": stored}
		if c.content {
			args["content"] = "Orpheus plays."
		}
		body, _ := json.Marshal(args)
		if status, payload := do(t, srv, http.MethodPost, "/v1/tools/"+c.tool, "notes-secret", string(body)); status != http.StatusOK {
			t.Fatalf("%s with the metadata it read: %d %s, want 200", c.tool, status, payload)
		}
		after := read()
		for k, v := range engine {
			if after[k] != v {
				t.Errorf("after %s, %s is %v, want the engine's %v", c.tool, k, after[k], v)
			}
		}
	}
	status, payload := do(t, srv, http.MethodPost, "/v1/tools/ingest_document", "notes-secret",
		`{"document_id":"d2","collection":"notes","content":"x","metadata":{"_Content_SHA256":"forged"}}`)
	if status != http.StatusForbidden {
		t.Errorf("another spelling of a mark: %d %s, want 403", status, payload)
	}
}
