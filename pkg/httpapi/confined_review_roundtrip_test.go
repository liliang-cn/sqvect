package httpapi

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// A document with a review decision carries the engine's _review_status,
// _reviewed_by and _reviewed_at, which the facade writes from the review
// fields only. So a confined key that sends back a document it wrote, these
// keys included, is not refused, and a copy that says "accepted" does not
// accept it.
func TestATypedKnowledgeWriteTakesTheReviewKeysTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "notes-secret",
		`{"knowledge_id":"k1","collection":"notes","content":"A note.","review_status":"proposed"}`, http.StatusOK)
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
	if stored["_review_status"] != "proposed" {
		t.Fatalf("the document's decision is %v, want proposed", stored["_review_status"])
	}
	stored["_review_status"], stored["_reviewed_by"], stored["_reviewed_at"] = "accepted", "notes", "2031-10-18T00:00:00Z"
	body, _ := json.Marshal(map[string]any{"knowledge_id": "k1", "collection": "notes", "content": "A note.", "metadata": stored})
	if status, payload := do(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret", string(body)); status != http.StatusOK {
		t.Fatalf("POST /v1/knowledge with the metadata it read: %d %s, want 200", status, payload)
	}
	after := read()
	if after["_review_status"] != "proposed" || after["_reviewed_by"] != nil || after["_reviewed_at"] != nil {
		t.Errorf("the decision is %v by %v at %v, want proposed and no reviewer", after["_review_status"], after["_reviewed_by"], after["_reviewed_at"])
	}
}

// The same for a memory: POST /v1/memory takes back the review keys of a
// memory a confined key wrote, and a copy that says "accepted" does not
// accept it. The marks of a document stay refused on a memory.
func TestATypedMemoryWriteTakesTheReviewKeysTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/tools/memory_save", "hermes-secret",
		`{"memory_id":"h1","user_id":"hermes","scope":"user","content":"Hermes takes the fast route.","review_status":"proposed"}`, http.StatusOK)
	read := func() (map[string]any, string) {
		t.Helper()
		status, payload := do(t, srv, http.MethodGet, "/v1/memory?memory_id=h1", "op-secret", "")
		if status != http.StatusOK {
			t.Fatalf("GET: %d %s", status, payload)
		}
		var got struct {
			Memory struct {
				Metadata     map[string]any `json:"metadata"`
				ReviewStatus string         `json:"review_status"`
			} `json:"memory"`
		}
		if err := json.Unmarshal(payload, &got); err != nil {
			t.Fatal(err)
		}
		return got.Memory.Metadata, got.Memory.ReviewStatus
	}
	stored, status := read()
	if status != "proposed" {
		t.Fatalf("the memory's decision is %q, want proposed (metadata %v)", status, stored)
	}
	if stored == nil {
		stored = map[string]any{}
	}
	stored["_review_status"], stored["_reviewed_by"], stored["_reviewed_at"] = "accepted", "hermes", "2031-10-18T00:00:00Z"
	body, _ := json.Marshal(map[string]any{"memory_id": "h1", "user_id": "hermes", "scope": "user", "content": "Hermes takes the fast route.", "metadata": stored})
	if code, payload := do(t, srv, http.MethodPost, "/v1/memory", "hermes-secret", string(body)); code != http.StatusOK {
		t.Fatalf("POST /v1/memory with the review keys it read: %d %s, want 200", code, payload)
	}
	if _, status := read(); status != "proposed" {
		t.Errorf("the decision is %q, want proposed", status)
	}
}

// The tool route takes back the review keys as the typed routes do: on the
// three document tools, and on memory_save, knowledge_memory_remember and
// memory_update for a memory.
// A copy that says "accepted" accepts nothing.
func TestAToolWriteTakesTheReviewKeysTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	forged := map[string]any{"team": "a", "_review_status": "accepted", "_reviewed_by": "notes", "_reviewed_at": "2031-10-18T00:00:00Z"}
	decision := func(path string, field string) any {
		t.Helper()
		status, payload := do(t, srv, http.MethodGet, path, "op-secret", "")
		if status != http.StatusOK {
			t.Fatalf("GET %s: %d %s", path, status, payload)
		}
		var got map[string]map[string]any
		if err := json.Unmarshal(payload, &got); err != nil {
			t.Fatal(err)
		}
		for _, row := range got {
			if md, _ := row["metadata"].(map[string]any); md != nil && md["_review_status"] != nil {
				return md["_review_status"]
			}
			return row[field]
		}
		return nil
	}
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "notes-secret",
		`{"knowledge_id":"k1","collection":"notes","content":"A note.","review_status":"proposed"}`, http.StatusOK)
	for _, tool := range []string{"ingest_document", "knowledge_save", "knowledge_update"} {
		args := map[string]any{"collection": "notes", "content": "A note.", "metadata": forged}
		if tool == "ingest_document" {
			args["document_id"] = "k1"
		} else {
			args["knowledge_id"] = "k1"
		}
		body, _ := json.Marshal(args)
		if status, payload := do(t, srv, http.MethodPost, "/v1/tools/"+tool, "notes-secret", string(body)); status != http.StatusOK {
			t.Fatalf("%s with the review keys: %d %s, want 200", tool, status, payload)
		}
		if got := decision("/v1/knowledge?knowledge_id=k1", "review_status"); got != "proposed" {
			t.Errorf("after %s the decision is %v, want proposed", tool, got)
		}
	}
	wantStatus(t, srv, http.MethodPost, "/v1/tools/memory_save", "hermes-secret",
		`{"memory_id":"h1","user_id":"hermes","scope":"user","content":"Hermes takes the fast route.","review_status":"proposed"}`, http.StatusOK)
	for _, tool := range []string{"memory_save", "knowledge_memory_remember", "memory_update"} {
		args := map[string]any{"memory_id": "h1", "content": "Hermes takes the fast route.", "metadata": forged}
		if tool != "memory_update" {
			args["user_id"], args["scope"] = "hermes", "user"
		}
		body, _ := json.Marshal(args)
		if status, payload := do(t, srv, http.MethodPost, "/v1/tools/"+tool, "hermes-secret", string(body)); status != http.StatusOK {
			t.Fatalf("%s with the review keys: %d %s, want 200", tool, status, payload)
		}
		if got := decision("/v1/memory?memory_id=h1", "review_status"); got != "proposed" {
			t.Errorf("after %s the decision is %v, want proposed", tool, got)
		}
	}
	// Another spelling is not rewritten, and keeps the refusal.
	wantStatus(t, srv, http.MethodPost, "/v1/tools/memory_update", "hermes-secret",
		`{"memory_id":"h1","metadata":{"_REVIEW_STATUS":"accepted"}}`, http.StatusForbidden)
}
