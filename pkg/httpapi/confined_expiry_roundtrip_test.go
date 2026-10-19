package httpapi

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// A document with an expiry carries the engine's _expires_at, which the
// facade writes from the expires_at field only. So a confined key that
// sends back a document it wrote, _expires_at included, is not refused,
// and the stored expiry stays what the field set, not what was sent.
func TestATypedKnowledgeWriteTakesTheExpiryKeyTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	const later = "2031-10-18T00:00:00Z"
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "notes-secret",
		`{"knowledge_id":"k1","collection":"notes","content":"A note.","expires_at":"`+later+`"}`, http.StatusOK)
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
	if stored["_expires_at"] != later {
		t.Fatalf("the document's expiry is %v, want %s", stored["_expires_at"], later)
	}
	stored["_expires_at"] = "2099-01-01T00:00:00Z"
	body, _ := json.Marshal(map[string]any{"knowledge_id": "k1", "collection": "notes", "content": "A note.", "metadata": stored})
	if status, payload := do(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret", string(body)); status != http.StatusOK {
		t.Fatalf("POST /v1/knowledge with the metadata it read: %d %s, want 200", status, payload)
	}
	if got := read()["_expires_at"]; got != later {
		t.Errorf("the expiry is %v, want the one the field set, %s", got, later)
	}
}
