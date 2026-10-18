package httpapi

import (
	"net/http"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// A confined key may ask ingest_document for dedupe, and the lookup stays in
// the key's collection: the same content stored in another collection is not
// a duplicate and is never named, so duplicate_of cannot tell a confined key
// what another collection holds. A copy within its own collection is named.
func TestAConfinedKeyDedupesWithinItsCollectionOnly(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	const content = "The vault code is kept in the second drawer."
	wantStatus(t, srv, http.MethodPost, "/v1/tools/ingest_document", "op-secret",
		`{"document_id":"sec","collection":"secrets","content":"`+content+`"}`, http.StatusOK)

	status, payload := do(t, srv, http.MethodPost, "/v1/tools/ingest_document", "notes-secret",
		`{"document_id":"n1","collection":"notes","content":"`+content+`","dedupe":true}`)
	if status != http.StatusOK || strings.Contains(string(payload), "duplicate_of") {
		t.Errorf("dedupe over a copy in another collection: %d %s, want 200 and no duplicate_of", status, payload)
	}
	status, payload = do(t, srv, http.MethodPost, "/v1/tools/ingest_document", "notes-secret",
		`{"document_id":"n2","collection":"notes","content":"`+content+`","dedupe":true}`)
	if status != http.StatusOK || !strings.Contains(string(payload), `"duplicate_of":"n1"`) {
		t.Errorf("dedupe over a copy in its own collection: %d %s, want 200 naming n1", status, payload)
	}
}
