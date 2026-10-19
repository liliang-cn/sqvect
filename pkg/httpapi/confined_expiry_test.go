package httpapi

import (
	"net/http"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// A confined key may set expires_at on the three tools that take it, in its
// own collection, and may not call sweep_expired, which removes expired rows
// of every store and every collection.
func TestAConfinedKeySetsAnExpiryButDoesNotSweep(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	const later = `"expires_at":"2031-10-18T00:00:00Z"`
	wantStatus(t, srv, http.MethodPost, "/v1/tools/ingest_document", "notes-secret",
		`{"document_id":"n1","collection":"notes","content":"A note about the ledger.",`+later+`}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "notes-secret",
		`{"knowledge_id":"n2","collection":"notes","content":"Another note.",`+later+`}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_update", "notes-secret",
		`{"knowledge_id":"n2","collection":"notes",`+later+`}`, http.StatusOK)
	msg := wantStatus(t, srv, http.MethodPost, "/v1/tools/sweep_expired", "notes-secret", `{}`, http.StatusForbidden)
	if !strings.Contains(msg, "sweep_expired") {
		t.Errorf("sweep_expired refusal: %s", msg)
	}
}
