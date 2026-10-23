package authz

import (
	"errors"
	"strings"
	"testing"
)

// knowledge_get reads knowledge_ids and ignores knowledge_id when both are
// set, while the row guard reads knowledge_id: a foreign knowledge_id beside a
// list of the caller's would answer "not found", a missing one would serve
// the list, and the pair would tell them apart. A confined key sends one form
// or the other. The list is data: the facade lists a document outside the
// caller's collection under missing.
func TestKnowledgeGetTakesOneFormFromAConfinedKey(t *testing.T) {
	notes := Key{ID: "notes", Scope: Scope{Collection: "notes"}}
	err := notes.AuthorizeToolArgs("knowledge_get", []byte(`{"knowledge_id":"hr-1","Knowledge_IDs":["n1"]}`))
	if !errors.Is(err, ErrDenied) || !strings.Contains(err.Error(), "knowledge_id") || !strings.Contains(err.Error(), "knowledge_ids") {
		t.Fatalf("both forms of knowledge_get: %v, want a denial naming both", err)
	}
	for _, args := range []string{`{"knowledge_id":"n1"}`, `{"knowledge_ids":["n1","n2"]}`} {
		if err := notes.AuthorizeToolArgs("knowledge_get", []byte(args)); err != nil {
			t.Errorf("%s: one form was refused: %v", args, err)
		}
	}
}
