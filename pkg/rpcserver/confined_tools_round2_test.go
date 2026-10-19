package rpcserver

import (
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// The second round of review on the allowlist, held over gRPC: a confined
// key cannot tell a row outside its scope from a row that does not exist (one
// wording, NOT_FOUND, and a list answers through missing in the order asked),
// declares no entity or relation, takes no engine key in metadata, runs no
// path retrieval and sends one form of memory_get. The operator's
// unconfined key keeps the tool's own wording.
func TestTheRoundTwoRulesOfTheAllowlistHoldOverGRPC(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	tools := rpcv1.NewToolsServiceClient(conn)
	call := func(secret, tool, args string) (string, error) {
		resp, err := tools.CallTool(asKey(secret), &rpcv1.CallToolRequest{Name: tool, ArgsJson: args})
		return resp.GetResultJson(), err
	}
	must := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	for _, w := range []struct{ tool, args string }{
		{"knowledge_save", `{"knowledge_id":"hr-salaries-2026","content":"Orpheus earns a secret salary.","collection":"secrets"}`},
		{"knowledge_save", `{"knowledge_id":"n1","content":"Orpheus plays the lyre on Fridays.","collection":"notes"}`},
		{"memory_save", `{"memory_id":"zeus-1","content":"Zeus keeps the thunderbolt in the drawer.","user_id":"zeus","scope":"user"}`},
		{"memory_save", `{"memory_id":"hermes-1","content":"Hermes tunes the lyre on Fridays.","user_id":"hermes","scope":"user"}`},
	} {
		_, err := call("op-secret", w.tool, w.args)
		must("operator writes "+w.args, err)
	}

	// 1. One wording. A foreign id and a missing id, on every tool that
	// names a row, come back NOT_FOUND with the message "not found" and
	// nothing else: no kind, no id, no hint of which it was.
	for _, c := range []struct{ what, secret, tool, args string }{
		{"memory_get foreign", "hermes-secret", "memory_get", `{"memory_id":"zeus-1"}`},
		{"memory_get missing", "hermes-secret", "memory_get", `{"memory_id":"no-such-memory"}`},
		{"memory_delete foreign", "hermes-secret", "memory_delete", `{"memory_id":"zeus-1"}`},
		{"memory_delete missing", "hermes-secret", "memory_delete", `{"memory_id":"no-such-memory"}`},
		{"supersedes foreign", "hermes-secret", "memory_save", `{"memory_id":"h-2","content":"newer","user_id":"hermes","scope":"user","supersedes":["zeus-1"]}`},
		{"supersedes missing", "hermes-secret", "memory_save", `{"memory_id":"h-2","content":"newer","user_id":"hermes","scope":"user","supersedes":["no-such-memory"]}`},
		{"knowledge_get foreign", "notes-secret", "knowledge_get", `{"knowledge_id":"hr-salaries-2026"}`},
		{"knowledge_get missing", "notes-secret", "knowledge_get", `{"knowledge_id":"no-such-doc"}`},
		{"knowledge_delete foreign", "notes-secret", "knowledge_delete", `{"knowledge_id":"hr-salaries-2026"}`},
		{"knowledge_delete missing", "notes-secret", "knowledge_delete", `{"knowledge_id":"no-such-doc"}`},
		{"knowledge_update foreign", "notes-secret", "knowledge_update", `{"knowledge_id":"hr-salaries-2026","collection":"notes","content":"overwritten"}`},
		{"knowledge_update missing", "notes-secret", "knowledge_update", `{"knowledge_id":"no-such-doc","collection":"notes","content":"overwritten"}`},
		{"memory_update foreign", "hermes-secret", "memory_update", `{"memory_id":"zeus-1","content":"overwritten"}`},
		{"memory_update missing", "hermes-secret", "memory_update", `{"memory_id":"no-such-memory","content":"overwritten"}`},
		// The four tools that create the row they name: a foreign id is
		// the same bare "not found" (a missing id is created, which the
		// body names).
		{"ingest_document foreign", "notes-secret", "ingest_document", `{"document_id":"hr-salaries-2026","collection":"notes","content":"overwritten"}`},
		{"knowledge_save foreign", "notes-secret", "knowledge_save", `{"knowledge_id":"hr-salaries-2026","collection":"notes","content":"overwritten"}`},
		{"memory_save foreign", "hermes-secret", "memory_save", `{"memory_id":"zeus-1","content":"overwritten","user_id":"hermes","scope":"user"}`},
		{"knowledge_memory_remember foreign", "hermes-secret", "knowledge_memory_remember", `{"memory_id":"zeus-1","content":"overwritten","user_id":"hermes","scope":"user"}`},
	} {
		_, err := call(c.secret, c.tool, c.args)
		if status.Code(err) != codes.NotFound || status.Convert(err).Message() != authz.ErrWithheld.Error() {
			t.Errorf("%s: %v, want NOT_FOUND %q", c.what, err, authz.ErrWithheld.Error())
		}
	}
	// Neither refused memory_save wrote its new row: a missing supersede
	// target is checked before the write, as a foreign one is, so the new
	// id does not tell the two apart.
	if _, err := call("op-secret", "memory_get", `{"memory_id":"h-2"}`); status.Code(err) != codes.NotFound {
		t.Errorf("a refused memory_save left its new row: %v", err)
	}
	// Only NOT_FOUND is reworded: any other refusal of the tool keeps its
	// code and its words for a confined key.
	empty := `{"memory_id":"h-4","content":"","user_id":"hermes","scope":"user"}`
	_, err = call("hermes-secret", "memory_save", empty)
	_, own := call("op-secret", "memory_save", empty)
	if code := status.Code(err); code == codes.NotFound || code == codes.OK || code != status.Code(own) || status.Convert(err).Message() != status.Convert(own).Message() {
		t.Errorf("an empty memory under a confined key: %v, want the tool's own refusal: %v", err, own)
	}
	// A session bucket of another user is NOT_FOUND from the tool, worded
	// "memory bucket not found": flattened like every other NOT_FOUND.
	must("operator opens zeus's session", func() error {
		_, err := call("op-secret", "memory_save", `{"memory_id":"zeus-s1","content":"Zeus naps.","user_id":"zeus","scope":"session","session_id":"zeus-session"}`)
		return err
	}())
	_, err = call("hermes-secret", "memory_save", `{"memory_id":"h-5","content":"x","user_id":"hermes","scope":"session","session_id":"zeus-session"}`)
	if status.Code(err) != codes.NotFound || status.Convert(err).Message() != authz.ErrWithheld.Error() {
		t.Errorf("a save into another user's session: %v, want NOT_FOUND %q", err, authz.ErrWithheld.Error())
	}
	// The operator is not confined: a missing row keeps the tool's own
	// wording, which says more than "not found".
	_, err = call("op-secret", "memory_get", `{"memory_id":"no-such-memory"}`)
	if msg := status.Convert(err).Message(); status.Code(err) != codes.NotFound || !strings.Contains(msg, "not found") || msg == authz.ErrWithheld.Error() {
		t.Errorf("operator reads a missing memory: %v, want the tool's own not-found wording", err)
	}

	// A list of memory ids: the foreign id goes to missing, in the order
	// asked, beside the id that does not exist.
	for _, ids := range []string{`["zeus-1","no-such-memory","hermes-1"]`, `["no-such-memory","zeus-1"]`} {
		got, err := call("hermes-secret", "memory_get", `{"memory_ids":`+ids+`}`)
		must("hermes reads a list", err)
		var resp struct {
			Memories []struct {
				ID string `json:"id"`
			} `json:"memories"`
			Missing []string `json:"missing"`
		}
		if err := json.Unmarshal([]byte(got), &resp); err != nil {
			t.Fatal(err)
		}
		want := strings.Trim(strings.ReplaceAll(strings.ReplaceAll(ids, `"hermes-1"`, ""), `"`, ""), "[],")
		if strings.Join(resp.Missing, ",") != strings.TrimSuffix(want, ",") {
			t.Errorf("memory_ids %s: missing %v, want %s", ids, resp.Missing, want)
		}
		served := 0
		for _, m := range resp.Memories {
			if m.ID != "hermes-1" {
				t.Errorf("memory_ids %s: served %s", ids, m.ID)
			}
			served++
		}
		if want := strings.Count(ids, "hermes-1"); served != want {
			t.Errorf("memory_ids %s: served %d of hermes's own memories, want %d", ids, served, want)
		}
	}
	// A search pinned to another collection's document finds nothing, as a
	// pin to a document that does not exist finds nothing: no error.
	for _, pin := range []string{"hr-salaries-2026", "no-such-doc"} {
		got, err := call("notes-secret", "knowledge_search", `{"query":"salary","collection":"notes","plan":{"filters":{"document_ids":["`+pin+`"]}}}`)
		must("notes searches pinned to "+pin, err)
		if strings.Contains(got, "secret salary") || strings.Contains(got, `"id":"hr-salaries-2026"`) {
			t.Errorf("pinned to %s, notes read the secret: %s", pin, got)
		}
	}

	// 2. Nothing that writes or walks the shared graph, no engine key in
	// metadata, one form of memory_get.
	for _, c := range []struct{ secret, tool, args, want string }{
		{"notes-secret", "knowledge_save", `{"knowledge_id":"n2","collection":"notes","content":"x","relations":[{"from":"a","to":"b"}]}`, "relations"},
		{"notes-secret", "knowledge_update", `{"knowledge_id":"n1","collection":"notes","entities":[{"name":"Orpheus"}]}`, "entities"},
		{"hermes-secret", "memory_save", `{"memory_id":"h-3","user_id":"hermes","content":"x","entities":[{"name":"Orpheus"}]}`, "entities"},
		{"notes-secret", "ingest_document", `{"document_id":"n2","collection":"notes","content":"x","metadata":{"document_id":"hr-salaries-2026"}}`, "metadata.document_id"},
		{"hermes-secret", "memory_save", `{"memory_id":"h-3","user_id":"hermes","content":"x","metadata":{"_table":"x"}}`, "metadata._table"},
		{"notes-secret", "search_graphrag_lexical", `{"collection":"notes","entity_names":["Orpheus"],"return_paths":true}`, "return_paths"},
		{"hermes-secret", "memory_get", `{"memory_id":"zeus-1","memory_ids":["hermes-1"]}`, "memory_id or memory_ids"},
	} {
		_, err := call(c.secret, c.tool, c.args)
		if status.Code(err) != codes.PermissionDenied || !strings.Contains(status.Convert(err).Message(), c.want) {
			t.Errorf("%s: %v, want a denial naming %s", c.args, err, c.want)
		}
	}
	_, err = call("notes-secret", "knowledge_save", `{"knowledge_id":"n2","collection":"notes","content":"Orpheus plays the lyre.","metadata":{"color":"blue"}}`)
	must("notes writes its own metadata key", err)

	// 3. Arguments that are null are refused, on a tool with a gate to
	// force too, and the server keeps serving: forcing a gate into them
	// once panicked inside the interceptor, which grpc-go does not recover.
	for _, name := range []string{"knowledge_memory_recall", "knowledge_search"} {
		if _, err := call("notes-secret", name, "null"); status.Code(err) != codes.PermissionDenied {
			t.Errorf("%s with null arguments: %v, want PERMISSION_DENIED", name, err)
		}
	}
	_, err = call("notes-secret", "knowledge_get", `{"knowledge_id":"n1"}`)
	must("the server still serves after null arguments", err)
}
