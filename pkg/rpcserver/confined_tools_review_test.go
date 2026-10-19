package rpcserver

import (
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// The ways out of a confinement that review found on the first wiring, each
// closed by the allowlist and held here over gRPC. hermes is confined to its
// user, notes to its collection; the operator writes the rows they must not
// reach.
func TestTheWaysOutOfAConfinementFoundInReviewAreClosedOverGRPC(t *testing.T) {
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
	denied := func(what, secret, tool, args, want string) {
		t.Helper()
		_, err := call(secret, tool, args)
		if status.Code(err) != codes.PermissionDenied || !strings.Contains(status.Convert(err).Message(), want) {
			t.Errorf("%s: %v, want a denial saying %q", what, err, want)
		}
	}
	_, err = call("op-secret", "knowledge_save", `{"knowledge_id":"hr-salaries-2026","content":"Orpheus earns a secret salary.","collection":"secrets"}`)
	must("operator writes the secret", err)
	_, err = call("op-secret", "knowledge_save", `{"knowledge_id":"n1","content":"Orpheus plays the lyre on Fridays.","collection":"notes"}`)
	must("operator writes a note", err)
	_, err = call("op-secret", "memory_save", `{"memory_id":"zeus-1","content":"Zeus keeps the thunderbolt in the drawer.","user_id":"zeus","scope":"user"}`)
	must("operator writes zeus-1", err)
	_, err = call("op-secret", "memory_save", `{"memory_id":"hermes-1","content":"Hermes tunes the lyre on Fridays.","user_id":"hermes","scope":"user"}`)
	must("operator writes hermes-1", err)

	// 1. A promotion under consolidate wrote another user's memory into a
	// document: the tool is off the allowlist.
	denied("consolidate", "hermes-secret", "knowledge_memory_consolidate",
		`{"user_id":"hermes","promote_to_knowledge":true,"promotion":{"memory_ids":["zeus-1"],"knowledge_id":"hr-salaries-2026","collection":"secrets"}}`,
		"not a tool a confined key may call")

	// 2. Recall reads both stores: the store the key is not confined in is
	// switched off for it, whatever the call asked.
	got, err := call("hermes-secret", "knowledge_memory_recall", `{"query":"lyre Fridays","user_id":"hermes","collection":"notes"}`)
	must("hermes recalls", err)
	if strings.Contains(got, "plays the lyre") || !strings.Contains(got, "tunes the lyre") {
		t.Errorf("hermes's recall reached knowledge, or lost its memory: %s", got)
	}
	got, err = call("notes-secret", "knowledge_memory_recall", `{"query":"lyre Fridays","collection":"notes","user_id":"hermes","disable_memory":false}`)
	must("notes recalls", err)
	if strings.Contains(got, "tunes the lyre") || !strings.Contains(got, "plays the lyre") {
		t.Errorf("notes's recall reached memory, or lost its note: %s", got)
	}

	// 3. cortex_query's graph prefetch searched chunks of every collection.
	denied("cortex_query", "notes-secret", "cortex_query", `{"query":"Orpheus","collection":"notes","entity_names":["Orpheus"]}`, "not a tool a confined key may call")

	// 4. extract_conversation read a session by its raw bucket id.
	denied("extract_conversation by session", "notes-secret", "extract_conversation",
		`{"collection":"notes","session_id":"memory:user:zeus:default"}`, "takes no session_id from a confined key")
	denied("extract_conversation persisting", "notes-secret", "extract_conversation",
		`{"collection":"notes","text":"Orpheus met Eurydice.","persist":true}`, "takes no persist from a confined key")
	_, err = call("notes-secret", "extract_conversation", `{"collection":"notes","text":"Orpheus met Eurydice in Thrace."}`)
	must("notes extracts from text", err)

	// 5. supersedes retired another user's memory, and told whether an id
	// existed: now both a foreign and a missing id are "not found".
	for _, target := range []string{"zeus-1", "no-such-memory"} {
		_, err := call("hermes-secret", "memory_save", `{"memory_id":"h-2","content":"newer","user_id":"hermes","scope":"user","supersedes":["`+target+`"]}`)
		if status.Code(err) != codes.NotFound || !strings.Contains(status.Convert(err).Message(), "not found") {
			t.Errorf("supersedes %s: %v, want not found", target, err)
		}
	}
	got, err = call("op-secret", "memory_get", `{"memory_id":"zeus-1"}`)
	must("operator reads zeus-1", err)
	if strings.Contains(got, "superseded_by") {
		t.Errorf("zeus-1 was retired: %s", got)
	}
	_, err = call("hermes-secret", "memory_save", `{"memory_id":"h-2","content":"newer","user_id":"hermes","scope":"user","supersedes":["hermes-1"]}`)
	must("hermes retires its own memory", err)

	// Also held: declared entities, which write the shared graph (chunk ids
	// on them linked it to chunks of any collection), and a search pinned to
	// another collection's document.
	denied("declared entities", "notes-secret", "knowledge_save",
		`{"knowledge_id":"n2","collection":"notes","content":"x","entities":[{"name":"Orpheus","chunk_ids":["chunk:hr-salaries-2026:000"]}]}`,
		"entities")
	got, err = call("notes-secret", "knowledge_search", `{"query":"salary","collection":"notes","plan":{"filters":{"document_ids":["hr-salaries-2026"]}}}`)
	must("notes searches pinned to the secret", err)
	if strings.Contains(got, "secret salary") {
		t.Errorf("notes pinned another collection's document and read it: %s", got)
	}
}
