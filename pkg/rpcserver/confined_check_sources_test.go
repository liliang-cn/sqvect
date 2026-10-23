package rpcserver

import (
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// A key confined to notes checks sentences against the notes it cites; a
// source of another collection comes back missing, by its id or by its
// chunk's, and its text never comes back. Another collection is refused.
func TestAConfinedKeyChecksAgainstItsOwnSourcesOverGRPC(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadOnly, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := rpcv1.NewToolsServiceClient(newPolicyConn(t, Options{Keys: ks}))
	call := func(secret, tool, args string) (string, error) {
		resp, err := tools.CallTool(asKey(secret), &rpcv1.CallToolRequest{Name: tool, ArgsJson: args})
		return resp.GetResultJson(), err
	}
	for _, args := range []string{
		`{"knowledge_id":"hr-salaries-2026","content":"Orpheus earns 9000 in Thrace.","collection":"secrets"}`,
		`{"knowledge_id":"n1","content":"Orpheus plays the lyre in Thrace.","collection":"notes"}`,
	} {
		if _, err := call("op-secret", "knowledge_save", args); err != nil {
			t.Fatal(err)
		}
	}
	got, err := call("notes-secret", "check_against_sources", `{"collection":"notes","sentences":[`+
		`{"text":"Orpheus earns 9000 in Thrace.","sources":["hr-salaries-2026","chunk:hr-salaries-2026:000"]},`+
		`{"text":"Orpheus plays in Thrace.","sources":["n1"]}]}`)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"results":[{"holds":false,"missing":["9000","Thrace"],"sources_missing":["hr-salaries-2026","chunk:hr-salaries-2026:000"]},` +
		`{"holds":true,"missing":[],"sources_missing":[]}],"counts":{"sentences":2,"hold":1}}`
	if got != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}
	_, err = call("notes-secret", "check_against_sources", `{"collection":"secrets","sentences":[{"text":"x","sources":["hr-salaries-2026"]}]}`)
	if status.Code(err) != codes.PermissionDenied || !strings.Contains(status.Convert(err).Message(), `asks for "secrets"`) {
		t.Errorf("another collection: %v, want a denial", err)
	}
}
