package rpcserver

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// A key confined to a collection reads a list of documents with
// knowledge_get: a document of another collection is listed under missing,
// at its place in the order asked, as an id that names nothing. It sends one
// form at a time. The operator's key reads every collection.
func TestAConfinedKeyListsOnlyItsCollectionWithKnowledgeGet(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := rpcv1.NewToolsServiceClient(newPolicyConn(t, Options{Keys: ks}))
	call := func(secret, args string) (served, missing []string, err error) {
		resp, err := tools.CallTool(asKey(secret), &rpcv1.CallToolRequest{Name: "knowledge_get", ArgsJson: args})
		if err != nil {
			return nil, nil, err
		}
		var got struct {
			Knowledges []struct {
				ID string `json:"id"`
			} `json:"knowledges"`
			Missing []string `json:"missing"`
		}
		if err := json.Unmarshal([]byte(resp.GetResultJson()), &got); err != nil {
			t.Fatal(err)
		}
		for _, k := range got.Knowledges {
			served = append(served, k.ID)
		}
		return served, got.Missing, nil
	}
	for _, args := range []string{
		`{"knowledge_id":"hr-1","content":"The salaries of 2026.","collection":"secrets"}`,
		`{"knowledge_id":"n1","content":"Orpheus plays the lyre.","collection":"notes"}`,
	} {
		if _, err := tools.CallTool(asKey("op-secret"), &rpcv1.CallToolRequest{Name: "knowledge_save", ArgsJson: args}); err != nil {
			t.Fatal(err)
		}
	}
	served, missing, err := call("notes-secret", `{"knowledge_ids":["hr-1","absent","n1"]}`)
	if err != nil || !reflect.DeepEqual(served, []string{"n1"}) || !reflect.DeepEqual(missing, []string{"hr-1", "absent"}) {
		t.Errorf("notes lists [hr-1 absent n1]: served %v, missing %v (%v); want [n1], [hr-1 absent]", served, missing, err)
	}
	served, missing, err = call("op-secret", `{"knowledge_ids":["hr-1","n1"]}`)
	if err != nil || !reflect.DeepEqual(served, []string{"hr-1", "n1"}) || len(missing) != 0 {
		t.Errorf("the operator lists [hr-1 n1]: served %v, missing %v (%v)", served, missing, err)
	}
	_, _, err = call("notes-secret", `{"knowledge_id":"hr-1","knowledge_ids":["n1"]}`)
	if status.Code(err) != codes.PermissionDenied || !strings.Contains(status.Convert(err).Message(), "knowledge_ids") {
		t.Errorf("both forms: %v, want PERMISSION_DENIED naming knowledge_ids", err)
	}
}
