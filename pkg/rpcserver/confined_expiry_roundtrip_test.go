package rpcserver

import (
	"encoding/json"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// The gRPC twin of the REST test: SaveKnowledge and UpdateKnowledge take
// back the _expires_at of a document a confined key wrote, and the stored
// expiry stays what the field set.
func TestATypedKnowledgeWriteTakesTheExpiryKeyTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)
	tools := rpcv1.NewToolsServiceClient(conn)
	const later = "2031-10-18T00:00:00Z"
	args, _ := json.Marshal(map[string]any{"knowledge_id": "n1", "collection": "notes", "content": "A note.", "expires_at": later})
	if _, err := tools.CallTool(asKey("notes-secret"), &rpcv1.CallToolRequest{Name: "knowledge_save", ArgsJson: string(args)}); err != nil {
		t.Fatal(err)
	}
	read := func() map[string]string {
		t.Helper()
		got, err := knowledge.GetKnowledge(asKey("op-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: "n1"})
		if err != nil {
			t.Fatal(err)
		}
		return got.GetKnowledge().GetMetadata()
	}
	sent := read()
	if sent["_expires_at"] != later {
		t.Fatalf("the document's expiry is %q, want %s", sent["_expires_at"], later)
	}
	sent["_expires_at"] = "2099-01-01T00:00:00Z"
	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n1", Collection: "notes", Content: "A note.", Metadata: sent}); err != nil {
		t.Fatalf("SaveKnowledge with the metadata it read: %v", err)
	}
	if _, err := knowledge.UpdateKnowledge(asKey("notes-secret"), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "n1", Collection: proto.String("notes"), Metadata: sent}); err != nil {
		t.Fatalf("UpdateKnowledge with the metadata it read: %v", err)
	}
	if got := read()["_expires_at"]; got != later {
		t.Errorf("the expiry is %q, want the one the field set, %s", got, later)
	}
}
