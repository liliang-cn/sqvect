package rpcserver

import (
	"encoding/json"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// The gRPC twin of the REST test: SaveKnowledge and UpdateKnowledge take
// back the review keys of a document a confined key wrote, and a copy that
// says "accepted" does not accept it.
func TestATypedKnowledgeWriteTakesTheReviewKeysTheFacadeReplaces(t *testing.T) {
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
	args, _ := json.Marshal(map[string]any{"knowledge_id": "n1", "collection": "notes", "content": "A note.", "review_status": "proposed"})
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
	if sent["_review_status"] != "proposed" {
		t.Fatalf("the document's decision is %q, want proposed", sent["_review_status"])
	}
	sent["_review_status"], sent["_reviewed_by"], sent["_reviewed_at"] = "accepted", "notes", "2031-10-18T00:00:00Z"
	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n1", Collection: "notes", Content: "A note.", Metadata: sent}); err != nil {
		t.Fatalf("SaveKnowledge with the metadata it read: %v", err)
	}
	if _, err := knowledge.UpdateKnowledge(asKey("notes-secret"), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "n1", Collection: proto.String("notes"), Metadata: sent}); err != nil {
		t.Fatalf("UpdateKnowledge with the metadata it read: %v", err)
	}
	if after := read(); after["_review_status"] != "proposed" || after["_reviewed_by"] != "" || after["_reviewed_at"] != "" {
		t.Errorf("the decision is %q by %q at %q, want proposed and no reviewer", after["_review_status"], after["_reviewed_by"], after["_reviewed_at"])
	}
}

// The gRPC twin for a memory: SaveMemory and UpdateMemory take back the
// review keys of a memory a confined key wrote, and the decision stays.
func TestATypedMemoryWriteTakesTheReviewKeysTheFacadeReplaces(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: ownershipKeys(t)})
	mem := rpcv1.NewMemoryServiceClient(conn)
	tools := rpcv1.NewToolsServiceClient(conn)
	args, _ := json.Marshal(map[string]any{"memory_id": "h1", "user_id": "hermes", "scope": "user", "content": "Hermes takes the fast route.", "review_status": "proposed"})
	if _, err := tools.CallTool(asKey("hermes-secret"), &rpcv1.CallToolRequest{Name: "memory_save", ArgsJson: string(args)}); err != nil {
		t.Fatal(err)
	}
	read := func() *rpcv1.MemoryRecord {
		t.Helper()
		got, err := mem.GetMemory(asKey("op-secret"), &rpcv1.GetMemoryRequest{MemoryId: "h1"})
		if err != nil {
			t.Fatal(err)
		}
		return got.GetMemory()
	}
	sent := read().GetMetadata().AsMap()
	if sent["_review_status"] != "proposed" {
		t.Fatalf("the memory's decision is %v, want proposed", sent["_review_status"])
	}
	sent["_review_status"], sent["_reviewed_by"], sent["_reviewed_at"] = "accepted", "hermes", "2031-10-18T00:00:00Z"
	md, err := structpb.NewStruct(sent)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "h1", UserId: "hermes", Scope: "user", Content: "Hermes takes the fast route.", Metadata: md}); err != nil {
		t.Fatalf("SaveMemory with the metadata it read: %v", err)
	}
	if _, err := mem.UpdateMemory(asKey("hermes-secret"), &rpcv1.UpdateMemoryRequest{MemoryId: "h1", Metadata: md}); err != nil {
		t.Fatalf("UpdateMemory with the metadata it read: %v", err)
	}
	if got := read().GetMetadata().AsMap(); got["_review_status"] != "proposed" || got["_reviewed_by"] != nil {
		t.Errorf("the decision is %v by %v, want proposed and no reviewer", got["_review_status"], got["_reviewed_by"])
	}
	// CallTool takes them back on memory_save and memory_update too. The
	// signature keeps its refusal on the tool route, as on main.
	delete(sent, "written_by")
	delete(sent, "updated_by")
	for _, tool := range []string{"memory_save", "memory_update"} {
		a := map[string]any{"memory_id": "h1", "content": "Hermes takes the fast route.", "metadata": sent}
		if tool == "memory_save" {
			a["user_id"], a["scope"] = "hermes", "user"
		}
		args, _ := json.Marshal(a)
		if _, err := tools.CallTool(asKey("hermes-secret"), &rpcv1.CallToolRequest{Name: tool, ArgsJson: string(args)}); err != nil {
			t.Fatalf("CallTool %s with the metadata it read: %v", tool, err)
		}
		if got := read().GetMetadata().AsMap(); got["_review_status"] != "proposed" || got["_reviewed_by"] != nil {
			t.Errorf("after CallTool %s the decision is %v by %v, want proposed and no reviewer", tool, got["_review_status"], got["_reviewed_by"])
		}
	}
}
