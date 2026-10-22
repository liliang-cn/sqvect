package rpcserver

import (
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// An id of blanks names a row like any other: the facade stores it as given,
// so the upsert guards read it rather than taking it for no id, and a
// confined key cannot take another scope's row by naming it, through
// SaveKnowledge, UpdateKnowledge or SaveMemory.
func TestABlankUpsertIDIsGuardedLikeAnyOtherOverGRPC(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)
	memory := rpcv1.NewMemoryServiceClient(conn)
	if _, err := knowledge.SaveKnowledge(asKey("op-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: " ", Collection: "secrets", Content: "The secret retainer."}); err != nil {
		t.Fatal(err)
	}
	if _, err := memory.SaveMemory(asKey("op-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "\t", UserId: "zeus", Scope: "user", Content: "Zeus keeps the thunderbolt."}); err != nil {
		t.Fatal(err)
	}
	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: " ", Collection: "notes", Content: "overwritten"}); status.Code(err) != codes.NotFound {
		t.Errorf("a blank id naming another collection's document: %v, want NOT_FOUND", err)
	}
	if _, err := memory.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "\t", UserId: "hermes", Scope: "user", Content: "stolen"}); status.Code(err) != codes.NotFound {
		t.Errorf("a blank id naming another user's memory: %v, want NOT_FOUND", err)
	}
	if _, err := knowledge.UpdateKnowledge(asKey("notes-secret"), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: " ", Collection: proto.String("notes"), Content: proto.String("moved")}); status.Code(err) != codes.NotFound {
		t.Errorf("an update of a blank id naming another collection's document: %v, want NOT_FOUND", err)
	}
	// No id at all is the facade's to refuse, as for any key.
	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{Collection: "notes", Content: "x"}); status.Code(err) != codes.InvalidArgument {
		t.Errorf("a document without an id: %v, want INVALID_ARGUMENT", err)
	}
	if _, err := memory.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{UserId: "hermes", Scope: "user", Content: "x"}); status.Code(err) != codes.InvalidArgument {
		t.Errorf("a memory without an id: %v, want INVALID_ARGUMENT", err)
	}
	got, err := knowledge.GetKnowledge(asKey("op-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: " "})
	if err != nil || got.GetKnowledge().GetCollection() != "secrets" {
		t.Errorf("the document moved: %v %v", got, err)
	}
}
