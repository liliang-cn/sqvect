package rpcserver

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// A typed write a confined key may call takes what the allowlist line of
// the tool it stands for takes, and nothing more: SaveKnowledge and
// UpdateKnowledge are knowledge_save and knowledge_update, InsertGraphDocument
// is ingest_document, SaveMemory and UpdateMemory are memory_save and
// memory_update. So declared entities and relations, an engine key in
// metadata, and a document's chunk keys are refused on both paths with the
// same reason, and the caller's own metadata passes on both. The operator's
// unconfined key is not held to the allowlist.
func TestATypedWriteTakesWhatTheAllowlistLineOfItsToolTakes(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
		{ID: "lab", Secret: "lab-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Namespace: "lab"}},
		{ID: "mine", Secret: "mine-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{MemoryScope: "user"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks}, cortexdb.WithEmbedder(fakeEmbedder{}))
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)
	memory := rpcv1.NewMemoryServiceClient(conn)
	graphrag := rpcv1.NewGraphRagServiceClient(conn)
	meta := func(m map[string]any) *structpb.Struct {
		s, err := structpb.NewStruct(m)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	entity := []*rpcv1.ToolEntityInput{{Name: "Project Nightshade", Description: "defaced"}}
	relation := []*rpcv1.ToolRelationInput{{From: "Project Nightshade", To: "Orpheus", Type: "funds"}}

	if _, err := memory.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "hermes-1", UserId: "hermes", Scope: "user", Content: "Hermes tunes the lyre."}); err != nil {
		t.Fatalf("hermes writes its memory: %v", err)
	}
	for _, c := range []struct {
		what string
		call func(ctx context.Context) error
		want string
	}{
		{"SaveKnowledge entities", func(ctx context.Context) error {
			_, err := knowledge.SaveKnowledge(ctx, &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n1", Collection: "notes", Content: "x", Entities: entity})
			return err
		}, "entities"},
		{"SaveKnowledge relations", func(ctx context.Context) error {
			_, err := knowledge.SaveKnowledge(ctx, &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n1", Collection: "notes", Content: "x", Relations: relation})
			return err
		}, "relations"},
		{"SaveKnowledge engine key", func(ctx context.Context) error {
			_, err := knowledge.SaveKnowledge(ctx, &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n1", Collection: "notes", Content: "x", Metadata: map[string]string{"_scope": "global"}})
			return err
		}, "metadata._scope"},
		{"UpdateKnowledge entities", func(ctx context.Context) error {
			_, err := knowledge.UpdateKnowledge(ctx, &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "n1", Collection: proto.String("notes"), Entities: entity})
			return err
		}, "entities"},
		{"UpdateKnowledge relations", func(ctx context.Context) error {
			_, err := knowledge.UpdateKnowledge(ctx, &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "n1", Collection: proto.String("notes"), Relations: relation})
			return err
		}, "relations"},
		{"UpdateKnowledge chunk key", func(ctx context.Context) error {
			_, err := knowledge.UpdateKnowledge(ctx, &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "n1", Collection: proto.String("notes"), Metadata: map[string]string{"document_id": "hr"}})
			return err
		}, "metadata.document_id"},
		{"InsertGraphDocument chunk key", func(ctx context.Context) error {
			_, err := graphrag.InsertGraphDocument(ctx, &rpcv1.InsertGraphDocumentRequest{Id: "n2", Collection: "notes", Content: "x", Metadata: map[string]string{"graph_kind": "entity"}})
			return err
		}, "metadata.graph_kind"},
		{"SaveMemory engine key", func(ctx context.Context) error {
			_, err := memory.SaveMemory(ctx, &rpcv1.SaveMemoryRequest{MemoryId: "hermes-2", UserId: "hermes", Scope: "user", Content: "x", Metadata: meta(map[string]any{"_scope": "global"})})
			return err
		}, "metadata._scope"},
		{"UpdateMemory engine key", func(ctx context.Context) error {
			_, err := memory.UpdateMemory(ctx, &rpcv1.UpdateMemoryRequest{MemoryId: "hermes-1", Metadata: meta(map[string]any{"_x": "y"})})
			return err
		}, "metadata._x"},
	} {
		key := "notes-secret"
		if strings.Contains(c.what, "Memory") {
			key = "hermes-secret"
		}
		err := c.call(asKey(key))
		if status.Code(err) != codes.PermissionDenied || !strings.Contains(status.Convert(err).Message(), c.want) {
			t.Errorf("%s: %v, want PERMISSION_DENIED naming %s", c.what, err, c.want)
		}
		if err := c.call(asKey("op-secret")); status.Code(err) == codes.PermissionDenied {
			t.Errorf("%s: the operator was held to the allowlist: %v", c.what, err)
		}
	}
	// The caller's own metadata, on each typed write.
	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n3", Collection: "notes", Content: "Orpheus plays.", Metadata: map[string]string{"color": "blue"}}); err != nil {
		t.Errorf("SaveKnowledge with the caller's metadata: %v", err)
	}
	if _, err := knowledge.UpdateKnowledge(asKey("notes-secret"), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "n3", Collection: proto.String("notes"), Content: proto.String("Orpheus plays on."), Metadata: map[string]string{"color": "red"}}); err != nil {
		t.Errorf("UpdateKnowledge with the caller's metadata: %v", err)
	}
	if _, err := graphrag.InsertGraphDocument(asKey("notes-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "n4", Collection: "notes", Content: "Eurydice sings.", Metadata: map[string]string{"color": "blue"}}); err != nil {
		t.Errorf("InsertGraphDocument with the caller's metadata: %v", err)
	}
	if got, err := knowledge.GetKnowledge(asKey("notes-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: "n4"}); err != nil || got.GetKnowledge().GetCollection() != "notes" {
		t.Errorf("the ingested document: %v, %v, want it in notes", got.GetKnowledge(), err)
	}
	if _, err := memory.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "hermes-3", UserId: "hermes", Scope: "user", Content: "x", Metadata: meta(map[string]any{"document_id": "a note of mine"})}); err != nil {
		t.Errorf("SaveMemory with the caller's metadata: %v", err)
	}
	if _, err := memory.UpdateMemory(asKey("hermes-secret"), &rpcv1.UpdateMemoryRequest{MemoryId: "hermes-1", Content: proto.String("Hermes tunes the lyre twice."), Metadata: meta(map[string]any{"mood": "calm"})}); err != nil {
		t.Errorf("UpdateMemory with the caller's metadata: %v", err)
	}
	// The scope fields of a typed SaveMemory reach the allowlist line too:
	// a key confined to a namespace, or to a memory scope, writes in its own.
	if _, err := memory.SaveMemory(asKey("lab-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "lab-1", UserId: "hermes", Scope: "user", Namespace: "lab", Content: "x"}); err != nil {
		t.Errorf("SaveMemory in the key's namespace: %v", err)
	}
	if _, err := memory.SaveMemory(asKey("mine-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "mine-1", UserId: "hermes", Scope: "user", Content: "x"}); err != nil {
		t.Errorf("SaveMemory in the key's memory scope: %v", err)
	}
}
