package rpcserver

import (
	"encoding/json"
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

// Every document carries the engine's marks (_content_sha256, _chunk_size,
// _chunk_overlap), which SaveKnowledge and UpdateKnowledge write from the
// content and replace whatever the caller sent. So a confined key that sends
// back the metadata of a document it read, marks included, is not refused,
// and the document keeps the engine's marks, not the ones sent. The tool
// route does the same for ingest_document, knowledge_save and
// knowledge_update, and InsertGraphDocument drops the marks it is sent. A
// memory write keeps the refusal: it does not write the marks.
func TestATypedKnowledgeWriteTakesTheMarksTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)
	memory := rpcv1.NewMemoryServiceClient(conn)

	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n1", Collection: "notes", Content: "Orpheus plays."}); err != nil {
		t.Fatal(err)
	}
	read := func() map[string]string {
		t.Helper()
		got, err := knowledge.GetKnowledge(asKey("notes-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: "n1"})
		if err != nil {
			t.Fatal(err)
		}
		return got.GetKnowledge().GetMetadata()
	}
	stored := read()
	hash := stored["_content_sha256"]
	sent, engine := map[string]string{}, map[string]string{}
	for k, v := range stored {
		sent[k] = v
	}
	for _, k := range cortexdb.EngineDocumentKeys() {
		if stored[k] == "" {
			t.Fatalf("the document carries no %s: %v", k, stored)
		}
		engine[k], sent[k] = stored[k], "forged"
	}
	same := func(what string) {
		t.Helper()
		after := read()
		for k, v := range engine {
			if after[k] != v {
				t.Errorf("after %s, %s is %q, want the engine's %q", what, k, after[k], v)
			}
		}
	}
	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n1", Collection: "notes", Content: "Orpheus plays.", Metadata: sent}); err != nil {
		t.Fatalf("SaveKnowledge with the metadata it read: %v", err)
	}
	same("SaveKnowledge")
	if _, err := knowledge.UpdateKnowledge(asKey("notes-secret"), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "n1", Collection: proto.String("notes"), Metadata: sent}); err != nil {
		t.Fatalf("UpdateKnowledge with the metadata it read: %v", err)
	}
	same("UpdateKnowledge")
	// The tool route refuses a signature, as on main, so the signed
	// record is sent back without it.
	delete(sent, "written_by")
	delete(sent, "updated_by")
	tools := rpcv1.NewToolsServiceClient(conn)
	for _, c := range []struct{ tool, id string }{{"ingest_document", "document_id"}, {"knowledge_save", "knowledge_id"}, {"knowledge_update", "knowledge_id"}} {
		args, _ := json.Marshal(map[string]any{c.id: "n1", "collection": "notes", "content": "Orpheus plays.", "metadata": sent})
		if _, err := tools.CallTool(asKey("notes-secret"), &rpcv1.CallToolRequest{Name: c.tool, ArgsJson: string(args)}); err != nil {
			t.Fatalf("CallTool %s with the metadata it read: %v", c.tool, err)
		}
		same(c.tool)
	}

	// InsertGraphDocument drops a caller's copy of the marks, so the
	// metadata read back passes, as on main. This server has no embedder,
	// so the call fails after the authorization; only a refusal fails here.
	graphrag := rpcv1.NewGraphRagServiceClient(conn)
	for _, k := range cortexdb.EngineDocumentKeys() {
		_, err = graphrag.InsertGraphDocument(asKey("notes-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "n3", Collection: "notes", Content: "x", Metadata: map[string]string{k: "x", "team": "a"}})
		if status.Code(err) == codes.PermissionDenied {
			t.Errorf("InsertGraphDocument refused the metadata it read, with %s: %v", k, err)
		}
	}
	if _, err = graphrag.InsertGraphDocument(asKey("notes-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "n3", Collection: "notes", Content: "x", Metadata: map[string]string{"_table": "x"}}); status.Code(err) != codes.PermissionDenied {
		t.Errorf("InsertGraphDocument took another engine key: %v, want PERMISSION_DENIED", err)
	}

	marked, err := structpb.NewStruct(map[string]any{"_content_sha256": hash})
	if err != nil {
		t.Fatal(err)
	}
	_, err = memory.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "h1", UserId: "hermes", Scope: "user", Content: "Hermes tunes the lyre.", Metadata: marked})
	if status.Code(err) != codes.PermissionDenied || !strings.Contains(status.Convert(err).Message(), "metadata._content_sha256") {
		t.Errorf("SaveMemory with a mark: %v, want PERMISSION_DENIED naming metadata._content_sha256", err)
	}
	_, err = knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n2", Collection: "notes", Content: "x", Metadata: map[string]string{"_Content_SHA256": "forged"}})
	if status.Code(err) != codes.PermissionDenied {
		t.Errorf("SaveKnowledge with another spelling of a mark: %v, want PERMISSION_DENIED", err)
	}
}
