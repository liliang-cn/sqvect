package rpcserver

import (
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// InsertGraphDocument names the document it writes by id, as SaveKnowledge
// does, so it is guarded as SaveKnowledge is: a key confined to a collection
// cannot overwrite a document of another collection, or move it into its own,
// and the answer is a withheld document's. A new id, its own document again,
// and the operator's unconfined key pass.
func TestInsertGraphDocumentCannotTakeADocumentOfAnotherCollection(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks}, cortexdb.WithEmbedder(fakeEmbedder{}))
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)
	graphrag := rpcv1.NewGraphRagServiceClient(conn)
	if _, err := knowledge.SaveKnowledge(asKey("op-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "sec", Collection: "secrets", Content: "The vault code."}); err != nil {
		t.Fatal(err)
	}
	_, err = graphrag.InsertGraphDocument(asKey("notes-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "sec", Collection: "notes", Content: "Overwritten by notes."})
	if want := `knowledge "sec" not found`; status.Code(err) != codes.NotFound || status.Convert(err).Message() != want {
		t.Errorf("InsertGraphDocument over another collection's document: %v, want NOT_FOUND %q", err, want)
	}
	// Empty content is refused by the guard first, as on SaveKnowledge.
	_, err = graphrag.InsertGraphDocument(asKey("notes-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "sec", Collection: "notes", Content: ""})
	if status.Code(err) != codes.NotFound {
		t.Errorf("InsertGraphDocument with empty content over another collection's document: %v, want NOT_FOUND", err)
	}
	read := func(id string) *rpcv1.KnowledgeRecord {
		t.Helper()
		got, err := knowledge.GetKnowledge(asKey("op-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: id})
		if err != nil {
			t.Fatal(err)
		}
		return got.GetKnowledge()
	}
	if k := read("sec"); k.GetCollection() != "secrets" || k.GetContent() != "The vault code." {
		t.Errorf("the document is now %q in %q, want it untouched", k.GetContent(), k.GetCollection())
	}
	for _, content := range []string{"Orpheus plays.", "Orpheus plays on."} {
		if _, err := graphrag.InsertGraphDocument(asKey("notes-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "mine", Collection: "notes", Content: content}); err != nil {
			t.Errorf("InsertGraphDocument of its own document %q: %v", content, err)
		}
		if k := read("mine"); k.GetCollection() != "notes" || k.GetContent() != content {
			t.Errorf("its own document is %q in %q, want %q in notes", k.GetContent(), k.GetCollection(), content)
		}
	}
	if _, err := graphrag.InsertGraphDocument(asKey("op-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "sec", Collection: "secrets", Content: "The vault code, changed."}); err != nil {
		t.Errorf("the operator rewrites the document: %v", err)
	}
}
