package rpcserver

import (
	"context"
	"net"
	"path/filepath"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// A document whose chunks are gone has no collection the facade can read
// (GetKnowledge names the default one), so the typed RPCs withhold it from a
// confined key, as the tool guard does, even from a key confined to the
// default collection.
func TestTheTypedRPCsWithholdADocumentTheyCannotPlace(t *testing.T) {
	ctx := context.Background()
	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(t.TempDir(), "chunkless.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if _, err := db.SaveKnowledge(ctx, cortexdb.KnowledgeSaveRequest{KnowledgeID: "secret-doc", Collection: "secrets", Content: "the retainer"}); err != nil {
		t.Fatal(err)
	}
	if err := db.Vector().DeleteByDocID(ctx, "secret-doc"); err != nil {
		t.Fatal(err)
	}
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "chunks", Secret: "chunks-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "graphrag_chunks"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv, err := NewWithPolicy(db, Options{Keys: ks, DBPath: filepath.Join(t.TempDir(), "unused.db")})
	if err != nil {
		t.Fatal(err)
	}
	lis := bufconn.Listen(1 << 20)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)

	if _, err := knowledge.GetKnowledge(asKey("op-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: "secret-doc"}); err != nil {
		t.Fatalf("the operator reads the document: %v", err)
	}
	for _, c := range []struct {
		what string
		call func() error
	}{
		{"GetKnowledge", func() error {
			_, err := knowledge.GetKnowledge(asKey("chunks-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: "secret-doc"})
			return err
		}},
		{"DeleteKnowledge", func() error {
			_, err := knowledge.DeleteKnowledge(asKey("chunks-secret"), &rpcv1.DeleteKnowledgeRequest{KnowledgeId: "secret-doc"})
			return err
		}},
		{"SaveKnowledge", func() error {
			_, err := knowledge.SaveKnowledge(asKey("chunks-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "secret-doc", Collection: "graphrag_chunks", Content: "overwritten"})
			return err
		}},
		{"UpdateKnowledge", func() error {
			_, err := knowledge.UpdateKnowledge(asKey("chunks-secret"), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "secret-doc", Collection: proto.String("graphrag_chunks"), Content: proto.String("overwritten")})
			return err
		}},
	} {
		if err := c.call(); status.Code(err) != codes.NotFound {
			t.Errorf("%s of a document with no chunks under a key confined to the default collection: %v, want NOT_FOUND", c.what, err)
		}
	}
	got, err := db.GetKnowledge(ctx, cortexdb.KnowledgeGetRequest{KnowledgeID: "secret-doc"})
	if err != nil {
		t.Fatalf("the document was deleted through the refused call: %v", err)
	}
	if got.Knowledge.Content != "the retainer" {
		t.Errorf("the document was overwritten through a refused call: %q", got.Knowledge.Content)
	}
}
