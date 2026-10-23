package rpcserver

import (
	"context"
	"net"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/liliang-cn/cortexdb/v2/pkg/rpc"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// A process that mounts cortexdb.v1 on a server of its own gets the reread
// too, through AuthInterceptorFromSource.
func TestAServerOfItsOwnRereadsTheKeyFileThroughAuthInterceptorFromSource(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyA), 0)
	src, err := authz.OpenKeySource(path)
	if err != nil {
		t.Fatal(err)
	}
	dbPath := filepath.Join(t.TempDir(), "own.db")
	db, err := cortexdb.Open(cortexdb.DefaultConfig(dbPath))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	srv := grpc.NewServer(append(rpc.ServerOptions(), grpc.UnaryInterceptor(AuthInterceptorFromSource(src, db)))...)
	Register(srv, db, Options{DBPath: dbPath})
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
	c := rpcv1.NewAdminServiceClient(conn)

	if got := healthAs(t, c, "a-secret"); got != codes.OK {
		t.Fatalf("a before the change: %v", got)
	}
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyB), time.Second)
	if got := healthAs(t, c, "a-secret"); got != codes.Unauthenticated {
		t.Errorf("a key removed from the file: %v, want Unauthenticated", got)
	}
	if got := healthAs(t, c, "b-secret"); got != codes.OK {
		t.Errorf("a key added to the file: %v, want OK", got)
	}
}

// An interceptor over a KeySource that was never opened refuses the call; it
// neither panics nor lets the call through.
func TestAuthInterceptorFromSourceRefusesOnASourceNeverOpened(t *testing.T) {
	for name, src := range map[string]*authz.KeySource{"zero": {}, "nil": nil} {
		intercept := AuthInterceptorFromSource(src, nil)
		ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs("authorization", "Bearer a-secret"))
		called := false
		_, err := intercept(ctx, nil, &grpc.UnaryServerInfo{FullMethod: rpcv1.AdminService_Health_FullMethodName},
			func(context.Context, any) (any, error) { called = true; return nil, nil })
		if called || status.Code(err) != codes.Unauthenticated {
			t.Errorf("%s source: handler called %v, error %v; want Unauthenticated", name, called, err)
		}
	}
}
