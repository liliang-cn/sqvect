package rpcserver

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

const (
	reloadKeyA = `{"id":"a","secret":"a-secret","clearance":"read-write"}`
	reloadKeyB = `{"id":"b","secret":"b-secret","clearance":"read-only"}`
	reloadKeyC = `{"id":"c","secret":"c-secret","clearance":"read-only"}`
)

func reloadKeyFile(keys ...string) string { return `{"keys":[` + strings.Join(keys, ",") + `]}` }

// rewriteKeyFile writes body to path and moves its time on by step, so the
// change shows even within one tick of the file system clock.
func rewriteKeyFile(t testing.TB, path, body string, step time.Duration) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	at := time.Now().Add(step)
	if err := os.Chtimes(path, at, at); err != nil {
		t.Fatal(err)
	}
}

func healthAs(t *testing.T, c rpcv1.AdminServiceClient, secret string) codes.Code {
	t.Helper()
	ctx := asKey(secret)
	if secret == "" {
		ctx = t.Context()
	}
	_, err := c.Health(ctx, &rpcv1.HealthRequest{})
	return status.Code(err)
}

func TestAKeyRemovedFromTheKeyFileIsRefusedWithoutARestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyA, reloadKeyB), 0)
	c := rpcv1.NewAdminServiceClient(newPolicyConn(t, Options{KeyFile: path}))
	if got := healthAs(t, c, "a-secret"); got != codes.OK {
		t.Fatalf("a before the change: %v", got)
	}
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyB, reloadKeyC), time.Second)
	if got := healthAs(t, c, "a-secret"); got != codes.Unauthenticated {
		t.Errorf("a key removed from the file: %v, want Unauthenticated", got)
	}
	if got := healthAs(t, c, "c-secret"); got != codes.OK {
		t.Errorf("a key added to the file: %v, want OK", got)
	}
	if got := healthAs(t, c, "b-secret"); got != codes.OK {
		t.Errorf("a key the file still holds: %v, want OK", got)
	}
}

// An emptied key file is a file that does not load: the keys in force stay,
// and a call without a key is still refused.
func TestAnEmptiedKeyFileDoesNotOpenTheGRPCServer(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyA), 0)
	c := rpcv1.NewAdminServiceClient(newPolicyConn(t, Options{KeyFile: path}))
	rewriteKeyFile(t, path, `{"keys":[]}`, time.Second)
	if got := healthAs(t, c, ""); got != codes.Unauthenticated {
		t.Errorf("a call without a key after the file was emptied: %v, want Unauthenticated", got)
	}
	if got := healthAs(t, c, "a-secret"); got != codes.OK {
		t.Errorf("the key in force after the file was emptied: %v, want OK", got)
	}
}

// Options.Keys is a key set the caller built: it is served as given, and
// the file KeyFile names is never read.
func TestAPreloadedKeySetIsNeverReread(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyC), 0)
	keys, err := authz.Parse([]byte(reloadKeyFile(reloadKeyA)))
	if err != nil {
		t.Fatal(err)
	}
	c := rpcv1.NewAdminServiceClient(newPolicyConn(t, Options{Keys: keys, KeyFile: path}))
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyB), time.Second)
	if got := healthAs(t, c, "a-secret"); got != codes.OK {
		t.Errorf("the preloaded key: %v, want OK", got)
	}
	for _, secret := range []string{"b-secret", "c-secret"} {
		if got := healthAs(t, c, secret); got != codes.Unauthenticated {
			t.Errorf("a key of the file beside Options.Keys: %v, want Unauthenticated", got)
		}
	}
}

// What the reread costs one call, against a key set given in Options.Keys.
// Run with go test -run '^$' -bench HealthWith -benchmem ./pkg/rpcserver/
func BenchmarkHealthWithAPreloadedKeySet(b *testing.B) {
	keys, err := authz.Parse([]byte(reloadKeyFile(reloadKeyA)))
	if err != nil {
		b.Fatal(err)
	}
	benchHealth(b, Options{Keys: keys})
}

func BenchmarkHealthWithAKeyFile(b *testing.B) {
	path := filepath.Join(b.TempDir(), "keys.json")
	rewriteKeyFile(b, path, reloadKeyFile(reloadKeyA), 0)
	benchHealth(b, Options{KeyFile: path})
}

func benchHealth(b *testing.B, opts Options) {
	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(b.TempDir(), "bench.db")))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = db.Close() })
	srv, err := NewWithPolicy(db, opts)
	if err != nil {
		b.Fatal(err)
	}
	lis := bufconn.Listen(1 << 20)
	go func() { _ = srv.Serve(lis) }()
	b.Cleanup(srv.Stop)
	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = conn.Close() })
	c := rpcv1.NewAdminServiceClient(conn)
	ctx := asKey("a-secret")
	for b.Loop() {
		if _, err := c.Health(ctx, &rpcv1.HealthRequest{}); err != nil {
			b.Fatal(err)
		}
	}
}
