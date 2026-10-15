package rpc

import (
	"context"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

type healthServer struct {
	rpcv1.UnimplementedAdminServiceServer
	calls atomic.Int32
	fail  codes.Code
}

func (h *healthServer) Health(context.Context, *rpcv1.HealthRequest) (*rpcv1.HealthResponse, error) {
	if h.calls.Add(1) == 1 && h.fail != codes.OK {
		return nil, status.Error(h.fail, "first call fails")
	}
	return &rpcv1.HealthResponse{Ok: true}, nil
}

func serve(t *testing.T, h *healthServer) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := grpc.NewServer(ServerOptions()...)
	rpcv1.RegisterAdminServiceServer(s, h)
	go func() { _ = s.Serve(lis) }()
	t.Cleanup(s.Stop)
	return lis.Addr().String()
}

// dropper forwards connections to a server and can lose them the way a
// failover or a dead path does: the server's side goes away, the client hears
// nothing, and the client's next packet is answered with a reset.
type dropper struct {
	lis      net.Listener
	upstream string
	mu       sync.Mutex
	pairs    []*droppedPair
}

type droppedPair struct {
	client, server net.Conn
	dead           atomic.Bool
	lastRead       atomic.Int64 // unix nanos of the client's last packet
}

func newDropper(t *testing.T, upstream string) *dropper {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	d := &dropper{lis: lis, upstream: upstream}
	t.Cleanup(func() { _ = lis.Close() })
	go d.accept()
	return d
}

func (d *dropper) accept() {
	for {
		c, err := d.lis.Accept()
		if err != nil {
			return
		}
		s, err := net.Dial("tcp", d.upstream)
		if err != nil {
			_ = c.Close()
			continue
		}
		p := &droppedPair{client: c, server: s}
		p.lastRead.Store(time.Now().UnixNano())
		d.mu.Lock()
		d.pairs = append(d.pairs, p)
		d.mu.Unlock()
		go func() { _, _ = io.Copy(c, s) }()
		go func() {
			buf := make([]byte, 32<<10)
			for {
				n, err := c.Read(buf)
				p.lastRead.Store(time.Now().UnixNano())
				if p.dead.Load() {
					_ = c.(*net.TCPConn).SetLinger(0)
					_ = c.Close()
					return
				}
				if n > 0 {
					if _, werr := s.Write(buf[:n]); werr != nil {
						return
					}
				}
				if err != nil {
					_ = s.Close()
					return
				}
			}
		}()
	}
}

// drop loses every open connection silently. It first waits for the client
// to go quiet, as a connection lost while idle does: gRPC still writes a few
// control frames (BDP pings, ping acks) right after a call, and one of those
// meeting the reset would let the client reconnect before its next call.
func (d *dropper) drop() {
	d.mu.Lock()
	defer d.mu.Unlock()
	const quiet = 100 * time.Millisecond
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		idle := true
		for _, p := range d.pairs {
			if time.Since(time.Unix(0, p.lastRead.Load())) < quiet {
				idle = false
			}
		}
		if idle {
			break
		}
	}
	for _, p := range d.pairs {
		p.dead.Store(true)
		_ = p.server.Close()
	}
}

func health(t *testing.T, conn *grpc.ClientConn) error {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := rpcv1.NewAdminServiceClient(conn).Health(ctx, &rpcv1.HealthRequest{})
	return err
}

func dialThrough(t *testing.T, addr string, opts ...grpc.DialOption) *grpc.ClientConn {
	t.Helper()
	conn, err := grpc.NewClient(addr, append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

func TestDialOptionsSurviveASilentlyDroppedConnection(t *testing.T) {
	d := newDropper(t, serve(t, &healthServer{}))
	conn := dialThrough(t, d.lis.Addr().String(), DialOptions()...)
	if err := health(t, conn); err != nil {
		t.Fatalf("first call: %v", err)
	}
	d.drop()
	if err := health(t, conn); err != nil {
		t.Fatalf("call after the connection was lost: %v", err)
	}
}

// The failure the options exist for: without them, the call after a silent
// drop goes down the dead socket and fails.
func TestPlainDialFailsAfterASilentlyDroppedConnection(t *testing.T) {
	d := newDropper(t, serve(t, &healthServer{}))
	conn := dialThrough(t, d.lis.Addr().String())
	if err := health(t, conn); err != nil {
		t.Fatalf("first call: %v", err)
	}
	d.drop()
	if err := health(t, conn); status.Code(err) != codes.Unavailable {
		t.Fatalf("call after the connection was lost = %v, want Unavailable", err)
	}
}

func TestDialOptionsRetryOnlyUnavailable(t *testing.T) {
	for _, tc := range []struct {
		fail  codes.Code
		calls int32
		ok    bool
	}{
		{codes.Unavailable, 2, true},
		{codes.Internal, 1, false},
		{codes.Unauthenticated, 1, false},
	} {
		h := &healthServer{fail: tc.fail}
		conn := dialThrough(t, serve(t, h), DialOptions()...)
		err := health(t, conn)
		if (err == nil) != tc.ok || h.calls.Load() != tc.calls {
			t.Errorf("%v: err=%v calls=%d, want ok=%v calls=%d", tc.fail, err, h.calls.Load(), tc.ok, tc.calls)
		}
	}
}
