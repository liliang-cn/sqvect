package rpcserver

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"

	"github.com/liliang-cn/cortexdb/v2/pkg/observability"
)

func runThrough(t *testing.T, o TraceOptions, work func(ctx context.Context)) string {
	t.Helper()
	var buf bytes.Buffer
	o.Log = log.New(&buf, "", 0)
	ic := RequestLogInterceptor(o)
	_, _ = ic(context.Background(), nil, &grpc.UnaryServerInfo{FullMethod: "/cortexdb.v1.MemoryService/SearchMemory"},
		func(ctx context.Context, _ any) (any, error) { work(ctx); return nil, nil })
	return buf.String()
}

// A slow call is logged with where its time went, without anyone turning
// tracing on: the 24-second recall that started this left no line at all.
func TestASlowCallIsLoggedWithItsStages(t *testing.T) {
	var file bytes.Buffer
	got := runThrough(t, TraceOptions{Slow: 20 * time.Millisecond, File: &file}, func(ctx context.Context) {
		done := observability.Time(ctx, "embed")
		time.Sleep(30 * time.Millisecond)
		done()
		observability.Time(ctx, "lexical")()
	})
	for _, want := range []string{"rpc=SearchMemory", "embed=0.0", "lexical=", "(slow)"} {
		if !strings.Contains(got, want) {
			t.Fatalf("log line lacks %q: %q", want, got)
		}
	}
	var rec traceRecord
	if err := json.Unmarshal(bytes.TrimSpace(file.Bytes()), &rec); err != nil || rec.Stages["embed"] < 30 {
		t.Fatalf("trace file record: %v %+v", err, rec)
	}
}

func TestAFastCallIsQuietUnlessTracingIsOn(t *testing.T) {
	quick := func(context.Context) {}
	if got := runThrough(t, TraceOptions{Slow: time.Second}, quick); got != "" {
		t.Fatalf("a fast call was logged: %q", got)
	}
	if got := runThrough(t, TraceOptions{All: true, Slow: time.Second}, quick); !strings.Contains(got, "rpc=SearchMemory") || strings.Contains(got, "(slow)") {
		t.Fatalf("with tracing on every call is logged: %q", got)
	}
}
