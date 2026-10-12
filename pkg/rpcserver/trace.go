package rpcserver

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"

	"github.com/liliang-cn/cortexdb/v2/pkg/observability"
)

// The request log: one line per call, with where its time went.
//
// CORTEXDB_TRACE=1 logs every call. Without it only slow calls are logged —
// CORTEXDB_SLOW_MS, 1000 by default, 0 to log none — because a server that
// says nothing about a 24-second recall leaves its clients guessing which of
// three machines to blame. CORTEXDB_TRACE_FILE appends the same records as
// JSON lines.

// TraceOptions configures RequestLogInterceptor.
type TraceOptions struct {
	All  bool          // log every call
	Slow time.Duration // log calls at least this slow; 0 logs none
	File io.Writer     // JSON lines, optional
	Log  *log.Logger   // defaults to the standard logger
}

// TraceOptionsFromEnv reads CORTEXDB_TRACE, CORTEXDB_SLOW_MS and
// CORTEXDB_TRACE_FILE.
func TraceOptionsFromEnv() TraceOptions {
	o := TraceOptions{Slow: time.Second}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("CORTEXDB_TRACE"))) {
	case "1", "true", "on", "yes":
		o.All = true
	}
	if v := strings.TrimSpace(os.Getenv("CORTEXDB_SLOW_MS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			o.Slow = time.Duration(n) * time.Millisecond
		}
	}
	if p := strings.TrimSpace(os.Getenv("CORTEXDB_TRACE_FILE")); p != "" {
		if f, err := os.OpenFile(p, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600); err == nil {
			o.File = f
		} else {
			log.Printf("cortexdb-trace: cannot open %s: %v", p, err)
		}
	}
	return o
}

type traceRecord struct {
	TS     string           `json:"ts"`
	Method string           `json:"method"`
	DurMs  int64            `json:"dur_ms"`
	Stages map[string]int64 `json:"stages_ms,omitempty"`
	Err    string           `json:"err,omitempty"`
}

// RequestLogInterceptor gives every call a stage collector and logs the call
// when the options say to.
func RequestLogInterceptor(o TraceOptions) grpc.UnaryServerInterceptor {
	logger := o.Log
	if logger == nil {
		logger = log.Default()
	}
	var fileMu sync.Mutex
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		ctx, stages := observability.WithStages(ctx)
		start := time.Now()
		resp, err := handler(ctx, req)
		dur := time.Since(start)
		if !o.All && (o.Slow <= 0 || dur < o.Slow) {
			return resp, err
		}
		method := info.FullMethod
		if i := strings.LastIndexByte(method, '/'); i >= 0 {
			method = method[i+1:]
		}
		line := "cortexdb-trace rpc=" + method + " dur=" + strconv.FormatFloat(dur.Seconds(), 'f', 2, 64) + "s"
		if s := stages.String(); s != "" {
			line += " " + s
		}
		if !o.All {
			line += " (slow)"
		}
		if err != nil {
			line += " err=" + strconv.Quote(err.Error())
		}
		logger.Print(line)
		if o.File != nil {
			rec := traceRecord{TS: start.UTC().Format(time.RFC3339Nano), Method: method, DurMs: dur.Milliseconds(), Stages: stages.Millis()}
			if err != nil {
				rec.Err = err.Error()
			}
			if b, jerr := json.Marshal(rec); jerr == nil {
				fileMu.Lock()
				_, _ = o.File.Write(append(b, '\n'))
				fileMu.Unlock()
			}
		}
		return resp, err
	}
}
