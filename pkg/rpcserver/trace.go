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

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/observability"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// The request log: one line per call, with where its time went.
//
// CORTEXDB_TRACE=1 logs every call. Without it only slow calls are logged —
// CORTEXDB_SLOW_MS, 1000 by default, 0 to log none — because a server that
// says nothing about a 24-second recall leaves its clients guessing which of
// three machines to blame. CORTEXDB_TRACE_FILE appends the same records as
// JSON lines. A CallTool's record also names the rows its arguments name
// (ids), so a trace file says what a key reached for, not only which tool.

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
	TS     string `json:"ts"`
	Method string `json:"method"`
	Key    string `json:"key,omitempty"`
	Tool   string `json:"tool,omitempty"`
	// IDs are the rows a CallTool names in its arguments, by path
	// (authz.RowIDsOf): the ids, distinct and sorted before the cut, at
	// most maxTraceIDs per path, each cut at maxTraceIDBytes. IDsOmitted
	// counts the ones left out.
	IDs        map[string][]string `json:"ids,omitempty"`
	IDsOmitted int                 `json:"ids_omitted,omitempty"`
	DurMs      int64               `json:"dur_ms"`
	Stages     map[string]int64    `json:"stages_ms,omitempty"`
	Err        string              `json:"err,omitempty"`
}

// maxTraceIDs bounds the ids a record keeps per path, and maxTraceIDBytes
// the length of each. The log runs before authentication, so without them
// any caller could write a list of any length to the trace file.
const (
	maxTraceIDs     = 100
	maxTraceIDBytes = 256
)

// callerTag is what the request log learns about a call after it has started
// timing it: which key spoke. The log runs outside the policy gate (see
// Options.Interceptors), so the key is not known when the log begins, and a
// context value set further down the chain never travels back up. The log
// plants this holder in the context instead, the auth interceptor fills it in
// once the key is resolved, and the log reads it back when the call returns.
//
// Only the key's id is kept, never its secret: the record is for an operator
// asking "who wrote this", and a trace file is not a place to keep a
// credential. Whether the call was then denied is already in err.
type callerTag struct {
	mu  sync.Mutex
	key string
}

type callerTagContextKey struct{}

// tagCaller records the id of the key a call authenticated as. Free when the
// request log is not installed.
func tagCaller(ctx context.Context, keyID string) {
	tag, _ := ctx.Value(callerTagContextKey{}).(*callerTag)
	if tag == nil {
		return
	}
	tag.mu.Lock()
	tag.key = keyID
	tag.mu.Unlock()
}

func (t *callerTag) keyID() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.key
}

// callToolMethod is the one RPC whose request names a tool, which the record
// keeps beside the method: "CallTool" alone says nothing about what ran.
const callToolMethod = "/cortexdb.v1.ToolsService/CallTool"

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
		tag := &callerTag{}
		ctx = context.WithValue(ctx, callerTagContextKey{}, tag)
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
		key := tag.keyID()
		tool := ""
		var ids map[string][]string
		if info.FullMethod == callToolMethod {
			tool, _ = requestToolName(req)()
			call, _ := req.(*rpcv1.CallToolRequest)
			ids = authz.RowIDsOf(tool, []byte(call.GetArgsJson()))
		}
		line := "cortexdb-trace rpc=" + method + " dur=" + strconv.FormatFloat(dur.Seconds(), 'f', 2, 64) + "s"
		if key != "" {
			line += " key=" + key
		}
		if tool != "" {
			line += " tool=" + tool
		}
		// The count only: the ids themselves go to the trace file.
		if n := idCount(ids); n > 0 {
			line += " ids=" + strconv.Itoa(n)
		}
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
			rec := traceRecord{TS: start.UTC().Format(time.RFC3339Nano), Method: method, Key: key, Tool: tool, DurMs: dur.Milliseconds(), Stages: stages.Millis()}
			rec.IDs, rec.IDsOmitted = boundIDs(ids)
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

func idCount(ids map[string][]string) int {
	n := 0
	for _, v := range ids {
		n += len(v)
	}
	return n
}

// boundIDs keeps the first maxTraceIDs ids of each path, cuts each at
// maxTraceIDBytes on a character boundary with a mark, and counts the ids
// left out.
func boundIDs(ids map[string][]string) (map[string][]string, int) {
	omitted := 0
	for p, v := range ids {
		if len(v) > maxTraceIDs {
			omitted += len(v) - maxTraceIDs
			v = v[:maxTraceIDs]
		}
		for i, id := range v {
			if len(id) > maxTraceIDBytes {
				v[i] = strings.ToValidUTF8(id[:maxTraceIDBytes], "") + "…"
			}
		}
		ids[p] = v
	}
	return ids, omitted
}
