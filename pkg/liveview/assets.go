package liveview

import (
	"compress/gzip"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"io"
	"net/http"
	"strings"
	"sync"
)

// graph3dJS is the page's 3D stack — 3d-force-graph, three and the bloom pass
// — built from web/ into one script (see web/package.json; `npm run build`
// there rewrites it). The page used to take 3d-force-graph from unpkg, which
// bundles its own three, and then import three again from unpkg for the bloom
// pass: two copies of three in one page, which three itself warns about, and
// a view that drew nothing without the internet. Served from the binary it is
// one copy and works offline.
//
//go:embed assets/graph3d.js
var graph3dJS []byte

// graph3dPath is where the page asks for the bundle. The name carries a hash
// of its content so it can be cached for good: a new build is a new URL.
var graph3dPath = func() string {
	sum := sha256.Sum256(graph3dJS)
	return "assets/graph3d-" + hex.EncodeToString(sum[:6]) + ".js"
}()

var (
	graph3dGzipOnce sync.Once
	graph3dGzip     []byte
)

// handleGraph3D serves the bundle, compressed once and kept.
func handleGraph3D(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	w.Header().Set("Vary", "Accept-Encoding")
	if acceptsGzip(r) {
		graph3dGzipOnce.Do(func() {
			var b strings.Builder
			zw, _ := gzip.NewWriterLevel(&b, gzip.BestCompression)
			_, _ = zw.Write(graph3dJS)
			_ = zw.Close()
			graph3dGzip = []byte(b.String())
		})
		w.Header().Set("Content-Encoding", "gzip")
		_, _ = w.Write(graph3dGzip)
		return
	}
	_, _ = w.Write(graph3dJS)
}

func acceptsGzip(r *http.Request) bool {
	return strings.Contains(r.Header.Get("Accept-Encoding"), "gzip")
}

// withGzip compresses the pages and the JSON the view serves. The graph alone
// is three quarters of a megabyte of JSON on the shared brain, read again on
// every reload; it compresses to a tenth. The event stream is left alone: it
// is flushed event by event, and a compressor would hold events back.
func withGzip(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !acceptsGzip(r) || strings.HasSuffix(r.URL.Path, "/api/stream") || strings.HasPrefix(strings.TrimPrefix(r.URL.Path, "/"), "assets/") {
			next.ServeHTTP(w, r)
			return
		}
		w.Header().Add("Vary", "Accept-Encoding")
		gw := &gzipResponseWriter{ResponseWriter: w}
		defer gw.close()
		next.ServeHTTP(gw, r)
	})
}

var gzipWriters = sync.Pool{New: func() any { w, _ := gzip.NewWriterLevel(io.Discard, gzip.DefaultCompression); return w }}

// gzipResponseWriter compresses a body once a handler starts writing one, and
// steps aside for responses that have no body to compress or arrive already
// encoded.
type gzipResponseWriter struct {
	http.ResponseWriter
	zw      *gzip.Writer
	decided bool
	plain   bool
}

func (g *gzipResponseWriter) WriteHeader(code int) {
	g.decide(code)
	g.ResponseWriter.WriteHeader(code)
}

func (g *gzipResponseWriter) decide(code int) {
	if g.decided {
		return
	}
	g.decided = true
	h := g.Header()
	if code == http.StatusNoContent || code == http.StatusNotModified || h.Get("Content-Encoding") != "" {
		g.plain = true
		return
	}
	h.Set("Content-Encoding", "gzip")
	h.Del("Content-Length")
	g.zw = gzipWriters.Get().(*gzip.Writer)
	g.zw.Reset(g.ResponseWriter)
}

func (g *gzipResponseWriter) Write(p []byte) (int, error) {
	g.decide(http.StatusOK)
	if g.plain {
		return g.ResponseWriter.Write(p)
	}
	return g.zw.Write(p)
}

// Flush lets a handler that streams still reach the client.
func (g *gzipResponseWriter) Flush() {
	if g.zw != nil {
		_ = g.zw.Flush()
	}
	if f, ok := g.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (g *gzipResponseWriter) close() {
	if g.zw != nil {
		_ = g.zw.Close()
		gzipWriters.Put(g.zw)
		g.zw = nil
	}
}
