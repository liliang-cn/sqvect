package cortexdb

import (
	"context"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/observability"
)

// stallingEmbedder answers until told to stall, then waits for its caller to
// give up — an embedding model on another machine that had to load first.
type stallingEmbedder struct{ stall atomic.Bool }

func (e *stallingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if e.stall.Load() {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	v := make([]float32, 8)
	for i, r := range text {
		v[i%8] += float32(r % 7)
	}
	return v, nil
}

func (e *stallingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, err := e.Embed(ctx, t)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func (e *stallingEmbedder) Dim() int { return 8 }

// A recall whose embedder stalls answers by keyword within the bound, instead
// of waiting for as long as the caller is willing to — which was 24 seconds —
// and the request's stages show where the time went.
func TestARecallWithAStalledEmbedderAnswersByKeywordInTime(t *testing.T) {
	queryEmbedTimeout = 200 * time.Millisecond
	t.Cleanup(func() { queryEmbedTimeout = 3 * time.Second })

	emb := &stallingEmbedder{}
	cfg := DefaultConfig(filepath.Join(t.TempDir(), "stall.db"))
	cfg.Dimensions = 8
	db, err := Open(cfg, WithEmbedder(emb))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	if _, err := db.SaveMemory(context.Background(), MemorySaveRequest{MemoryID: "m1", Scope: "global", Content: "the gateway listens on 47821"}); err != nil {
		t.Fatalf("save: %v", err)
	}

	emb.stall.Store(true)
	ctx, stages := observability.WithStages(context.Background())
	start := time.Now()
	res, err := db.SearchMemory(ctx, MemorySearchRequest{Query: "gateway", Scope: "global", TopK: 3})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if took := time.Since(start); took > 2*time.Second {
		t.Fatalf("recall waited %s on a stalled embedder", took)
	}
	if len(res.Results) == 0 || res.Results[0].Memory.ID != "m1" {
		t.Fatalf("keyword fallback did not find the memory: %+v", res.Results)
	}
	if ms := stages.Millis(); ms["embed"] < 150 || ms["lexical"] < 0 {
		t.Fatalf("stages do not show the stalled embed: %v", ms)
	}
}
