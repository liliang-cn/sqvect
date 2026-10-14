package cortexdb

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type cacheProbeEmbedder struct {
	calls atomic.Int32
	delay time.Duration
	fail  atomic.Bool
}

func (e *cacheProbeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	e.calls.Add(1)
	select {
	case <-time.After(e.delay):
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if e.fail.Load() {
		return nil, errors.New("model unavailable")
	}
	return []float32{float32(len(text)), 1}, nil
}

func (e *cacheProbeEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

func (e *cacheProbeEmbedder) Dim() int { return 2 }

// The same text is embedded once; the second request is answered from memory,
// and what it gets back is its own copy.
func TestCachedEmbedderAnswersARepeatFromMemory(t *testing.T) {
	inner := &cacheProbeEmbedder{}
	c := newCachedEmbedder(inner)
	ctx := context.Background()
	a, err := c.Embed(ctx, "drbd failover")
	if err != nil {
		t.Fatal(err)
	}
	a[0] = -1
	b, err := c.Embed(ctx, "drbd failover")
	if err != nil {
		t.Fatal(err)
	}
	if inner.calls.Load() != 1 {
		t.Fatalf("embedder called %d times for one text", inner.calls.Load())
	}
	if b[0] != float32(len("drbd failover")) {
		t.Fatalf("a caller's change reached the cache: %v", b)
	}
}

// A burst of the same question makes one call to the model.
func TestCachedEmbedderSharesOneCallAcrossABurst(t *testing.T) {
	inner := &cacheProbeEmbedder{delay: 50 * time.Millisecond}
	c := newCachedEmbedder(inner)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := c.Embed(context.Background(), "what did the hive decide"); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if n := inner.calls.Load(); n != 1 {
		t.Fatalf("16 concurrent requests made %d calls", n)
	}
}

// A failure is not remembered: the next request asks the model again.
func TestCachedEmbedderDoesNotCacheAFailure(t *testing.T) {
	inner := &cacheProbeEmbedder{}
	inner.fail.Store(true)
	c := newCachedEmbedder(inner)
	if _, err := c.Embed(context.Background(), "q"); err == nil {
		t.Fatal("want the model's error")
	}
	inner.fail.Store(false)
	if _, err := c.Embed(context.Background(), "q"); err != nil {
		t.Fatalf("after the model recovered: %v", err)
	}
	if inner.calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", inner.calls.Load())
	}
}

// A request that gives up does not take the call down with it: the answer
// still lands in the cache, and the next request gets it without asking.
func TestCachedEmbedderKeepsAnAnswerItsCallerGaveUpOn(t *testing.T) {
	inner := &cacheProbeEmbedder{delay: 100 * time.Millisecond}
	c := newCachedEmbedder(inner)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, err := c.Embed(ctx, "slow"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want the caller's deadline, got %v", err)
	}
	time.Sleep(200 * time.Millisecond)
	if _, err := c.Embed(context.Background(), "slow"); err != nil {
		t.Fatal(err)
	}
	if inner.calls.Load() != 1 {
		t.Fatalf("calls = %d, want the first call's answer reused", inner.calls.Load())
	}
}

// The cache keeps the most recently used texts and no more.
func TestCachedEmbedderIsBounded(t *testing.T) {
	inner := &cacheProbeEmbedder{}
	c := newCachedEmbedder(inner)
	ctx := context.Background()
	for i := 0; i < embedCacheEntries+10; i++ {
		if _, err := c.Embed(ctx, string(rune('a'+i%26))+time.Duration(i).String()); err != nil {
			t.Fatal(err)
		}
	}
	c.cache.mu.Lock()
	n := c.cache.order.Len()
	c.cache.mu.Unlock()
	if n != embedCacheEntries {
		t.Fatalf("cache holds %d entries, want %d", n, embedCacheEntries)
	}
}
