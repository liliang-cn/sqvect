package cortexdb

import (
	"container/list"
	"context"
	"sync"
)

// cachedEmbedder remembers the vectors of the texts it embedded, and makes
// concurrent requests for one text share a single call.
//
// A shared brain answers many agents asking the same question at once — a
// hive's queen and its workers recall on the same task text — and each recall
// embeds its query on a model that may be on another, busy machine. Those
// calls queue behind each other there: a burst of identical questions took
// seconds when one call takes a tenth of one. The same model maps the same
// text to the same vector, so the second request has nothing to learn from
// the model.
//
// Only Embed is cached. EmbedBatch is the write path's — documents and
// memories being stored, rarely repeated — and goes straight through.
type cachedEmbedder struct {
	Embedder
	cache *embedCache
}

// embedCacheEntries bounds the cache: 2048 768-dimension vectors are 6 MB.
const embedCacheEntries = 2048

type embedCache struct {
	mu       sync.Mutex
	entries  map[string]*list.Element
	order    *list.List // front is most recently used
	inflight map[string]*embedCall
}

type embedCacheEntry struct {
	text   string
	vector []float32
}

type embedCall struct {
	done   chan struct{}
	vector []float32
	err    error
}

func newCachedEmbedder(e Embedder) cachedEmbedder {
	return cachedEmbedder{Embedder: e, cache: &embedCache{
		entries:  map[string]*list.Element{},
		order:    list.New(),
		inflight: map[string]*embedCall{},
	}}
}

func (c cachedEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	cache := c.cache
	cache.mu.Lock()
	if el, ok := cache.entries[text]; ok {
		cache.order.MoveToFront(el)
		v := el.Value.(*embedCacheEntry).vector
		cache.mu.Unlock()
		return append([]float32(nil), v...), nil
	}
	if call, ok := cache.inflight[text]; ok {
		cache.mu.Unlock()
		select {
		case <-call.done:
			if call.err != nil {
				return nil, call.err
			}
			return append([]float32(nil), call.vector...), nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	call := &embedCall{done: make(chan struct{})}
	cache.inflight[text] = call
	cache.mu.Unlock()

	// The call runs on a context of its own: a request that gives up first
	// must not cancel it for the others waiting on the same text, and an
	// answer that arrives after everyone left still fills the cache for the
	// next one. It is bounded all the same, so a stuck model cannot hold an
	// entry in flight forever.
	callCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), embedCallTimeout)
	go func() {
		defer cancel()
		call.vector, call.err = c.Embedder.Embed(callCtx, text)
		cache.mu.Lock()
		delete(cache.inflight, text)
		if call.err == nil && len(call.vector) > 0 {
			if el, ok := cache.entries[text]; ok {
				cache.order.Remove(el)
			}
			cache.entries[text] = cache.order.PushFront(&embedCacheEntry{text: text, vector: call.vector})
			for cache.order.Len() > embedCacheEntries {
				oldest := cache.order.Back()
				cache.order.Remove(oldest)
				delete(cache.entries, oldest.Value.(*embedCacheEntry).text)
			}
		}
		cache.mu.Unlock()
		close(call.done)
	}()

	select {
	case <-call.done:
		if call.err != nil {
			return nil, call.err
		}
		return append([]float32(nil), call.vector...), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
