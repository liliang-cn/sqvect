package semanticrouter

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

func TestCachedEmbedderConcurrentOperations(t *testing.T) {
	embedder := NewCachedEmbedder(NewMockEmbedder(8))
	start := make(chan struct{})
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			<-start
			for i := 0; i < 20; i++ {
				query := fmt.Sprintf("worker-%d-%d", worker, i)
				vec, err := embedder.Embed(context.Background(), query)
				if err != nil || len(vec) != 8 {
					t.Errorf("Embed: len=%d err=%v", len(vec), err)
				}
				batch, err := embedder.EmbedBatch(context.Background(), []string{query, "shared"})
				if err != nil || len(batch) != 2 || len(batch[0]) != 8 || len(batch[1]) != 8 {
					t.Errorf("EmbedBatch: %+v err=%v", batch, err)
				}
				_ = embedder.CacheSize()
				embedder.ClearCache()
			}
		}(worker)
	}
	close(start)
	wg.Wait()
}
