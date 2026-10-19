package cortexdb

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

// BenchmarkDocumentGate measures what the expiry gate adds to one vector
// search: 10 000 documents, 100 of them expired, 50 hits to filter, every
// hit on an expired document.
func BenchmarkDocumentGate(b *testing.B) {
	db, err := Open(DefaultConfig(filepath.Join(b.TempDir(), "gate-bench.db")))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	past := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	for i := 0; i < 10000; i++ {
		doc := &core.Document{ID: fmt.Sprintf("d%05d", i), Title: "t", Content: "c"}
		if i%100 == 0 {
			doc.Metadata = map[string]any{expiresAtKey: past}
		}
		if err := db.store.CreateDocument(ctx, doc); err != nil {
			b.Fatal(err)
		}
	}
	hits := make([]core.ScoredEmbedding, 50)
	for i := range hits {
		hits[i] = core.ScoredEmbedding{Embedding: core.Embedding{ID: fmt.Sprintf("e%d", i), DocID: fmt.Sprintf("d%05d", i*100)}}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if kept := db.filterScoredEmbeddings(ctx, hits, nil); len(kept) != 0 {
			b.Fatalf("%d hits kept, want 0", len(kept))
		}
	}
}
