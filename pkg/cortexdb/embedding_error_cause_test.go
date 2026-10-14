package cortexdb

import (
	"context"
	"errors"
	"testing"
)

type embeddingProviderFailure struct{}

func (*embeddingProviderFailure) Error() string { return "provider temporarily unavailable" }

type failingCauseEmbedder struct{ err error }

func (e failingCauseEmbedder) Embed(context.Context, string) ([]float32, error) { return nil, e.err }
func (e failingCauseEmbedder) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, e.err
}
func (failingCauseEmbedder) Dim() int { return 2 }

func TestTextOperationsPreserveEmbeddingFailureCause(t *testing.T) {
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded, &embeddingProviderFailure{}} {
		t.Run(cause.Error(), func(t *testing.T) {
			db := &DB{embedder: failingCauseEmbedder{cause}}
			ctx := context.Background()
			for name, operation := range map[string]func() error{
				"quick add":     func() error { _, err := db.Quick().AddText(ctx, "text", nil); return err },
				"insert":        func() error { return db.InsertText(ctx, "id", "text", nil) },
				"insert batch":  func() error { return db.InsertTextBatch(ctx, map[string]string{"id": "text"}, nil) },
				"search":        func() error { _, err := db.SearchTextInCollection(ctx, "collection", "query", 2); return err },
				"hybrid search": func() error { _, err := db.HybridSearchText(ctx, "query", 2); return err },
				"range search":  func() error { _, err := db.RangeSearchText(ctx, "query", VectorRangeOptions{Radius: 1}); return err },
			} {
				t.Run(name, func(t *testing.T) {
					err := operation()
					if !errors.Is(err, ErrEmbeddingFailed) {
						t.Fatalf("embedding failure sentinel lost: %v", err)
					}
					if !errors.Is(err, cause) {
						t.Errorf("embedding error cause lost: got %v, want errors.Is(..., %v)", err, cause)
					}
					if _, ok := cause.(*embeddingProviderFailure); ok {
						var provider *embeddingProviderFailure
						if !errors.As(err, &provider) {
							t.Error("typed provider error lost")
						}
					}
				})
			}
		})
	}
}
