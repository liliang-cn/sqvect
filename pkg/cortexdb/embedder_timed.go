package cortexdb

import (
	"context"

	"github.com/liliang-cn/cortexdb/v2/pkg/observability"
)

// timedEmbedder adds what each embedding took to the request's stages, so a
// slow call says when the time went to the embedder — which may be a model on
// another machine, behind a relay.
type timedEmbedder struct{ Embedder }

func (e timedEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	defer observability.Time(ctx, "embed")()
	return e.Embedder.Embed(ctx, text)
}

func (e timedEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	defer observability.Time(ctx, "embed")()
	return e.Embedder.EmbedBatch(ctx, texts)
}
