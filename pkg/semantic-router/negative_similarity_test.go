package semanticrouter

import (
	"context"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

type fixedScoreEmbedder struct{ vectors map[string][]float32 }

func (e fixedScoreEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	return e.vectors[text], nil
}
func (e fixedScoreEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		out[i], _ = e.Embed(ctx, text)
	}
	return out, nil
}
func (e fixedScoreEmbedder) Dimensions() int { return 1 }

func TestRouterSelectsNonpositiveSimilarity(t *testing.T) {
	for _, tc := range []struct {
		name            string
		fn              core.SimilarityFunc
		near, far       []float32
		threshold, want float64
	}{
		{"exact Euclidean", core.EuclideanDist, []float32{1}, []float32{5}, 0, 0},
		{"negative Euclidean", core.EuclideanDist, []float32{2}, []float32{5}, -1.5, -1},
		{"negative dot product", core.DotProduct, []float32{-1}, []float32{-5}, -1.5, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, err := NewRouter(fixedScoreEmbedder{map[string][]float32{"query": {1}, "near": tc.near, "far": tc.far}}, WithSimilarityFunc(tc.fn), WithThreshold(tc.threshold))
			if err != nil {
				t.Fatal(err)
			}
			if err := router.AddBatch([]*Route{{Name: "far", Utterances: []string{"far"}}, {Name: "near", Utterances: []string{"near"}}}); err != nil {
				t.Fatal(err)
			}
			got, err := router.Route(context.Background(), "query")
			if err != nil {
				t.Fatal(err)
			}
			if got.RouteName != "near" || !got.Matched || got.Score != tc.want {
				t.Fatalf("got %+v; want near matched at %v", got, tc.want)
			}
		})
	}
}

func TestRouterWithoutCandidatesKeepsZeroScore(t *testing.T) {
	router, _ := NewRouter(fixedScoreEmbedder{map[string][]float32{"query": {1}}}, WithSimilarityFunc(core.EuclideanDist), WithThreshold(-1))
	got, err := router.Route(context.Background(), "query")
	if err != nil {
		t.Fatal(err)
	}
	if got.Score != 0 || got.Matched || got.RouteName != "" {
		t.Fatalf("empty router: %+v", got)
	}
}
