package semanticrouter

import (
	"context"
	"errors"
	"testing"
)

type retryEmbedder struct {
	failUtterance  bool
	utteranceCalls int
}

func (e *retryEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if text == "utterance" {
		e.utteranceCalls++
		if e.failUtterance {
			return nil, errors.New("temporary embedding failure")
		}
	}
	return []float32{1}, nil
}
func (e *retryEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		var err error
		out[i], err = e.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
func (e *retryEmbedder) Dimensions() int { return 1 }

func TestRouterRetriesFailedUtteranceEmbedding(t *testing.T) {
	embedder := &retryEmbedder{failUtterance: true}
	router, _ := NewRouter(embedder)
	if err := router.Add(&Route{Name: "retry", Utterances: []string{"utterance"}}); err != nil {
		t.Fatal(err)
	}
	first, err := router.Route(context.Background(), "query")
	if err != nil || first.Matched {
		t.Fatalf("failed utterance should not match: %+v, %v", first, err)
	}
	embedder.failUtterance = false
	second, err := router.Route(context.Background(), "query")
	if err != nil || !second.Matched || second.RouteName != "retry" {
		t.Fatalf("route did not recover: %+v, %v", second, err)
	}
	calls := embedder.utteranceCalls
	embedder.failUtterance = true
	third, err := router.Route(context.Background(), "query")
	if err != nil || !third.Matched || embedder.utteranceCalls != calls {
		t.Fatalf("successful embedding cache not retained: %+v, %v, calls %d -> %d", third, err, calls, embedder.utteranceCalls)
	}
}
