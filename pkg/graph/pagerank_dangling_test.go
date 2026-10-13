package graph

import (
	"context"
	"math"
	"testing"
)

func TestPageRankRedistributesDanglingMass(t *testing.T) {
	for _, edge := range []bool{false, true} {
		t.Run(map[bool]string{false: "isolated", true: "one edge"}[edge], func(t *testing.T) {
			_, g, cleanup := setupTestGraph(t)
			defer cleanup()
			ctx := context.Background()
			for _, id := range []string{"a", "b"} {
				if err := g.UpsertNode(ctx, &GraphNode{ID: id, NodeType: "entity"}); err != nil {
					t.Fatal(err)
				}
			}
			if edge {
				if err := g.UpsertEdge(ctx, &GraphEdge{ID: "a-b", FromNodeID: "a", ToNodeID: "b", EdgeType: "link", Weight: 1}); err != nil {
					t.Fatal(err)
				}
			}
			for _, damping := range []float64{0.5, 0.85} {
				results, err := g.PageRank(ctx, 100, damping)
				if err != nil {
					t.Fatal(err)
				}
				scores := map[string]float64{}
				sum := 0.0
				for _, res := range results {
					scores[res.NodeID] = res.Score
					sum += res.Score
				}
				if math.Abs(sum-1) > 1e-10 {
					t.Fatalf("PageRank mass=%g, want 1: %v", sum, scores)
				}
				wantA := 0.5
				if edge {
					wantA = 1 / (2 + damping)
				} // Stationary solution of the 2x2 Google matrix.
				if math.Abs(scores["a"]-wantA) > 1e-5 || math.Abs(scores["b"]-(1-wantA)) > 1e-5 {
					t.Fatalf("got %v, want a=%g b=%g", scores, wantA, 1-wantA)
				}
			}
		})
	}
}
