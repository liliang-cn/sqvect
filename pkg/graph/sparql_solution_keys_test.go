package graph

import (
	"context"
	"testing"
)

func TestSPARQLDistinctDoesNotMergeDelimiterBearingValues(t *testing.T) {
	_, g, cleanup := setupTestGraph(t)
	defer cleanup()
	res, err := g.ExecuteSPARQL(context.Background(), `SELECT DISTINCT ?x ?y WHERE {
 VALUES (?x ?y) { ("a||;y=literal|b" "c") ("a" "b||;y=literal|c") ("a" "b||;y=literal|c") }
 }`)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Bindings) != 2 {
		t.Fatalf("got %d distinct rows, want 2: %v", len(res.Bindings), res.Bindings)
	}
}

func TestSPARQLGroupingDoesNotMergeDelimiterBearingValues(t *testing.T) {
	_, g, cleanup := setupTestGraph(t)
	defer cleanup()
	res, err := g.ExecuteSPARQL(context.Background(), `SELECT ?x ?y (COUNT(*) AS ?n) WHERE {
 VALUES (?x ?y) { ("a||;y=literal|b" "c") ("a" "b||;y=literal|c") ("a" "b||;y=literal|c") }
 } GROUP BY ?x ?y`)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Bindings) != 2 {
		t.Fatalf("got %d groups, want 2: %v", len(res.Bindings), res.Bindings)
	}
	counts := map[string]string{}
	for _, row := range res.Bindings {
		counts[row["x"].Value] = row["n"].Value
	}
	if counts["a||;y=literal|b"] != "1" || counts["a"] != "2" {
		t.Fatalf("incorrect group counts: %v", counts)
	}
}
