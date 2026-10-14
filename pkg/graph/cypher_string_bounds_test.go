package graph

import (
	"context"
	"fmt"
	"math"
	"testing"
)

func TestCypherStringPositionsHaveBoundedIntegerArguments(t *testing.T) {
	_, g, cleanup := setupTestGraph(t)
	defer cleanup()
	for _, query := range []string{
		"RETURN left('abc', $n)", "RETURN right('abc', $n)",
		"RETURN substring('abc', $n)", "RETURN substring('abc', 1, $n)",
	} {
		for _, n := range []int64{math.MaxInt32 + 1, math.MaxInt64} {
			t.Run(fmt.Sprintf("%s/%d", query, n), func(t *testing.T) {
				defer func() {
					if p := recover(); p != nil {
						t.Fatalf("query panicked instead of returning an argument error: %v", p)
					}
				}()
				_, err := g.QueryCypher(context.Background(), CypherRequest{Query: query, Params: map[string]any{"n": n}})
				if err == nil {
					t.Fatal("out-of-range string position was accepted")
				}
			})
		}
	}
	for _, query := range []string{"RETURN left('abc', $n)", "RETURN right('abc', $n)", "RETURN substring('abc', 0, $n)"} {
		res, err := g.QueryCypher(context.Background(), CypherRequest{Query: query, Params: map[string]any{"n": int64(math.MaxInt32)}})
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Rows) != 1 || res.Rows[0][0] != "abc" {
			t.Fatalf("boundary length: %v", res.Rows)
		}
	}
}
