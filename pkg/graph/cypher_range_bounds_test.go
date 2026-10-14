package graph

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func TestCypherRangeStopsAtIntegerBounds(t *testing.T) {
	_, g, cleanup := setupTestGraph(t)
	defer cleanup()
	cases := []struct {
		name           string
		from, to, step int64
		want           []any
	}{
		{"max singleton", math.MaxInt64, math.MaxInt64, 1, []any{int64(math.MaxInt64)}},
		{"min singleton", math.MinInt64, math.MinInt64, -1, []any{int64(math.MinInt64)}},
		{"max pair", math.MaxInt64 - 1, math.MaxInt64, 1, []any{int64(math.MaxInt64 - 1), int64(math.MaxInt64)}},
		{"min pair", math.MinInt64 + 1, math.MinInt64, -1, []any{int64(math.MinInt64 + 1), int64(math.MinInt64)}},
		{"step overshoots max", math.MaxInt64 - 1, math.MaxInt64, 3, []any{int64(math.MaxInt64 - 1)}},
		{"step overshoots min", math.MinInt64 + 1, math.MinInt64, -3, []any{int64(math.MinInt64 + 1)}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res, err := g.QueryCypher(context.Background(), CypherRequest{Query: "RETURN range($from, $to, $step) AS xs", Params: map[string]any{"from": c.from, "to": c.to, "step": c.step}})
			if err != nil {
				t.Fatal(err)
			}
			if len(res.Rows) != 1 || !reflect.DeepEqual(res.Rows[0][0], c.want) {
				t.Fatalf("got %v, want %v", res.Rows, c.want)
			}
		})
	}
}
