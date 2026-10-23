package graph

import (
	"context"
	"testing"
)

// edgeIndexes is the names of the indexes on graph_edges, in this backend's
// catalogue.
func edgeIndexes(t *testing.T, b backend) map[string]bool {
	t.Helper()
	q := `SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'graph_edges'`
	if b.name == "postgres" {
		q = `SELECT indexname FROM pg_indexes WHERE tablename = 'graph_edges' AND schemaname = current_schema()`
	}
	rows, err := b.store.db.QueryContext(context.Background(), q)
	if err != nil {
		t.Fatalf("list indexes: %v", err)
	}
	defer func() { _ = rows.Close() }()
	out := map[string]bool{}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatalf("scan: %v", err)
		}
		out[name] = true
	}
	return out
}

func TestTheSingleColumnEdgeIndexesAreGoneOnBothBackends(t *testing.T) {
	for _, b := range backends(t) {
		t.Run(b.name, func(t *testing.T) {
			ctx := context.Background()
			if err := b.store.createGraphSchema(ctx); err != nil {
				t.Fatalf("schema: %v", err)
			}
			check := func(when string) {
				t.Helper()
				got := edgeIndexes(t, b)
				for _, name := range []string{"idx_edges_from", "idx_edges_to"} {
					if got[name] {
						t.Errorf("%s: %s is still there", when, name)
					}
				}
				for _, name := range []string{"idx_edges_composite", "idx_edges_walk_from", "idx_edges_walk_to"} {
					if !got[name] {
						t.Errorf("%s: %s is missing", when, name)
					}
				}
			}
			check("fresh schema")

			// A brain written by an earlier release carries both indexes.
			for _, stmt := range []string{
				`CREATE INDEX IF NOT EXISTS idx_edges_from ON graph_edges(from_node_id)`,
				`CREATE INDEX IF NOT EXISTS idx_edges_to ON graph_edges(to_node_id)`,
			} {
				if _, err := b.store.db.ExecContext(ctx, stmt); err != nil {
					t.Fatalf("%s: %v", stmt, err)
				}
			}
			if err := b.store.createGraphSchema(ctx); err != nil {
				t.Fatalf("schema over an earlier brain: %v", err)
			}
			check("schema over an earlier brain")
		})
	}
}
