package cortexdb

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
)

// plainIndex is an index SQLite can use for a lookup on its leading columns:
// no expression column and no WHERE clause.
type plainIndex struct {
	name    string
	columns []string
	unique  bool
}

// plainIndexes lists, table by table, every index of the brain whose keys are
// plain columns. An index on an expression or with a WHERE clause is left out:
// another index never stands in for it.
func plainIndexes(t *testing.T, db *sql.DB) map[string][]plainIndex {
	t.Helper()
	tables := queryStrings(t, db, `SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'`)
	out := map[string][]plainIndex{}
	for _, table := range tables {
		rows, err := db.Query(fmt.Sprintf(`PRAGMA index_list(%q)`, table))
		if err != nil {
			t.Fatalf("index_list(%s): %v", table, err)
		}
		type entry struct {
			name    string
			unique  bool
			partial bool
		}
		var entries []entry
		for rows.Next() {
			var seq int
			var name, origin string
			var unique, partial bool
			if err := rows.Scan(&seq, &name, &unique, &origin, &partial); err != nil {
				t.Fatalf("scan index_list(%s): %v", table, err)
			}
			entries = append(entries, entry{name, unique, partial})
		}
		_ = rows.Close()
		for _, e := range entries {
			if e.partial {
				continue
			}
			cols, plain := indexColumns(t, db, e.name)
			if plain {
				out[table] = append(out[table], plainIndex{e.name, cols, e.unique})
			}
		}
	}
	return out
}

// indexColumns is the key columns of an index, and false when one of them is
// an expression.
func indexColumns(t *testing.T, db *sql.DB, index string) ([]string, bool) {
	t.Helper()
	rows, err := db.Query(fmt.Sprintf(`PRAGMA index_xinfo(%q)`, index))
	if err != nil {
		t.Fatalf("index_xinfo(%s): %v", index, err)
	}
	defer func() { _ = rows.Close() }()
	var cols []string
	plain := true
	for rows.Next() {
		var seqno, cid, desc, key int
		var name sql.NullString
		var coll string
		if err := rows.Scan(&seqno, &cid, &name, &desc, &coll, &key); err != nil {
			t.Fatalf("scan index_xinfo(%s): %v", index, err)
		}
		if key == 0 {
			continue
		}
		if cid == -2 {
			plain = false
		}
		cols = append(cols, name.String)
	}
	return cols, plain
}

func queryStrings(t *testing.T, db *sql.DB, q string) []string {
	t.Helper()
	rows, err := db.Query(q)
	if err != nil {
		t.Fatalf("%s: %v", q, err)
	}
	defer func() { _ = rows.Close() }()
	var out []string
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			t.Fatalf("scan: %v", err)
		}
		out = append(out, s)
	}
	return out
}

// redundantIndexes names each non-unique index whose columns are the leading
// columns of another index on the same table: SQLite can answer every lookup
// it serves from the longer one, and every write still pays for both. Column
// names only: no index of the schema sets a collation or DESC.
func redundantIndexes(byTable map[string][]plainIndex) []string {
	var out []string
	for table, indexes := range byTable {
		for _, a := range indexes {
			if a.unique {
				continue
			}
			for _, b := range indexes {
				if a.name == b.name || len(b.columns) < len(a.columns) {
					continue
				}
				if strings.Join(b.columns[:len(a.columns)], ",") != strings.Join(a.columns, ",") {
					continue
				}
				if len(b.columns) == len(a.columns) && a.name < b.name {
					continue // two identical indexes: name the second one only
				}
				out = append(out, fmt.Sprintf("%s.%s (%s) leads %s (%s)", table, a.name,
					strings.Join(a.columns, ", "), b.name, strings.Join(b.columns, ", ")))
				break
			}
		}
	}
	return out
}

// openWrittenBrain opens a brain and writes through the paths that create
// their tables on first use (documents, chunks, entities, relations, the
// change feed), so that their indexes are there to inspect. The tables of
// pkg/agentmem are not created here; a scan of every CREATE INDEX in pkg, cmd
// and internal finds no pair of plain-column indexes there either.
func openWrittenBrain(t *testing.T, path string) *DB {
	t.Helper()
	db, err := Open(DefaultConfig(path))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	ctx := context.Background()
	if _, err := db.GraphRAGTools().IngestDocument(ctx, ToolIngestDocumentRequest{
		DocumentID: "doc-1", Content: "Ada Lovelace wrote the first program for the Analytical Engine.",
	}); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	upsertProseEntities(t, db, "doc-1", []string{"Ada Lovelace", "Analytical Engine"}, "chunk:doc-1:000")
	if _, err := db.GraphRAGTools().UpsertRelations(ctx, ToolUpsertRelationsRequest{
		Relations: []ToolRelationInput{{From: "Ada Lovelace", To: "Analytical Engine", Type: "programmed"}},
	}); err != nil {
		t.Fatalf("relations: %v", err)
	}
	return db
}

func TestNoIndexOfABrainIsTheLeadingPartOfAnother(t *testing.T) {
	db := openWrittenBrain(t, filepath.Join(t.TempDir(), "brain.db"))
	defer func() { _ = db.Close() }()
	byTable := plainIndexes(t, db.SQL())
	if len(byTable["graph_edges"]) == 0 {
		t.Fatal("no index found on graph_edges: the inspection reads nothing")
	}
	if got := redundantIndexes(byTable); len(got) > 0 {
		t.Fatalf("indexes another index already serves:\n%s", strings.Join(got, "\n"))
	}
}

func TestABrainWrittenWithTheSingleColumnEdgeIndexesLosesThemOnOpen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.db")
	db := openWrittenBrain(t, path)
	for _, stmt := range []string{
		`CREATE INDEX IF NOT EXISTS idx_edges_from ON graph_edges(from_node_id)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_to ON graph_edges(to_node_id)`,
	} {
		if _, err := db.SQL().Exec(stmt); err != nil {
			t.Fatalf("%s: %v", stmt, err)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	db, err := Open(DefaultConfig(path))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = db.Close() }()
	left := queryStrings(t, db.SQL(), `SELECT name FROM sqlite_master WHERE type = 'index' AND name IN ('idx_edges_from', 'idx_edges_to')`)
	if len(left) > 0 {
		t.Fatalf("still there after reopening: %v", left)
	}
	edges := queryStrings(t, db.SQL(), `SELECT id FROM graph_edges WHERE edge_type = 'programmed'`)
	if len(edges) != 1 {
		t.Fatalf("the relation written before the migration: got %v", edges)
	}
}

// queryPlan is the EXPLAIN QUERY PLAN of q, one detail per line.
func queryPlan(t *testing.T, db *sql.DB, q string, args ...any) string {
	t.Helper()
	rows, err := db.Query("EXPLAIN QUERY PLAN "+q, args...)
	if err != nil {
		t.Fatalf("explain %s: %v", q, err)
	}
	defer func() { _ = rows.Close() }()
	var lines []string
	for rows.Next() {
		var id, parent, notused int
		var detail string
		if err := rows.Scan(&id, &parent, &notused, &detail); err != nil {
			t.Fatalf("scan plan: %v", err)
		}
		lines = append(lines, detail)
	}
	return strings.Join(lines, "\n")
}

// Without idx_edges_from and idx_edges_to, every lookup of graph_edges by one
// endpoint must still seek an index: idx_edges_composite on the from side,
// idx_edges_walk_to on the to side. The shapes are the engine's own: a read
// that the walk index does not cover (GetEdges), the endpoint rewrite of an
// entity merge, the foreign-key cascade of a node delete, and the typed edges
// of one node that a supersede reads.
//
// It names the exact indexes, so a legitimate change of index, or SQLite
// choosing another index with the same leading column, fails it without any
// SCAN: read the plan before relaxing it. The plan fragments are those of
// SQLite 3.36 and later (modernc embeds 3.50.3).
func TestEveryLookupOfAnEdgeByOneEndpointSeeksAnIndex(t *testing.T) {
	db := openWrittenBrain(t, filepath.Join(t.TempDir(), "brain.db"))
	defer func() { _ = db.Close() }()
	cases := []struct {
		name, query string
		args        []any
		want        []string
	}{
		{"read by from", `SELECT id, to_node_id, edge_type, weight, properties FROM graph_edges WHERE from_node_id = ? ORDER BY id`,
			[]any{"n"}, []string{"SEARCH graph_edges USING INDEX idx_edges_composite (from_node_id=?)"}},
		{"read by to", `SELECT id, from_node_id, edge_type, weight, properties FROM graph_edges WHERE to_node_id = ? ORDER BY id`,
			[]any{"n"}, []string{"SEARCH graph_edges USING INDEX idx_edges_walk_to (to_node_id=?)"}},
		{"merge rewrites from", `UPDATE graph_edges SET from_node_id = ? WHERE from_node_id = ?`,
			[]any{"a", "b"}, []string{"(from_node_id=?)"}},
		{"merge rewrites to", `UPDATE graph_edges SET to_node_id = ? WHERE to_node_id = ?`,
			[]any{"a", "b"}, []string{"(to_node_id=?)"}},
		{"node delete, foreign-key cascade", `DELETE FROM graph_nodes WHERE id = ?`,
			[]any{"n"}, []string{"graph_edges USING COVERING INDEX idx_edges_walk_to (to_node_id=?)",
				"graph_edges USING COVERING INDEX idx_edges_composite (from_node_id=?)"}},
		{"supersede, typed edges from one node (graphflow/temporal.go)", `SELECT id FROM graph_edges WHERE from_node_id = ? AND edge_type = ?
			AND json_extract(properties, '$.valid_from') IS NOT NULL AND json_extract(properties, '$.valid_to') IS NULL`,
			[]any{"n", "mentions"}, []string{"idx_edges_composite (from_node_id=? AND edge_type=?)"}},
	}
	for _, c := range cases {
		plan := queryPlan(t, db.SQL(), c.query, c.args...)
		if strings.Contains(plan, "SCAN graph_edges") {
			t.Errorf("%s scans graph_edges:\n%s", c.name, plan)
		}
		for _, w := range c.want {
			if !strings.Contains(plan, w) {
				t.Errorf("%s: plan lacks %q:\n%s", c.name, w, plan)
			}
		}
	}
}
