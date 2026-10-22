package sqldialect_test

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/sqldialect"

	_ "modernc.org/sqlite"
)

// JSONArrayContains has to survive a row whose JSON column holds no JSON.
//
// `properties` is a TEXT column and a node written without any carries the
// empty string — which is the case every other helper in this file guards
// against, and the reason JSONTextGuarded exists at all. The containment
// helper read the column unguarded, and SQLite's json_extract(”) raises
// "malformed JSON" over the whole statement rather than yielding a miss for
// that one row.
//
// The caller is document deletion: pkg/cortexdb's deleteDocumentGraphTx finds
// the entities that cite a document with this predicate, and its own comment
// states that every JSON predicate there is guarded because "edges and nodes
// written without properties carry an empty string, and reading a JSON field
// out of one is an error rather than a miss". One such node anywhere in
// graph_nodes failed every document deletion in the database.
func TestJSONArrayContainsSurvivesAColumnWithoutJSON(t *testing.T) {
	db, err := sql.Open("sqlite", filepath.Join(t.TempDir(), "guard.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer func() { _ = db.Close() }()

	if _, err := db.Exec(`CREATE TABLE graph_nodes (id TEXT PRIMARY KEY, properties TEXT)`); err != nil {
		t.Fatalf("create: %v", err)
	}
	for _, row := range []struct{ id, properties string }{
		{"cites", `{"source_document_ids":["doc-1","doc-2"]}`},
		{"no-properties", ""},               // written without properties
		{"scalar-properties", `"not here"`}, // valid JSON, not an object
		{"other-document", `{"source_document_ids":["doc-9"]}`},
	} {
		if _, err := db.Exec(`INSERT INTO graph_nodes (id, properties) VALUES (?, ?)`, row.id, row.properties); err != nil {
			t.Fatalf("insert %s: %v", row.id, err)
		}
	}

	query := `SELECT id FROM graph_nodes WHERE ` +
		sqldialect.For(sqldialect.SQLite).JSONArrayContains("properties", "source_document_ids")
	rows, err := db.Query(query, "doc-1")
	if err != nil {
		t.Fatalf("a row holding no JSON failed the whole statement: %v\nquery: %s", err, query)
	}
	defer func() { _ = rows.Close() }()

	var got []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatalf("scan: %v", err)
		}
		got = append(got, id)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("a row holding no JSON failed the scan: %v", err)
	}
	if len(got) != 1 || got[0] != "cites" {
		t.Errorf("matched %v, want only [cites]", got)
	}
}
