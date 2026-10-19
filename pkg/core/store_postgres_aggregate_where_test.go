package core

import (
	"reflect"
	"testing"
)

// The PostgreSQL store renders the excluded documents as one subquery, each
// ? of the condition bound to the next $n after the arguments before it.
// Rendering needs no server.
func TestThePostgresAggregateWhereExcludesDocumentsByOneSubquery(t *testing.T) {
	args := &pgArgs{}
	where := pgAggregateWhere(AggregationRequest{
		Collection:           "c",
		ExcludeDocuments:     "kind = ? AND rank < ?",
		ExcludeDocumentsArgs: []interface{}{"gone", 5},
	}, args)

	want := []string{
		"collection_id = (SELECT id FROM collections WHERE name = $1)",
		"(doc_id IS NULL OR doc_id NOT IN (SELECT id FROM documents WHERE kind = $2 AND rank < $3))",
	}
	if !reflect.DeepEqual(where, want) {
		t.Fatalf("where = %q, want %q", where, want)
	}
	if !reflect.DeepEqual(args.vals, []any{"c", "gone", 5}) {
		t.Fatalf("args = %v, want [c gone 5]", args.vals)
	}

	if where := pgAggregateWhere(AggregationRequest{}, &pgArgs{}); len(where) != 0 {
		t.Fatalf("no exclusion rendered %q", where)
	}
}

// A condition that keeps only some documents renders as its own subquery,
// after the exclusion, and its ? take the numbers that follow.
func TestThePostgresAggregateWhereKeepsOnlyTheSelectedDocuments(t *testing.T) {
	args := &pgArgs{}
	where := pgAggregateWhere(AggregationRequest{
		ExcludeDocuments:     "kind = ?",
		ExcludeDocumentsArgs: []interface{}{"gone"},
		OnlyDocuments:        "status = ? OR status = ?",
		OnlyDocumentsArgs:    []interface{}{"proposed", "rejected"},
	}, args)

	want := []string{
		"(doc_id IS NULL OR doc_id NOT IN (SELECT id FROM documents WHERE kind = $1))",
		"doc_id IN (SELECT id FROM documents WHERE status = $2 OR status = $3)",
	}
	if !reflect.DeepEqual(where, want) {
		t.Fatalf("where = %q, want %q", where, want)
	}
	if !reflect.DeepEqual(args.vals, []any{"gone", "proposed", "rejected"}) {
		t.Fatalf("args = %v, want [gone proposed rejected]", args.vals)
	}
}
