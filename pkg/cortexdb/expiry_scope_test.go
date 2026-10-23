package cortexdb

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

func TestTheDocumentCollectionsQueryReadsTheCollectionOfEveryChunk(t *testing.T) {
	query, args := documentCollectionsQuery([]string{"a1", "b1"})
	for _, part := range []string{"SELECT e.doc_id, COALESCE(c.name, '') FROM embeddings e", "LEFT JOIN collections c ON c.id = e.collection_id", "WHERE e.doc_id IN (?,?)"} {
		if !strings.Contains(query, part) {
			t.Fatalf("query %q lacks %q", query, part)
		}
	}
	if !reflect.DeepEqual(args, []any{"a1", "b1"}) {
		t.Fatalf("args = %v, want the two ids", args)
	}
}

func TestADocumentIsOutOfScopeUnlessEveryChunkIsInAPermittedCollection(t *testing.T) {
	permits := func(c string) bool { return c == "alpha" }
	collections := map[string][]string{
		"own":     {"alpha", "alpha"},
		"foreign": {"beta"},
		"mixed":   {"alpha", "beta"},
		"default": {""},
	}
	got := outOfScope([]string{"own", "foreign", "mixed", "default", "chunkless"}, collections, permits)
	want := map[string]struct{}{"foreign": {}, "mixed": {}, "default": {}, "chunkless": {}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("out of scope = %v, want %v", got, want)
	}
}

// A scope with no knowledge rule reads nothing: the brain may be closed and
// nothing is hidden.
func TestAScopeWithNoKnowledgeRuleHidesNoDocument(t *testing.T) {
	db := scopedGraphBrain(t)
	_ = db.Close()
	for label, ctx := range map[string]context.Context{
		"no scope":          context.Background(),
		"no knowledge rule": WithCallerScope(context.Background(), CallerScope{Key: "k", Memory: func(MemoryBucket) bool { return false }}),
	} {
		hidden, err := db.outOfScopeDocuments(ctx, []string{"a1", "b1"})
		if err != nil || len(hidden) != 0 {
			t.Fatalf("%s: outOfScopeDocuments = %v, %v, want nothing read and nothing hidden", label, hidden, err)
		}
	}
	scoped := WithCallerScope(context.Background(), alphaScope())
	if hidden, err := db.outOfScopeDocuments(scoped, nil); err != nil || len(hidden) != 0 {
		t.Fatalf("no ids: outOfScopeDocuments = %v, %v, want nothing read", hidden, err)
	}
	if _, err := db.outOfScopeDocuments(scoped, []string{"a1"}); err == nil {
		t.Fatal("a closed brain under a knowledge rule reported no error")
	}
}
