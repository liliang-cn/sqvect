package cortexdb

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// The graph lane of cortex_query walks entity nodes, which are shared by
// every collection. A chunk it reaches must still belong to the collection
// queried: with Collection set, the lane leaves the other collections' chunks
// out; without it, it searches the whole brain as before; when the chunks'
// collection cannot be read, the query fails rather than serve them.
func TestTheGraphLaneOfAQueryStaysInTheCollection(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "query-scope.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	tools := db.GraphRAGTools()
	for id, collection := range map[string]string{"n": "notes", "s": "secrets"} {
		ingested, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: id, Collection: collection, Content: "Orpheus keeps the " + collection + " ledger."})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tools.UpsertEntities(ctx, ToolUpsertEntitiesRequest{DocumentID: id, Entities: []ToolEntityInput{{Name: "Orpheus", ChunkIDs: ingested.ChunkNodeIDs}}}); err != nil {
			t.Fatal(err)
		}
	}
	query := func(collection string) ([]string, error) {
		resp, err := db.Query(ctx, QueryRequest{
			Collection: collection, Limit: 10,
			Prefetch: []QueryPrefetch{{Name: "graph", Kind: QueryPrefetchGraph, EntityNames: []string{"Orpheus"}, Weight: 1, Limit: 10, MaxHops: 1}},
		})
		if err != nil {
			return nil, err
		}
		docs := map[string]bool{}
		for _, r := range resp.Results {
			docs[r.DocID] = true
		}
		out := make([]string, 0, len(docs))
		for d := range docs {
			out = append(out, d)
		}
		return out, nil
	}

	docs, err := query("")
	if err != nil || len(docs) != 2 {
		t.Fatalf("without a collection the lane searches the whole brain: %v %v", docs, err)
	}
	docs, err = query("notes")
	if err != nil || strings.Join(docs, ",") != "n" {
		t.Fatalf("with collection notes the lane served %v (%v)", docs, err)
	}

	if _, err := db.SQL().Exec(`ALTER TABLE collections RENAME COLUMN name TO name_was`); err != nil {
		t.Fatal(err)
	}
	if _, err := query("notes"); err == nil || !strings.Contains(err.Error(), "load walked chunks") {
		t.Fatalf("an unreadable collection must fail the query, got %v", err)
	}
}
