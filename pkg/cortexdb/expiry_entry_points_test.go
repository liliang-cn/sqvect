package cortexdb

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

// gatedBrain holds three expired documents and one live one, all with the
// same text, so every search ranks them together and an expired document
// can take the one place a caller asks for.
func gatedBrain(t *testing.T) (*DB, context.Context, string) {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "entry.db")), WithEmbedder(NewMockEmbedder(8)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	content := "The zebra ledger reconciles the quarterly accounts."
	past := pastInstant()
	for _, id := range []string{"a-old", "b-old", "c-old"} {
		if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: id, Content: content, ExpiresAt: strPtr(past)}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "z-live", Content: content}); err != nil {
		t.Fatal(err)
	}
	return db, ctx, content
}

func docsOf(hits []core.ScoredEmbedding) string {
	var ids []string
	for _, h := range hits {
		ids = append(ids, h.DocID)
	}
	return strings.Join(ids, ",")
}

// Every exported search of the facade, which the typed RPCs (SearchText,
// HybridSearchText) and library callers reach without a tool, leaves an
// expired document out, and a hidden hit never takes the place of a
// visible one: one asked for, one live document back.
func TestEveryExportedSearchLeavesExpiredDocumentsOutAndFillsTheBudget(t *testing.T) {
	db, ctx, content := gatedBrain(t)
	vector, err := db.embedder.Embed(ctx, content)
	if err != nil {
		t.Fatal(err)
	}
	for name, search := range map[string]func() ([]core.ScoredEmbedding, error){
		"SearchText":             func() ([]core.ScoredEmbedding, error) { return db.SearchText(ctx, "zebra ledger", 1) },
		"SearchTextInCollection": func() ([]core.ScoredEmbedding, error) { return db.SearchTextInCollection(ctx, "", "zebra ledger", 1) },
		"HybridSearchText":       func() ([]core.ScoredEmbedding, error) { return db.HybridSearchText(ctx, "zebra ledger", 1) },
		"HybridSearchTextWithOptions": func() ([]core.ScoredEmbedding, error) {
			return db.HybridSearchTextWithOptions(ctx, "zebra ledger", TextSearchOptions{TopK: 1})
		},
		"SearchTextOnly": func() ([]core.ScoredEmbedding, error) {
			return db.SearchTextOnly(ctx, "zebra ledger", TextSearchOptions{TopK: 1})
		},
		"RangeSearchVector": func() ([]core.ScoredEmbedding, error) {
			return db.RangeSearchVector(ctx, vector, VectorRangeOptions{Radius: 0.5, MaxResults: 1})
		},
		"RangeSearchText": func() ([]core.ScoredEmbedding, error) {
			return db.RangeSearchText(ctx, "zebra ledger", VectorRangeOptions{Radius: 2, MaxResults: 1})
		},
		"Quick.Search": func() ([]core.ScoredEmbedding, error) { return db.Quick().Search(ctx, vector, 1) },
		"SearchWithFacets": func() ([]core.ScoredEmbedding, error) {
			hits, _, err := db.SearchWithFacets(ctx, vector, core.FacetedSearchOptions{SearchOptions: core.SearchOptions{TopK: 1}})
			return hits, err
		},
	} {
		hits, err := search()
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if got := docsOf(hits); got != "z-live" {
			t.Errorf("%s returned %q, want the live document alone", name, got)
		}
	}
}

// search_vector_range says truncated only when visible rows were left out:
// three expired rows inside the radius do not make one live row a
// truncated answer.
func TestSearchVectorRangeCountsOnlyVisibleRowsForTruncated(t *testing.T) {
	db, ctx, _ := gatedBrain(t)
	resp, err := db.GraphRAGTools().SearchVectorRange(ctx, ToolSearchVectorRangeRequest{Query: "zebra ledger", Radius: 2, MaxResults: 1})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Count != 1 || resp.Matches[0].DocumentID != "z-live" || resp.Truncated {
		t.Fatalf("got %d matches (%+v), truncated=%v; want the live row, not truncated", resp.Count, resp.Matches, resp.Truncated)
	}
}

// The aggregates exclude expired documents by one condition on the
// documents table, not by a list of their ids: past 32 766 unswept expired
// documents (SQLite's bound parameters) a list fails the whole aggregate.
func TestTheAggregatesHoldPastTheBoundParameterLimit(t *testing.T) {
	db, ctx, _ := gatedBrain(t)
	if _, err := db.SQL().ExecContext(ctx, `
		WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 33000)
		INSERT INTO documents (id, title, content, metadata)
		SELECT 'bulk-' || i, 'bulk', 'bulk', '{"_expires_at":"2001-01-01T00:00:00Z"}' FROM n`); err != nil {
		t.Fatal(err)
	}
	count, err := db.Aggregate(ctx, core.AggregationRequest{Type: core.AggregationCount})
	if err != nil {
		t.Fatalf("aggregate_metadata past the parameter limit: %v", err)
	}
	if got := count.Results[0].Count; got != 1 {
		t.Errorf("count = %d, want the live document's one chunk", got)
	}
	if _, err := db.VectorAggregate(ctx, core.VectorAggregateRequest{Kind: core.VectorMedoid}); err != nil {
		t.Fatalf("representative_records past the parameter limit: %v", err)
	}
}

// The expiry lives on the record alone: the gate reads the documents table,
// so a copy on a chunk or on the document node would only go stale when
// knowledge_update changes the expiry and nothing else.
func TestTheExpiryIsKeptOnTheRecordOnly(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "record-only")
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "i", Content: "The zebra ledger.", ExpiresAt: strPtr(futureInstant())}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "k", Content: "The zebra ledger.", ExpiresAt: strPtr(futureInstant())}); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"i", "k"} {
		if doc, err := db.store.GetDocument(ctx, id); err != nil || doc.Metadata[expiresAtKey] == nil {
			t.Fatalf("%s: the record lost its expiry: %v %v", id, doc, err)
		}
		chunks, err := db.store.GetByDocID(ctx, id)
		if err != nil || len(chunks) == 0 {
			t.Fatalf("%s: chunks %v %v", id, chunks, err)
		}
		for _, c := range chunks {
			if _, ok := c.Metadata[expiresAtKey]; ok {
				t.Errorf("%s: chunk %s carries a copy of the expiry", id, c.ID)
			}
		}
		node, err := db.graph.GetNode(ctx, graphDocumentNodeID(id))
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := node.Properties[expiresAtKey]; ok {
			t.Errorf("%s: the document node carries a copy of the expiry", id)
		}
	}
}

// gatedSearch asks again while hidden rows leave fewer than topK, keeps no
// more than topK of what the larger answer brings, and reports a search
// that fails.
func TestGatedSearchTrimsToTheBudgetAndReportsAFailure(t *testing.T) {
	db, ctx, _ := gatedBrain(t)
	hit := func(id, doc string) core.ScoredEmbedding {
		return core.ScoredEmbedding{Embedding: core.Embedding{ID: id, DocID: doc}}
	}
	var asked []int
	got, err := db.gatedSearch(ctx, 1, func(k int) ([]core.ScoredEmbedding, error) {
		asked = append(asked, k)
		if k == 1 {
			return []core.ScoredEmbedding{hit("e1", "a-old")}, nil
		}
		return []core.ScoredEmbedding{hit("e1", "a-old"), hit("e2", "z-live"), hit("e3", ""), hit("e4", "")}, nil
	})
	if err != nil || len(got) != 1 || got[0].ID != "e2" || fmt.Sprint(asked) != "[1 4]" {
		t.Fatalf("got %+v, %v after asking %v; want e2 alone after asking 1 then 4", got, err, asked)
	}
	boom := errors.New("store down")
	if _, err := db.gatedSearch(ctx, 1, func(int) ([]core.ScoredEmbedding, error) { return nil, boom }); !errors.Is(err, boom) {
		t.Fatalf("a failing search: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := db.SearchWithFacets(ctx, []float32{1, 0, 0, 0, 0, 0, 0, 0}, core.FacetedSearchOptions{SearchOptions: core.SearchOptions{TopK: 1}}); err == nil {
		t.Fatal("a faceted search on a closed brain reported no error")
	}
}
