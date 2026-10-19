package core

import (
	"context"
	"fmt"
	"testing"
)

// Both stores keep only the rows of the selected documents in a count, a
// group_by and a vector aggregate, and leave out the rows that belong to no
// document.
func TestBothStoresKeepOnlyTheSelectedDocumentsInAggregates(t *testing.T) {
	for _, s := range storesUnderTest(t) {
		t.Run(s.name, func(t *testing.T) {
			ctx := context.Background()
			for _, doc := range []string{"kept", "other"} {
				if err := s.store.CreateDocument(ctx, &Document{ID: doc, Title: doc}); err != nil {
					t.Fatalf("CreateDocument: %v", err)
				}
				for i := 0; i < 2; i++ {
					err := s.store.Upsert(ctx, &Embedding{
						ID:       fmt.Sprintf("%s-%d", doc, i),
						Vector:   []float32{1, float32(i), 0, 0},
						Content:  doc,
						DocID:    doc,
						Metadata: map[string]string{"kind": "chunk"},
					})
					if err != nil {
						t.Fatalf("Upsert: %v", err)
					}
				}
			}
			if err := s.store.Upsert(ctx, &Embedding{ID: "loose", Vector: []float32{0, 1, 0, 0}, Content: "loose", Metadata: map[string]string{"kind": "chunk"}}); err != nil {
				t.Fatalf("Upsert loose: %v", err)
			}

			count, err := s.store.Aggregate(ctx, AggregationRequest{Type: AggregationCount, OnlyDocuments: "id = ?", OnlyDocumentsArgs: []interface{}{"kept"}})
			if err != nil {
				t.Fatalf("Aggregate: %v", err)
			}
			if got := count.Results[0].Count; got != 2 {
				t.Errorf("count of kept only = %d, want its two chunks", got)
			}
			both, err := s.store.Aggregate(ctx, AggregationRequest{Type: AggregationGroupBy, GroupBy: []string{"kind"},
				OnlyDocuments: "id IN (?, ?)", OnlyDocumentsArgs: []interface{}{"kept", "other"},
				ExcludeDocuments: "id = ?", ExcludeDocumentsArgs: []interface{}{"other"}})
			if err != nil {
				t.Fatalf("Aggregate group_by: %v", err)
			}
			if len(both.Results) != 1 || both.Results[0].Count != 2 {
				t.Errorf("group_by of kept and other, other excluded = %+v, want kept's two chunks", both.Results)
			}
			medoid, err := s.store.VectorAggregate(ctx, VectorAggregateRequest{Kind: VectorMedoid, OnlyDocuments: "id = ?", OnlyDocumentsArgs: []interface{}{"other"}})
			if err != nil {
				t.Fatalf("VectorAggregate: %v", err)
			}
			if len(medoid.Groups) != 1 || medoid.Groups[0].Count != 2 {
				t.Errorf("medoid of other only = %+v, want one group of two", medoid.Groups)
			}
		})
	}
}
