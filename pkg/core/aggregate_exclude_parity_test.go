package core

import (
	"context"
	"fmt"
	"testing"
)

// Both stores leave the rows of the excluded documents out of a count and
// of a vector aggregate, and keep the rows that belong to no document.
func TestBothStoresLeaveExcludedDocumentsOutOfAggregates(t *testing.T) {
	for _, s := range storesUnderTest(t) {
		t.Run(s.name, func(t *testing.T) {
			ctx := context.Background()
			for _, doc := range []string{"hidden", "shown"} {
				if err := s.store.CreateDocument(ctx, &Document{ID: doc, Title: doc}); err != nil {
					t.Fatalf("CreateDocument: %v", err)
				}
				for i := 0; i < 2; i++ {
					err := s.store.Upsert(ctx, &Embedding{
						ID:       fmt.Sprintf("%s-%d", doc, i),
						Vector:   []float32{1, 0, 0, 0},
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

			count, err := s.store.Aggregate(ctx, AggregationRequest{Type: AggregationCount, ExcludeDocuments: "id = ?", ExcludeDocumentsArgs: []interface{}{"hidden"}})
			if err != nil {
				t.Fatalf("Aggregate: %v", err)
			}
			if got := count.Results[0].Count; got != 3 {
				t.Errorf("count with hidden excluded = %d, want 3: two shown chunks and the loose row", got)
			}
			grouped, err := s.store.Aggregate(ctx, AggregationRequest{Type: AggregationGroupBy, GroupBy: []string{"kind"}, ExcludeDocuments: "id IN (?, ?)", ExcludeDocumentsArgs: []interface{}{"hidden", "shown"}})
			if err != nil {
				t.Fatalf("Aggregate group_by: %v", err)
			}
			if len(grouped.Results) != 1 || grouped.Results[0].Count != 1 {
				t.Errorf("group_by with both excluded = %+v, want the loose row alone", grouped.Results)
			}

			medoid, err := s.store.VectorAggregate(ctx, VectorAggregateRequest{Kind: VectorMedoid, ExcludeDocuments: "id IN (?, ?)", ExcludeDocumentsArgs: []interface{}{"hidden", "shown"}})
			if err != nil {
				t.Fatalf("VectorAggregate: %v", err)
			}
			if len(medoid.Groups) != 1 || medoid.Groups[0].Count != 1 || medoid.Groups[0].MemberID != "loose" {
				t.Errorf("medoid with both documents excluded = %+v, want the loose row", medoid.Groups)
			}
		})
	}
}
