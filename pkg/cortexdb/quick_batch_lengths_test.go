package cortexdb

import (
	"context"
	"testing"
)

func TestQuickBatchRejectsMismatchedEmptySlices(t *testing.T) {
	quick := (&DB{}).Quick()
	for _, tc := range []struct {
		name     string
		vectors  [][]float32
		contents []string
	}{
		{"missing contents", [][]float32{{1, 0}}, nil},
		{"missing vectors", nil, []string{"content"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := quick.AddBatchWithVectorsToCollection(context.Background(), "collection", tc.vectors, tc.contents, nil)
			if err == nil || ids != nil {
				t.Fatalf("mismatched batch silently accepted: ids=%v err=%v", ids, err)
			}
		})
	}
	ids, err := quick.AddBatchWithVectors(context.Background(), nil, nil, nil)
	if err != nil || ids != nil {
		t.Fatalf("empty matched batch: ids=%v err=%v", ids, err)
	}
}
