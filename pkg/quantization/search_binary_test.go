package quantization

import (
	"testing"
	"time"
)

// SearchBinary must scale like a sort, not like a quadratic selection pass,
// and codes at equal distance must come back in database order.
func TestSearchBinarySortsLargeDatabaseInOrder(t *testing.T) {
	bq := NewBinaryQuantizer(8)
	const n = 60000
	db := make([][]byte, n)
	for i := range db {
		db[i] = []byte{byte(i % 9)} // few distinct distances, many ties
	}
	start := time.Now()
	idx, dist := bq.SearchBinary([]byte{0}, db, n)
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("SearchBinary over %d codes took %v", n, elapsed)
	}
	if len(idx) != n {
		t.Fatalf("got %d results, want %d", len(idx), n)
	}
	for i := 1; i < n; i++ {
		if dist[i] < dist[i-1] || (dist[i] == dist[i-1] && idx[i] < idx[i-1]) {
			t.Fatalf("results %d/%d out of order: (%d,%d) then (%d,%d)", i-1, i, idx[i-1], dist[i-1], idx[i], dist[i])
		}
	}
}
