package index

import "testing"

// bucketEntries counts the IDs held across all hash tables.
func bucketEntries(lsh *LSHIndex) int {
	n := 0
	for _, table := range lsh.hashTables {
		for _, bucket := range table {
			n += len(bucket)
		}
	}
	return n
}

func TestLSHReinsertReplacesBucketEntries(t *testing.T) {
	lsh := NewLSHIndex(LSHConfig{Dimension: 2, NumTables: 10, NumHashFuncs: 8, Seed: 1})
	if err := lsh.Insert("a", []float32{1, 1}); err != nil {
		t.Fatal(err)
	}
	if err := lsh.Insert("a", []float32{-1, -1}); err != nil {
		t.Fatal(err)
	}
	if got := bucketEntries(lsh); got != lsh.numTables {
		t.Fatalf("one vector should occupy one bucket per table (%d), found %d entries", lsh.numTables, got)
	}
	if !lsh.Delete("a") {
		t.Fatal("expected Delete to find a")
	}
	if got := bucketEntries(lsh); got != 0 {
		t.Fatalf("expected empty buckets after Delete, found %d entries", got)
	}
}

func TestLSHInsertCopiesVector(t *testing.T) {
	lsh := NewLSHIndex(LSHConfig{Dimension: 2, NumTables: 10, NumHashFuncs: 8, Seed: 1})
	v := []float32{1, 1}
	if err := lsh.Insert("a", v); err != nil {
		t.Fatal(err)
	}
	v[0], v[1] = -5, -5 // the caller reuses its buffer
	if !lsh.Delete("a") {
		t.Fatal("expected Delete to find a")
	}
	if got := bucketEntries(lsh); got != 0 {
		t.Fatalf("expected empty buckets after Delete, found %d entries", got)
	}
}
