package index

import "testing"

func TestIVFAddReplacesExistingID(t *testing.T) {
	ivf := NewIVFIndex(2, 2)
	if err := ivf.Train([][]float32{{0, 0}, {10, 10}}); err != nil {
		t.Fatal(err)
	}
	if err := ivf.Add("a", []float32{0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := ivf.Add("b", []float32{1, 1}); err != nil {
		t.Fatal(err)
	}
	// Re-adding "a" with a vector near the other centroid must replace the old
	// entry rather than leave a stale one behind.
	if err := ivf.Add("a", []float32{10, 10}); err != nil {
		t.Fatal(err)
	}
	ivf.SetNProbe(2)

	if got := ivf.Size(); got != 2 {
		t.Fatalf("Size = %d after re-adding an ID, want 2", got)
	}
	ids, dists, err := ivf.Search([]float32{10, 10}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != "a" || dists[0] != 0 {
		t.Fatalf("Search = %v %v, want [a b] led by an exact match", ids, dists)
	}
	for _, id := range ids[1:] {
		if id == "a" {
			t.Fatalf("stale entry for a returned: %v", ids)
		}
	}

	if err := ivf.Delete("a"); err != nil {
		t.Fatal(err)
	}
	ids, _, err = ivf.Search([]float32{10, 10}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != "b" {
		t.Fatalf("Search after Delete(a) = %v, want [b]", ids)
	}
}

func TestIVFAddReplacesIDAfterTrailingDelete(t *testing.T) {
	ivf := NewIVFIndex(2, 1)
	if err := ivf.Train([][]float32{{0, 0}}); err != nil {
		t.Fatal(err)
	}
	for i, v := range [][]float32{{0, 0}, {1, 1}, {2, 2}} {
		if err := ivf.Add(string(rune('a'+i)), v); err != nil {
			t.Fatal(err)
		}
	}
	if err := ivf.Delete("c"); err != nil {
		t.Fatal(err)
	}
	if err := ivf.Add("b", []float32{9, 9}); err != nil {
		t.Fatal(err)
	}
	if got := ivf.Size(); got != 2 {
		t.Fatalf("Size = %d, want 2", got)
	}
	ids, _, err := ivf.Search([]float32{9, 9}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != "b" || ids[1] != "a" {
		t.Fatalf("Search = %v, want [b a]", ids)
	}
}
