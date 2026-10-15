package index

import "testing"

// A caller can pass any k. Zero or a negative count must come back as an
// empty result, never a panic.
func TestSearchNonPositiveKReturnsNothing(t *testing.T) {
	vecs := map[string][]float32{"a": {1, 0}, "b": {0, 1}, "c": {1, 1}}
	q := []float32{1, 0}

	flat := NewFlatIndex(2, EuclideanDistance)
	lsh := NewLSHIndex(LSHConfig{Dimension: 2, Seed: 1})
	ivf := NewIVFIndex(2, 1)
	hnsw := NewHNSW(8, 32, EuclideanDistance)
	if err := ivf.Train([][]float32{{0, 0}}); err != nil {
		t.Fatal(err)
	}
	for id, v := range vecs {
		if err := flat.Insert(id, v); err != nil {
			t.Fatal(err)
		}
		if err := lsh.Insert(id, v); err != nil {
			t.Fatal(err)
		}
		if err := ivf.Add(id, v); err != nil {
			t.Fatal(err)
		}
		if err := hnsw.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}

	for _, k := range []int{0, -1} {
		searches := map[string]func() int{
			"flat": func() int { ids, _ := flat.Search(q, k); return len(ids) },
			"lsh": func() int {
				r, _ := lsh.Search(q, k)
				return len(r)
			},
			"lsh multiprobe": func() int {
				r, _ := lsh.SearchWithMultiProbe(q, k, 2)
				return len(r)
			},
			"ivf": func() int {
				ids, _, _ := ivf.Search(q, k)
				return len(ids)
			},
			"hnsw": func() int { ids, _ := hnsw.Search(q, k, 16); return len(ids) },
		}
		for name, search := range searches {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("%s with k=%d panicked: %v", name, k, r)
					}
				}()
				if n := search(); n != 0 {
					t.Errorf("%s with k=%d returned %d results, want 0", name, k, n)
				}
			}()
		}
	}
}
