package quantization

import "testing"

func trainedPQ(t *testing.T) *ProductQuantizer {
	t.Helper()
	pq, err := NewProductQuantizer(4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	vectors := [][]float32{{1, 2, 3, 4}, {4, 3, 2, 1}, {0, 0, 0, 0}, {5, 5, 5, 5}, {1, 1, 1, 1}}
	if err := pq.Train(vectors); err != nil {
		t.Fatal(err)
	}
	return pq
}

func TestProductQuantizerComputeDistanceRejectsBadInput(t *testing.T) {
	pq := trainedPQ(t)
	good := []float32{1, 2, 3, 4}
	for name, c := range map[string]struct {
		codes []byte
		query []float32
	}{
		"short query": {[]byte{0, 0}, []float32{1}},
		"long query":  {[]byte{0, 0}, []float32{1, 2, 3, 4, 5}},
		"short codes": {[]byte{0}, good},
		"long codes":  {[]byte{0, 0, 0}, good},
		"bad code":    {[]byte{0, 9}, good},
	} {
		if _, err := pq.ComputeDistance(c.codes, c.query); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := pq.ComputeDistance([]byte{0, 1}, good); err != nil {
		t.Fatalf("valid input failed: %v", err)
	}
}

func TestProductQuantizerSearchPQIgnoresBadInput(t *testing.T) {
	pq := trainedPQ(t)
	good := []float32{1, 2, 3, 4}

	if idx, _ := pq.SearchPQ([]float32{1}, [][]byte{{0, 0}}, 1); len(idx) != 0 {
		t.Errorf("short query returned %v", idx)
	}
	for _, k := range []int{0, -1} {
		if idx, _ := pq.SearchPQ(good, [][]byte{{0, 0}}, k); len(idx) != 0 {
			t.Errorf("topK %d returned %v", k, idx)
		}
	}

	// Codes that do not fit the quantizer are skipped; the rest keep their
	// positions in the input.
	codes := [][]byte{{0}, {0, 9}, {0, 1}, {1, 0, 0}}
	idx, dist := pq.SearchPQ(good, codes, 10)
	if len(idx) != 1 || idx[0] != 2 || len(dist) != 1 {
		t.Fatalf("expected only code 2 to be scored, got %v %v", idx, dist)
	}
}
