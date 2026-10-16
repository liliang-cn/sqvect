package quantization

import "testing"

func TestNewProductQuantizerRejectsNonPositiveSizes(t *testing.T) {
	for _, c := range [][3]int{{8, 0, 4}, {8, -2, 4}, {8, 2, 0}, {8, 2, -1}, {0, 1, 4}, {-8, 2, 4}} {
		pq, err := NewProductQuantizer(c[0], c[1], c[2])
		if err == nil || pq != nil {
			t.Errorf("NewProductQuantizer(%d, %d, %d) = %v, %v; want an error", c[0], c[1], c[2], pq, err)
		}
	}
}

func TestProductQuantizerTrainRejectsWrongDimensions(t *testing.T) {
	pq, err := NewProductQuantizer(4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	vectors := [][]float32{{1, 2, 3, 4}, {4, 3, 2, 1}, {0, 0, 0, 0}, {5, 5, 5, 5}, {1, 1, 1, 1}}
	for name, bad := range map[string][]float32{"short": {1}, "long": {1, 2, 3, 4, 5}} {
		in := append([][]float32{bad}, vectors...)
		if err := pq.Train(in); err == nil {
			t.Errorf("%s vector: Train returned nil, want an error", name)
		}
		if pq.Trained || pq.TrainSize != 0 {
			t.Errorf("%s vector: rejected training left the quantizer marked trained", name)
		}
	}
	if err := pq.Train(vectors); err != nil {
		t.Fatalf("valid training failed: %v", err)
	}
}
