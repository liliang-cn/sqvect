package index

import "testing"

// Training vectors of the wrong dimension must be rejected with an error;
// k-means would otherwise index past the end of a shorter vector and panic.
func TestIVFTrainRejectsWrongDimension(t *testing.T) {
	for name, vectors := range map[string][][]float32{
		"short": {{0, 0}, {10}},
		"long":  {{0, 0}, {10, 10, 10}},
		"nil":   {{0, 0}, nil},
	} {
		ivf := NewIVFIndex(2, 2)
		if err := ivf.Train(vectors); err == nil {
			t.Errorf("%s: Train accepted a vector of the wrong dimension", name)
		}
		if ivf.Trained {
			t.Errorf("%s: index marked trained after a rejected Train", name)
		}
	}
}
