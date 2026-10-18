package quantization

import "testing"

// A centroid that attracts no vector must keep its position. Resetting it to
// the origin plants a codeword in a region no training data occupies, and it
// then wins encodes for any vector that happens to lie near the origin.
func TestKMeansEmptyClusterKeepsCentroid(t *testing.T) {
	// Two distinct points and four centroids: at least two clusters are
	// always empty. Initialisation is random, so repeat.
	for run := 0; run < 50; run++ {
		var vectors [][]float32
		for i := 0; i < 10; i++ {
			vectors = append(vectors, []float32{100, 100}, []float32{200, 200})
		}
		centroids, err := kMeans(vectors, 4, 20)
		if err != nil {
			t.Fatal(err)
		}
		for i, c := range centroids {
			if (c[0] != 100 && c[0] != 200) || c[0] != c[1] {
				t.Fatalf("run %d: centroid %d = %v, want a trained position (an empty cluster was reset to the origin)", run, i, c)
			}
		}
	}
}
