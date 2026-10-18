package quantization

import "testing"

// A vector longer than the input dimension has components no projection
// row covers. Project must report that with a nil result instead of
// panicking with an index out of range.
func TestOptimizedBinaryQuantizerProjectRejectsOverlongVector(t *testing.T) {
	obq := NewOptimizedBinaryQuantizer(4, 8)

	if got := obq.Project(make([]float32, 5)); got != nil {
		t.Fatalf("Project of a 5-dimensional vector = %v, want nil", got)
	}
	if got := obq.Project([]float32{1, 2, 3, 4}); len(got) != 8 {
		t.Fatalf("Project of a 4-dimensional vector has %d outputs, want 8", len(got))
	}
}
