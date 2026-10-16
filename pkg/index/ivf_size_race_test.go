package index

import (
	"strconv"
	"sync"
	"testing"
)

func TestIVFSizeIsSafeDuringAdd(t *testing.T) {
	ivf := NewIVFIndex(2, 1)
	if err := ivf.Train([][]float32{{0, 0}}); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 500; i++ {
			if err := ivf.Add(strconv.Itoa(i), []float32{float32(i), 0}); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 500; i++ {
			_ = ivf.Size()
		}
	}()
	wg.Wait()

	if got := ivf.Size(); got != 500 {
		t.Fatalf("Size = %d, want 500", got)
	}
}
