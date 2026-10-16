package index

import (
	"bytes"
	"testing"
)

func TestHNSWLoadTruncatedSnapshotLeavesIndexUntouched(t *testing.T) {
	src := NewHNSW(4, 16, EuclideanDistance)
	for i := 0; i < 20; i++ {
		if err := src.Insert(string(rune('a'+i)), []float32{float32(i), float32(i * i % 7)}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := src.Save(&buf); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()

	h := NewHNSW(4, 16, EuclideanDistance)
	if err := h.Load(bytes.NewReader(data[:len(data)*2/3])); err == nil {
		t.Fatal("expected an error loading a truncated snapshot")
	}
	if len(h.Nodes) != 0 || h.EntryPoint != "" {
		t.Fatalf("failed Load left %d nodes and entry point %q behind", len(h.Nodes), h.EntryPoint)
	}
	// The caller falls back to rebuilding on the same index.
	if err := h.Insert("x", []float32{1, 1}); err != nil {
		t.Fatal(err)
	}
	if ids, _ := h.Search([]float32{1, 1}, 1, 16); len(ids) != 1 || ids[0] != "x" {
		t.Fatalf("expected [x], got %v", ids)
	}

	if err := h.Load(bytes.NewReader(data)); err != nil {
		t.Fatalf("a complete snapshot must load: %v", err)
	}
	if h.Size() != 20 {
		t.Fatalf("expected 20 nodes, got %d", h.Size())
	}
}

func TestHNSWLoadRejectsDanglingReferences(t *testing.T) {
	src := NewHNSW(4, 16, EuclideanDistance)
	for i := 0; i < 5; i++ {
		if err := src.Insert(string(rune('a'+i)), []float32{float32(i), 1}); err != nil {
			t.Fatal(err)
		}
	}
	save := func(h *HNSW) *bytes.Reader {
		var buf bytes.Buffer
		if err := h.Save(&buf); err != nil {
			t.Fatal(err)
		}
		return bytes.NewReader(buf.Bytes())
	}

	src.EntryPoint = "missing"
	if err := NewHNSW(4, 16, EuclideanDistance).Load(save(src)); err == nil {
		t.Error("expected an error for an entry point that is not a node")
	}

	src.EntryPoint = "a"
	src.Nodes["b"].Neighbors[0] = append(src.Nodes["b"].Neighbors[0], "missing")
	if err := NewHNSW(4, 16, EuclideanDistance).Load(save(src)); err == nil {
		t.Error("expected an error for a link to a node that is not there")
	}
}

func TestIVFLoadRejectsBadSnapshotsWithoutChangingIndex(t *testing.T) {
	src := NewIVFIndex(2, 2)
	if err := src.Train([][]float32{{0, 0}, {5, 5}}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		if err := src.Add(string(rune('a'+i)), []float32{float32(i), float32(i)}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := src.Save(&buf); err != nil {
		t.Fatal(err)
	}
	data := append([]byte(nil), buf.Bytes()...)

	ivf := NewIVFIndex(2, 2)
	if err := ivf.Load(bytes.NewReader(data[:len(data)*3/4])); err == nil {
		t.Fatal("expected an error loading a truncated snapshot")
	}
	if ivf.Trained || len(ivf.IDs) != 0 || len(ivf.Centroids) != 0 {
		t.Fatalf("failed Load left state behind: trained=%v ids=%d centroids=%d", ivf.Trained, len(ivf.IDs), len(ivf.Centroids))
	}

	// A list that points past the stored vectors would panic in Search.
	src.Invlists[0] = append(src.Invlists[0], 99)
	buf.Reset()
	if err := src.Save(&buf); err != nil {
		t.Fatal(err)
	}
	if err := NewIVFIndex(2, 2).Load(&buf); err == nil {
		t.Error("expected an error for an inverted list entry out of range")
	}

	if err := ivf.Load(bytes.NewReader(data)); err != nil {
		t.Fatalf("a complete snapshot must load: %v", err)
	}
	ids, _, err := ivf.Search([]float32{1, 1}, 3)
	if err != nil || len(ids) == 0 {
		t.Fatalf("search after load: %v %v", ids, err)
	}
}
