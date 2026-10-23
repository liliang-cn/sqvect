package index

import (
	"bytes"
	"testing"
)

// Upsert replaces the vector a live node holds; Insert still refuses it.
func TestHNSWUpsertReplacesLiveNode(t *testing.T) {
	h := NewHNSW(8, 32, CosineDistance)
	for id, v := range map[string][]float32{"a": {1, 0, 0}, "b": {0, 1, 0}, "c": {0, 0, 1}} {
		if err := h.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	if err := h.Upsert("a", []float32{0, 0.1, 1}); err != nil {
		t.Fatal(err)
	}
	if got := h.Nodes["a"].Vector; got[2] != 1 {
		t.Fatalf("a still holds %v", got)
	}
	if h.Size() != 3 {
		t.Fatalf("expected 3 live nodes, got %d", h.Size())
	}
	ids, _ := h.Search([]float32{0, 0.1, 1}, 1, 16)
	if len(ids) != 1 || ids[0] != "a" {
		t.Fatalf("expected [a], got %v", ids)
	}
	for _, node := range h.Nodes {
		for _, links := range node.Neighbors {
			for _, to := range links {
				if to == node.ID {
					t.Fatalf("node %s links to itself", node.ID)
				}
			}
		}
	}
}

// Writing the vector a node already holds is not a change, so it does not
// make a saved snapshot stale.
func TestHNSWUpsertSameVectorIsNoMutation(t *testing.T) {
	h := NewHNSW(8, 32, CosineDistance)
	if err := h.Insert("a", []float32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	before := h.Mutations()
	if err := h.Upsert("a", []float32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if h.Mutations() != before {
		t.Fatal("an unchanged upsert counted as a mutation")
	}
}

// A replacement of the wrong dimension is refused and keeps the old vector.
func TestHNSWUpsertRejectedKeepsOldVector(t *testing.T) {
	h := NewHNSW(8, 32, CosineDistance)
	for id, v := range map[string][]float32{"a": {1, 0}, "b": {0, 1}} {
		if err := h.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	if err := h.Upsert("a", []float32{1, 0, 0}); err == nil {
		t.Fatal("expected a dimension error")
	}
	if node := h.Nodes["a"]; node == nil || node.Deleted || len(node.Vector) != 2 {
		t.Fatal("a rejected upsert lost the node it would have replaced")
	}
}

// Load reads the dimension off the nodes. It used to leave it at zero, so the
// first insert after a reload decided it: one vector of another size written
// first, and every write of the right size was refused until the next restart.
func TestHNSWLoadKeepsDimension(t *testing.T) {
	h := NewHNSW(8, 32, CosineDistance)
	for id, v := range map[string][]float32{"a": {1, 0, 0}, "b": {0, 1, 0}} {
		if err := h.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := h.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded := NewHNSW(8, 32, CosineDistance)
	if err := loaded.Load(&buf); err != nil {
		t.Fatal(err)
	}
	if err := loaded.Insert("lexical", []float32{1, 0}); err == nil {
		t.Fatal("a 2-d vector went into a 3-d index")
	}
	if err := loaded.Insert("c", []float32{0, 0, 1}); err != nil {
		t.Fatalf("a 3-d vector was refused after the reload: %v", err)
	}
}
