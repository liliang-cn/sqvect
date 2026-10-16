package index

import "testing"

// A store that deletes a row and upserts it again re-inserts the same ID.
// Delete only marks the node, so Insert used to refuse the ID for good and the
// re-added vector never became searchable.
func TestHNSWInsertAfterDelete(t *testing.T) {
	for name, insert := range map[string]func(h *HNSW, id string, v []float32) error{
		"Insert": func(h *HNSW, id string, v []float32) error { return h.Insert(id, v) },
		"InsertBatch": func(h *HNSW, id string, v []float32) error {
			return h.InsertBatch([]struct {
				ID     string
				Vector []float32
			}{{id, v}})
		},
	} {
		t.Run(name, func(t *testing.T) {
			h := NewHNSW(8, 32, EuclideanDistance)
			if err := h.Insert("a", []float32{1, 0}); err != nil {
				t.Fatal(err)
			}
			if err := h.Insert("b", []float32{0, 1}); err != nil {
				t.Fatal(err)
			}
			if err := h.Delete("a"); err != nil {
				t.Fatal(err)
			}
			if err := insert(h, "a", []float32{1, 0}); err != nil {
				t.Fatalf("re-inserting a deleted ID: %v", err)
			}
			ids, _ := h.Search([]float32{1, 0}, 2, 16)
			if len(ids) != 2 || ids[0] != "a" {
				t.Fatalf("expected the re-inserted a first, got %v", ids)
			}
			if h.Size() != 2 {
				t.Fatalf("expected 2 live nodes, got %d", h.Size())
			}
			if err := h.Insert("a", []float32{1, 0}); err == nil {
				t.Fatal("inserting a live ID must still fail")
			}
		})
	}
}

// Deleting the only node leaves it as the entry point; re-inserting its ID
// must not leave the graph pointing at a node that was dropped.
func TestHNSWInsertAfterDeletingOnlyNode(t *testing.T) {
	h := NewHNSW(8, 32, EuclideanDistance)
	if err := h.Insert("a", []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := h.Delete("a"); err != nil {
		t.Fatal(err)
	}
	if err := h.Insert("a", []float32{0, 1}); err != nil {
		t.Fatal(err)
	}
	ids, _ := h.Search([]float32{0, 1}, 1, 16)
	if len(ids) != 1 || ids[0] != "a" {
		t.Fatalf("expected [a], got %v", ids)
	}
}

// A rejected insert must not cost the deleted node it would have replaced.
func TestHNSWRejectedInsertKeepsDeletedNode(t *testing.T) {
	h := NewHNSW(8, 32, EuclideanDistance)
	for id, v := range map[string][]float32{"a": {1, 0}, "b": {0, 1}} {
		if err := h.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	if err := h.Delete("a"); err != nil {
		t.Fatal(err)
	}
	if err := h.Insert("a", []float32{1, 0, 0}); err == nil {
		t.Fatal("expected a dimension error")
	}
	if err := h.Insert("a", nil); err == nil {
		t.Fatal("expected an empty-vector error")
	}
	if _, ok := h.Nodes["a"]; !ok {
		t.Fatal("rejected inserts dropped the deleted node")
	}
	if ids, _ := h.Search([]float32{0, 1}, 2, 16); len(ids) != 1 || ids[0] != "b" {
		t.Fatalf("expected [b], got %v", ids)
	}
}

// Node levels are random, so the graph shape differs run to run. Links left
// pointing at the dropped node would resolve to the re-inserted one and make
// it its own neighbour, hiding the rest of the graph from a search.
func TestHNSWReinsertedNodeStaysConnected(t *testing.T) {
	for i := 0; i < 200; i++ {
		h := NewHNSW(8, 32, EuclideanDistance)
		for id, v := range map[string][]float32{"a": {1, 0}, "b": {0, 1}, "c": {1, 1}} {
			if err := h.Insert(id, v); err != nil {
				t.Fatal(err)
			}
		}
		if err := h.Delete("a"); err != nil {
			t.Fatal(err)
		}
		if err := h.Insert("a", []float32{1, 0}); err != nil {
			t.Fatal(err)
		}
		for _, links := range h.Nodes["a"].Neighbors {
			for _, to := range links {
				if to == "a" {
					t.Fatalf("iteration %d: a is its own neighbour", i)
				}
			}
		}
		if ids, _ := h.Search([]float32{1, 0}, 3, 16); len(ids) != 3 {
			t.Fatalf("iteration %d: expected all 3 nodes, got %v", i, ids)
		}
	}
}
