package quantization

import (
	"bytes"
	"encoding/gob"
	"testing"
)

func gobBytes(t *testing.T, v interface{}) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(v); err != nil {
		t.Fatal(err)
	}
	return &buf
}

func TestScalarQuantizerLoadRejectsMalformedState(t *testing.T) {
	cases := map[string]*ScalarQuantizer{
		"short min and max": {Dimension: 4, NBits: 8, Min: []float32{0}, Max: []float32{1}, Trained: true},
		"missing max":       {Dimension: 2, NBits: 8, Min: []float32{0, 0}, Trained: true},
		"zero nbits":        {Dimension: 1, NBits: 0, Min: []float32{0}, Max: []float32{1}, Trained: true},
		"nbits over 8":      {Dimension: 1, NBits: 99, Min: []float32{0}, Max: []float32{1}, Trained: true},
		"negative size":     {Dimension: -1, NBits: 8, Trained: true},
	}
	for name, bad := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Load or use panicked: %v", r)
				}
			}()
			sq := &ScalarQuantizer{}
			if err := sq.Load(gobBytes(t, bad)); err == nil {
				t.Fatal("expected Load to reject the malformed quantizer")
			}
			if sq.Trained || sq.Dimension != 0 {
				t.Fatalf("rejected Load must leave the receiver unchanged, got %+v", sq)
			}
		})
	}
}

func TestBinaryQuantizerLoadRejectsMalformedState(t *testing.T) {
	cases := map[string]*BinaryQuantizer{
		"short thresholds": {Dimension: 8, Threshold: []float32{0}, Trained: true},
		"negative size":    {Dimension: -3, Trained: true},
	}
	for name, bad := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Load or use panicked: %v", r)
				}
			}()
			bq := &BinaryQuantizer{}
			if err := bq.Load(gobBytes(t, bad)); err == nil {
				t.Fatal("expected Load to reject the malformed quantizer")
			}
			if bq.Trained || bq.Dimension != 0 {
				t.Fatalf("rejected Load must leave the receiver unchanged, got %+v", bq)
			}
		})
	}
}

func TestQuantizerLoadAcceptsSavedState(t *testing.T) {
	sq, _ := NewScalarQuantizer(3, 4)
	if err := sq.Train([][]float32{{0, 1, 2}, {1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := sq.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded := &ScalarQuantizer{}
	if err := loaded.Load(&buf); err != nil {
		t.Fatal(err)
	}
	if _, err := loaded.Encode([]float32{0.5, 1.5, 2.5}); err != nil {
		t.Fatal(err)
	}

	bq := NewBinaryQuantizer(3)
	if err := bq.Train([][]float32{{0, 1, 2}, {1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	buf.Reset()
	if err := bq.Save(&buf); err != nil {
		t.Fatal(err)
	}
	lb := &BinaryQuantizer{}
	if err := lb.Load(&buf); err != nil {
		t.Fatal(err)
	}
	if _, err := lb.Encode([]float32{0.5, 1.5, 2.5}); err != nil {
		t.Fatal(err)
	}
}
