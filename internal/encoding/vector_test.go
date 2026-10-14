package encoding

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// legacyEncode is the encoder this package used before it wrote bytes
// directly; stored vectors were written by it, so the formats must agree.
func legacyEncode(vector []float32) []byte {
	buf := new(bytes.Buffer)
	_ = binary.Write(buf, binary.LittleEndian, int32(len(vector)))
	for _, v := range vector {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func TestVectorEncodingMatchesTheStoredFormat(t *testing.T) {
	v := []float32{0, 1, -1, 0.5, float32(math.Pi), math.MaxFloat32, -math.SmallestNonzeroFloat32, float32(math.Inf(1))}
	got, err := EncodeVector(v)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, legacyEncode(v)) {
		t.Fatalf("encoding changed:\n got %x\nwant %x", got, legacyEncode(v))
	}
	back, err := DecodeVector(legacyEncode(v))
	if err != nil {
		t.Fatal(err)
	}
	for i := range v {
		if math.Float32bits(back[i]) != math.Float32bits(v[i]) {
			t.Fatalf("value %d: %v != %v", i, back[i], v[i])
		}
	}
	if empty, err := DecodeVector(legacyEncode([]float32{})); err != nil || len(empty) != 0 {
		t.Fatalf("empty vector: %v %v", empty, err)
	}
	for _, bad := range [][]byte{nil, {1, 0}, {2, 0, 0, 0, 1, 2, 3, 4}, {0xff, 0xff, 0xff, 0xff}} {
		if _, err := DecodeVector(bad); err == nil {
			t.Fatalf("%x: want an error", bad)
		}
	}
	if _, err := EncodeVector(nil); err == nil {
		t.Fatal("nil vector: want an error")
	}
}

func BenchmarkDecodeVector768(b *testing.B) {
	v := make([]float32, 768)
	data, _ := EncodeVector(v)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = DecodeVector(data)
	}
}
