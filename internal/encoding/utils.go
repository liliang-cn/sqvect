package encoding

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

// ErrInvalidVector is returned when a vector is invalid
var ErrInvalidVector = errors.New("invalid vector")

// EncodeVector encodes a float32 vector as its length (int32) followed by
// its values (float32), all little-endian.
//
// The bytes are written directly. This used to go through binary.Write one
// value at a time, an interface call and an allocation per float, and a scan
// that decodes every stored vector paid that once per element per row.
func EncodeVector(vector []float32) ([]byte, error) {
	if vector == nil {
		return nil, ErrInvalidVector
	}
	if len(vector) > math.MaxInt32 {
		return nil, fmt.Errorf("vector too large: %d elements exceeds maximum", len(vector))
	}
	out := make([]byte, 4+4*len(vector))
	binary.LittleEndian.PutUint32(out, uint32(int32(len(vector))))
	for i, v := range vector {
		binary.LittleEndian.PutUint32(out[4+4*i:], math.Float32bits(v))
	}
	return out, nil
}

// DecodeVector decodes what EncodeVector wrote.
func DecodeVector(data []byte) ([]float32, error) {
	if len(data) < 4 {
		return nil, ErrInvalidVector
	}
	length := int32(binary.LittleEndian.Uint32(data))
	if length < 0 {
		return nil, ErrInvalidVector
	}
	if length == 0 {
		return []float32{}, nil
	}
	body := data[4:]
	if len(body) < int(length)*4 {
		return nil, ErrInvalidVector
	}
	vector := make([]float32, length)
	for i := range vector {
		vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[4*i:]))
	}
	return vector, nil
}

// encodeMetadata converts metadata map to JSON string
// EncodeMetadata encodes metadata to JSON string
func EncodeMetadata(metadata map[string]string) (string, error) {
	if metadata == nil {
		return "", nil
	}

	data, err := json.Marshal(metadata)
	if err != nil {
		return "", fmt.Errorf("failed to encode metadata: %w", err)
	}

	return string(data), nil
}

// decodeMetadata converts JSON string back to metadata map
// DecodeMetadata decodes JSON string to metadata
func DecodeMetadata(jsonStr string) (map[string]string, error) {
	if jsonStr == "" {
		return nil, nil
	}

	var metadata map[string]string
	if err := json.Unmarshal([]byte(jsonStr), &metadata); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %w", err)
	}

	return metadata, nil
}

// ValidateEmbedding validates an embedding's vector against expected dimension.
// If expectedDim is 0 (auto-detect mode), only validates vector values but not dimension.
func ValidateEmbedding(vector []float32, expectedDim int) error {
	if expectedDim > 0 && len(vector) != expectedDim {
		return fmt.Errorf("vector dimension mismatch: got %d, expected %d", len(vector), expectedDim)
	}
	return ValidateVector(vector)
}

// ValidateVector validates a vector
func ValidateVector(vector []float32) error {
	if vector == nil {
		return ErrInvalidVector
	}

	if len(vector) == 0 {
		return ErrInvalidVector
	}

	for _, val := range vector {
		if val != val { // NaN check
			return ErrInvalidVector
		}
		// Check for infinity
		if math.IsInf(float64(val), 0) {
			return ErrInvalidVector
		}
	}

	return nil
}
