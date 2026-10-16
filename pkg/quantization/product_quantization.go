// Package quantization provides vector compression techniques
package quantization

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// ProductQuantizer implements Product Quantization for vector compression
type ProductQuantizer struct {
	M         int           // Number of subspaces
	K         int           // Number of centroids per subspace
	D         int           // Original dimension
	SubDim    int           // Dimension per subspace (D/M)
	Codebooks [][][]float32 // M codebooks, each K x SubDim
	Trained   bool
	TrainSize int
}

// NewProductQuantizer creates a new PQ instance
func NewProductQuantizer(dimension, numSubspaces, numCentroids int) (*ProductQuantizer, error) {
	if dimension <= 0 || numSubspaces <= 0 || numCentroids <= 0 {
		return nil, fmt.Errorf("dimension, numSubspaces and numCentroids must be positive, got %d, %d, %d", dimension, numSubspaces, numCentroids)
	}
	if dimension%numSubspaces != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by numSubspaces %d", dimension, numSubspaces)
	}

	if numCentroids > 256 {
		return nil, errors.New("numCentroids must be <= 256 for byte encoding")
	}

	return &ProductQuantizer{
		M:         numSubspaces,
		K:         numCentroids,
		D:         dimension,
		SubDim:    dimension / numSubspaces,
		Codebooks: make([][][]float32, numSubspaces),
	}, nil
}

// Train learns the codebooks from training data
func (pq *ProductQuantizer) Train(vectors [][]float32) error {
	if len(vectors) < pq.K*pq.M {
		return fmt.Errorf("need at least %d vectors for training, got %d", pq.K*pq.M, len(vectors))
	}

	for i, vec := range vectors {
		if len(vec) != pq.D {
			return fmt.Errorf("training vector %d has dimension %d, quantizer expects %d", i, len(vec), pq.D)
		}
	}

	pq.TrainSize = len(vectors)

	// Train each subspace independently
	for m := 0; m < pq.M; m++ {
		// Extract subvectors for this subspace
		subvectors := make([][]float32, len(vectors))
		for i, vec := range vectors {
			start := m * pq.SubDim
			end := start + pq.SubDim
			subvectors[i] = vec[start:end]
		}

		// Run k-means on subvectors
		centroids, err := kMeans(subvectors, pq.K, 20)
		if err != nil {
			return fmt.Errorf("k-means failed for subspace %d: %w", m, err)
		}

		pq.Codebooks[m] = centroids
	}

	pq.Trained = true
	return nil
}

// Encode compresses a vector to PQ codes
func (pq *ProductQuantizer) Encode(vector []float32) ([]byte, error) {
	if !pq.Trained {
		return nil, errors.New("quantizer not trained")
	}

	if len(vector) != pq.D {
		return nil, fmt.Errorf("vector dimension %d doesn't match quantizer dimension %d", len(vector), pq.D)
	}

	codes := make([]byte, pq.M)

	// Encode each subvector
	for m := 0; m < pq.M; m++ {
		start := m * pq.SubDim
		end := start + pq.SubDim
		subvec := vector[start:end]

		// Find nearest centroid
		minDist := float32(math.MaxFloat32)
		minIdx := 0

		for k := 0; k < pq.K; k++ {
			dist := euclideanDistance(subvec, pq.Codebooks[m][k])
			if dist < minDist {
				minDist = dist
				minIdx = k
			}
		}

		codes[m] = byte(minIdx)
	}

	return codes, nil
}

// Decode reconstructs a vector from PQ codes
func (pq *ProductQuantizer) Decode(codes []byte) ([]float32, error) {
	if !pq.Trained {
		return nil, errors.New("quantizer not trained")
	}

	if len(codes) != pq.M {
		return nil, fmt.Errorf("codes length %d doesn't match number of subspaces %d", len(codes), pq.M)
	}

	vector := make([]float32, pq.D)

	// Decode each subvector
	for m := 0; m < pq.M; m++ {
		centroidIdx := int(codes[m])
		if centroidIdx >= pq.K {
			return nil, fmt.Errorf("invalid code %d for subspace %d", centroidIdx, m)
		}

		start := m * pq.SubDim
		centroid := pq.Codebooks[m][centroidIdx]

		for i := 0; i < pq.SubDim; i++ {
			vector[start+i] = centroid[i]
		}
	}

	return vector, nil
}

// ComputeDistance computes approximate distance between PQ codes and a query vector
func (pq *ProductQuantizer) ComputeDistance(codes []byte, query []float32) (float32, error) {
	if !pq.Trained {
		return 0, errors.New("quantizer not trained")
	}

	if len(query) != pq.D {
		return 0, fmt.Errorf("query dimension %d doesn't match quantizer dimension %d", len(query), pq.D)
	}
	if err := pq.checkCodes(codes); err != nil {
		return 0, err
	}

	// Precompute distance table
	distTable := pq.computeDistanceTable(query)

	// Sum distances from each subspace
	totalDist := float32(0)
	for m := 0; m < pq.M; m++ {
		totalDist += distTable[m][codes[m]]
	}

	return totalDist, nil
}

// checkCodes reports whether codes has one entry per subspace and each entry
// names a centroid.
func (pq *ProductQuantizer) checkCodes(codes []byte) error {
	if len(codes) != pq.M {
		return fmt.Errorf("codes length %d doesn't match number of subspaces %d", len(codes), pq.M)
	}
	for m, c := range codes {
		if int(c) >= pq.K {
			return fmt.Errorf("invalid code %d for subspace %d", c, m)
		}
	}
	return nil
}

// computeDistanceTable precomputes distances between query and all centroids
func (pq *ProductQuantizer) computeDistanceTable(query []float32) [][]float32 {
	table := make([][]float32, pq.M)

	for m := 0; m < pq.M; m++ {
		table[m] = make([]float32, pq.K)
		start := m * pq.SubDim
		end := start + pq.SubDim
		subquery := query[start:end]

		for k := 0; k < pq.K; k++ {
			table[m][k] = euclideanDistance(subquery, pq.Codebooks[m][k])
		}
	}

	return table
}

// SearchPQ performs approximate nearest neighbor search on PQ-compressed vectors
func (pq *ProductQuantizer) SearchPQ(query []float32, codes [][]byte, topK int) ([]int, []float32) {
	if !pq.Trained || len(codes) == 0 || len(query) != pq.D || topK <= 0 {
		return nil, nil
	}

	// Precompute distance table
	distTable := pq.computeDistanceTable(query)

	// Compute distances for all vectors
	type result struct {
		idx  int
		dist float32
	}

	// A code that does not belong to this quantizer cannot be scored; skip it
	// rather than index past its codebook. Indices still refer to codes.
	results := make([]result, 0, len(codes))
	for i, code := range codes {
		if pq.checkCodes(code) != nil {
			continue
		}
		dist := float32(0)
		for m := 0; m < pq.M; m++ {
			dist += distTable[m][code[m]]
		}
		results = append(results, result{idx: i, dist: dist})
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	// Return top-K
	k := topK
	if k > len(results) {
		k = len(results)
	}

	indices := make([]int, k)
	distances := make([]float32, k)
	for i := 0; i < k; i++ {
		indices[i] = results[i].idx
		distances[i] = results[i].dist
	}

	return indices, distances
}

// CompressionRatio returns the compression ratio achieved by PQ
func (pq *ProductQuantizer) CompressionRatio() float32 {
	originalSize := pq.D * 4 // 4 bytes per float32
	compressedSize := pq.M   // 1 byte per subspace
	return float32(originalSize) / float32(compressedSize)
}

// SerializeCodebooks serializes codebooks to bytes for storage
func (pq *ProductQuantizer) SerializeCodebooks() []byte {
	if !pq.Trained {
		return nil
	}

	// Calculate total size
	size := 4 * 4                       // M, K, D, SubDim (4 int32s)
	size += pq.M * pq.K * pq.SubDim * 4 // Codebook data

	buf := make([]byte, size)
	offset := 0

	// Write header
	binary.LittleEndian.PutUint32(buf[offset:], uint32(pq.M))
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], uint32(pq.K))
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], uint32(pq.D))
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], uint32(pq.SubDim))
	offset += 4

	// Write codebooks
	for m := 0; m < pq.M; m++ {
		for k := 0; k < pq.K; k++ {
			for d := 0; d < pq.SubDim; d++ {
				binary.LittleEndian.PutUint32(buf[offset:], math.Float32bits(pq.Codebooks[m][k][d]))
				offset += 4
			}
		}
	}

	return buf
}

// DeserializeCodebooks loads codebooks from bytes. The data is checked against
// its own header before anything is allocated, and a rejected load leaves the
// quantizer unchanged.
func (pq *ProductQuantizer) DeserializeCodebooks(data []byte) error {
	const headerSize = 16
	if len(data) < headerSize {
		return errors.New("invalid codebook data")
	}

	m := int(binary.LittleEndian.Uint32(data[0:]))
	k := int(binary.LittleEndian.Uint32(data[4:]))
	d := int(binary.LittleEndian.Uint32(data[8:]))
	subDim := int(binary.LittleEndian.Uint32(data[12:]))

	if m == 0 || k == 0 || subDim == 0 {
		return errors.New("invalid codebook data: header has a zero size")
	}
	if k > 256 {
		return fmt.Errorf("invalid codebook data: %d centroids exceed the byte encoding limit of 256", k)
	}

	// m*k*subDim float32 values must fill the body exactly. Divide instead of
	// multiplying so a hostile header cannot overflow the check.
	body := len(data) - headerSize
	if body%4 != 0 {
		return errors.New("invalid codebook data: body is not a whole number of float32 values")
	}
	values := body / 4
	if m > values/k || values%(m*k) != 0 || values/(m*k) != subDim {
		return fmt.Errorf("invalid codebook data: header declares %d x %d x %d values but the body holds %d", m, k, subDim, values)
	}
	if d != m*subDim {
		return fmt.Errorf("invalid codebook data: dimension %d is not %d subspaces of %d", d, m, subDim)
	}

	codebooks := make([][][]float32, m)
	offset := headerSize
	for i := range codebooks {
		codebooks[i] = make([][]float32, k)
		for j := range codebooks[i] {
			codebooks[i][j] = make([]float32, subDim)
			for x := range codebooks[i][j] {
				codebooks[i][j][x] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
				offset += 4
			}
		}
	}

	pq.M, pq.K, pq.D, pq.SubDim = m, k, d, subDim
	pq.Codebooks = codebooks
	pq.Trained = true
	return nil
}

// kMeans performs k-means clustering
func kMeans(vectors [][]float32, k int, maxIters int) ([][]float32, error) {
	if len(vectors) < k {
		return nil, fmt.Errorf("need at least %d vectors, got %d", k, len(vectors))
	}

	dim := len(vectors[0])

	// Initialize centroids randomly
	centroids := make([][]float32, k)
	perm := rand.Perm(len(vectors))
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, dim)
		copy(centroids[i], vectors[perm[i]])
	}

	assignments := make([]int, len(vectors))

	for iter := 0; iter < maxIters; iter++ {
		// Assign vectors to nearest centroid
		changed := false
		for i, vec := range vectors {
			minDist := float32(math.MaxFloat32)
			minIdx := 0

			for j, centroid := range centroids {
				dist := euclideanDistance(vec, centroid)
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}

			if assignments[i] != minIdx {
				changed = true
				assignments[i] = minIdx
			}
		}

		if !changed {
			break
		}

		// Update centroids
		counts := make([]int, k)
		for i := range centroids {
			centroids[i] = make([]float32, dim)
		}

		for i, vec := range vectors {
			cluster := assignments[i]
			counts[cluster]++
			for j := 0; j < dim; j++ {
				centroids[cluster][j] += vec[j]
			}
		}

		for i := range centroids {
			if counts[i] > 0 {
				for j := 0; j < dim; j++ {
					centroids[i][j] /= float32(counts[i])
				}
			}
		}
	}

	return centroids, nil
}

// euclideanDistance computes Euclidean distance between two vectors
func euclideanDistance(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}
