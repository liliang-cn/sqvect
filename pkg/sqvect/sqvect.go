// Package sqvect provides a lightweight SQLite-based vector database for Go AI projects
package sqvect

import (
	"context"
	"fmt"

	"github.com/liliang-cn/sqvect/pkg/core"
	"github.com/liliang-cn/sqvect/pkg/graph"
)

// DB represents a SQLite vector database instance
type DB struct {
	store *core.SQLiteStore
	graph *graph.GraphStore
}

// Config represents database configuration
type Config struct {
	Path         string            // Database file path
	Dimensions   int               // Vector dimensions (0 for auto-detect)
	SimilarityFn core.SimilarityFunc // Similarity function (default: cosine)
}

// DefaultConfig returns default configuration
func DefaultConfig(path string) Config {
	return Config{
		Path:         path,
		Dimensions:   0, // Auto-detect
		SimilarityFn: core.CosineSimilarity,
	}
}

// Open opens or creates a vector database
func Open(config Config) (*DB, error) {
	coreConfig := core.Config{
		Path:           config.Path,
		VectorDim:      config.Dimensions,
		SimilarityFn:   config.SimilarityFn,
		AutoDimAdapt:   core.SmartAdapt,
		HNSW:           core.DefaultHNSWConfig(),
		TextSimilarity: core.DefaultTextSimilarityConfig(),
	}

	store, err := core.NewWithConfig(coreConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create store: %w", err)
	}

	// Initialize the database
	ctx := context.Background()
	if err := store.Init(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize store: %w", err)
	}

	// Create graph store
	graphStore := graph.NewGraphStore(store)

	return &DB{
		store: store,
		graph: graphStore,
	}, nil
}

// Vector returns the core vector store interface
func (db *DB) Vector() core.Store {
	return db.store
}

// Graph returns the graph store interface
func (db *DB) Graph() *graph.GraphStore {
	return db.graph
}

// Close closes the database
func (db *DB) Close() error {
	return db.store.Close()
}

// Quick is a simplified interface for common operations
type Quick struct {
	db *DB
}

// NewQuick creates a simple interface for quick operations
func (db *DB) Quick() *Quick {
	return &Quick{db: db}
}

// Add adds a vector with automatic ID generation
func (q *Quick) Add(ctx context.Context, vector []float32, content string) (string, error) {
	return q.AddToCollection(ctx, "", vector, content)
}

// AddToCollection adds a vector to a specific collection with automatic ID generation
func (q *Quick) AddToCollection(ctx context.Context, collection string, vector []float32, content string) (string, error) {
	id := generateID()
	embedding := &core.Embedding{
		ID:         id,
		Collection: collection,
		Vector:     vector,
		Content:    content,
	}
	
	err := q.db.store.Upsert(ctx, embedding)
	return id, err
}

// Search performs similarity search
func (q *Quick) Search(ctx context.Context, query []float32, topK int) ([]core.ScoredEmbedding, error) {
	return q.SearchInCollection(ctx, "", query, topK)
}

// SearchInCollection performs similarity search within a collection
func (q *Quick) SearchInCollection(ctx context.Context, collection string, query []float32, topK int) ([]core.ScoredEmbedding, error) {
	opts := core.SearchOptions{
		Collection: collection,
		TopK:       topK,
		Threshold:  0.0,
	}
	
	return q.db.store.Search(ctx, query, opts)
}

// generateID generates a simple ID for embeddings
func generateID() string {
	// Use a simple counter or UUID - keeping it lightweight
	return fmt.Sprintf("emb_%d", generateCounter())
}

var idCounter int64

func generateCounter() int64 {
	idCounter++
	return idCounter
}