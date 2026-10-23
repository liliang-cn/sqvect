package cortexdb

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

// ReembedOptions controls a re-embedding pass.
type ReembedOptions struct {
	// Limit caps how many rows are processed (0 = all of them).
	Limit int
	// BatchSize is how many texts go to the embedder per call (0 = 16).
	BatchSize int
	// DryRun reports what would change without writing anything.
	DryRun bool
}

// ReembedReport is the outcome of a re-embedding pass.
type ReembedReport struct {
	TargetDim  int  `json:"targetDim"`
	Candidates int  `json:"candidates"`
	Reembedded int  `json:"reembedded"`
	Failed     int  `json:"failed"`
	DryRun     bool `json:"dryRun"`
	// Collections whose declared dimension was brought in line with their contents.
	Reconciled int `json:"reconciled"`
	// Chunk rows an earlier repair had stripped of their document and
	// metadata, given them back from the chunk id and the document record.
	Restored int      `json:"restored"`
	Errors   []string `json:"errors,omitempty"`
}

// ReembedMismatchedVectors recomputes vectors whose stored dimensionality differs from
// the configured embedder's.
//
// Such rows appear whenever a store outlives an embedding model. They cannot enter the
// vector index — a graph holds one dimensionality — so they quietly stop being
// retrievable by similarity while lexical search still finds them, which masks the
// problem. The numbers cannot be salvaged by truncating or padding: vectors from two
// models occupy unrelated spaces, so the only honest repair is to embed the stored text
// again with the current model.
//
// Returns ErrEmbedderNotConfigured when the DB has no embedder.
func (db *DB) ReembedMismatchedVectors(ctx context.Context, opts ReembedOptions) (*ReembedReport, error) {
	if db.embedder == nil {
		return nil, ErrEmbedderNotConfigured
	}
	targetDim := db.embedder.Dim()
	if targetDim <= 0 {
		return nil, fmt.Errorf("cortexdb: embedder reports dimension %d", targetDim)
	}
	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = 16
	}

	stale, err := db.store.MismatchedEmbeddings(ctx, targetDim, opts.Limit)
	if err != nil {
		return nil, err
	}
	report := &ReembedReport{TargetDim: targetDim, Candidates: len(stale), DryRun: opts.DryRun}
	if opts.DryRun {
		return report, nil
	}
	// Note there is no early return for an empty candidate list: metadata still needs
	// reconciling below when a previous pass already fixed the vectors.

	for start := 0; start < len(stale); start += batchSize {
		end := start + batchSize
		if end > len(stale) {
			end = len(stale)
		}
		batch := stale[start:end]
		texts := make([]string, len(batch))
		for i, emb := range batch {
			texts[i] = emb.Content
		}
		vectors, err := db.embedder.EmbedBatch(ctx, texts)
		if err != nil {
			report.Failed += len(batch)
			report.Errors = append(report.Errors, fmt.Sprintf("embed batch at %d: %v", start, err))
			continue
		}
		if len(vectors) != len(batch) {
			report.Failed += len(batch)
			report.Errors = append(report.Errors,
				fmt.Sprintf("embed batch at %d returned %d vectors for %d texts", start, len(vectors), len(batch)))
			continue
		}
		updates := make([]*core.Embedding, 0, len(batch))
		for i, emb := range batch {
			if len(vectors[i]) != targetDim {
				report.Failed++
				report.Errors = append(report.Errors,
					fmt.Sprintf("row %s: embedder returned %d dimensions, want %d", emb.ID, len(vectors[i]), targetDim))
				continue
			}
			emb.Vector = vectors[i]
			updates = append(updates, emb)
		}
		if len(updates) == 0 {
			continue
		}
		if err := db.store.UpsertBatch(ctx, updates); err != nil {
			report.Failed += len(updates)
			report.Errors = append(report.Errors, fmt.Sprintf("write batch at %d: %v", start, err))
			continue
		}
		report.Reembedded += len(updates)
	}
	// A collection records the dimension it was created with. Leaving it stale would keep
	// the drift report flagging rows that are now correct. This runs on every real pass,
	// not only when rows were rewritten: vectors repaired by an earlier pass (or written
	// directly) leave exactly this state behind.
	if !opts.DryRun {
		reconciled, err := db.store.ReconcileCollectionDimensions(ctx, targetDim)
		if err != nil {
			report.Errors = append(report.Errors, fmt.Sprintf("reconcile collection dimensions: %v", err))
		} else {
			report.Reconciled = reconciled
		}
		restored, err := db.restoreStrippedChunks(ctx)
		report.Restored = restored
		if err != nil {
			report.Errors = append(report.Errors, fmt.Sprintf("restore stripped chunks: %v", err))
		}
	}
	return report, nil
}

// restoreStrippedChunks gives back to chunk rows what a repair before v2.137.0
// took from them. That repair wrote each row back with its id, content and
// collection alone, so the chunk lost the document it belongs to and its
// metadata: search could no longer report where a hit came from, and the
// expiry and review gates, which find a chunk's document by doc_id, let it
// through. A chunk's id names its document (graphChunkNodeID), and the
// document record keeps its title and metadata, which is what ingest copied
// onto each chunk, so both come back. Its ACL does not: no other row kept it.
// A row whose document is gone, or whose id is not a chunk id, is left alone.
func (db *DB) restoreStrippedChunks(ctx context.Context) (int, error) {
	rows, err := db.query(ctx, `SELECT id FROM embeddings WHERE (doc_id IS NULL OR doc_id = '') AND id LIKE 'chunk:%'`)
	if err != nil {
		return 0, err
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			_ = rows.Close()
			return 0, err
		}
		ids = append(ids, id)
	}
	if err := rows.Close(); err != nil {
		return 0, err
	}

	restored := 0
	var restoredRows []*core.Embedding
	for _, id := range ids {
		docID, index, ok := parseChunkID(id)
		if !ok {
			continue
		}
		doc, err := db.store.GetDocument(ctx, docID)
		if errors.Is(err, core.ErrNotFound) {
			continue
		}
		if err != nil {
			return restored, err
		}
		row, err := db.store.GetByID(ctx, id)
		if err != nil {
			return restored, err
		}
		row.DocID = docID
		if len(row.Metadata) == 0 {
			metadata := map[string]string{
				"graph_kind":  "chunk",
				"document_id": docID,
				"chunk_index": strconv.Itoa(index),
				"title":       doc.Title,
			}
			for k, v := range anyMapToStringMap(doc.Metadata) {
				if !recordOnlyKey(k) {
					metadata[k] = v
				}
			}
			row.Metadata = metadata
		}
		restoredRows = append(restoredRows, row)
		restored++
	}
	if len(restoredRows) == 0 {
		return 0, nil
	}
	if err := db.store.UpsertBatch(ctx, restoredRows); err != nil {
		return 0, err
	}
	return restored, nil
}

// parseChunkID reads the document id and the chunk index out of a chunk id
// made by graphChunkNodeID: "chunk:<document id>:<index>". The document id
// may itself hold colons; the index is what follows the last one.
func parseChunkID(id string) (string, int, bool) {
	rest, ok := strings.CutPrefix(id, "chunk:")
	if !ok {
		return "", 0, false
	}
	i := strings.LastIndexByte(rest, ':')
	if i <= 0 {
		return "", 0, false
	}
	index, err := strconv.Atoi(rest[i+1:])
	if err != nil || index < 0 {
		return "", 0, false
	}
	return rest[:i], index, true
}

// DimensionReport surfaces vector-dimension drift across the store, so a caller can see
// whether a re-embedding pass is needed.
func (db *DB) DimensionReport(ctx context.Context) (*core.DimensionReport, error) {
	return db.store.DimensionReport(ctx)
}

// VectorDimensionRepairRequest is the `vector_dimension_repair` MCP tool input.
type VectorDimensionRepairRequest struct {
	DryRun    *bool `json:"dry_run,omitempty"`
	Limit     int   `json:"limit,omitempty"`
	BatchSize int   `json:"batch_size,omitempty"`
}

// VectorDimensionRepairResponse pairs the drift report with the repair outcome.
type VectorDimensionRepairResponse struct {
	Report *core.DimensionReport `json:"report"`
	Repair *ReembedReport        `json:"repair,omitempty"`
}

// RepairVectorDimensions backs the `vector_dimension_repair` MCP tool. It always reports
// the drift; it only rewrites vectors when dry_run is explicitly false.
func (db *DB) RepairVectorDimensions(ctx context.Context, req VectorDimensionRepairRequest) (*VectorDimensionRepairResponse, error) {
	report, err := db.DimensionReport(ctx)
	if err != nil {
		return nil, err
	}
	response := &VectorDimensionRepairResponse{Report: report}
	dryRun := true
	if req.DryRun != nil {
		dryRun = *req.DryRun
	}
	repair, err := db.ReembedMismatchedVectors(ctx, ReembedOptions{
		Limit:     req.Limit,
		BatchSize: req.BatchSize,
		DryRun:    dryRun,
	})
	if err != nil {
		return response, err
	}
	response.Repair = repair
	return response, nil
}

// ReembedMemoryVectors embeds stored memories whose vector is missing or has
// the wrong dimensionality for the configured embedder.
//
// ReembedMismatchedVectors covers knowledge chunks only; memories live in the
// messages table and were left out, so a store that ran without an embedder
// accumulated memories with no vector — invisible to semantic recall while
// lexical search still found them, which masked the gap. Only memory buckets
// (session ids under the `memory:` prefix) are touched, not chat history.
func (db *DB) ReembedMemoryVectors(ctx context.Context, opts ReembedOptions) (*ReembedReport, error) {
	if db.embedder == nil {
		return nil, ErrEmbedderNotConfigured
	}
	targetDim := db.embedder.Dim()
	if targetDim <= 0 {
		return nil, fmt.Errorf("cortexdb: embedder reports dimension %d", targetDim)
	}
	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = 16
	}

	clause, args := db.missingMemoryVectorClause()
	query := `SELECT id, content FROM messages WHERE ` + clause + ` ORDER BY created_at DESC`
	if opts.Limit > 0 {
		query += " LIMIT ?"
		args = append(args, opts.Limit)
	}
	rows, err := db.query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list memories needing vectors: %w", err)
	}
	type pending struct{ id, content string }
	var stale []pending
	for rows.Next() {
		var p pending
		if err := rows.Scan(&p.id, &p.content); err != nil {
			_ = rows.Close()
			return nil, fmt.Errorf("scan memory: %w", err)
		}
		stale = append(stale, p)
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return nil, err
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}

	report := &ReembedReport{TargetDim: targetDim, Candidates: len(stale), DryRun: opts.DryRun}
	if opts.DryRun {
		return report, nil
	}

	for start := 0; start < len(stale); start += batchSize {
		end := start + batchSize
		if end > len(stale) {
			end = len(stale)
		}
		batch := stale[start:end]
		texts := make([]string, len(batch))
		for i, p := range batch {
			texts[i] = p.content
		}
		vectors, err := db.embedder.EmbedBatch(ctx, texts)
		if err != nil {
			report.Failed += len(batch)
			report.Errors = append(report.Errors, fmt.Sprintf("embed batch at %d: %v", start, err))
			continue
		}
		if len(vectors) != len(batch) {
			report.Failed += len(batch)
			report.Errors = append(report.Errors,
				fmt.Sprintf("embed batch at %d returned %d vectors for %d texts", start, len(vectors), len(batch)))
			continue
		}
		for i, p := range batch {
			if len(vectors[i]) != targetDim {
				report.Failed++
				report.Errors = append(report.Errors,
					fmt.Sprintf("memory %s: embedder returned %d dimensions, want %d", p.id, len(vectors[i]), targetDim))
				continue
			}
			encoded, err := encodeMemoryVector(db.Dialect(), vectors[i])
			if err != nil {
				report.Failed++
				report.Errors = append(report.Errors, fmt.Sprintf("memory %s: encode vector: %v", p.id, err))
				continue
			}
			if _, err := db.exec(ctx,
				`UPDATE messages SET vector = ? WHERE id = ?`,
				memoryVectorArg(db.Dialect(), encoded), p.id); err != nil {
				report.Failed++
				report.Errors = append(report.Errors, fmt.Sprintf("memory %s: write vector: %v", p.id, err))
				continue
			}
			report.Reembedded++
		}
	}
	return report, nil
}
