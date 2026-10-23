package core

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/liliang-cn/cortexdb/v2/internal/encoding"
)

// A loaded snapshot is checked against the table before it is trusted.
//
// The snapshot is the index as it stood when it was written, and nothing used
// to compare it with the rows it indexes. Whatever had gone wrong with the
// index in memory — a write that did not reach it, a vector it refused, a row
// another process added or deleted — was saved with it and loaded back on
// every open after, for good. Measured on a copy of a live brain: of 529
// rows, 316 were not in the loaded index at all, 66 nodes still held the
// 64-d lexical vectors their rows had long since been re-embedded away from,
// and 1,296 of the graph's 1,509 nodes were tombstones; a question whose
// nearest chunk scored 0.476 came back with chunks at 0.35 and below.
//
// So after a load the store reconciles: rows the index lacks are inserted,
// ids the table no longer has are removed, and rows written since the
// snapshot are written again, which replaces a vector that changed under the
// same id (and costs nothing when it did not). An index that cannot be
// patched — vectors of a dimension the table does not hold, more tombstones
// than live nodes, or a difference larger than snapshotRebuildFraction of
// the table — is rebuilt from the rows instead.
//
// The cost, and why it is paid. The check reads ids and vector lengths only:
// length() of a BLOB is answered from the record header, so no vector is read
// for a row the index already holds. Counts are compared first, so an index
// that is plainly far off is rebuilt without the id pass. Vectors are read
// only for the rows that need them. On the 529-row brain the whole check takes
// a few milliseconds; at 100,000 rows it is one scan of the id column, small
// beside decoding the snapshot itself. The pass cannot see a vector changed
// under the same id by another process before the snapshot was written; the
// window it refreshes (snapshotRefreshMargin before the snapshot's time) is
// there for clock skew between the nodes of a failover pair, not for that.

// snapshotRebuildFraction is how much of the table a reconciliation may
// touch before rebuilding is the better answer: a snapshot that far from the
// table is evidence of a fault rather than of a few writes it missed, and
// patching it would leave a graph built mostly by repair.
const snapshotRebuildFraction = 0.3

// snapshotRefreshMargin widens the window of rows rewritten after a load, to
// cover a clock that differs between the process that wrote the snapshot and
// the one that wrote the rows.
const snapshotRefreshMargin = 10 * time.Minute

// reconcileChunk is how many vectors are fetched per query when patching.
const reconcileChunk = 256

// loadedIndex is what reconciliation needs of an index it did not build.
type loadedIndex struct {
	kind       string
	live       map[string]int // id -> dimension of the vector it holds, 0 unknown
	tombstones int
	upsert     func(id string, vector []float32) error
	remove     func(id string) error
}

// reconcileResult says what reconciliation did, or why it gave up.
type reconcileResult struct {
	unusable  string // non-empty: rebuild instead, for this reason
	added     int
	removed   int
	refreshed int
	failed    int
	elapsed   time.Duration
}

func (r reconcileResult) changed() bool { return r.added+r.removed+r.refreshed+r.failed > 0 }

// tableVectorDims counts the stored rows by vector dimension, reading only
// the length of each vector.
func (s *SQLiteStore) tableVectorDims(ctx context.Context) (map[int]int, int, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT (length(vector) - 4) / 4, count(*) FROM embeddings GROUP BY 1`)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = rows.Close() }()
	dims := map[int]int{}
	total := 0
	for rows.Next() {
		var dim, n int
		if err := rows.Scan(&dim, &n); err != nil {
			return nil, 0, err
		}
		dims[dim] += n
		total += n
	}
	return dims, total, rows.Err()
}

// majorityDim is the dimension most rows hold, 0 for an empty table.
func majorityDim(dims map[int]int) int {
	best, bestN := 0, 0
	for dim, n := range dims {
		if dim > 0 && (n > bestN || (n == bestN && dim > best)) {
			best, bestN = dim, n
		}
	}
	return best
}

// snapshotRefreshCutoff is the time from which rows are rewritten after a
// load, as SQLite's datetime text, or "" when the snapshot has no time.
func (s *SQLiteStore) snapshotRefreshCutoff(ctx context.Context, indexType string) string {
	lo, hi := chunkRange(indexType)
	var cutoff sql.NullString
	err := s.db.QueryRowContext(ctx,
		`SELECT datetime(min(created_at), ?) FROM index_snapshots WHERE type = ? OR (type >= ? AND type < ?)`,
		fmt.Sprintf("-%d seconds", int(snapshotRefreshMargin.Seconds())), indexType, lo, hi).Scan(&cutoff)
	if err != nil || !cutoff.Valid {
		return ""
	}
	return cutoff.String
}

// reconcileLoadedIndex brings a loaded index into line with the table; see
// the comment at the top of this file.
func (s *SQLiteStore) reconcileLoadedIndex(ctx context.Context, idx loadedIndex) (res reconcileResult, err error) {
	start := time.Now()
	defer func() { res.elapsed = time.Since(start) }()

	tableDims, tableRows, err := s.tableVectorDims(ctx)
	if err != nil {
		return res, fmt.Errorf("count stored vectors: %w", err)
	}

	// The dimension the index holds. Nodes of two dimensions, or of one the
	// table has none of, cannot be patched into anything a query can use.
	indexDim := 0
	for _, dim := range idx.live {
		if dim == 0 {
			continue
		}
		if indexDim == 0 {
			indexDim = dim
		} else if dim != indexDim {
			res.unusable = "it holds vectors of more than one dimension"
			return res, nil
		}
	}
	if indexDim > 0 && tableDims[indexDim] == 0 && tableRows > 0 {
		res.unusable = fmt.Sprintf("it holds %d-d vectors and the table has none", indexDim)
		return res, nil
	}
	if indexDim == 0 {
		indexDim = majorityDim(tableDims)
	}
	if idx.tombstones > len(idx.live) {
		res.unusable = fmt.Sprintf("%d of its %d nodes are deleted", idx.tombstones, idx.tombstones+len(idx.live))
		return res, nil
	}

	// Counts first: the rows the index could hold against the ids it holds.
	indexable := tableRows
	if indexDim > 0 {
		indexable = tableDims[indexDim]
	}
	limit := int(snapshotRebuildFraction * float64(max(indexable, len(idx.live), 1)))
	if diff := indexable - len(idx.live); diff > limit || -diff > limit {
		res.unusable = fmt.Sprintf("it holds %d vectors and the table %d", len(idx.live), indexable)
		return res, nil
	}

	// Then the ids, streamed: one pass, no vectors read.
	cutoff := s.snapshotRefreshCutoff(ctx, idx.kind)
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, (length(vector) - 4) / 4, coalesce(datetime(created_at) >= ?, 0) FROM embeddings`, cutoff)
	if err != nil {
		return res, fmt.Errorf("list stored ids: %w", err)
	}
	unseen := make(map[string]struct{}, len(idx.live))
	for id := range idx.live {
		unseen[id] = struct{}{}
	}
	var missing, refresh []string
	for rows.Next() {
		var id string
		var dim int
		var recent bool
		if err := rows.Scan(&id, &dim, &recent); err != nil {
			_ = rows.Close()
			return res, fmt.Errorf("list stored ids: %w", err)
		}
		if _, ok := unseen[id]; ok {
			delete(unseen, id)
			if recent && cutoff != "" {
				refresh = append(refresh, id)
			}
			continue
		}
		if indexDim > 0 && dim != indexDim {
			continue // a vector this index cannot hold; a rebuild would skip it too
		}
		missing = append(missing, id)
	}
	if err := rows.Close(); err != nil {
		return res, fmt.Errorf("list stored ids: %w", err)
	}
	if err := rows.Err(); err != nil {
		return res, fmt.Errorf("list stored ids: %w", err)
	}

	if touched := len(missing) + len(unseen); touched > limit {
		res.unusable = fmt.Sprintf("%d of its ids are missing and %d are no longer stored", len(missing), len(unseen))
		return res, nil
	}

	for id := range unseen {
		if err := idx.remove(id); err != nil {
			res.failed++
			continue
		}
		res.removed++
	}
	apply := func(ids []string, count *int) error {
		for len(ids) > 0 {
			n := min(len(ids), reconcileChunk)
			batch := ids[:n]
			ids = ids[n:]
			vectors, err := s.vectorsByID(ctx, batch)
			if err != nil {
				return err
			}
			for _, id := range batch {
				vec, ok := vectors[id]
				if !ok {
					continue // deleted since the id pass
				}
				before := s.indexMutations()
				if err := idx.upsert(id, vec); err != nil {
					s.logger.Warn("failed to reconcile vector into index", "type", idx.kind, "id", id, "error", err)
					res.failed++
					continue
				}
				if s.indexMutations() != before {
					*count++
				}
			}
		}
		return nil
	}
	if err := apply(missing, &res.added); err != nil {
		return res, fmt.Errorf("fetch missing vectors: %w", err)
	}
	if err := apply(refresh, &res.refreshed); err != nil {
		return res, fmt.Errorf("fetch recent vectors: %w", err)
	}
	return res, nil
}

// vectorsByID reads the stored vectors of ids.
func (s *SQLiteStore) vectorsByID(ctx context.Context, ids []string) (map[string][]float32, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	args := make([]any, len(ids))
	for i, id := range ids {
		args[i] = id
	}
	query := `SELECT id, vector FROM embeddings WHERE id IN (?` + strings.Repeat(",?", len(ids)-1) + `)`
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := make(map[string][]float32, len(ids))
	for rows.Next() {
		var id string
		var blob []byte
		if err := rows.Scan(&id, &blob); err != nil {
			return nil, err
		}
		vec, err := encoding.DecodeVector(blob)
		if err != nil {
			s.logger.Warn("failed to decode vector during index reconciliation", "id", id, "error", err)
			continue
		}
		out[id] = vec
	}
	return out, rows.Err()
}

// notice logs what an operator should see at startup. The facade opens the
// store with no logger at all, so what reconciliation did would otherwise
// be invisible exactly where it matters, on a server; the standard logger is
// what the CJK backfill already writes to for the same reason.
func (s *SQLiteStore) notice(msg string, keyvals ...any) {
	if _, silent := s.logger.(nopLogger); !silent {
		s.logger.Info(msg, keyvals...)
		return
	}
	var b strings.Builder
	b.WriteString("cortexdb: ")
	b.WriteString(msg)
	for i := 0; i+1 < len(keyvals); i += 2 {
		fmt.Fprintf(&b, " %v=%v", keyvals[i], keyvals[i+1])
	}
	log.Print(b.String())
}

// reportReconcile logs the outcome of a reconciliation.
func (s *SQLiteStore) reportReconcile(kind string, res reconcileResult) {
	switch {
	case res.unusable != "":
		s.notice(kind+" snapshot does not match the stored vectors, rebuilding", "reason", res.unusable,
			"took", res.elapsed.Round(100*time.Microsecond))
	case res.changed():
		s.notice(kind+" snapshot reconciled with the stored vectors",
			"added", res.added, "removed", res.removed, "refreshed", res.refreshed,
			"failed", res.failed, "took", res.elapsed.Round(100*time.Microsecond))
	default:
		s.logger.Info(kind+" snapshot matches the stored vectors", "took", res.elapsed.Round(100*time.Microsecond))
	}
}
