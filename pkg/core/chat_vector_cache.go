package core

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/liliang-cn/cortexdb/v2/internal/encoding"
)

// A memory recall scores the query against every vector in one session (a
// memory bucket). Reading those vectors out of SQLite and decoding them was
// most of a recall on a small node: a thousand 768-dimension vectors are 3 MB
// of overflow pages, read through the page cache of whichever pooled
// connection the query landed on. The decoded vectors are kept here instead,
// per session, and checked against a version SQLite itself bumps on every
// write that could change them — an insert, a delete, an update of the vector
// or of the session — whoever makes it, this process or another. A check is
// one primary-key lookup; a stale entry is reloaded, never served.

// messageVectorSchema is the version table, the triggers that keep it, and
// the FTS update triggers narrowed to the one column the indexes hold.
//
// The FTS triggers used to fire on an update of any column, so every recall
// counter bumped in a memory's metadata deleted and re-inserted its text in
// both full-text indexes: two index rewrites per recalled memory, on every
// recall, under the write lock the next search was waiting for.
const messageVectorSchema = `
CREATE TABLE IF NOT EXISTS message_vector_versions (
	session_id TEXT PRIMARY KEY,
	version INTEGER NOT NULL
);
CREATE TRIGGER IF NOT EXISTS messages_vv_ai AFTER INSERT ON messages WHEN new.vector IS NOT NULL BEGIN
	INSERT INTO message_vector_versions(session_id, version) VALUES (new.session_id, 1)
	ON CONFLICT(session_id) DO UPDATE SET version = version + 1;
END;
CREATE TRIGGER IF NOT EXISTS messages_vv_ad AFTER DELETE ON messages WHEN old.vector IS NOT NULL BEGIN
	INSERT INTO message_vector_versions(session_id, version) VALUES (old.session_id, 1)
	ON CONFLICT(session_id) DO UPDATE SET version = version + 1;
END;
CREATE TRIGGER IF NOT EXISTS messages_vv_au AFTER UPDATE OF vector, session_id ON messages BEGIN
	INSERT INTO message_vector_versions(session_id, version) VALUES (old.session_id, 1)
	ON CONFLICT(session_id) DO UPDATE SET version = version + 1;
	INSERT INTO message_vector_versions(session_id, version) VALUES (new.session_id, 1)
	ON CONFLICT(session_id) DO UPDATE SET version = version + 1;
END;

DROP TRIGGER IF EXISTS messages_au;
CREATE TRIGGER messages_au AFTER UPDATE OF content ON messages BEGIN
	INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.rowid, old.content);
	INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
DROP TRIGGER IF EXISTS messages_cjk_au;
CREATE TRIGGER messages_cjk_au AFTER UPDATE OF content ON messages BEGIN
	INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content) VALUES('delete', old.rowid, old.content);
	INSERT INTO messages_fts_cjk(rowid, content) VALUES (new.rowid, new.content);
END;
`

// maxCachedMessageVectors bounds the cache: 30,000 768-dimension vectors are
// about 90 MB. Past it the least recently searched sessions are dropped.
const maxCachedMessageVectors = 30000

type sessionVectors struct {
	version  int64
	ids      []string
	vectors  [][]float32
	lastUsed time.Time
}

type messageVectorCache struct {
	mu       sync.Mutex
	sessions map[string]*sessionVectors
	count    int
}

func (s *SQLiteStore) initMessageVectorCache(ctx context.Context) error {
	if _, err := s.db.ExecContext(ctx, messageVectorSchema); err != nil {
		return fmt.Errorf("failed to create message vector versions: %w", err)
	}
	return nil
}

// messageVectorVersion is the session's vector version; 0 when nothing has
// been written to it since the table existed.
func (s *SQLiteStore) messageVectorVersion(ctx context.Context, sessionID string) (int64, error) {
	var version int64
	err := s.db.QueryRowContext(ctx, `SELECT version FROM message_vector_versions WHERE session_id = ?`, sessionID).Scan(&version)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, nil
	}
	return version, err
}

// sessionMessageVectors returns the ids and decoded vectors of a session's
// messages that have one, from the cache when it is current.
func (s *SQLiteStore) sessionMessageVectors(ctx context.Context, sessionID string) ([]string, [][]float32, error) {
	version, err := s.messageVectorVersion(ctx, sessionID)
	if err != nil {
		return nil, nil, err
	}
	c := &s.vectorCache
	c.mu.Lock()
	if e, ok := c.sessions[sessionID]; ok && e.version == version {
		e.lastUsed = time.Now()
		ids, vectors := e.ids, e.vectors
		c.mu.Unlock()
		return ids, vectors, nil
	}
	c.mu.Unlock()

	// Loaded after the version was read, so a write landing in between makes
	// this entry newer than its version, and the next search reloads it: the
	// cache can lag a write by one reload, never miss one.
	rows, err := s.db.QueryContext(ctx, `SELECT id, vector FROM messages WHERE session_id = ? AND vector IS NOT NULL`, sessionID)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	var ids []string
	var vectors [][]float32
	for rows.Next() {
		var id string
		var vectorBytes []byte
		if err := rows.Scan(&id, &vectorBytes); err != nil {
			return nil, nil, err
		}
		vector, err := encoding.DecodeVector(vectorBytes)
		if err != nil || len(vector) == 0 {
			continue
		}
		ids = append(ids, id)
		vectors = append(vectors, vector)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.sessions == nil {
		c.sessions = map[string]*sessionVectors{}
	}
	if old, ok := c.sessions[sessionID]; ok {
		c.count -= len(old.ids)
		delete(c.sessions, sessionID)
	}
	if len(ids) <= maxCachedMessageVectors {
		for c.count+len(ids) > maxCachedMessageVectors {
			c.evictOldest()
		}
		c.sessions[sessionID] = &sessionVectors{version: version, ids: ids, vectors: vectors, lastUsed: time.Now()}
		c.count += len(ids)
	}
	return ids, vectors, nil
}

func (c *messageVectorCache) evictOldest() {
	var oldest string
	var at time.Time
	for id, e := range c.sessions {
		if oldest == "" || e.lastUsed.Before(at) {
			oldest, at = id, e.lastUsed
		}
	}
	if oldest == "" {
		c.count = 0
		return
	}
	c.count -= len(c.sessions[oldest].ids)
	delete(c.sessions, oldest)
}
