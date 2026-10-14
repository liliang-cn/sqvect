package core

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/internal/encoding"
)

func cacheTestStore(t *testing.T) *SQLiteStore {
	t.Helper()
	store, err := New(filepath.Join(t.TempDir(), "chat.db"), 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	for _, id := range []string{"a", "b"} {
		if err := store.CreateSession(context.Background(), &Session{ID: id, UserID: "u"}); err != nil {
			t.Fatal(err)
		}
	}
	return store
}

func topIDs(t *testing.T, s *SQLiteStore, session string, q []float32) []string {
	t.Helper()
	scored, err := s.SearchChatHistoryScored(context.Background(), q, session, 10)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(scored))
	for i, m := range scored {
		ids[i] = m.Message.ID
	}
	return ids
}

func sameMessageIDs(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The cached vectors must follow every write that can change them, through
// this store's methods or straight SQL, or a recall ranks by a vector that is
// no longer in the database.
func TestVectorCacheFollowsEveryWrite(t *testing.T) {
	ctx := context.Background()
	s := cacheTestStore(t)
	q := []float32{1, 0, 0, 0}
	add := func(id string, v []float32) {
		if err := s.AddMessage(ctx, &Message{ID: id, SessionID: "a", Role: "user", Content: id, Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	add("near", []float32{1, 0.1, 0, 0})
	add("far", []float32{0, 1, 0, 0})
	if got := topIDs(t, s, "a", q); !sameMessageIDs(got, []string{"near", "far"}) {
		t.Fatalf("first search: %v", got)
	}

	add("exact", []float32{1, 0, 0, 0})
	if got := topIDs(t, s, "a", q); !sameMessageIDs(got, []string{"exact", "near", "far"}) {
		t.Fatalf("after insert: %v", got)
	}

	if _, err := s.db.ExecContext(ctx, `DELETE FROM messages WHERE id = 'exact'`); err != nil {
		t.Fatal(err)
	}
	if got := topIDs(t, s, "a", q); !sameMessageIDs(got, []string{"near", "far"}) {
		t.Fatalf("after delete: %v", got)
	}

	data, err := encoding.EncodeVector([]float32{1, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.ExecContext(ctx, `UPDATE messages SET vector = ? WHERE id = 'far'`, data); err != nil {
		t.Fatal(err)
	}
	if got := topIDs(t, s, "a", q); !sameMessageIDs(got, []string{"far", "near"}) {
		t.Fatalf("after a vector update by plain SQL: %v", got)
	}

	if _, err := s.db.ExecContext(ctx, `UPDATE messages SET session_id = 'b' WHERE id = 'far'`); err != nil {
		t.Fatal(err)
	}
	if got := topIDs(t, s, "a", q); !sameMessageIDs(got, []string{"near"}) {
		t.Fatalf("session a after a move: %v", got)
	}
	if got := topIDs(t, s, "b", q); !sameMessageIDs(got, []string{"far"}) {
		t.Fatalf("session b after a move: %v", got)
	}

	// A vector handed back is a copy: changing it changes nothing stored.
	scored, _ := s.SearchChatHistoryScored(ctx, q, "a", 1)
	scored[0].Message.Vector[0] = -1
	if got := topIDs(t, s, "a", q); !sameMessageIDs(got, []string{"near"}) {
		t.Fatalf("after mutating a result: %v", got)
	}
	again, _ := s.SearchChatHistoryScored(ctx, q, "a", 1)
	if again[0].Message.Vector[0] != 1 {
		t.Fatalf("a caller's change reached the cache: %v", again[0].Message.Vector)
	}
}

// Recall counters live in a memory's metadata. Updating them must not touch
// the full-text index, which holds only content.
func TestMetadataUpdateLeavesTheTextIndexAlone(t *testing.T) {
	ctx := context.Background()
	s := cacheTestStore(t)
	if err := s.AddMessage(ctx, &Message{ID: "m", SessionID: "a", Role: "user", Content: "drbd failover with haify"}); err != nil {
		t.Fatal(err)
	}
	var before int
	if err := s.db.QueryRowContext(ctx, `SELECT count(*) FROM messages_fts_data`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		if _, err := s.db.ExecContext(ctx, `UPDATE messages SET metadata = json_set(coalesce(metadata,'{}'), '$.recall_count', ?) WHERE id = 'm'`, i); err != nil {
			t.Fatal(err)
		}
	}
	var after int
	if err := s.db.QueryRowContext(ctx, `SELECT count(*) FROM messages_fts_data`).Scan(&after); err != nil {
		t.Fatal(err)
	}
	if after != before {
		t.Fatalf("metadata updates rewrote the text index: %d segments before, %d after", before, after)
	}
	if _, err := s.db.ExecContext(ctx, `UPDATE messages SET content = 'linstor satellite' WHERE id = 'm'`); err != nil {
		t.Fatal(err)
	}
	var hits int
	if err := s.db.QueryRowContext(ctx, `SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'satellite'`).Scan(&hits); err != nil {
		t.Fatal(err)
	}
	if hits != 1 {
		t.Fatalf("a content update must still reach the text index: %d hits", hits)
	}
}
