package cortexdb

import (
	"context"
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// knowledgeGetList calls knowledge_get through the tool surface, as a port
// does, and reads the ids it served and the ids it listed as missing.
func knowledgeGetList(t *testing.T, db *DB, ctx context.Context, args string) (served, missing []string, first string, err error) {
	t.Helper()
	out, err := db.GraphRAGTools().Call(ctx, "knowledge_get", json.RawMessage(args))
	if err != nil {
		return nil, nil, "", err
	}
	raw, _ := json.Marshal(out)
	var resp struct {
		Knowledge struct {
			ID string `json:"id"`
		} `json:"knowledge"`
		Knowledges []struct {
			ID string `json:"id"`
		} `json:"knowledges"`
		Missing []string `json:"missing"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatal(err)
	}
	for _, k := range resp.Knowledges {
		served = append(served, k.ID)
	}
	return served, resp.Missing, resp.Knowledge.ID, nil
}

func openListBrain(t *testing.T, opts ...Option) *DB {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "knowledge-list.db")), opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// knowledge_get with knowledge_ids serves the documents in the order asked,
// and lists an id that names no document, the empty one included, under
// missing instead of failing the call. knowledge is the first one served, as memory is for memory_get.
// With both forms the list is read, as memory_get reads its list.
func TestKnowledgeGetListsSeveralInTheOrderAsked(t *testing.T) {
	db := openListBrain(t)
	ctx := context.Background()
	for _, id := range []string{"k1", "k2"} {
		if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: id, Content: "The text of " + id + "."}); err != nil {
			t.Fatal(err)
		}
	}
	served, missing, first, err := knowledgeGetList(t, db, ctx, `{"knowledge_ids":["k2","absent","k1",""]}`)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(served, []string{"k2", "k1"}) || !reflect.DeepEqual(missing, []string{"absent", ""}) || first != "k2" {
		t.Errorf("served %v, missing %q, first %q; want [k2 k1], [absent \"\"], k2", served, missing, first)
	}
	served, _, _, err = knowledgeGetList(t, db, ctx, `{"knowledge_id":"k1","knowledge_ids":["k2"]}`)
	if err != nil || !reflect.DeepEqual(served, []string{"k2"}) {
		t.Errorf("both forms: served %v (%v), want the list's [k2]", served, err)
	}
	served, missing, first, err = knowledgeGetList(t, db, ctx, `{"knowledge_ids":["absent"]}`)
	if err != nil || served != nil || !reflect.DeepEqual(missing, []string{"absent"}) || first != "" {
		t.Errorf("nothing found: served %v, missing %q, first %q (%v); want none, [absent], none", served, missing, first, err)
	}
}

// Under a caller scope, the list form lists a document of a collection the
// caller may not read under missing, at its place in the order asked, exactly
// as an id that names nothing, and so a document the caller cannot place
// (one without chunks, which sits in no collection). The single form stays
// unscoped on purpose: the row guard of the ports reads through it. A
// document that cannot be read is an error, never missing.
func TestKnowledgeGetListHidesAnotherCollectionAsMissing(t *testing.T) {
	db := openListBrain(t)
	ctx := context.Background()
	for _, k := range []KnowledgeSaveRequest{
		{KnowledgeID: "notes-1", Collection: "notes", Content: "Orpheus plays the lyre."},
		{KnowledgeID: "notes-bare", Collection: "notes", Content: "A document that will lose its chunks."},
		{KnowledgeID: "ops-1", Collection: "ops", Content: "The salaries of 2026."},
	} {
		if _, err := db.SaveKnowledge(ctx, k); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.SQL().Exec(`DELETE FROM embeddings WHERE doc_id = 'notes-bare'`); err != nil {
		t.Fatal(err)
	}
	// The scope reaches the default collection too, where the facade
	// places a document without chunks: such a document is still not
	// listed, as the row guard withholds it from every confined key.
	notes := WithCallerScope(ctx, CallerScope{Key: "notes", Knowledge: func(c string) bool { return c == "notes" || c == defaultGraphRAGCollection }})
	for _, ids := range [][]string{{"ops-1", "absent", "notes-1"}, {"absent", "notes-1", "ops-1", "notes-bare"}} {
		args, _ := json.Marshal(map[string]any{"knowledge_ids": ids})
		served, missing, _, err := knowledgeGetList(t, db, notes, string(args))
		if err != nil {
			t.Fatalf("%v: %v", ids, err)
		}
		var want []string
		for _, id := range ids {
			if id != "notes-1" {
				want = append(want, id)
			}
		}
		if !reflect.DeepEqual(served, []string{"notes-1"}) || !reflect.DeepEqual(missing, want) {
			t.Errorf("%v: served %v, missing %v; want [notes-1], %v", ids, served, missing, want)
		}
	}
	// Unconfined, the list serves every collection; under the scope the
	// single form still reads the foreign row, for the guard.
	if served, missing, _, err := knowledgeGetList(t, db, ctx, `{"knowledge_ids":["ops-1"]}`); err != nil || len(missing) != 0 || !reflect.DeepEqual(served, []string{"ops-1"}) {
		t.Errorf("unconfined list read: served %v, missing %v, %v", served, missing, err)
	}
	// A scope with a memory rule and no knowledge rule reaches every
	// collection, as knowledgeReadPermitted reads it.
	memoryOnly := WithCallerScope(ctx, CallerScope{Key: "memory-only", Memory: func(MemoryBucket) bool { return false }})
	if served, missing, _, err := knowledgeGetList(t, db, memoryOnly, `{"knowledge_ids":["ops-1","notes-bare"]}`); err != nil || len(missing) != 0 || !reflect.DeepEqual(served, []string{"ops-1", "notes-bare"}) {
		t.Errorf("memory-only scope: served %v, missing %v, %v; want both", served, missing, err)
	}
	if got, err := db.GetKnowledge(notes, KnowledgeGetRequest{KnowledgeID: "ops-1"}); err != nil || got.Knowledge.ID != "ops-1" {
		t.Errorf("the single form was scoped, which blinds the row guard: %v", err)
	}
	if _, err := db.SQL().Exec(`ALTER TABLE documents RENAME COLUMN title TO title_hidden`); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := knowledgeGetList(t, db, notes, `{"knowledge_ids":["notes-1"]}`); err == nil || strings.Contains(err.Error(), "not found") {
		t.Errorf("an unreadable document: %v, want the read error", err)
	}
}

// The list form serves what the single form serves for the same id: a
// document that has expired, and a proposed one under the review gate, which
// knowledge_get reads by id (ReviewGateNote).
func TestKnowledgeGetListServesWhatTheSingleFormServes(t *testing.T) {
	db := openListBrain(t, WithReviewGate())
	ctx := context.Background()
	past := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	proposed := "proposed"
	if _, err := db.GraphRAGTools().IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "old", Content: "An expired notice.", ExpiresAt: &past}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.GraphRAGTools().IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "draft", Content: "A proposed page.", ReviewStatus: proposed}); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"old", "draft"} {
		if _, err := db.GetKnowledge(ctx, KnowledgeGetRequest{KnowledgeID: id}); err != nil {
			t.Fatalf("the single form does not serve %s: %v", id, err)
		}
	}
	served, missing, _, err := knowledgeGetList(t, db, ctx, `{"knowledge_ids":["old","draft"]}`)
	if err != nil || len(missing) != 0 || !reflect.DeepEqual(served, []string{"old", "draft"}) {
		t.Errorf("served %v, missing %v (%v), want both, as the single form serves them", served, missing, err)
	}
}
