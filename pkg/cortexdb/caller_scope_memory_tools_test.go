package cortexdb

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
)

// memory_list_all under a caller scope lists the memories of the buckets the
// scope permits, and only those: zeus's memory is not listed under hermes's
// scope, and its text is nowhere in the answer. Without a scope both are
// listed, as before.
func TestMemoryListAllListsOnlyWhatTheCallerScopePermits(t *testing.T) {
	db, _, _, hermes := openExtractScopeDB(t)
	tb := db.GraphRAGTools()
	ids := func(ctx context.Context) ([]string, string) {
		out, err := tb.Call(ctx, "memory_list_all", json.RawMessage(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		body, _ := json.Marshal(out)
		var got []string
		for _, m := range out.(*MemoryListAllResponse).Memories {
			got = append(got, m.ID)
		}
		return got, string(body)
	}
	got, body := ids(hermes)
	if strings.Join(got, ",") != "hermes-1" || strings.Contains(body, "Thunderbolt") {
		t.Errorf("under hermes's scope memory_list_all listed %v: %s", got, body)
	}
	if got, _ := ids(context.Background()); len(got) != 2 {
		t.Errorf("without a scope memory_list_all listed %v, want both", got)
	}
}

// knowledge_memory_promote_to_knowledge answers a memory outside the scope
// exactly as one that does not exist, and writes no document. The caller's
// own memory, and any memory without a scope, are promoted as before.
func TestPromoteToKnowledgeAnswersAMemoryOutsideTheScopeAsMissing(t *testing.T) {
	db, _, _, hermes := openExtractScopeDB(t)
	tb := db.GraphRAGTools()
	ctx := context.Background()
	promote := func(ctx context.Context, memoryID, knowledgeID string) error {
		_, err := tb.Call(ctx, "knowledge_memory_promote_to_knowledge", json.RawMessage(`{"memory_ids":["`+memoryID+`"],"knowledge_id":"`+knowledgeID+`"}`))
		return err
	}
	errMissing := promote(hermes, "no-such-memory", "k-missing")
	if errMissing == nil {
		t.Fatal("a missing memory was promoted")
	}
	err := promote(hermes, "zeus-1", "k-zeus")
	if err == nil || err.Error() != errMissing.Error() {
		t.Errorf("zeus's memory under hermes's scope: %v, want the missing answer %v", err, errMissing)
	}
	if _, err := db.GetKnowledge(ctx, KnowledgeGetRequest{KnowledgeID: "k-zeus"}); err == nil {
		t.Error("zeus's memory was written as a document under hermes's scope")
	}
	for _, c := range []struct {
		ctx              context.Context
		memory, document string
	}{{hermes, "hermes-1", "k-hermes"}, {ctx, "zeus-1", "k-zeus-open"}} {
		if err := promote(c.ctx, c.memory, c.document); err != nil {
			t.Errorf("promote %s: %v", c.memory, err)
		}
	}
}

// knowledge_memory_consolidate promotes through the same check: a promotion
// that names a memory outside the scope fails as a missing one does and
// writes no document, where the caller's own memory is promoted. Its
// reflection reads through recall, which the scope already holds, so zeus's
// text is not in a reflection under hermes's scope.
func TestConsolidateAndReflectReadNothingOutsideTheScope(t *testing.T) {
	db, _, _, hermes := openExtractScopeDB(t)
	tb := db.GraphRAGTools()
	recall := `{"query":"Lyre Apollo Thunderbolt","user_id":"hermes","scope":"user"}`
	out, err := tb.Call(hermes, "knowledge_memory_reflect", json.RawMessage(`{"recall":`+recall+`}`))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(out)
	if !strings.Contains(string(body), "Lyre") || strings.Contains(string(body), "Olympus") {
		t.Errorf("a reflection under hermes's scope: %s, want hermes's memory and not zeus's", body)
	}
	consolidate := func(memoryID, knowledgeID string) error {
		_, err := tb.Call(hermes, "knowledge_memory_consolidate", json.RawMessage(`{"reflect":{"recall":`+recall+`},"user_id":"hermes","scope":"user","promotion":{"memory_ids":["`+memoryID+`"],"knowledge_id":"`+knowledgeID+`"}}`))
		return err
	}
	if err := consolidate("hermes-1", "k-own"); err != nil {
		t.Fatalf("consolidate with hermes's own memory: %v", err)
	}
	errMissing := consolidate("no-such-memory", "k-missing")
	if err := consolidate("zeus-1", "k-zeus"); err == nil || errMissing == nil || err.Error() != errMissing.Error() {
		t.Errorf("consolidate promoting zeus's memory under hermes's scope: %v, want the missing answer %v", err, errMissing)
	}
	if _, err := db.GetKnowledge(context.Background(), KnowledgeGetRequest{KnowledgeID: "k-zeus"}); err == nil {
		t.Error("consolidate wrote zeus's memory as a document under hermes's scope")
	}
}

// A bucket that cannot be read is an error on both tools, never a pass.
func TestTheMemoryToolsFailWhenABucketCannotBeRead(t *testing.T) {
	db, _, _, hermes := openExtractScopeDB(t)
	if _, err := db.SQL().Exec(`UPDATE sessions SET created_at = 'not a time'`); err != nil {
		t.Fatal(err)
	}
	tb := db.GraphRAGTools()
	if out, err := tb.Call(hermes, "memory_list_all", json.RawMessage(`{}`)); err == nil {
		t.Errorf("memory_list_all passed an unreadable bucket: %+v", out)
	}
	if _, err := tb.Call(hermes, "knowledge_memory_promote_to_knowledge", json.RawMessage(`{"memory_ids":["hermes-1"]}`)); err == nil || strings.Contains(err.Error(), "not found") {
		t.Errorf("promote with an unreadable bucket: %v, want the read error", err)
	}
}

// memory_list_all fills a page from the rows past the ones the scope skips,
// in the first batch it reads and in the next ones. hermes's memory is the
// oldest, behind zeus-1 and three newer memories of zeus: with limit 1 the
// first batch (limit+1 rows) holds none of hermes's, with limit 2 the first
// batch holds three of zeus's. Either way the page is [hermes-1], and
// nothing is left after it.
func TestMemoryListAllFillsAPagePastTheRowsTheScopeSkips(t *testing.T) {
	db, _, _, hermes := openExtractScopeDB(t)
	ctx := context.Background()
	for i := 2; i <= 4; i++ {
		id := fmt.Sprintf("zeus-%d", i)
		if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: id, UserID: "zeus", Scope: "user", Content: "Zeus keeps another secret."}); err != nil {
			t.Fatal(err)
		}
	}
	for id, minutes := range map[string]int{"hermes-1": 5, "zeus-1": 4, "zeus-2": 3, "zeus-3": 2, "zeus-4": 1} {
		if _, err := db.SQL().Exec(`UPDATE messages SET created_at = datetime('now', ?) WHERE id = ?`, fmt.Sprintf("-%d minutes", minutes), id); err != nil {
			t.Fatal(err)
		}
	}
	for _, limit := range []int{1, 2} {
		resp, err := db.GraphRAGTools().Call(hermes, "memory_list_all", json.RawMessage(fmt.Sprintf(`{"limit":%d}`, limit)))
		if err != nil {
			t.Fatal(err)
		}
		page := resp.(*MemoryListAllResponse)
		var got []string
		for _, m := range page.Memories {
			got = append(got, m.ID)
		}
		if strings.Join(got, ",") != "hermes-1" || page.Truncated {
			t.Errorf("limit %d: listed %v, truncated %v; want [hermes-1] and nothing after it", limit, got, page.Truncated)
		}
	}
}

// A promotion of several memories fails when any of them is outside the
// scope, the foreign one second included, and writes nothing.
func TestPromoteToKnowledgeRefusesAListWithAMemoryOutsideTheScope(t *testing.T) {
	db, _, _, hermes := openExtractScopeDB(t)
	if _, err := db.GraphRAGTools().Call(hermes, "knowledge_memory_promote_to_knowledge", json.RawMessage(`{"memory_ids":["hermes-1","zeus-1"],"knowledge_id":"k-mixed"}`)); err == nil {
		t.Error("a promotion of hermes-1 and zeus-1 passed under hermes's scope")
	}
	if _, err := db.GetKnowledge(context.Background(), KnowledgeGetRequest{KnowledgeID: "k-mixed"}); err == nil {
		t.Error("the mixed promotion wrote a document")
	}
}

// Under the review gate, a proposed memory outside the scope answers as a
// missing one, not with the gate's refusal, which would tell that it exists
// and that it is proposed.
func TestPromoteToKnowledgeAnswersAForeignProposedMemoryAsMissing(t *testing.T) {
	ctx := context.Background()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "promote-gate.db")), WithReviewGate())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "zeus-p", UserID: "zeus", Scope: "user", Content: "Zeus proposes a secret.", ReviewStatus: "proposed"}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "hermes-1", UserID: "hermes", Scope: "user", Content: "Hermes tuned the Lyre."}); err != nil {
		t.Fatal(err)
	}
	hermes := WithCallerScope(ctx, CallerScope{Key: "hermes", Memory: func(b MemoryBucket) bool { return b.UserID == "hermes" }})
	promote := func(id string) error {
		_, err := db.GraphRAGTools().Call(hermes, "knowledge_memory_promote_to_knowledge", json.RawMessage(`{"memory_ids":["`+id+`"]}`))
		return err
	}
	errMissing, err := promote("no-such-memory"), promote("zeus-p")
	if err == nil || errMissing == nil || err.Error() != errMissing.Error() {
		t.Errorf("zeus's proposed memory under hermes's scope: %v, want the missing answer %v", err, errMissing)
	}
	// Unconfined, the gate still refuses it in its own words.
	if _, err := db.GraphRAGTools().Call(ctx, "knowledge_memory_promote_to_knowledge", json.RawMessage(`{"memory_ids":["zeus-p"]}`)); err == nil || !strings.Contains(err.Error(), "proposed") {
		t.Errorf("unconfined, zeus's proposed memory: %v, want the gate's refusal", err)
	}
}

// memory_list_all asks the scope once per bucket in a call, not once per
// row: five rows in two buckets cost two answers, also when limit 1 makes
// the call read several batches.
func TestMemoryListAllAsksTheScopeOncePerBucket(t *testing.T) {
	db, _, _, _ := openExtractScopeDB(t)
	ctx := context.Background()
	for i := 2; i <= 4; i++ {
		if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: fmt.Sprintf("zeus-%d", i), UserID: "zeus", Scope: "user", Content: "Zeus keeps another secret."}); err != nil {
			t.Fatal(err)
		}
	}
	for _, args := range []string{`{}`, `{"limit":1}`} {
		asked := 0
		counted := WithCallerScope(ctx, CallerScope{Key: "hermes", Memory: func(b MemoryBucket) bool { asked++; return b.UserID == "hermes" }})
		if _, err := db.GraphRAGTools().Call(counted, "memory_list_all", json.RawMessage(args)); err != nil {
			t.Fatal(err)
		}
		if asked != 2 {
			t.Errorf("%s: the scope was asked %d times for 5 rows in 2 buckets, want 2", args, asked)
		}
	}
}
