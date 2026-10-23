package cortexdb

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// openExtractScopeDB saves one memory for zeus and one for hermes, and returns
// the brain, the two session ids, and a context confined to hermes's memory.
func openExtractScopeDB(t *testing.T) (*DB, string, string, context.Context) {
	t.Helper()
	ctx := context.Background()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "extract-scope.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	zeus, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "zeus-1", UserID: "zeus", Scope: "user", Content: "Zeus hid the Thunderbolt under Olympus. Hera must not know."})
	if err != nil {
		t.Fatal(err)
	}
	hermes, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "hermes-1", UserID: "hermes", Scope: "user", Content: "Hermes tuned the Lyre for Apollo."})
	if err != nil {
		t.Fatal(err)
	}
	confined := WithCallerScope(ctx, CallerScope{Key: "hermes", Memory: func(b MemoryBucket) bool { return b.UserID == "hermes" }})
	return db, zeus.Memory.SessionID, hermes.Memory.SessionID, confined
}

// Under a caller scope, a session outside it answers exactly as a session
// that does not exist, so the answer does not tell that it exists.
func TestExtractConversationAnswersASessionOutsideTheCallerScopeAsMissing(t *testing.T) {
	db, zeusSession, _, hermes := openExtractScopeDB(t)
	tb := db.GraphRAGTools()

	_, errMissing := tb.Call(hermes, "extract_conversation", json.RawMessage(`{"session_id":"no-such-session"}`))
	if errMissing == nil {
		t.Fatal("a missing session was summarized")
	}
	out, err := tb.Call(hermes, "extract_conversation", json.RawMessage(`{"session_id":"`+zeusSession+`"}`))
	if err == nil {
		body, _ := json.Marshal(out)
		t.Fatalf("zeus's session was summarized under hermes's scope: %s", body)
	}
	if err.Error() != errMissing.Error() {
		t.Errorf("a foreign session answers %q, a missing one %q", err, errMissing)
	}
	if strings.Contains(err.Error(), "Thunderbolt") {
		t.Errorf("the refusal carries the session's text: %v", err)
	}
}

// persist writes nothing for a session outside the scope: no summary
// document, no entity in the shared graph.
func TestExtractConversationWithPersistWritesNothingForASessionOutsideTheScope(t *testing.T) {
	db, zeusSession, _, hermes := openExtractScopeDB(t)
	ctx := context.Background()

	if _, err := db.GraphRAGTools().Call(hermes, "extract_conversation", json.RawMessage(`{"session_id":"`+zeusSession+`","persist":true}`)); err == nil {
		t.Fatal("zeus's session was extracted under hermes's scope")
	}
	if _, err := db.GetKnowledge(ctx, KnowledgeGetRequest{KnowledgeID: "conversation:" + zeusSession}); err == nil {
		t.Error("a summary of zeus's session was written")
	}
	if node, _ := db.Graph().GetNode(ctx, "entity:thunderbolt"); node != nil {
		t.Error("an entity of zeus's session was written to the graph")
	}
}

// The check refuses only what the scope does not permit: the caller's own
// session, and any session without a scope, are summarized as before.
func TestExtractConversationStillReadsTheCallersOwnSession(t *testing.T) {
	db, zeusSession, hermesSession, hermes := openExtractScopeDB(t)
	tb := db.GraphRAGTools()

	for _, c := range []struct {
		name    string
		ctx     context.Context
		session string
		want    string
	}{
		{"hermes under hermes", hermes, hermesSession, "Lyre"},
		{"zeus without a scope", context.Background(), zeusSession, "Thunderbolt"},
	} {
		resp, err := tb.ExtractConversation(c.ctx, ToolExtractConversationRequest{SessionID: c.session})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if !strings.Contains(resp.Summary, c.want) {
			t.Errorf("%s: summary %q, want it to name %s", c.name, resp.Summary, c.want)
		}
	}
}

// A session bucket that cannot be read is an error, never a pass.
// created_at no longer scans as a time, so the bucket does not load; the
// messages still do.
func TestExtractConversationFailsWhenTheSessionBucketCannotBeRead(t *testing.T) {
	db, _, hermesSession, hermes := openExtractScopeDB(t)
	if _, err := db.SQL().Exec(`UPDATE sessions SET created_at = 'not a time'`); err != nil {
		t.Fatal(err)
	}
	resp, err := db.GraphRAGTools().ExtractConversation(hermes, ToolExtractConversationRequest{SessionID: hermesSession})
	if err == nil {
		t.Fatalf("an unreadable bucket passed as readable: %+v", resp)
	}
	if !strings.Contains(err.Error(), "extract_conversation: load session") {
		t.Errorf("error %q does not say the session did not load", err)
	}
}
