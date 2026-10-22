package httpapi

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// POST /v1/knowledge and POST /v1/memory upsert the row their id names, as
// SaveKnowledge and SaveMemory do over gRPC, so they guard it the same way
// (guardKnowledgeUpsert, guardMemoryUpsert): an id that names another scope's
// row is 404, and the row is neither overwritten nor moved into the caller's
// scope. An id that names nothing is a create, and the caller's own row is an
// update; both pass.
func TestAConfinedKeyCannotUpsertAnotherScopesRowOverHTTP(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "op-secret",
		`{"knowledge_id":"secret-doc","collection":"secrets","content":"The secret retainer."}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/memory", "op-secret",
		`{"memory_id":"zeus-1","user_id":"zeus","scope":"user","content":"Zeus keeps the thunderbolt."}`, http.StatusOK)

	if msg := wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret",
		`{"knowledge_id":"secret-doc","collection":"notes","content":"overwritten"}`, http.StatusNotFound); msg != `knowledge "secret-doc" not found` {
		t.Errorf("another collection's document answered %q", msg)
	}
	if msg := wantStatus(t, srv, http.MethodPost, "/v1/memory", "hermes-secret",
		`{"memory_id":"zeus-1","user_id":"hermes","scope":"user","content":"stolen"}`, http.StatusNotFound); msg != `memory "zeus-1" not found` {
		t.Errorf("another user's memory answered %q", msg)
	}
	if _, body := do(t, srv, http.MethodGet, "/v1/knowledge?knowledge_id=secret-doc", "op-secret", ""); !strings.Contains(string(body), "The secret retainer.") || !strings.Contains(string(body), `"secrets"`) {
		t.Errorf("the document changed: %s", body)
	}
	if _, body := do(t, srv, http.MethodGet, "/v1/memory?memory_id=zeus-1", "op-secret", ""); !strings.Contains(string(body), "thunderbolt") || !strings.Contains(string(body), `"zeus"`) {
		t.Errorf("the memory changed: %s", body)
	}

	// No id: the guard has no row to read and leaves the route its own answer.
	if msg := wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret",
		`{"collection":"notes","content":"A note with no id."}`, http.StatusBadRequest); msg != "knowledge_id is required" {
		t.Errorf("a document with no id answered %q", msg)
	}
	// A new id, then the caller's own row again.
	for i := 0; i < 2; i++ {
		wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret",
			`{"knowledge_id":"n-1","collection":"notes","content":"A note."}`, http.StatusOK)
		wantStatus(t, srv, http.MethodPost, "/v1/memory", "hermes-secret",
			`{"memory_id":"hermes-1","user_id":"hermes","scope":"user","content":"Hermes tunes the lyre."}`, http.StatusOK)
	}
}

// encoding/json reads any spelling of a field into the request, so every
// spelling of the id is guarded, not only the canonical one.
func TestEverySpellingOfAnUpsertIDIsGuarded(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "op-secret",
		`{"knowledge_id":"secret-doc","collection":"secrets","content":"The secret retainer."}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret",
		`{"KNOWLEDGE_ID":"secret-doc","collection":"notes","content":"overwritten"}`, http.StatusNotFound)
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret",
		`{"knowledge_id":"n-1","Knowledge_Id":"secret-doc","collection":"notes","content":"overwritten"}`, http.StatusNotFound)
	if _, body := do(t, srv, http.MethodGet, "/v1/knowledge?knowledge_id=secret-doc", "op-secret", ""); !strings.Contains(string(body), "The secret retainer.") {
		t.Errorf("the document changed: %s", body)
	}
}

// Each confinement holds on each kind of row, and a row the key's
// confinement cannot express (a memory under a collection, a document under a
// user) is a refusal. A read that fails is reported, never taken as a create.
func TestTheUpsertRowRuleHoldsEveryConfinedField(t *testing.T) {
	ctx := context.Background()
	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(t.TempDir(), "rows.db")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, cortexdb.KnowledgeSaveRequest{KnowledgeID: "doc", Collection: "notes", Content: "A note."}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveMemory(ctx, cortexdb.MemorySaveRequest{MemoryID: "mem", UserID: "hermes", Scope: "user", Namespace: "ops", Content: "Hermes tunes the lyre."}); err != nil {
		t.Fatal(err)
	}
	// The rows nobody owns: a memory of the global bucket and a document
	// saved with no collection, which lands in the default one.
	if _, err := db.SaveMemory(ctx, cortexdb.MemorySaveRequest{MemoryID: "glob", Scope: "global", Content: "The sky is wide."}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, cortexdb.KnowledgeSaveRequest{KnowledgeID: "loose", Content: "A loose note."}); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		scope            authz.Scope
		kind, id         string
		found, permitted bool
	}{
		{authz.Scope{Collection: "notes"}, "knowledge", "doc", true, true},
		{authz.Scope{Collection: "other"}, "knowledge", "doc", true, false},
		{authz.Scope{UserID: "hermes"}, "knowledge", "doc", true, false},
		{authz.Scope{MemoryScope: "user"}, "knowledge", "doc", true, false},
		{authz.Scope{Namespace: "ops"}, "knowledge", "doc", true, false},
		{authz.Scope{UserID: "hermes", MemoryScope: "user", Namespace: "ops"}, "memory", "mem", true, true},
		{authz.Scope{UserID: "zeus"}, "memory", "mem", true, false},
		{authz.Scope{MemoryScope: "session"}, "memory", "mem", true, false},
		{authz.Scope{Namespace: "dev"}, "memory", "mem", true, false},
		{authz.Scope{Collection: "notes"}, "memory", "mem", true, false},
		{authz.Scope{UserID: "hermes"}, "memory", "glob", true, false},
		{authz.Scope{Collection: "notes"}, "knowledge", "loose", true, false},
		{authz.Scope{Collection: "other"}, "knowledge", "no-such", false, false},
		{authz.Scope{UserID: "zeus"}, "memory", "no-such", false, false},
	} {
		found, permitted, err := rowPermitted(ctx, db, c.scope, c.kind, c.id)
		if err != nil || found != c.found || permitted != c.permitted {
			t.Errorf("%+v on %s %s: found %v permitted %v (%v), want %v %v", c.scope, c.kind, c.id, found, permitted, err, c.found, c.permitted)
		}
	}
	_ = db.Close()
	for _, kind := range []string{"knowledge", "memory"} {
		if found, _, err := rowPermitted(ctx, db, authz.Scope{Collection: "notes"}, kind, "doc"); found || err == nil {
			t.Errorf("a %s read on a closed brain: %v %v, want an error", kind, found, err)
		}
	}
}

// A row the guard cannot read is a refusal reported as a server fault, not a
// create and not a 404.
func TestAnUpsertWhoseRowCannotBeReadIsRefused(t *testing.T) {
	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(t.TempDir(), "closed.db")))
	if err != nil {
		t.Fatal(err)
	}
	_ = db.Close()
	req := httptest.NewRequest(http.MethodPost, "/v1/knowledge", strings.NewReader(`{"knowledge_id":"doc","collection":"notes","content":"x"}`))
	rec := httptest.NewRecorder()
	if guardUpsert(rec, req, db, authz.Scope{Collection: "notes"}, upsertRoutes["POST /v1/knowledge"]) || rec.Code != http.StatusInternalServerError {
		t.Fatalf("an unreadable row answered %d: %s", rec.Code, rec.Body.String())
	}
	// A supersedes target is read too: with no id to read first, the target's
	// read is the one that fails.
	req = httptest.NewRequest(http.MethodPost, "/v1/memory", strings.NewReader(`{"user_id":"hermes","scope":"user","content":"x","supersedes":["zeus-1"]}`))
	rec = httptest.NewRecorder()
	if guardUpsert(rec, req, db, authz.Scope{UserID: "hermes"}, upsertRoutes["POST /v1/memory"]) || rec.Code != http.StatusInternalServerError {
		t.Fatalf("an unreadable supersede target answered %d: %s", rec.Code, rec.Body.String())
	}
}

// supersedes names rows too: SaveMemory marks each target retired. So each
// target is guarded under every spelling, before anything is written: a
// target in another scope and a target that does not exist both get the same
// 404, and the new memory is not left behind.
func TestASupersedeTargetOutsideTheKeysScopeIsNotFoundOverHTTP(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/memory", "op-secret",
		`{"memory_id":"zeus-1","user_id":"zeus","scope":"user","content":"Zeus keeps the thunderbolt."}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/memory", "hermes-secret",
		`{"memory_id":"hermes-1","user_id":"hermes","scope":"user","content":"Hermes tunes the lyre."}`, http.StatusOK)

	answers := map[string]string{}
	for _, c := range []struct{ name, id, body string }{
		{"foreign", "hermes-2", `{"memory_id":"hermes-2","user_id":"hermes","scope":"user","content":"x","supersedes":["zeus-1"]}`},
		{"missing", "hermes-3", `{"memory_id":"hermes-3","user_id":"hermes","scope":"user","content":"x","supersedes":["no-such"]}`},
		{"foreign, other spelling", "hermes-4", `{"memory_id":"hermes-4","user_id":"hermes","scope":"user","content":"x","SUPERSEDES":["zeus-1"]}`},
		{"foreign after its own", "hermes-5", `{"memory_id":"hermes-5","user_id":"hermes","scope":"user","content":"x","supersedes":["hermes-1"," zeus-1 "]}`},
		// The id the handler keeps is the last spelling: hermes-7 is not the
		// new memory, so it is a target, and a missing one.
		{"another spelling's id as a target", "hermes-8", `{"Memory_Id":"hermes-7","memory_id":"hermes-8","user_id":"hermes","scope":"user","content":"x","supersedes":["hermes-7"]}`},
	} {
		msg := wantStatus(t, srv, http.MethodPost, "/v1/memory", "hermes-secret", c.body, http.StatusNotFound)
		answers[c.name] = strings.NewReplacer("zeus-1", "<id>", "no-such", "<id>", "hermes-7", "<id>").Replace(msg)
		if st, _ := do(t, srv, http.MethodGet, "/v1/memory?memory_id="+c.id, "op-secret", ""); st != http.StatusNotFound {
			t.Errorf("%s: the new memory was written although its target was refused (%d)", c.name, st)
		}
	}
	if want := `supersede "<id>": not found`; answers["missing"] != want {
		t.Errorf("a missing target answered %q, want %q", answers["missing"], want)
	}
	for name, msg := range answers {
		if msg != answers["missing"] {
			t.Errorf("%s answered %q and a missing target %q", name, msg, answers["missing"])
		}
	}
	if _, body := do(t, srv, http.MethodGet, "/v1/memory?memory_id=zeus-1", "op-secret", ""); strings.Contains(string(body), "superseded_by") {
		t.Errorf("zeus's memory was retired: %s", body)
	}
	// Its own target, padded as the facade trims it, passes; itself and a
	// blank target are skipped, as the facade skips them.
	wantStatus(t, srv, http.MethodPost, "/v1/memory", "hermes-secret",
		`{"memory_id":"hermes-6","user_id":"hermes","scope":"user","content":"newer","supersedes":[" hermes-1 ","hermes-6","  "]}`, http.StatusOK)
}

// An id of blanks names a row like any other: the facade stores it as given,
// so the guard reads it rather than taking it for no id.
func TestABlankUpsertIDIsGuardedLikeAnyOther(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "op-secret", `{"knowledge_id":" ","collection":"secrets","content":"The secret retainer."}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/memory", "op-secret", `{"memory_id":"\t","user_id":"zeus","scope":"user","content":"Zeus keeps the thunderbolt."}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret", `{"knowledge_id":" ","collection":"notes","content":"overwritten"}`, http.StatusNotFound)
	wantStatus(t, srv, http.MethodPost, "/v1/memory", "hermes-secret", `{"Memory_ID":"\t","user_id":"hermes","scope":"user","content":"stolen"}`, http.StatusNotFound)
}

// The guard is for confined keys: the operator's key writes any row, as on
// main.
func TestAnUnconfinedKeyUpsertsAnyRowOverHTTP(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "notes-secret", `{"knowledge_id":"n-1","collection":"notes","content":"A note."}`, http.StatusOK)
	wantStatus(t, srv, http.MethodPost, "/v1/knowledge", "op-secret", `{"knowledge_id":"n-1","collection":"secrets","content":"moved by the operator"}`, http.StatusOK)
	// A missing target is the facade's own answer: the guard did not run for
	// this key. SaveMemory checks its targets before it writes (#72), so the
	// new memory is not stored either.
	wantStatus(t, srv, http.MethodPost, "/v1/memory", "op-secret", `{"memory_id":"m-1","user_id":"zeus","scope":"user","content":"x","supersedes":["no-such"]}`, http.StatusNotFound)
	if st, _ := do(t, srv, http.MethodGet, "/v1/memory?memory_id=m-1", "op-secret", ""); st != http.StatusNotFound {
		t.Errorf("a save whose supersede target is missing left its memory behind (%d)", st)
	}
}

// Only the facade's sentinel means a missing row: a read error whose text
// says "not found" is still a read error, and the guard reports it.
func TestOnlyTheNotFoundSentinelIsAMissingRow(t *testing.T) {
	if err := readError(fmt.Errorf("load: %w", core.ErrNotFound)); err != nil {
		t.Errorf("the sentinel is a missing row, got %v", err)
	}
	if err := readError(errors.New("index not found")); err == nil {
		t.Error("an error whose text says not found was taken for a missing row")
	}
}
