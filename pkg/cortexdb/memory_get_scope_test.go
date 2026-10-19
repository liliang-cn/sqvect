package cortexdb

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

// memory_get with a list answers a confined caller about a row outside its
// scope exactly as about a row that does not exist: under missing, in the
// order asked. The single-id form stays unscoped on purpose: the row guard
// of the ports (authz.Scope.GuardToolRows) reads through it, and a facade
// that hid the row there would let the guard take a foreign row for a new
// one and pass the write.
func TestMemoryGetByListHidesRowsOutsideTheCallerScopeAsMissing(t *testing.T) {
	ctx := context.Background()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "scope.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	for _, m := range []MemorySaveRequest{
		{MemoryID: "zeus-1", UserID: "zeus", Scope: "user", Content: "the drawer key"},
		{MemoryID: "hermes-1", UserID: "hermes", Scope: "user", Content: "the lyre"},
	} {
		if _, err := db.SaveMemory(ctx, m); err != nil {
			t.Fatal(err)
		}
	}
	hermes := WithCallerScope(ctx, CallerScope{Key: "hermes", Memory: func(b MemoryBucket) bool { return b.UserID == "hermes" }})

	// Two orders, so that missing follows the request and not the kind of
	// absence.
	for _, ids := range [][]string{{"zeus-1", "no-such", "hermes-1"}, {"no-such", "zeus-1"}} {
		resp, err := db.GetMemory(hermes, MemoryGetRequest{MemoryIDs: ids})
		if err != nil {
			t.Fatalf("%v: %v", ids, err)
		}
		var wantMissing []string
		for _, id := range ids {
			if id != "hermes-1" {
				wantMissing = append(wantMissing, id)
			}
		}
		if strings.Join(resp.Missing, ",") != strings.Join(wantMissing, ",") {
			t.Errorf("%v: missing %v, want %v", ids, resp.Missing, wantMissing)
		}
		for _, m := range resp.Memories {
			if m.ID != "hermes-1" {
				t.Errorf("%v: served %s outside the scope", ids, m.ID)
			}
		}
	}
	// Unconfined, and the single form under the scope: the row is served.
	if resp, err := db.GetMemory(ctx, MemoryGetRequest{MemoryIDs: []string{"zeus-1"}}); err != nil || len(resp.Missing) != 0 || len(resp.Memories) != 1 {
		t.Fatalf("unconfined list read: %+v %v", resp, err)
	}
	if resp, err := db.GetMemory(hermes, MemoryGetRequest{MemoryID: "zeus-1"}); err != nil || resp.Memory.ID != "zeus-1" {
		t.Fatalf("the single form was scoped, which blinds the row guard: %+v %v", resp, err)
	}
	// A bucket that cannot be read is an error, never a pass. The row itself
	// still loads (its query reads the session's user_id only); the bucket
	// does not (created_at no longer scans as a time).
	if _, err := db.SQL().Exec(`UPDATE sessions SET created_at = 'not a time'`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.GetMemory(hermes, MemoryGetRequest{MemoryIDs: []string{"hermes-1"}}); err == nil {
		t.Fatal("an unreadable bucket passed as readable")
	}
}

// A memory_save whose supersedes names a row that does not exist writes
// nothing: the new row is not there afterwards. Otherwise a confined
// caller tells a missing target (row written, then "not found") from a
// foreign one (refused before any write, row absent) by reading the new id.
// Under a caller scope, a supersede target outside it is not found, as a
// missing one is: the new memory is not written and the target is not
// retired. The facade holds this itself, so a library caller that sets a
// CallerScope gets it without a port in front.
func TestASupersedeTargetOutsideTheCallerScopeIsNotFound(t *testing.T) {
	ctx := context.Background()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "supersede-scope.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "zeus-1", UserID: "zeus", Scope: "user", Content: "the drawer key"}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "hermes-1", UserID: "hermes", Scope: "user", Content: "the lyre"}); err != nil {
		t.Fatal(err)
	}
	hermes := WithCallerScope(ctx, CallerScope{Key: "hermes", Memory: func(b MemoryBucket) bool { return b.UserID == "hermes" }})
	_, foreign := db.SaveMemory(hermes, MemorySaveRequest{MemoryID: "hermes-2", UserID: "hermes", Scope: "user", Content: "newer", Supersedes: []string{"zeus-1"}})
	_, missing := db.SaveMemory(hermes, MemorySaveRequest{MemoryID: "hermes-2", UserID: "hermes", Scope: "user", Content: "newer", Supersedes: []string{"no-such"}})
	if foreign == nil || !errors.Is(foreign, core.ErrNotFound) || strings.Replace(foreign.Error(), "zeus-1", "no-such", 1) != missing.Error() {
		t.Fatalf("a foreign target: %v; a missing one: %v; want the same not-found", foreign, missing)
	}
	got, err := db.GetMemory(ctx, MemoryGetRequest{MemoryID: "zeus-1"})
	if err != nil {
		t.Fatal(err)
	}
	if by, _ := got.Memory.Metadata["superseded_by"].(string); by != "" {
		t.Fatalf("zeus-1 was retired by hermes: superseded_by=%q", by)
	}
	if _, err := db.GetMemory(ctx, MemoryGetRequest{MemoryID: "hermes-2"}); err == nil {
		t.Fatal("the new memory was written although its target is outside the scope")
	}
	// A target whose bucket cannot be read is an error, never a pass and
	// never a plain "not found".
	if _, err := db.SQL().Exec(`UPDATE sessions SET created_at = 'not a time' WHERE id LIKE 'memory:user:zeus:%'`); err != nil {
		t.Fatal(err)
	}
	_, err = db.SaveMemory(hermes, MemorySaveRequest{MemoryID: "hermes-3", UserID: "hermes", Scope: "user", Content: "newer", Supersedes: []string{"zeus-1"}})
	if err == nil || errors.Is(err, core.ErrNotFound) {
		t.Fatalf("a target in an unreadable bucket: %v, want the read error", err)
	}
	if _, err := db.GetMemory(ctx, MemoryGetRequest{MemoryID: "hermes-3"}); err == nil {
		t.Fatal("the new memory was written although its target's bucket could not be read")
	}
	if _, err := db.SaveMemory(hermes, MemorySaveRequest{MemoryID: "hermes-2", UserID: "hermes", Scope: "user", Content: "newer", Supersedes: []string{"hermes-1"}}); err != nil {
		t.Fatalf("hermes retires its own memory: %v", err)
	}
}

func TestAMissingSupersedeTargetLeavesNoNewMemory(t *testing.T) {
	ctx := context.Background()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "supersede.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	_, err = db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "new-1", UserID: "hermes", Scope: "user", Content: "the lyre", Supersedes: []string{"no-such"}})
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("a missing supersede target passed: %v", err)
	}
	if _, err := db.GetMemory(ctx, MemoryGetRequest{MemoryID: "new-1"}); err == nil {
		t.Fatal("the new memory was written although its supersede target does not exist")
	}
	// A blank target and the memory itself are skipped, as on main: they
	// name no other row, so they are not missing ones.
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "new-2", UserID: "hermes", Scope: "user", Content: "the lyre", Supersedes: []string{" ", "new-2"}}); err != nil {
		t.Fatalf("a blank target or the memory itself was taken for a missing one: %v", err)
	}
}
