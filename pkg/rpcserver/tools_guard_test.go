package rpcserver

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// The two halves of the CallTool check fail closed on their own, without the
// interceptor in front of them: a request that is not a CallToolRequest, a
// tool the toolbox does not define, arguments that are not an object, and a
// read that fails each come out as a refusal or as the failure itself, never
// as a pass.
func TestTheCallToolChecksFailClosedOnTheirOwn(t *testing.T) {
	hermes := authz.Key{ID: "hermes", Secret: "h", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}}

	// authorizeToolArgs over a message that is not a CallToolRequest: the
	// request-message denial, not a pass.
	if err := authorizeToolArgs(hermes, &rpcv1.HealthRequest{}); !errors.Is(err, authz.ErrDenied) {
		t.Fatalf("a non-CallTool message passed the tool argument check: %v", err)
	}

	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(t.TempDir(), "guard.db")))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.WithValue(context.Background(), callerKeyContextKey{}, hermes)

	if err := guardToolRows(ctx, db, "no_such_tool", `{"memory_id":"x"}`); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("an unknown tool passed the row guard: %v", err)
	}
	if err := guardToolRows(ctx, db, "memory_save", `[1]`); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("non-object arguments passed the row guard: %v", err)
	}
	// A read that fails (here: a cancelled context) is reported as itself,
	// neither NOT_FOUND nor a pass.
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	err = guardToolRows(cancelled, db, "memory_save", `{"memory_id":"h-1","user_id":"hermes"}`)
	if err == nil || status.Code(err) == codes.NotFound || status.Code(err) == codes.PermissionDenied {
		t.Fatalf("a failed read was not reported as one: %v", err)
	}
	// Unconfined: no check at all, whatever the arguments.
	root := context.WithValue(context.Background(), callerKeyContextKey{}, authz.Key{ID: "root", Secret: "s", Clearance: authz.ReadWrite})
	if err := guardToolRows(root, db, "no_such_tool", `[1]`); err != nil {
		t.Fatalf("an unconfined caller was checked: %v", err)
	}
}
