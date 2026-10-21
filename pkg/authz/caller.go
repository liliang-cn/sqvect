package authz

import (
	"context"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// CallerContext carries what the facade must know about the key behind a call
// past the port that authenticated it. Both ports, gRPC and REST, hand it the
// same context, so a check made in pkg/cortexdb holds the same on either.
//
// It names the key as the writer, so what the call saves records it as
// written_by. For a confined key it also hands the facade the key's scope as
// row rules, the same rules the typed RPCs apply to a row read by id
// (rpcserver/ownership.go): a memory bucket must match every confined field,
// and has no collection; a knowledge collection must be among the key's, and
// knowledge has no user_id, scope or namespace. The request's fields are
// checked by the policy already, but what they resolve to is checked here: a
// session_id picks another user's bucket, a recall's knowledge half reaches
// every collection.
func (k Key) CallerContext(ctx context.Context) context.Context {
	ctx = cortexdb.WithWriter(ctx, k.ID)
	if k.Scope.IsZero() {
		return ctx
	}
	scope := k.Scope
	return cortexdb.WithCallerScope(ctx, cortexdb.CallerScope{
		Key: k.ID,
		Memory: func(b cortexdb.MemoryBucket) bool {
			return scope.Permits(FieldUserID, b.UserID) &&
				scope.Permits(FieldScope, b.Scope) &&
				scope.Permits(FieldNamespace, b.Namespace) &&
				scope.Permits(FieldCollection, "")
		},
		Knowledge: func(collection string) bool {
			return scope.Permits(FieldCollection, collection) &&
				scope.Permits(FieldUserID, "") &&
				scope.Permits(FieldScope, "") &&
				scope.Permits(FieldNamespace, "")
		},
	})
}
