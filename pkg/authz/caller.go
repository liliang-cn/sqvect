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
// written_by. For a key confined to some user_ids it also confines the memory
// buckets the call may resolve to: the request's user_id is checked here
// already, but a session_id picks the bucket, and a session another user owns
// would otherwise answer to a key that only sent its own user_id beside it.
func (k Key) CallerContext(ctx context.Context) context.Context {
	ctx = cortexdb.WithWriter(ctx, k.ID)
	if k.Scope.IsZero() {
		return ctx
	}
	scope := k.Scope
	return cortexdb.WithMemoryOwners(ctx, func(owner string) bool {
		return scope.Permits(FieldUserID, owner)
	})
}
