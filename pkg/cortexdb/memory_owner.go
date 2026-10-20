package cortexdb

import (
	"context"
	"errors"
	"fmt"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

type memoryOwnerContextKey struct{}

// WithMemoryOwners confines what a call may read or write in memory to the
// buckets whose owner permits accepts. A server puts its confined caller's rule
// here, and SaveMemory and SearchMemory, and everything built on them (recall,
// context packs, the memory tools), hold the bucket a request resolves to
// against it.
//
// The check is on the bucket, not on the request's user_id, because the bucket
// is what decides which rows a call touches. A request that names user_id=hermes
// and session_id=zeus-sess resolves to Zeus's session bucket: the user_id passes
// a policy that reads fields, and the session then reads Zeus's rows. A bucket's
// owner is the user_id it was created with, and a bucket nobody owns (the
// global one, a session saved without a user_id) belongs to the shared pool a
// confined caller was confined away from, so permits sees "" and refuses it.
//
// Without a rule in ctx, nothing is checked: an embedded library, an unkeyed
// server and an unconfined key behave as before.
func WithMemoryOwners(ctx context.Context, permits func(owner string) bool) context.Context {
	if permits == nil {
		return ctx
	}
	return context.WithValue(ctx, memoryOwnerContextKey{}, permits)
}

// errMemoryBucketWithheld is what a confined caller is told about a bucket it
// may not write to. "not found", the wording a missing row gets, so the ports
// answer it the way they answer a missing memory.
var errMemoryBucketWithheld = errors.New("memory bucket not found")

// memoryBucketPermitted reports whether the caller in ctx may touch bucketID.
// creator is who a write would create the bucket as when it does not exist yet;
// a read passes "" and is permitted on a missing bucket, which has no rows to
// give away.
func (db *DB) memoryBucketPermitted(ctx context.Context, bucketID, creator string, isWrite bool) (bool, error) {
	permits, _ := ctx.Value(memoryOwnerContextKey{}).(func(string) bool)
	if permits == nil {
		return true, nil
	}
	session, err := db.store.GetSession(ctx, bucketID)
	switch {
	case errors.Is(err, core.ErrNotFound):
		return !isWrite || permits(creator), nil
	case err != nil:
		return false, fmt.Errorf("load memory bucket: %w", err)
	}
	return permits(session.UserID), nil
}
