package cortexdb

import (
	"context"
	"errors"
	"fmt"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

// CallerScope is what a confined caller may reach, as the facade checks it. A
// server builds one from the key it authenticated (authz.Key.CallerContext)
// and puts it in the call's context with WithCallerScope.
//
// The checks are on what a request resolves to, not on its fields, because
// that is what decides which rows a call touches. A request that names
// user_id=hermes and session_id=zeus-sess resolves to Zeus's session bucket:
// the user_id passes a policy that reads fields, and the session then reads
// Zeus's rows. A recall names a user_id for its memory half and reaches every
// collection with its knowledge half.
type CallerScope struct {
	// Key names the caller. Identical searches that arrive together share
	// one run (memory_search_coalesce.go), and two callers with different
	// scopes must not share an answer.
	Key string
	// Memory reports whether the caller may touch a memory bucket. A bucket
	// nobody owns (the global one, a session saved without a user_id) has
	// UserID "", the shared pool a user-confined caller was confined away
	// from.
	Memory func(MemoryBucket) bool
	// Knowledge reports whether the caller may read knowledge in a
	// collection; "" is every collection.
	Knowledge func(collection string) bool
}

// MemoryBucket is a memory bucket as a CallerScope sees it.
type MemoryBucket struct {
	UserID    string
	Scope     string
	Namespace string
}

type callerScopeContextKey struct{}

// WithCallerScope confines the call in ctx to what scope permits. Without one,
// nothing is checked: an embedded library, an unkeyed server and an
// unconfined key behave as before.
func WithCallerScope(ctx context.Context, scope CallerScope) context.Context {
	return context.WithValue(ctx, callerScopeContextKey{}, scope)
}

func callerScopeOf(ctx context.Context) (CallerScope, bool) {
	scope, ok := ctx.Value(callerScopeContextKey{}).(CallerScope)
	return scope, ok
}

// errMemoryBucketWithheld is what a confined caller is told about a bucket it
// may not write to. "not found", the wording a missing row gets, so the ports
// answer it the way they answer a missing memory.
var errMemoryBucketWithheld = errors.New("memory bucket not found")

// memoryBucketOf is the bucket bucketID names: as stored when it exists, and
// as a write would create it otherwise.
func (db *DB) memoryBucketOf(ctx context.Context, bucketID string, created MemoryBucket) (MemoryBucket, bool, error) {
	session, err := db.store.GetSession(ctx, bucketID)
	if errors.Is(err, core.ErrNotFound) {
		return created, false, nil
	}
	if err != nil {
		return MemoryBucket{}, false, fmt.Errorf("load memory bucket: %w", err)
	}
	stored := MemoryBucket{UserID: session.UserID}
	stored.Scope, _ = session.Metadata["scope"].(string)
	stored.Namespace, _ = session.Metadata["namespace"].(string)
	return stored, true, nil
}

// checkMemoryWrite decides whether a save may go into bucketID. created is the
// bucket the save would create, owned by nobody when it is the global one.
//
// One rule holds for every caller, confined or not: a save that names a
// user_id does not land in a bucket another user owns. Without it a confined
// key could take a session id before its user did, and own every row anybody
// later saved under it.
func (db *DB) checkMemoryWrite(ctx context.Context, bucketID, userID string, created MemoryBucket) error {
	bucket, exists, err := db.memoryBucketOf(ctx, bucketID, created)
	if err != nil {
		return err
	}
	// A confined caller hears "not found" about any bucket it may not have,
	// before anything that would tell it the bucket exists.
	if scope, ok := callerScopeOf(ctx); ok && scope.Memory != nil && !scope.Memory(bucket) {
		return errMemoryBucketWithheld
	}
	if exists && userID != "" && bucket.UserID != "" && bucket.UserID != userID {
		return fmt.Errorf("memory bucket %q belongs to another user: invalid user_id %q for it", bucketID, userID)
	}
	return nil
}

// memoryReadPermitted reports whether the caller may read bucketID. A bucket
// that does not exist has no rows to give away.
func (db *DB) memoryReadPermitted(ctx context.Context, bucketID string) (bool, error) {
	scope, ok := callerScopeOf(ctx)
	if !ok || scope.Memory == nil {
		return true, nil
	}
	bucket, exists, err := db.memoryBucketOf(ctx, bucketID, MemoryBucket{})
	if err != nil || !exists {
		return err == nil, err
	}
	return scope.Memory(bucket), nil
}

// knowledgeReadPermitted reports whether the caller may read knowledge in a
// collection, "" meaning every collection.
func knowledgeReadPermitted(ctx context.Context, collection string) bool {
	scope, ok := callerScopeOf(ctx)
	return !ok || scope.Knowledge == nil || scope.Knowledge(collection)
}

// knowledgeListed reports whether the caller may read a document the list
// form of GetKnowledge loaded. Under a knowledge scope a document without
// chunks sits in no collection, and the row guard of the ports withholds it
// from every confined key, so it is not listed either.
func knowledgeListed(ctx context.Context, record *KnowledgeRecord) bool {
	scope, ok := callerScopeOf(ctx)
	if !ok || scope.Knowledge == nil {
		return true
	}
	return len(record.ChunkIDs) > 0 && scope.Knowledge(record.Collection)
}
