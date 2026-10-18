package rpcserver

import (
	"context"
	"encoding/json"
	"errors"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// Ownership is the second half of the scope model, and it lives here rather
// than in pkg/authz because it needs something authz deliberately refuses to
// need: the row.
//
// authz decides from the request message alone, which is what makes it cheap,
// total and auditable — and is also why it could say nothing at all about the
// RPCs that carry only an id. GetMemory names a memory_id and nothing else, so
// the interceptor's honest answer was "I cannot tell whose row that is", and
// the honest answer to that was to refuse. Safe, and a real loss: an agent
// confined to its own user_id could not fetch back the memory it had written a
// moment earlier, which is most of what an agent does with its own memory.
//
// The fix is the standard shape and it has a cost worth naming: the row is read
// before the caller is known to be allowed to see it, and then withheld. The
// server touches data the caller may not have. That is unavoidable — the owner
// is *in* the row — so what matters is that nothing about the row escapes when
// the answer is no, which is what the single withheld() answer below is for.
//
// Everything here is inert for an unconfined key. The legacy single token maps
// onto exactly one unconfined read-write key (authz.LegacyToken), so every
// deployment in the wild takes the same code path it took before this file
// existed: no extra read, no extra check, no changed error.

// callerKeyContextKey is the private type under which the auth interceptor
// parks the resolved key. Unexported so nothing outside this package can forge
// a caller identity by writing to a context.
type callerKeyContextKey struct{}

// withCallerKey records the key a request authenticated as, so handlers that
// have to check the row itself can see what the caller is confined to. The
// interceptor has already resolved and authorised the key by this point; this
// only carries the decision forward.
func withCallerKey(ctx context.Context, key authz.Key) context.Context {
	return context.WithValue(ctx, callerKeyContextKey{}, key)
}

// confinedCaller reports the calling key's scope, and whether it confines
// anything at all.
//
// The false case covers three situations that must all behave identically to
// how they behaved before ownership existed: authentication disabled, the
// legacy token, and an explicitly unconfined key in a key file.
func confinedCaller(ctx context.Context) (authz.Scope, bool) {
	key, ok := ctx.Value(callerKeyContextKey{}).(authz.Key)
	if !ok || key.Scope.IsZero() {
		return authz.Scope{}, false
	}
	return key.Scope, true
}

// idAddressedMethods are the RPCs whose request names a row by id and carries
// no scope field to check it against. The interceptor waives its row check for
// these and the handler enforces ownership against the stored row instead.
//
// This is a table and not a heuristic for the same reason authz.methods is:
// waiving the interceptor check is a hole unless a handler closes it, so the
// set of methods that get the waiver has to be something a person can read,
// and something a test can walk. TestEveryIdAddressedMethodIsGuardedByAHandler
// walks it.
var idAddressedMethods = map[string]struct{}{
	"/cortexdb.v1.MemoryService/GetMemory":          {},
	"/cortexdb.v1.MemoryService/UpdateMemory":       {},
	"/cortexdb.v1.MemoryService/DeleteMemory":       {},
	"/cortexdb.v1.KnowledgeService/GetKnowledge":    {},
	"/cortexdb.v1.KnowledgeService/DeleteKnowledge": {},
}

// authorizeRequestRows is the interceptor's row check, with the id-addressed
// RPCs handed over to their handlers.
//
// The waiver is not a weakening: authz.AuthorizeRows can only ever deny these
// (the field it is confined to is not on the message), and the handler check
// that replaces it is strictly stronger — it compares against the row that
// exists rather than against a field the caller filled in.
func authorizeRequestRows(key authz.Key, fullMethod string, req any) error {
	if err := authorizeTypedArgs(key, req); err != nil {
		return err
	}
	if _, deferred := idAddressedMethods[fullMethod]; deferred {
		return nil
	}
	if fullMethod == authz.CallToolMethod {
		return authorizeToolArgs(key, req)
	}
	return key.AuthorizeRows(fullMethod, requestFieldLookup(req))
}

// authorizeToolArgs is the row check for CallTool. The request message carries
// the tool's arguments as a string of JSON, which requestFieldLookup cannot see
// into, so the arguments are parsed and checked against the allowlist line of
// the named tool (authz.AuthorizeToolArgs). Unconfined keys skip all of it, as
// they skip every row check.
//
// The gates a confined key needs (disable_memory for a key confined to a
// collection on a recall, and the reverse) are forced into the request first,
// so the handler calls the tool with exactly the arguments the policy checked.
func authorizeToolArgs(key authz.Key, req any) error {
	if key.Scope.IsZero() {
		return nil
	}
	call, ok := req.(*rpcv1.CallToolRequest)
	if !ok {
		return key.AuthorizeRows(authz.CallToolMethod, nil)
	}
	args := key.ConfineToolArgs(call.GetName(), []byte(call.GetArgsJson()))
	if err := key.AuthorizeToolArgs(call.GetName(), args); err != nil {
		return err
	}
	call.ArgsJson = string(args)
	return nil
}

// authorizeTypedArgs holds a typed write to the allowlist line of the tool
// it stands for, so the typed RPC and CallTool cannot disagree on what a
// confined key may send: declared entities and relations write the shared
// graph, and some metadata keys are the engine's. Only the arguments the two
// share are handed over; the scope fields and the row are checked as for any
// typed RPC. A request that is none of these five passes untouched.
//
// Two differences from CallTool, both because of what the typed RPC is.
// SaveKnowledge, UpdateKnowledge, SaveMemory and UpdateMemory are signed by
// the facade (cortexdb.WithWriter replaces written_by and updated_by), so
// those two keys are not handed over: a row read back and sent again is not
// refused for the signature it carries. InsertGraphDocument is not signed and
// keeps the refusal. And a memory write from a key that reaches no memory row
// (one confined to a collection) is left to the row checks, which answer it
// as main does: the interceptor refuses SaveMemory, and UpdateMemory is
// NOT_FOUND like GetMemory and DeleteMemory.
func authorizeTypedArgs(key authz.Key, req any) error {
	if key.Scope.IsZero() {
		return nil
	}
	var tool string
	args := map[string]any{}
	stringMap := func(m map[string]string) {
		if len(m) == 0 {
			return
		}
		meta := make(map[string]any, len(m))
		for k, v := range m {
			meta[k] = v
		}
		args["metadata"] = meta
	}
	signed := func(m map[string]any) map[string]any {
		delete(m, "written_by")
		delete(m, "updated_by")
		return m
	}
	graph := func(entities []*rpcv1.ToolEntityInput, relations []*rpcv1.ToolRelationInput) {
		if len(entities) > 0 {
			args["entities"] = []any{}
		}
		if len(relations) > 0 {
			args["relations"] = []any{}
		}
	}
	switch r := req.(type) {
	case *rpcv1.SaveKnowledgeRequest:
		tool, args["collection"] = "knowledge_save", r.GetCollection()
		stringMap(r.GetMetadata())
		graph(r.GetEntities(), r.GetRelations())
		if m, ok := args["metadata"].(map[string]any); ok {
			signed(m)
		}
	case *rpcv1.UpdateKnowledgeRequest:
		tool, args["collection"] = "knowledge_update", r.GetCollection()
		stringMap(r.GetMetadata())
		graph(r.GetEntities(), r.GetRelations())
		if m, ok := args["metadata"].(map[string]any); ok {
			signed(m)
		}
	case *rpcv1.InsertGraphDocumentRequest:
		// InsertGraphDocument drops a caller's copy of the marks, as the
		// ingest_document line of the allowlist expects.
		tool, args["collection"] = "ingest_document", r.GetCollection()
		stringMap(r.GetMetadata())
	case *rpcv1.SaveMemoryRequest:
		if !key.Scope.Permits(authz.FieldCollection, "") {
			return nil
		}
		tool = "memory_save"
		for field, v := range map[string]string{authz.FieldUserID: r.GetUserId(), authz.FieldScope: r.GetScope(), authz.FieldNamespace: r.GetNamespace()} {
			if v != "" {
				args[field] = v
			}
		}
		if m := r.GetMetadata().AsMap(); len(m) > 0 {
			args["metadata"] = signed(m)
		}
	case *rpcv1.UpdateMemoryRequest:
		if !key.Scope.Permits(authz.FieldCollection, "") {
			return nil
		}
		tool = "memory_update"
		if m := r.GetMetadata().AsMap(); len(m) > 0 {
			args["metadata"] = signed(m)
		}
	default:
		return nil
	}
	// Strings, and a Struct's own values: always JSON.
	raw, _ := json.Marshal(args)
	return key.AuthorizeToolArgs(tool, raw)
}

// withheld is the single answer a confined key gets for a row it may not have.
//
// NOT_FOUND, not PERMISSION_DENIED, and the same wording either way. Two
// different answers here would turn the id space into an oracle: an agent
// confined to its own rows could walk ids and learn which ones exist in
// somebody else's, and on a shared brain the ids are often meaningful strings
// ("hermes-deploy-key-rotation") — existence alone is the leak.
//
// NOT_FOUND rather than PERMISSION_DENIED because of which mistake it makes
// when it is wrong. The frequent legitimate case is a key asking for its own
// row that has expired or was never written, and "not found" is the true
// answer to that; answering PERMISSION_DENIED would tell an agent it lacks
// rights to its own memory and send whoever debugs it to the key file. The
// other direction only ever misinforms a caller reaching for a row it is not
// entitled to know about, which is the point.
func withheld(kind, id string) error {
	return status.Errorf(codes.NotFound, "%s %q not found", kind, id)
}

// scopePermitsMemory reports whether a scope may touch a memory row.
//
// Every confined field must be set on the row and equal. An empty field on the
// row — the row that predates scoping, whose session carries no user_id — does
// not match anything, so a confined key is refused it. That is the fail-closed
// call and it is the right one twice over: an unowned row belongs to the shared
// pool a confined key was confined away from, and a confined key cannot create
// one (its own writes are forced to carry its user_id by the interceptor), so
// allowing unowned rows could only ever hand a key rows it did not write. The
// operator's unconfined key still reaches them, which is who the pre-scoping
// rows belonged to all along.
//
// A confinement the row cannot express — a collection-confined key against a
// memory, which has no collection — is likewise a refusal rather than a pass.
// Ignoring an uncheckable constraint is how a scope quietly becomes wider than
// it reads.
func scopePermitsMemory(scope authz.Scope, record cortexdb.MemoryRecord) bool {
	// Memories carry no collection, so a collection-confined key has nothing
	// to match and is refused. The rule lives in authz so the toolbox guard
	// (authz.Scope.GuardToolRows) and this one cannot disagree.
	return scope.PermitsMemory(record.UserID, record.Scope, record.Namespace)
}

// scopePermitsKnowledge is scopePermitsMemory for knowledge documents, which
// carry a collection and none of the memory scoping fields. A user_id-confined
// key therefore reaches no knowledge row at all — the same answer the
// interceptor already gives it for SearchKnowledge, which declares no user_id.
//
// Membership is the whole rule (authz.Scope.Permits), and it is what makes the
// two hard cases fall out rather than needing to be special-cased: an unowned
// row's empty value is never among a confinement's values, and neither is the
// empty string passed in for a field the row's kind does not have at all. Both
// are refusals because they are not matches, not because a branch remembered
// to refuse them.
//
// A document whose chunks are gone has no collection the facade can read
// (it names the default one), so no confined key may have it, as the tool
// guard decides (authz.Scope.GuardToolRows).
func scopePermitsKnowledge(scope authz.Scope, record cortexdb.KnowledgeRecord) bool {
	return len(record.ChunkIDs) > 0 && scope.PermitsKnowledge(record.Collection)
}

// guardToolRows is the row check for CallTool: a tool that names a row by id
// may only touch a row within the caller's scope. The arguments were checked
// by the interceptor (authorizeToolArgs); the row is checked here, where the
// database is at hand, the way every id-addressed typed RPC checks its row.
// A row the key may not have is NOT_FOUND, same as a row that does not exist.
func guardToolRows(ctx context.Context, db *cortexdb.DB, tool, args string) error {
	scope, confined := confinedCaller(ctx)
	if !confined {
		return nil
	}
	err := scope.GuardToolRows(ctx, db, tool, []byte(args))
	switch {
	case err == nil:
		return nil
	case errors.Is(err, authz.ErrWithheld):
		return status.Error(codes.NotFound, err.Error())
	case errors.Is(err, authz.ErrDenied):
		return status.Error(codes.PermissionDenied, err.Error())
	default:
		return toStatus(err)
	}
}

// oneNotFound spells every NOT_FOUND a tool returns to a confined caller the
// way guardToolRows spells a row the caller may not have: "not found" and
// nothing else. The tool's own wording names the kind of row, or the store
// that missed it ("embedding not found"), and a confined caller that could
// read it would tell a row that does not exist from a row it may not have:
// the oracle the guard exists to close. An unconfined caller keeps the
// tool's wording, which is its diagnostic.
func oneNotFound(ctx context.Context, st error) error {
	if _, confined := confinedCaller(ctx); confined && status.Code(st) == codes.NotFound {
		return status.Error(codes.NotFound, authz.ErrWithheld.Error())
	}
	return st
}

// withheldIfMissing is oneNotFound for a typed RPC: a NOT_FOUND the facade
// returns to a confined caller is spelled as withheld spells a row the caller
// may not have, id included. UpdateKnowledge needs it because its guard lets a
// missing id through (guardKnowledgeUpsert), and the facade's own wording
// names the store that missed it ("embedding not found").
func withheldIfMissing(ctx context.Context, kind, id string, st error) error {
	if _, confined := confinedCaller(ctx); confined && status.Code(st) == codes.NotFound {
		return withheld(kind, id)
	}
	return st
}

// guardMemory is the pre-flight for every id-addressed memory RPC. It reports
// the row when the caller has been shown to own it, so a Get need not read it
// twice, and read is false when nothing was read because nothing needed to be.
//
// Check-then-mutate is not atomic here, and cannot be made so from this layer:
// cortexdb.DeleteMemory and cortexdb.UpdateMemory are themselves load-then-write
// with no surrounding transaction, so an ownership read folded into the same
// statement would need the facade to grow a transactional path. The exposure
// that leaves is narrow and worth stating rather than hiding. A memory's owner
// is the user_id of the session it hangs off, and the only call that can move a
// row to another session is SaveMemory upserting the same id — which a confined
// key may only ever do into its own user_id. So the window requires a second,
// differently-scoped key to re-home that exact id between the check and the
// write, and its outcome is a mutation the confined key was entitled to a
// moment before. The alternative — no ownership check at all — is what this
// file exists to stop.
func guardMemory(ctx context.Context, db *cortexdb.DB, id string) (record cortexdb.MemoryRecord, read bool, err error) {
	scope, confined := confinedCaller(ctx)
	if !confined {
		return cortexdb.MemoryRecord{}, false, nil
	}
	resp, err := db.GetMemory(ctx, cortexdb.MemoryGetRequest{MemoryID: id})
	if err != nil {
		st := toStatus(err)
		// A missing row and a forbidden row have to leave by the same door,
		// down to the message. Anything else — a broken database, a cancelled
		// context — says nothing about whether the row exists and is reported
		// as itself, or the caller cannot tell an outage from an empty brain.
		if status.Code(st) == codes.NotFound {
			return cortexdb.MemoryRecord{}, false, withheld("memory", id)
		}
		return cortexdb.MemoryRecord{}, false, st
	}
	if !scopePermitsMemory(scope, resp.Memory) {
		return cortexdb.MemoryRecord{}, false, withheld("memory", id)
	}
	return resp.Memory, true, nil
}

// guardKnowledge is guardMemory for knowledge documents; see its comment for
// the ordering and race reasoning, which is the same. cortexdb.DeleteKnowledge
// is also a load-then-delete without a transaction.
func guardKnowledge(ctx context.Context, db *cortexdb.DB, id string) (record cortexdb.KnowledgeRecord, read bool, err error) {
	scope, confined := confinedCaller(ctx)
	if !confined {
		return cortexdb.KnowledgeRecord{}, false, nil
	}
	resp, err := db.GetKnowledge(ctx, cortexdb.KnowledgeGetRequest{KnowledgeID: id})
	if err != nil {
		st := toStatus(err)
		if status.Code(st) == codes.NotFound {
			return cortexdb.KnowledgeRecord{}, false, withheld("knowledge", id)
		}
		return cortexdb.KnowledgeRecord{}, false, st
	}
	if !scopePermitsKnowledge(scope, resp.Knowledge) {
		return cortexdb.KnowledgeRecord{}, false, withheld("knowledge", id)
	}
	return resp.Knowledge, true, nil
}

// An upsert is a write to whatever id it names, and ids on a shared brain are
// guessable strings. Without this, a key confined to user_id="hermes" could
// SaveMemory with somebody else's id and the ON CONFLICT DO UPDATE behind it
// would overwrite that row *and* re-home it into hermes's bucket — the row's
// owner becomes the caller, so the theft also hides itself. Verified against a
// running server before this existed: openclaw's memory came back as
// user_id="hermes" with hermes's content.
//
// The rule is the mirror of guardMemory: an id that names an existing row the
// caller may not have is refused; an id that names nothing is a create and is
// allowed. Not-found is therefore success here, not a withholding, which is
// the one place the two guards differ.
func guardMemoryUpsert(ctx context.Context, db *cortexdb.DB, id string) error {
	// The facade refuses only an empty id; an id of blanks names a row.
	if id == "" {
		return nil
	}
	_, _, err := guardMemory(ctx, db, id)
	if err == nil {
		return nil
	}
	// guardMemory collapses "missing" and "forbidden" into one answer so a
	// reader cannot probe the id space. A writer must tell them apart, and can
	// do so safely: it is about to create the row either way, so learning that
	// the id was free tells it nothing it will not know a moment later.
	if status.Code(err) == codes.NotFound {
		if _, gerr := db.GetMemory(ctx, cortexdb.MemoryGetRequest{MemoryID: id}); gerr != nil {
			return nil
		}
	}
	return err
}

// guardKnowledgeUpsert is guardMemoryUpsert for documents. UpdateKnowledge
// needs it too, and for a second reason: it can set Collection, so without this
// a collection-confined key could move any document it can name into its own
// collection.
func guardKnowledgeUpsert(ctx context.Context, db *cortexdb.DB, id string) error {
	if id == "" {
		return nil
	}
	_, _, err := guardKnowledge(ctx, db, id)
	if err == nil {
		return nil
	}
	if status.Code(err) == codes.NotFound {
		if _, gerr := db.GetKnowledge(ctx, cortexdb.KnowledgeGetRequest{KnowledgeID: id}); gerr != nil {
			return nil
		}
	}
	return err
}
