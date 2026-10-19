package authz

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// Row ownership for tool calls.
//
// Key.AuthorizeToolArgs (allowlist.go) checks the arguments of a tool call
// against the key's scope: the collection asked for is one the key may name.
// That is the whole check a search needs. A tool that names a row by id — knowledge_update,
// ingest_document, memory_save — needs one more: the row that id already
// names must itself lie within the scope. Otherwise a key confined to
// `team-acme` passes collection=team-acme and the id of a document in
// `secrets`, and the write lands: the document moves, or is overwritten, in a
// collection the key was confined away from. The typed RPCs have guarded
// their rows this way since scoping existed (rpcserver/ownership.go); this
// file is the same rule for the toolbox, shared by the gRPC and REST servers.
//
// A row that does not exist yet is not guarded: the write creates it, and
// the argument check has already forced the confined fields onto it. A row
// the key may not have is answered with ErrWithheld, bare: "not found" and
// nothing else, not the kind, not the id. The servers turn it into NOT_FOUND
// / 404 and spell a missing row's "not found", which comes from the tool, the
// same way for a confined key, so a read, an update or a delete answers a
// foreign id and a missing one alike. A tool that creates the row it names
// cannot: a missing id is created and a foreign one is refused, so the
// caller learns that the id is taken. Ids are global; that is the price of
// refusing the write rather than letting it move the row.
//
// Between this check and the tool's write there is no lock: a row created
// under the same id by another caller in that window is written by the tool.
//
// The guard reads rows through the facade it protects (RowLoader is
// *cortexdb.DB). The facade confines a caller's searches and bucket writes
// to its CallerScope, and must keep its single-row reads by id unconfined:
// were a foreign row hidden from GetMemory or GetKnowledge, the guard would
// take it for a row that does not exist and let the write through, moving or
// overwriting the row. cortexdb.DB.GetMemory hides foreign rows in its list
// form only, which the guard does not use; the test
// TestTheRowGuardSeesForeignRowsUnderACallerScope holds the single form.

// ErrWithheld is the one answer for a row the key may not have.
var ErrWithheld = errors.New("not found")

// PermitsKnowledge reports whether the scope may touch a knowledge row (a
// document) in the given collection. Knowledge carries a collection and none
// of the memory fields, so a key confined on any memory field reaches no
// knowledge row: an uncheckable confinement is a refusal, not a pass.
func (s Scope) PermitsKnowledge(collection string) bool {
	return s.Permits(FieldCollection, collection) &&
		s.Permits(FieldUserID, "") &&
		s.Permits(FieldScope, "") &&
		s.Permits(FieldNamespace, "")
}

// PermitsMemory is PermitsKnowledge for a memory row, which carries user_id,
// scope and namespace and no collection.
func (s Scope) PermitsMemory(userID, memoryScope, namespace string) bool {
	return s.Permits(FieldUserID, userID) &&
		s.Permits(FieldScope, memoryScope) &&
		s.Permits(FieldNamespace, namespace) &&
		s.Permits(FieldCollection, "")
}

// RowLoader is what GuardToolRows needs from the database: the two id reads.
// *cortexdb.DB satisfies it.
type RowLoader interface {
	GetKnowledge(ctx context.Context, req cortexdb.KnowledgeGetRequest) (*cortexdb.KnowledgeGetResponse, error)
	GetMemory(ctx context.Context, req cortexdb.MemoryGetRequest) (*cortexdb.MemoryGetResponse, error)
}

// GuardToolRows checks that every row the tool call names by id, and that
// already exists, lies within the scope. The ids are read off the paths the
// allowlist lists for the tool, nested ones included. It is a no-op for an
// unconfined scope. ErrWithheld wraps the refusal; any other error is a read
// failure.
func (s Scope) GuardToolRows(ctx context.Context, db RowLoader, tool string, args []byte) error {
	if s.IsZero() {
		return nil
	}
	rule, ok := confinedTools[tool]
	if !ok {
		// Never reached behind AuthorizeToolArgs, which refuses a tool off the
		// allowlist first; refused here too, so the guard fails closed on its own.
		return fmt.Errorf("%w: %s is not a tool a confined key may call", ErrDenied, tool)
	}
	for _, field := range rule.ids {
		ids, err := JSONPathValues(args, field.path)
		if err != nil {
			return fmt.Errorf("%w: %v", ErrDenied, err)
		}
		for _, id := range ids {
			if err := s.guardRow(ctx, db, field.kind, id); err != nil {
				return err
			}
		}
	}
	return nil
}

// guardRow loads one row and holds it to the scope. A row that does not exist
// passes: the write creates it, inside the scope the argument check forced.
func (s Scope) guardRow(ctx context.Context, db RowLoader, kind idKind, id string) error {
	if kind == idMemory {
		resp, err := db.GetMemory(ctx, cortexdb.MemoryGetRequest{MemoryID: id})
		if err != nil {
			if isNotFound(err) {
				return nil
			}
			return fmt.Errorf("read memory %q: %w", id, err)
		}
		m := resp.Memory
		if !s.PermitsMemory(m.UserID, m.Scope, m.Namespace) {
			return ErrWithheld
		}
		return nil
	}
	resp, err := db.GetKnowledge(ctx, cortexdb.KnowledgeGetRequest{KnowledgeID: id})
	if err != nil {
		if isNotFound(err) {
			return nil
		}
		return fmt.Errorf("read knowledge %q: %w", id, err)
	}
	// The facade reads a document's collection off its chunks and names the
	// default one when there are none, so a document without chunks cannot
	// be placed: withheld, whatever the key.
	if len(resp.Knowledge.ChunkIDs) == 0 || !s.PermitsKnowledge(resp.Knowledge.Collection) {
		return ErrWithheld
	}
	return nil
}

// isNotFound is the facade's answer for a row that does not exist. Any
// other error, whatever its text says, is a read the guard could not do.
func isNotFound(err error) bool {
	return errors.Is(err, core.ErrNotFound) || errors.Is(err, sql.ErrNoRows)
}
