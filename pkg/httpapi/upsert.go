package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// upsertRoute is a route that writes the row its id names, creating it when
// the id names nothing: kind is the row's name in a refusal, field the body
// field that carries the id, and supersedes whether the body may also name
// rows to retire.
type upsertRoute struct {
	kind, field string
	supersedes  bool
}

// upsertRoutes are the REST twins of SaveKnowledge and SaveMemory. An upsert
// is a write to whatever id it names, and ids on a shared brain are guessable
// strings, so a confined key that names another scope's row would overwrite
// it and move it into its own scope. The gRPC side guards the same writes
// (guardKnowledgeUpsert, guardMemoryUpsert); this is that rule over REST.
// POST /v1/memory also takes supersedes, which the gRPC message does not:
// each target is marked retired, so each is a row the key writes.
var upsertRoutes = map[string]upsertRoute{
	"POST /v1/knowledge": {kind: "knowledge", field: "knowledge_id"},
	"POST /v1/memory":    {kind: "memory", field: "memory_id", supersedes: true},
}

// guardUpsert checks the rows a confined key's upsert names. An id that
// names an existing row outside the scope is 404, worded as the gRPC side
// words it, and an id that names nothing is a create and passes. A supersedes
// target must be a row the key may have: one outside the scope and one that
// does not exist get the same 404, before anything is written. Every spelling
// of each field is checked, since encoding/json reads any of them into the
// request. A read that fails is 500, never a pass.
//
// The guard runs before the handler, so for a confined key its 404 comes
// before the handler's own 400s (empty content, no memory_id).
//
// The reads go through GetKnowledge and GetMemory by id, which the caller
// scope does not filter: a foreign row reads as itself, not as missing. If
// those reads ever honour CallerScope, a foreign row would pass here as a
// create; the 404 tests would then fail.
//
// The check reads, then the handler writes; the window between them is the
// one the gRPC guards document (guardMemory).
//
// It returns false when it has written a response.
func guardUpsert(w http.ResponseWriter, r *http.Request, db *cortexdb.DB, scope authz.Scope, u upsertRoute) bool {
	// bodyFieldLookup has read this body a moment ago, within the size limit,
	// and refused one that is not a JSON object, so this second read of the
	// buffer it put back cannot fail.
	body, _ := bufferBody(w, r)
	fields := map[string]any{}
	_ = json.Unmarshal(body, &fields)
	for _, id := range spellings(fields, u.field) {
		// The facade refuses only an empty id; an id of blanks names a row.
		if id == "" {
			continue
		}
		found, permitted, err := rowPermitted(r.Context(), db, scope, u.kind, id)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return false
		}
		if found && !permitted {
			writeError(w, http.StatusNotFound, fmt.Sprintf("%s %q not found", u.kind, id))
			return false
		}
	}
	if !u.supersedes {
		return true
	}
	// The request the handler decodes, so the new memory and its targets are
	// the ones SaveMemory will see, whichever spelling the body uses. A body
	// that does not decode is refused by the handler before anything is
	// written.
	var save cortexdb.MemorySaveRequest
	_ = json.Unmarshal(body, &save)
	for _, target := range save.Supersedes {
		// The targets SaveMemory skips: blank, or the new memory itself.
		target = strings.TrimSpace(target)
		if target == "" || target == save.MemoryID {
			continue
		}
		found, permitted, err := rowPermitted(r.Context(), db, scope, "memory", target)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return false
		}
		if !found || !permitted {
			writeError(w, http.StatusNotFound, fmt.Sprintf("supersede %q: not found", target))
			return false
		}
	}
	return true
}

// spellings returns the string values the object carries under every
// spelling of field, the exact one first.
func spellings(obj map[string]any, field string) []string {
	top, _ := obj[field].(string)
	return append([]string{top}, otherSpellings(obj, field)...)
}

// rowPermitted reads the row id names and reports whether it exists and, if
// it does, whether the scope may write it. Only the facade's not-found
// sentinel means a missing row; any other read failure is returned.
func rowPermitted(ctx context.Context, db *cortexdb.DB, scope authz.Scope, kind, id string) (found, permitted bool, err error) {
	if kind == "knowledge" {
		resp, err := db.GetKnowledge(ctx, cortexdb.KnowledgeGetRequest{KnowledgeID: id})
		if err != nil {
			return false, false, readError(err)
		}
		return true, scope.Permits(authz.FieldCollection, resp.Knowledge.Collection) &&
			scope.Permits(authz.FieldUserID, "") &&
			scope.Permits(authz.FieldScope, "") &&
			scope.Permits(authz.FieldNamespace, ""), nil
	}
	resp, err := db.GetMemory(ctx, cortexdb.MemoryGetRequest{MemoryID: id})
	if err != nil {
		return false, false, readError(err)
	}
	return true, scope.Permits(authz.FieldUserID, resp.Memory.UserID) &&
		scope.Permits(authz.FieldScope, resp.Memory.Scope) &&
		scope.Permits(authz.FieldNamespace, resp.Memory.Namespace) &&
		scope.Permits(authz.FieldCollection, ""), nil
}

// readError is nil for a missing row and the read's own error otherwise.
func readError(err error) error {
	if errors.Is(err, core.ErrNotFound) {
		return nil
	}
	return err
}
