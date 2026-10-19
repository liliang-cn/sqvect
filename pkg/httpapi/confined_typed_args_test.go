package httpapi

import (
	"net/http"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// A typed write a confined key may call takes what the allowlist line of
// the tool it stands for takes, and nothing more: POST /v1/knowledge is
// knowledge_save and POST /v1/memory is memory_save. So declared entities
// and relations, an engine key in metadata, and a document's chunk keys
// are refused on both paths with the same reason, and the caller's own
// metadata passes on both. The operator's unconfined key is not held to the
// allowlist. The typed writes of the shared graph name no scope field, so a
// confined key is refused them whole, as before.
func TestATypedWriteTakesWhatTheAllowlistLineOfItsToolTakes(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
		{ID: "lab", Secret: "lab-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Namespace: "lab"}},
		{ID: "mine", Secret: "mine-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{MemoryScope: "user"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	for _, c := range []struct{ secret, path, body, want string }{
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n1","collection":"notes","content":"x","entities":[{"name":"Project Nightshade","description":"defaced"}]}`, "entities"},
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n1","collection":"notes","content":"x","relations":[{"from":"Project Nightshade","to":"Orpheus","type":"funds"}]}`, "relations"},
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n1","collection":"notes","content":"x","metadata":{"_scope":"global"}}`, "metadata._scope"},
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n1","collection":"notes","content":"x","metadata":{"Chunk_Index":"0"}}`, "metadata.chunk_index"},
		{"hermes-secret", "/v1/memory", `{"memory_id":"h1","user_id":"hermes","scope":"user","content":"x","metadata":{"_scope":"global"}}`, "metadata._scope"},
		// The handler's decoder matches a field name by Unicode case folding,
		// so "entitieſ" (a long s, U+017F) is its Entities and "Metadata"
		// its Metadata. The check reads the body as the decoder does.
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n1","collection":"notes","content":"x","entitieſ":[{"name":"Project Nightshade","description":"defaced"}]}`, "entities"},
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n1","collection":"notes","content":"x","relationſ":[{"from":"Project Nightshade","to":"Orpheus","type":"funds"}]}`, "relations"},
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n1","collection":"notes","content":"x","METADATA":{"_scope":"global"}}`, "metadata._scope"},
		{"hermes-secret", "/v1/memory", `{"memory_id":"h1","user_id":"hermes","scope":"user","content":"x","entitieſ":[{"name":"Project Nightshade","description":"defaced"}]}`, "entities"},
		{"hermes-secret", "/v1/memory", `{"memory_id":"h1","user_id":"hermes","scope":"user","content":"x","Relationſ":[{"from":"Project Nightshade","to":"Orpheus","type":"funds"}]}`, "relations"},
		// The decoder keeps the last of two spellings, so an empty one after
		// the key's collection leaves the write in no collection of the key.
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n1","collection":"notes","Collection":"","content":"x"}`, "leaves collection unset"},
	} {
		status, payload := do(t, srv, http.MethodPost, c.path, c.secret, c.body)
		if status != http.StatusForbidden || !strings.Contains(errorMessage(t, payload), c.want) {
			t.Errorf("%s %s: %d %s, want 403 naming %s", c.path, c.body, status, payload, c.want)
		}
		if status, payload := do(t, srv, http.MethodPost, c.path, "op-secret", c.body); status == http.StatusForbidden {
			t.Errorf("%s: the operator was held to the allowlist: %s", c.path, payload)
		}
	}
	for _, c := range []struct{ secret, path, body string }{
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n2","collection":"notes","content":"Orpheus plays.","metadata":{"color":"blue"}}`},
		{"hermes-secret", "/v1/memory", `{"memory_id":"h2","user_id":"hermes","scope":"user","content":"Hermes tunes the lyre.","metadata":{"document_id":"a note of mine"}}`},
		// Empty lists declare nothing, so the typed routes take them as main does.
		{"notes-secret", "/v1/knowledge", `{"knowledge_id":"n3","collection":"notes","content":"Orpheus sings.","entities":[],"relations":[]}`},
		{"hermes-secret", "/v1/memory", `{"memory_id":"h3","user_id":"hermes","scope":"user","content":"Hermes runs.","entities":[],"relations":[]}`},
		// Each memory scope field the key is confined to is handed to the check.
		{"lab-secret", "/v1/memory", `{"memory_id":"l1","user_id":"u","scope":"user","namespace":"lab","content":"A lab note."}`},
		{"mine-secret", "/v1/memory", `{"memory_id":"u1","user_id":"u","scope":"user","content":"A note of mine."}`},
	} {
		if status, payload := do(t, srv, http.MethodPost, c.path, c.secret, c.body); status != http.StatusOK {
			t.Errorf("%s with the caller's metadata: %d %s", c.path, status, payload)
		}
	}
	for _, path := range []string{"/v1/graph/entities", "/v1/graph/relations", "/v1/graph/triples"} {
		if status, payload := do(t, srv, http.MethodPost, path, "notes-secret", `{}`); status != http.StatusForbidden {
			t.Errorf("%s under a confined key: %d %s, want 403", path, status, payload)
		}
	}
}
