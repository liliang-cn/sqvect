package rpcserver

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// newPolicyConn runs a real server behind a key policy and returns a client
// connection to it. The end-to-end shape matters here: a scope check that only
// holds in a unit test of the interceptor is a scope check that has not been
// wired up.
func newPolicyConn(t *testing.T, opts Options, dbOpts ...cortexdb.Option) *grpc.ClientConn {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "scoped.db")
	db, err := cortexdb.Open(cortexdb.DefaultConfig(dbPath), dbOpts...)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	opts.DBPath = dbPath
	srv, err := NewWithPolicy(db, opts)
	if err != nil {
		t.Fatalf("build server: %v", err)
	}
	lis := bufconn.Listen(1 << 20)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial bufnet: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

func asKey(secret string) context.Context {
	return metadata.AppendToOutgoingContext(context.Background(), "authorization", "Bearer "+secret)
}

// hermesAndZeus is the shape the shared-brain docs describe: two agents against
// one brain, each confined to its own user_id, plus the operator's full key.
func hermesAndZeus(t *testing.T) *authz.KeySet {
	t.Helper()
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite,
			Scope: authz.Scope{UserID: "hermes"}},
		{ID: "hermes-reader", Secret: "hermes-ro", Clearance: authz.ReadOnly,
			Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatalf("build keys: %v", err)
	}
	return ks
}

func TestAKeyConfinedToOneUserCannotReadAnotherUsersMemory(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	client := rpcv1.NewMemoryServiceClient(conn)

	if _, err := client.SaveMemory(asKey("op-secret"), &rpcv1.SaveMemoryRequest{
		MemoryId: "z1", UserId: "zeus", Scope: "user",
		Content: "Zeus keeps the thunderbolts in the second drawer.",
	}); err != nil {
		t.Fatalf("operator save: %v", err)
	}

	_, err := client.SearchMemory(asKey("hermes-secret"), &rpcv1.SearchMemoryRequest{
		Query: "thunderbolts", UserId: "zeus", TopK: 5,
	})
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("hermes read zeus's memory: %v", err)
	}

	// And its own rows are still reachable, or the confinement is useless.
	if _, err := client.SearchMemory(asKey("hermes-secret"), &rpcv1.SearchMemoryRequest{
		Query: "thunderbolts", UserId: "hermes", TopK: 5,
	}); err != nil {
		t.Fatalf("hermes was refused its own rows: %v", err)
	}
}

func TestAKeyConfinedToOneUserCannotWriteIntoAnotherUsersMemory(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	client := rpcv1.NewMemoryServiceClient(conn)

	_, err := client.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{
		MemoryId: "z2", UserId: "zeus", Scope: "user",
		Content: "Planted by hermes.",
	})
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("hermes wrote into zeus's rows: %v", err)
	}

	if _, err := client.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{
		MemoryId: "h1", UserId: "hermes", Scope: "user",
		Content: "Hermes prefers the fast route.",
	}); err != nil {
		t.Fatalf("hermes was refused its own write: %v", err)
	}
}

func TestAReadOnlyKeyCannotDeleteSomebodyElsesMemory(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	client := rpcv1.NewMemoryServiceClient(conn)

	if _, err := client.SaveMemory(asKey("op-secret"), &rpcv1.SaveMemoryRequest{
		MemoryId: "z3", UserId: "zeus", Scope: "user", Content: "Zeus's note.",
	}); err != nil {
		t.Fatalf("operator save: %v", err)
	}

	_, err := client.DeleteMemory(asKey("hermes-ro"), &rpcv1.DeleteMemoryRequest{MemoryId: "z3"})
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("a read-only key deleted a memory: %v", err)
	}

	got, err := client.GetMemory(asKey("op-secret"), &rpcv1.GetMemoryRequest{MemoryId: "z3"})
	if err != nil || got.GetMemory().GetId() != "z3" {
		t.Fatalf("the memory did not survive the refused delete: %v", err)
	}
}

// The gap this file used to document — a confined key refused every
// id-addressed RPC, its own rows included, because the request does not say
// whose row an id names — is closed. The handler reads the row and checks its
// owner; see ownership.go and ownership_test.go, where
// TestAConfinedKeyGetsItsOwnMemoryButNotAnothersAndCannotTellThemApart
// replaces TestAConfinedKeyCannotReachAMemoryByIdAlone.

func TestAConfinedKeyCanStillAskWhetherTheServerIsUp(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	admin := rpcv1.NewAdminServiceClient(conn)
	resp, err := admin.Health(asKey("hermes-ro"), &rpcv1.HealthRequest{})
	if err != nil || !resp.GetOk() {
		t.Fatalf("health probe refused to a confined key: %v", err)
	}
}

func TestTheLegacySingleTokenDeploymentStillHasFullAccess(t *testing.T) {
	// The regression that would break every deployment in the wild: only
	// Options.Token set, no key file, everything still permitted.
	conn := newPolicyConn(t, Options{Token: "s3cret"})
	memory := rpcv1.NewMemoryServiceClient(conn)
	ctx := asKey("s3cret")

	if _, err := memory.SaveMemory(ctx, &rpcv1.SaveMemoryRequest{
		MemoryId: "m1", UserId: "anyone", Scope: "user", Content: "Anything at all.",
	}); err != nil {
		t.Fatalf("legacy token save: %v", err)
	}
	if _, err := memory.GetMemory(ctx, &rpcv1.GetMemoryRequest{MemoryId: "m1"}); err != nil {
		t.Fatalf("legacy token get: %v", err)
	}
	if _, err := memory.SearchMemory(ctx, &rpcv1.SearchMemoryRequest{
		Query: "anything", TopK: 3,
	}); err != nil {
		t.Fatalf("legacy token search with no user_id: %v", err)
	}
	del, err := memory.DeleteMemory(ctx, &rpcv1.DeleteMemoryRequest{MemoryId: "m1"})
	if err != nil || !del.GetDeleted() {
		t.Fatalf("legacy token delete: %v", err)
	}

	graph := rpcv1.NewKnowledgeGraphServiceClient(conn)
	if _, err := graph.ListNamespaces(ctx, &rpcv1.ListNamespacesRequest{}); err != nil {
		t.Fatalf("legacy token graph read: %v", err)
	}
}

func TestARevokedKeyIsRefused(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	client := rpcv1.NewMemoryServiceClient(conn)
	_, err := client.SearchMemory(asKey("a-secret-nobody-issued"), &rpcv1.SearchMemoryRequest{
		Query: "anything", UserId: "hermes",
	})
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("want UNAUTHENTICATED, got %v", err)
	}
}

func TestAKeyConfinedToACollectionCannotSearchAnother(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite,
			Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	client := rpcv1.NewKnowledgeServiceClient(conn)

	if _, err := client.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{
		KnowledgeId: "k1", Title: "A note", Content: "Something worth keeping.",
		Collection: "notes",
	}); err != nil {
		t.Fatalf("save into the permitted collection: %v", err)
	}
	_, err = client.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{
		KnowledgeId: "k2", Title: "Elsewhere", Content: "Not yours.",
		Collection: "secrets",
	})
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("wrote into another collection: %v", err)
	}
	_, err = client.SearchKnowledge(asKey("notes-secret"), &rpcv1.SearchKnowledgeRequest{
		Query: "note", Collection: "secrets", TopK: 3,
	})
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("searched another collection: %v", err)
	}
}

func TestAConfinedKeyCannotReachTheRdfGraphAtAll(t *testing.T) {
	// The RDF store is one global graph with no user_id, scope, namespace or
	// collection on its requests, so there is nothing for a scope to check
	// against. SPARQL in particular can read everything in one query. Fail
	// closed: a confined key is refused rather than handed the whole graph.
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	graph := rpcv1.NewKnowledgeGraphServiceClient(conn)
	_, err := graph.QuerySparql(asKey("hermes-secret"), &rpcv1.QuerySparqlRequest{
		Query: "SELECT ?s WHERE { ?s ?p ?o }",
	})
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("a confined key reached the shared RDF graph: %v", err)
	}
	// The operator's unconfined key is unaffected.
	if _, err := graph.QuerySparql(asKey("op-secret"), &rpcv1.QuerySparqlRequest{
		Query: "SELECT ?s WHERE { ?s ?p ?o }",
	}); err != nil {
		t.Fatalf("operator sparql: %v", err)
	}
}

func TestAKeyFileOnDiskDrivesTheRunningServer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "keys.json")
	body := `{"keys":[
	  {"id":"hermes","secret":"hermes-secret","clearance":"read-only","scope":{"user_id":"hermes"}}
	]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{KeyFile: path, Token: "ignored-legacy-token"})
	client := rpcv1.NewMemoryServiceClient(conn)

	if _, err := client.SearchMemory(asKey("hermes-secret"), &rpcv1.SearchMemoryRequest{
		Query: "anything", UserId: "hermes", TopK: 3,
	}); err != nil {
		t.Fatalf("the key from the file was refused: %v", err)
	}
	_, err := client.SearchMemory(asKey("ignored-legacy-token"), &rpcv1.SearchMemoryRequest{
		Query: "anything", UserId: "hermes", TopK: 3,
	})
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("the environment token outranked the key file: %v", err)
	}
}

// The MCP shared-brain client proxies every tool call through CallTool. Until
// the arguments could be checked, a confined key was refused every tool and so
// could not use the shared brain at all. Now the arguments are checked against
// the scope fields the named tool declares: hermes reads and writes its own
// memory through the tools, is refused another user's, is refused a call that
// leaves user_id unset, and is refused a tool off the allowlist.
func TestAConfinedKeyMayCallToolsWithinItsScope(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	tools := rpcv1.NewToolsServiceClient(conn)
	call := func(secret, tool, args string) error {
		_, err := tools.CallTool(asKey(secret), &rpcv1.CallToolRequest{Name: tool, ArgsJson: args})
		return err
	}

	if err := call("hermes-secret", "memory_save",
		`{"memory_id":"h1","content":"hermes deploys on Fridays","user_id":"hermes","scope":"user"}`); err != nil {
		t.Fatalf("hermes could not save its own memory through the tools: %v", err)
	}
	if err := call("hermes-secret", "memory_search", `{"query":"deploys","user_id":"hermes"}`); err != nil {
		t.Fatalf("hermes could not search its own memory through the tools: %v", err)
	}

	for name, c := range map[string]struct{ tool, args, want string }{
		"another user":        {"memory_search", `{"query":"deploys","user_id":"zeus"}`, `asks for "zeus"`},
		"user unset":          {"memory_search", `{"query":"deploys"}`, "leaves user_id unset"},
		"tool without a user": {"graph_statistics", `{}`, "not a tool a confined key may call"},
		"nested disagreement": {"knowledge_memory_recall", `{"query":"q","user_id":"hermes","plan":{"filters":{"user_id":"zeus"}}}`, "nested user_id"},
	} {
		err := call("hermes-secret", c.tool, c.args)
		if status.Code(err) != codes.PermissionDenied {
			t.Fatalf("%s: a confined key got through CallTool: %v", name, err)
		}
		if msg := status.Convert(err).Message(); !strings.Contains(msg, c.want) {
			t.Fatalf("%s: denial %q should say %q", name, msg, c.want)
		}
	}

	// Arguments that are not a JSON object cannot be checked: denied, not served.
	if err := call("hermes-secret", "memory_search", `[1,2,3]`); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("non-object arguments were served to a confined key: %v", err)
	}

	// The operator's unconfined key is unaffected.
	if err := call("op-secret", "graph_statistics", `{}`); err != nil {
		t.Fatalf("operator graph_statistics: %v", err)
	}
}

func TestAKeyConfinedToACollectionMayIngestAndSearchItThroughTheTools(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite,
			Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	tools := rpcv1.NewToolsServiceClient(conn)
	call := func(tool, args string) error {
		_, err := tools.CallTool(asKey("notes-secret"), &rpcv1.CallToolRequest{Name: tool, ArgsJson: args})
		return err
	}

	if err := call("ingest_document",
		`{"document_id":"n1","content":"A note about the ledger migration.","collection":"notes"}`); err != nil {
		t.Fatalf("ingest into the permitted collection: %v", err)
	}
	if err := call("knowledge_search", `{"query":"ledger","collection":"notes","top_k":3}`); err != nil {
		t.Fatalf("search the permitted collection: %v", err)
	}
	if err := call("ingest_document",
		`{"document_id":"n2","content":"Not yours.","collection":"secrets"}`); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("ingested into another collection: %v", err)
	}
	if err := call("knowledge_search", `{"query":"ledger","top_k":3}`); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("searched every collection: %v", err)
	}
	if err := call("delete_document_graph", `{"document_id":"n1"}`); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("a tool that takes no collection was served to a collection-confined key: %v", err)
	}
}

// One key, several collections: the shared-brain shape where an agent reads
// its team's knowledge and its own. Both ports apply the same scope, so the
// typed RPC and the tool call agree on what is in and what is out.
func TestAKeyConfinedToSeveralCollectionsReachesEachOfThem(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "lf", Secret: "lf-secret", Clearance: authz.ReadWrite,
			Scope: authz.Scope{Collections: []string{"team-acme", "user-lf"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)
	tools := rpcv1.NewToolsServiceClient(conn)

	for _, c := range []string{"team-acme", "user-lf"} {
		if _, err := knowledge.SaveKnowledge(asKey("lf-secret"), &rpcv1.SaveKnowledgeRequest{
			KnowledgeId: "k-" + c, Title: c, Content: "Kept in " + c + ".", Collection: c,
		}); err != nil {
			t.Fatalf("save into %s: %v", c, err)
		}
		if _, err := tools.CallTool(asKey("lf-secret"), &rpcv1.CallToolRequest{
			Name: "knowledge_search", ArgsJson: `{"query":"kept","collection":"` + c + `","top_k":3}`,
		}); err != nil {
			t.Fatalf("tool search in %s: %v", c, err)
		}
	}
	_, err = knowledge.SearchKnowledge(asKey("lf-secret"), &rpcv1.SearchKnowledgeRequest{
		Query: "kept", Collection: "secrets", TopK: 3,
	})
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("searched a collection outside the list: %v", err)
	}
	_, err = tools.CallTool(asKey("lf-secret"), &rpcv1.CallToolRequest{
		Name: "knowledge_search", ArgsJson: `{"query":"kept","collection":"secrets"}`,
	})
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("tool searched a collection outside the list: %v", err)
	}
}

// The argument check lets a confined key name its own collection; the row
// check keeps it from naming another collection's row with it. Found by the
// end-to-end trial: lf passed collection=team-acme and the id of a document
// in secrets, and the document was overwritten and moved.
func TestAConfinedKeyCannotReachAnotherRowByIDThroughTheTools(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "lf", Secret: "lf-secret", Clearance: authz.ReadWrite,
			Scope: authz.Scope{Collections: []string{"team-acme", "user-lf"}}},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite,
			Scope: authz.Scope{UserID: "hermes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	tools := rpcv1.NewToolsServiceClient(conn)
	call := func(secret, tool, args string) error {
		_, err := tools.CallTool(asKey(secret), &rpcv1.CallToolRequest{Name: tool, ArgsJson: args})
		return err
	}
	must := func(what string, err error) {
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	must("operator writes secret-doc", call("op-secret", "knowledge_save", `{"knowledge_id":"secret-doc","content":"The secret retainer.","collection":"secrets"}`))
	must("operator writes zeus-1", call("op-secret", "memory_save", `{"memory_id":"zeus-1","content":"Zeus private.","user_id":"zeus","scope":"user"}`))
	must("lf writes its own", call("lf-secret", "knowledge_save", `{"knowledge_id":"acme-1","content":"Acme spec.","collection":"team-acme"}`))

	for name, c := range map[string]struct{ tool, args string }{
		"knowledge_update": {"knowledge_update", `{"knowledge_id":"secret-doc","collection":"team-acme","content":"overwritten"}`},
		"knowledge_save":   {"knowledge_save", `{"knowledge_id":"secret-doc","collection":"team-acme","content":"overwritten"}`},
		"ingest_document":  {"ingest_document", `{"document_id":"secret-doc","collection":"team-acme","content":"overwritten"}`},
	} {
		if err := call("lf-secret", c.tool, c.args); status.Code(err) != codes.NotFound {
			t.Errorf("%s: lf reached another collection's row: %v", name, err)
		}
	}
	if err := call("hermes-secret", "memory_save", `{"memory_id":"zeus-1","content":"overwritten","user_id":"hermes","scope":"user"}`); status.Code(err) != codes.NotFound {
		t.Errorf("hermes reached zeus's memory: %v", err)
	}
	// Own rows, by id, still work: update and re-ingest within the scope.
	must("lf updates its own", call("lf-secret", "knowledge_update", `{"knowledge_id":"acme-1","collection":"team-acme","content":"Acme spec v2."}`))
	must("hermes saves its own", call("hermes-secret", "memory_save", `{"memory_id":"h-1","content":"mine","user_id":"hermes","scope":"user"}`))

	// Nothing moved: the operator reads the rows as written.
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)
	got, err := knowledge.GetKnowledge(asKey("op-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: "secret-doc"})
	must("operator reads secret-doc", err)
	if got.GetKnowledge().GetCollection() != "secrets" || !strings.Contains(got.GetKnowledge().GetContent(), "retainer") {
		t.Fatalf("secret-doc was changed: %v", got.GetKnowledge())
	}
}

func TestASaveIsSignedByTheKeyThatMadeItWhateverItClaims(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	tools := rpcv1.NewToolsServiceClient(conn)

	resp, err := tools.CallTool(asKey("op-secret"), &rpcv1.CallToolRequest{
		Name: "memory_save",
		ArgsJson: `{"memory_id":"o1","scope":"global",` +
			`"content":"The cache moved to node-b.","metadata":{"written_by":"hermes"}}`,
	})
	if err != nil {
		t.Fatalf("operator save: %v", err)
	}
	var saved cortexdb.MemorySaveResponse
	if err := json.Unmarshal([]byte(resp.GetResultJson()), &saved); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got := saved.Memory.Metadata["written_by"]; got != "operator" {
		t.Fatalf("written_by = %v, want the key operator, not the hermes it claimed", got)
	}
}

func TestASessionIDCannotTakeAUserConfinedKeyIntoAnotherUsersSession(t *testing.T) {
	conn := newPolicyConn(t, Options{Keys: hermesAndZeus(t)})
	client := rpcv1.NewMemoryServiceClient(conn)
	for _, m := range []*rpcv1.SaveMemoryRequest{
		{MemoryId: "zs-1", UserId: "zeus", SessionId: "zeus-sess", Content: "Zeus keeps the thunderbolt in the drawer."},
		{MemoryId: "g-1", Scope: "global", Content: "The shared thunderbolt policy."},
	} {
		if _, err := client.SaveMemory(asKey("op-secret"), m); err != nil {
			t.Fatalf("operator save %s: %v", m.MemoryId, err)
		}
	}

	for name, req := range map[string]*rpcv1.SearchMemoryRequest{
		"session_id":              {Query: "thunderbolt", UserId: "hermes", SessionId: "zeus-sess", TopK: 5},
		"plan.filters":            {Query: "thunderbolt", UserId: "hermes", TopK: 5, Plan: &rpcv1.RetrievalPlan{Query: "thunderbolt", Filters: &rpcv1.RetrievalFilters{UserId: "hermes", SessionId: "zeus-sess"}}},
		"global, owned by no one": {Query: "thunderbolt", UserId: "hermes", Scope: "global", TopK: 5},
	} {
		resp, err := client.SearchMemory(asKey("hermes-secret"), req)
		if err != nil {
			t.Fatalf("%s: search: %v", name, err)
		}
		if len(resp.GetResults()) != 0 {
			t.Fatalf("%s: hermes read %d rows of a bucket it does not own: %v", name, len(resp.GetResults()), resp.GetResults())
		}
	}

	_, err := client.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{
		MemoryId: "planted", UserId: "hermes", SessionId: "zeus-sess", Content: "Planted by hermes.",
	})
	if status.Code(err) != codes.NotFound {
		t.Fatalf("hermes wrote into zeus's session: %v", err)
	}

	// Its own session still works, or the confinement is useless.
	if _, err := client.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{
		MemoryId: "hs-1", UserId: "hermes", SessionId: "hermes-sess", Content: "Hermes tunes the lyre.",
	}); err != nil {
		t.Fatalf("hermes was refused its own session: %v", err)
	}
	resp, err := client.SearchMemory(asKey("hermes-secret"), &rpcv1.SearchMemoryRequest{
		Query: "lyre", UserId: "hermes", SessionId: "hermes-sess", TopK: 5,
	})
	if err != nil || len(resp.GetResults()) != 1 {
		t.Fatalf("hermes could not read its own session: %v %v", err, resp.GetResults())
	}
	// And the operator, unconfined, still reads every bucket.
	resp, err = client.SearchMemory(asKey("op-secret"), &rpcv1.SearchMemoryRequest{Query: "thunderbolt", SessionId: "zeus-sess", TopK: 5})
	if err != nil || len(resp.GetResults()) != 1 {
		t.Fatalf("operator lost zeus's session: %v %v", err, resp.GetResults())
	}
}
