package rpcserver

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// The facade signs SaveKnowledge, UpdateKnowledge, SaveMemory, UpdateMemory
// and InsertGraphDocument itself (cortexdb.WithWriter): whatever a caller
// sends under written_by or updated_by is replaced. So these five typed
// writes take the two keys from a confined key, and a row read back and sent
// again is not refused for the signature it carries. A key the facade does
// not replace (another spelling) is still refused.
func TestATypedWriteTakesTheSignatureTheFacadeReplaces(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	conn := newPolicyConn(t, Options{Keys: ks})
	knowledge := rpcv1.NewKnowledgeServiceClient(conn)
	memory := rpcv1.NewMemoryServiceClient(conn)
	graphrag := rpcv1.NewGraphRagServiceClient(conn)
	forged := map[string]string{"written_by": "operator", "updated_by": "operator"}

	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n1", Collection: "notes", Content: "Orpheus plays.", Metadata: forged}); err != nil {
		t.Fatalf("SaveKnowledge with a signature: %v", err)
	}
	if _, err := knowledge.UpdateKnowledge(asKey("notes-secret"), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "n1", Collection: proto.String("notes"), Content: proto.String("Orpheus plays on."), Metadata: forged}); err != nil {
		t.Fatalf("UpdateKnowledge with a signature: %v", err)
	}
	got, err := knowledge.GetKnowledge(asKey("op-secret"), &rpcv1.GetKnowledgeRequest{KnowledgeId: "n1"})
	if err != nil {
		t.Fatal(err)
	}
	if m := got.GetKnowledge().GetMetadata(); m["written_by"] != "notes" || m["updated_by"] != "notes" {
		t.Errorf("the document is signed %q / %q, want notes / notes", m["written_by"], m["updated_by"])
	}

	sig, err := structpb.NewStruct(map[string]any{"written_by": "operator", "updated_by": "operator"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := memory.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "hermes-1", UserId: "hermes", Scope: "user", Content: "Hermes tunes the lyre.", Metadata: sig}); err != nil {
		t.Fatalf("SaveMemory with a signature: %v", err)
	}
	// Read the row back and send its metadata again: the round trip of a
	// client that edits what it read.
	row, err := memory.GetMemory(asKey("hermes-secret"), &rpcv1.GetMemoryRequest{MemoryId: "hermes-1"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := memory.UpdateMemory(asKey("hermes-secret"), &rpcv1.UpdateMemoryRequest{MemoryId: "hermes-1", Content: proto.String("Hermes tunes the lyre twice."), Metadata: row.GetMemory().GetMetadata()}); err != nil {
		t.Fatalf("UpdateMemory with the metadata it read: %v", err)
	}
	row, err = memory.GetMemory(asKey("op-secret"), &rpcv1.GetMemoryRequest{MemoryId: "hermes-1"})
	if err != nil {
		t.Fatal(err)
	}
	if m := row.GetMemory().GetMetadata().AsMap(); m["written_by"] != "hermes" || m["updated_by"] != "hermes" {
		t.Errorf("the memory is signed %v / %v, want hermes / hermes", m["written_by"], m["updated_by"])
	}

	// InsertGraphDocument is signed by the facade too, so it takes both
	// keys. This server has no embedder: the call fails after the
	// authorization, and only a refusal fails here.
	if _, err := graphrag.InsertGraphDocument(asKey("notes-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "n2", Collection: "notes", Content: "x", Metadata: forged}); status.Code(err) == codes.PermissionDenied {
		t.Errorf("InsertGraphDocument refused the signature the facade replaces: %v", err)
	}

	for _, c := range []struct {
		what string
		err  error
		want string
	}{
		{"InsertGraphDocument, another spelling", func() error {
			_, err := graphrag.InsertGraphDocument(asKey("notes-secret"), &rpcv1.InsertGraphDocumentRequest{Id: "n2", Collection: "notes", Content: "x", Metadata: map[string]string{"Written_By": "operator"}})
			return err
		}(), "metadata.written_by"},
		{"SaveKnowledge, another spelling", func() error {
			_, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "n3", Collection: "notes", Content: "x", Metadata: map[string]string{"Written_By": "operator"}})
			return err
		}(), "metadata.written_by"},
	} {
		if status.Code(c.err) != codes.PermissionDenied || !strings.Contains(status.Convert(c.err).Message(), c.want) {
			t.Errorf("%s: %v, want PERMISSION_DENIED naming %s", c.what, c.err, c.want)
		}
	}
}

// A typed UpdateKnowledge from a confined key answers a missing id as it
// answers a foreign one, in main's words for a withheld document, so the
// typed RPC cannot tell the two apart where CallTool does not. The operator
// keeps the facade's wording.
func TestATypedUpdateKnowledgeAnswersAMissingIdAsAForeignOne(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	knowledge := rpcv1.NewKnowledgeServiceClient(newPolicyConn(t, Options{Keys: ks}))
	if _, err := knowledge.SaveKnowledge(asKey("op-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "sec", Collection: "secrets", Content: "The vault code."}); err != nil {
		t.Fatal(err)
	}
	update := func(secret, id string) error {
		_, err := knowledge.UpdateKnowledge(asKey(secret), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: id, Collection: proto.String("notes"), Content: proto.String("overwritten")})
		return err
	}
	for _, id := range []string{"sec", "no-such-doc"} {
		err := update("notes-secret", id)
		if want := fmt.Sprintf("knowledge %q not found", id); status.Code(err) != codes.NotFound || status.Convert(err).Message() != want {
			t.Errorf("UpdateKnowledge %s: %v, want NOT_FOUND %q", id, err, want)
		}
	}
	err = update("op-secret", "no-such-doc")
	if status.Code(err) != codes.NotFound || status.Convert(err).Message() == `knowledge "no-such-doc" not found` {
		t.Errorf("operator updates a missing document: %v, want the facade's own wording", err)
	}
	// Only NOT_FOUND is reworded: another refusal of the facade keeps its
	// code and its words for a confined key, as the operator gets them.
	if _, err := knowledge.SaveKnowledge(asKey("notes-secret"), &rpcv1.SaveKnowledgeRequest{KnowledgeId: "own", Collection: "notes", Content: "Orpheus plays."}); err != nil {
		t.Fatal(err)
	}
	empty := func(secret string) error {
		_, err := knowledge.UpdateKnowledge(asKey(secret), &rpcv1.UpdateKnowledgeRequest{KnowledgeId: "own", Collection: proto.String("notes"), Content: proto.String("")})
		return err
	}
	confined, own := empty("notes-secret"), empty("op-secret")
	if code := status.Code(confined); code == codes.OK || code == codes.NotFound || code != status.Code(own) || status.Convert(confined).Message() != status.Convert(own).Message() {
		t.Errorf("an empty update of its own document: %v, want the facade's own refusal: %v", confined, own)
	}
}

// The typed memory RPCs hold every memory field of a confinement against the
// stored row, through the rule the policy exports: a key confined to a
// namespace reaches only rows of that namespace, and a key confined to a
// memory scope only rows of that scope. Another row is NOT_FOUND, as a
// missing one is.
func TestTheTypedMemoryRowChecksHoldEveryMemoryField(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "lab", Secret: "lab-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Namespace: "lab"}},
		{ID: "mine", Secret: "mine-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{MemoryScope: "user"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	memory := rpcv1.NewMemoryServiceClient(newPolicyConn(t, Options{Keys: ks}))
	for _, m := range []*rpcv1.SaveMemoryRequest{
		{MemoryId: "lab-1", UserId: "hermes", Scope: "user", Namespace: "lab", Content: "In the lab."},
		{MemoryId: "kitchen-1", UserId: "hermes", Scope: "user", Namespace: "kitchen", Content: "In the kitchen."},
		{MemoryId: "global-1", Scope: "global", Content: "For everyone."},
	} {
		if _, err := memory.SaveMemory(asKey("op-secret"), m); err != nil {
			t.Fatalf("operator saves %s: %v", m.GetMemoryId(), err)
		}
	}
	for _, c := range []struct {
		secret, id string
		reach      bool
	}{
		{"lab-secret", "lab-1", true},
		{"lab-secret", "kitchen-1", false},
		{"mine-secret", "lab-1", true},
		{"mine-secret", "global-1", false},
	} {
		_, gerr := memory.GetMemory(asKey(c.secret), &rpcv1.GetMemoryRequest{MemoryId: c.id})
		_, uerr := memory.UpdateMemory(asKey(c.secret), &rpcv1.UpdateMemoryRequest{MemoryId: c.id, Content: proto.String("changed by " + c.secret)})
		for what, err := range map[string]error{"GetMemory": gerr, "UpdateMemory": uerr} {
			if c.reach && err != nil {
				t.Errorf("%s %s under %s: %v, want its own row", what, c.id, c.secret, err)
			}
			if !c.reach && status.Code(err) != codes.NotFound {
				t.Errorf("%s %s under %s: %v, want NOT_FOUND", what, c.id, c.secret, err)
			}
		}
	}
}

// A key confined to a collection reaches no memory row, so a memory write
// from it is answered by the row checks, as on main: UpdateMemory is
// NOT_FOUND like GetMemory and DeleteMemory, and SaveMemory is the
// interceptor's refusal. The allowlist line of the tool is not consulted:
// it would name a gate that memory_update does not have.
func TestAMemoryWriteFromACollectionKeyIsAnsweredByTheRowChecks(t *testing.T) {
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "hermes", Secret: "hermes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{UserID: "hermes"}},
		{ID: "notes", Secret: "notes-secret", Clearance: authz.ReadWrite, Scope: authz.Scope{Collection: "notes"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	memory := rpcv1.NewMemoryServiceClient(newPolicyConn(t, Options{Keys: ks}))
	if _, err := memory.SaveMemory(asKey("hermes-secret"), &rpcv1.SaveMemoryRequest{MemoryId: "hermes-1", UserId: "hermes", Scope: "user", Content: "Hermes tunes the lyre."}); err != nil {
		t.Fatal(err)
	}
	meta, err := structpb.NewStruct(map[string]any{"mood": "calm"})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"hermes-1", "no-such-memory"} {
		_, uerr := memory.UpdateMemory(asKey("notes-secret"), &rpcv1.UpdateMemoryRequest{MemoryId: id, Content: proto.String("overwritten"), Metadata: meta})
		_, gerr := memory.GetMemory(asKey("notes-secret"), &rpcv1.GetMemoryRequest{MemoryId: id})
		if status.Code(uerr) != codes.NotFound || status.Convert(uerr).Message() != status.Convert(gerr).Message() {
			t.Errorf("UpdateMemory %s: %v, want what GetMemory answers: %v", id, uerr, gerr)
		}
	}
	req := &rpcv1.SaveMemoryRequest{MemoryId: "notes-1", UserId: "hermes", Scope: "user", Content: "x", Metadata: meta}
	notes, _ := ks.Lookup("notes-secret")
	want := notes.AuthorizeRows("/cortexdb.v1.MemoryService/SaveMemory", requestFieldLookup(req))
	_, err = memory.SaveMemory(asKey("notes-secret"), req)
	if want == nil || status.Code(err) != codes.PermissionDenied || status.Convert(err).Message() != want.Error() {
		t.Errorf("SaveMemory: %v, want the interceptor's refusal %v", err, want)
	}
}
