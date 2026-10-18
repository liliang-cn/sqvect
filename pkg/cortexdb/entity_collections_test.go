package cortexdb

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// Two collections on one server share the entity "Acme": each has a document
// that names it. A search confined to one collection must not bring back the
// other collection's chunks through that shared entity, in any retrieval
// mode that walks the graph.
func TestASearchInOneCollectionDoesNotReachAnotherThroughASharedEntity(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "entity-collections.db")))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	c := context.Background()
	save := func(id, collection, content string) {
		_, err := db.SaveKnowledge(c, KnowledgeSaveRequest{
			KnowledgeID: id, Collection: collection, Content: content,
			// The chunk mentions the entity, so the graph can walk from the
			// chunk to Acme and from Acme to every other chunk that names it.
			Entities: []ToolEntityInput{{Name: "Acme", Type: "organization", ChunkIDs: []string{graphChunkNodeID(id, 0)}}},
		})
		if err != nil {
			t.Fatalf("save %s: %v", id, err)
		}
	}
	save("notes-1", "notes", "Acme asked for a quote on the ledger migration.")
	save("secret-1", "secrets", "Acme pays a hidden retainer of forty thousand.")

	// One node for both, which records the two collections that mention it.
	node, err := db.graph.GetNode(c, graphEntityNodeID("Acme"))
	if err != nil {
		t.Fatalf("entity node: %v", err)
	}
	if got := strings.Join(toStringSlice(node.Properties[entityCollectionsKey]), ","); got != "notes,secrets" && got != "secrets,notes" {
		t.Fatalf("entity collections %q, want notes and secrets", got)
	}
	if !entityKnownIn([]string{"notes"}, "notes") || entityKnownIn([]string{"notes"}, "secrets") ||
		!entityKnownIn(nil, "secrets") || !entityKnownIn([]string{"notes"}, "") {
		t.Fatal("entityKnownIn disagrees with the rule")
	}

	for _, mode := range []string{RetrievalModeLexical, RetrievalModeGraph, RetrievalModeAuto, RetrievalModePPR} {
		resp, err := db.SearchKnowledge(c, KnowledgeSearchRequest{
			Query: "Acme retainer quote", Collection: "notes", TopK: 10, MaxHops: 3,
			RetrievalMode: mode, EntityNames: []string{"Acme"},
		})
		if err != nil {
			t.Fatalf("%s: search: %v", mode, err)
		}
		ids := make([]string, 0, len(resp.Results))
		for _, h := range resp.Results {
			ids = append(ids, h.KnowledgeID)
		}
		for _, ch := range resp.Chunks {
			ids = append(ids, ch.DocumentID)
		}
		got := strings.Join(ids, ",")
		if strings.Contains(got, "secret") {
			t.Errorf("%s: a search in notes reached secrets: %s", mode, got)
		}
		if !strings.Contains(got, "notes-1") {
			t.Errorf("%s: a search in notes missed its own document: %s", mode, got)
		}
	}
}

// The entity rule on its own. An entity that only `secrets` ever saved gets a
// mention edge from a `notes` chunk by hand, through upsert_entities, which
// has no collection and so leaves the entity's list alone. The walk from the
// notes chunk reaches that entity and must stop there: `notes` does not know
// it, so nothing it mentions in `secrets` comes back.
func TestTheWalkDoesNotPassThroughAnEntityTheCollectionDoesNotKnow(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "entity-unknown.db")))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	c := context.Background()

	if _, err := db.SaveKnowledge(c, KnowledgeSaveRequest{
		KnowledgeID: "secret-1", Collection: "secrets", Content: "Globex pays a hidden retainer of forty thousand.",
		Entities: []ToolEntityInput{{Name: "Globex", Type: "organization", ChunkIDs: []string{graphChunkNodeID("secret-1", 0)}}},
	}); err != nil {
		t.Fatalf("save secret-1: %v", err)
	}
	if _, err := db.SaveKnowledge(c, KnowledgeSaveRequest{
		KnowledgeID: "notes-1", Collection: "notes", Content: "The ledger migration quote is due on Friday.",
	}); err != nil {
		t.Fatalf("save notes-1: %v", err)
	}
	// The notes chunk never names Globex, so extraction adds nothing. The
	// hand-made mention has no collection, so Globex stays known to secrets only.
	if _, err := db.GraphRAGTools().UpsertEntities(c, ToolUpsertEntitiesRequest{
		Entities: []ToolEntityInput{{Name: "Globex", Type: "organization", ChunkIDs: []string{graphChunkNodeID("notes-1", 0)}}},
	}); err != nil {
		t.Fatalf("upsert_entities: %v", err)
	}
	node, err := db.graph.GetNode(c, graphEntityNodeID("Globex"))
	if err != nil {
		t.Fatalf("entity node: %v", err)
	}
	if got := strings.Join(toStringSlice(node.Properties[entityCollectionsKey]), ","); got != "secrets" {
		t.Fatalf("entity collections %q, want secrets only", got)
	}

	resp, err := db.SearchKnowledge(c, KnowledgeSearchRequest{
		Query: "Globex retainer quote", Collection: "notes", TopK: 10, MaxHops: 3,
		RetrievalMode: RetrievalModeGraph, EntityNames: []string{"Globex"},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	ids := make([]string, 0, len(resp.Results)+len(resp.Chunks))
	for _, h := range resp.Results {
		ids = append(ids, h.KnowledgeID)
	}
	for _, ch := range resp.Chunks {
		ids = append(ids, ch.DocumentID)
	}
	got := strings.Join(ids, ",")
	if strings.Contains(got, "secret") {
		t.Fatalf("the walk passed through an entity notes does not know: %s", got)
	}
	if !strings.Contains(got, "notes-1") {
		t.Fatalf("the collection's own chunk is missing: %s", got)
	}
}
