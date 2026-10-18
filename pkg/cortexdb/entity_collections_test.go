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
