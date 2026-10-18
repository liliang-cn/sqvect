package cortexdb

import "github.com/liliang-cn/cortexdb/v2/pkg/graph"

// Entity nodes and collections.
//
// An entity node's id is its normalised name (graphEntityNodeID), so "Acme"
// in one collection and "Acme" in another are one node, with the mention
// edges and relations of both. That is what makes a glossary work across a
// whole brain, and what let a graph walk confined to one collection reach the
// other collection's chunks through the shared node. The node now records
// which collections mention it, in the property below, and the walk keeps to
// entities the searched collection knows (entityKnownIn).
//
// The list is a union that grows with every ingest; delete_document_graph
// detaches the entity from the document but does not shrink the list. An
// entity with no list — written before this property existed, or by a tool
// that has no collection such as upsert_entities — is known everywhere.

// entityCollectionsKey is the entity node property that lists the
// collections whose chunks mention the entity.
const entityCollectionsKey = "collections"

// mergeEntityCollections adds collection to each node's list, keeping what
// prior writes recorded. Mirrors mergeEntitySourceDocuments.
func mergeEntityCollections(nodes []*graph.GraphNode, prior map[string]*graph.GraphNode, collection string) {
	if collection == "" {
		return
	}
	for _, node := range nodes {
		if node.Properties == nil {
			node.Properties = map[string]interface{}{}
		}
		var before []string
		if was, ok := prior[node.ID]; ok && was.Properties != nil {
			before = toStringSlice(was.Properties[entityCollectionsKey])
		}
		node.Properties[entityCollectionsKey] = unionStrings(before, []string{collection})
	}
}

// entityKnownIn reports whether an entity with these collections may be
// walked through by a search confined to collection. No confinement, or an
// entity with no list, is a yes.
func entityKnownIn(collections []string, collection string) bool {
	if collection == "" || len(collections) == 0 {
		return true
	}
	for _, c := range collections {
		if c == collection {
			return true
		}
	}
	return false
}
