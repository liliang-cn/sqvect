package cortexdb

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// findNodeIDs is the ids find_nodes resolved one name to, in its order, and
// the strength of its answer.
func findNodeIDs(t *testing.T, tools *GraphRAGToolbox, ctx context.Context, name string, limit int, nodeTypes ...string) ([]string, string) {
	t.Helper()
	resp, err := tools.FindNodes(ctx, ToolFindNodesRequest{Names: []string{name}, Limit: limit, NodeTypes: nodeTypes})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Matches) != 1 {
		t.Fatalf("find_nodes %q: %d matches, want 1", name, len(resp.Matches))
	}
	ids := make([]string, 0, len(resp.Matches[0].Nodes))
	for _, node := range resp.Matches[0].Nodes {
		ids = append(ids, node.ID)
	}
	return ids, resp.Matches[0].Match
}

// find_nodes searches the content of every node, the text of chunks
// included: the chunk and document nodes of an expired document are left
// out, as get_nodes leaves them out, and the live document's stay.
func TestFindNodesLeavesOutTheNodesOfAnExpiredDocument(t *testing.T) {
	db := openReviewBrain(t, "find-expired")
	ctx := context.Background()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "old", Content: "Zephyr Gate opened EXPIREDTEXT at dawn.", ExpiresAt: strPtr(pastInstant())}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "live", Content: "Zephyr Gate closes at dusk."}); err != nil {
		t.Fatal(err)
	}
	resp, err := db.GraphRAGTools().FindNodes(ctx, ToolFindNodesRequest{Names: []string{"Zephyr Gate"}})
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(resp)
	for _, leak := range []string{"EXPIREDTEXT", graphChunkNodeID("old", 0), graphDocumentNodeID("old")} {
		if strings.Contains(string(body), leak) {
			t.Errorf("find_nodes served the expired document (%s): %.300s", leak, body)
		}
	}
	if !strings.Contains(string(body), graphChunkNodeID("live", 0)) {
		t.Errorf("find_nodes lost the live document: %.300s", body)
	}
}

// Under the review gate, a proposed document and a proposed memory are left
// out of find_nodes, as they are left out of get_nodes and the walks; the
// unreviewed ones stay.
func TestFindNodesLeavesOutAProposedDocumentUnderTheReviewGate(t *testing.T) {
	db := openReviewBrain(t, "find-review", WithReviewGate())
	ctx := context.Background()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "draft", Content: "Zephyr Gate opened PROPOSEDTEXT at dawn.", ReviewStatus: ReviewProposed}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "kept", Content: "Zephyr Gate closes at dusk."}); err != nil {
		t.Fatal(err)
	}
	mention := []ToolEntityInput{{Name: "Zephyr Gate"}}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "m-draft", UserID: "u", Content: "Zephyr Gate PROPOSEDMEMORY note.", Entities: mention, ReviewStatus: ReviewProposed}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "m-kept", UserID: "u", Content: "Zephyr Gate kept note.", Entities: mention}); err != nil {
		t.Fatal(err)
	}
	resp, err := db.GraphRAGTools().FindNodes(ctx, ToolFindNodesRequest{Names: []string{"Zephyr Gate"}, Limit: 20})
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(resp)
	for _, leak := range []string{"PROPOSEDTEXT", "PROPOSEDMEMORY", graphChunkNodeID("draft", 0), memoryGraphNodeID("m-draft")} {
		if strings.Contains(string(body), leak) {
			t.Errorf("find_nodes served what the review gate hides (%s): %.300s", leak, body)
		}
	}
	for _, want := range []string{graphChunkNodeID("kept", 0), memoryGraphNodeID("m-kept")} {
		if !strings.Contains(string(body), want) {
			t.Errorf("find_nodes lost %s: %.300s", want, body)
		}
	}
}

// An entity is a fact between documents, not a passage: one that only a
// hidden chunk mentions is still found, as get_nodes still serves it.
func TestFindNodesStillFindsAnEntityThatAHiddenChunkMentions(t *testing.T) {
	db := openReviewBrain(t, "find-entity")
	ctx := context.Background()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{
		KnowledgeID: "old", Content: "Zephyr Gate opened at dawn.", ExpiresAt: strPtr(pastInstant()),
		Entities: []ToolEntityInput{{Name: "Zephyr Gate", ChunkIDs: []string{graphChunkNodeID("old", 0)}}},
	}); err != nil {
		t.Fatal(err)
	}
	ids, match := findNodeIDs(t, db.GraphRAGTools(), ctx, "Zephyr Gate", 0)
	if strings.Join(ids, ",") != EntityNodeID("Zephyr Gate") || match != "exact" {
		t.Errorf("find_nodes = %v (%s), want the entity alone, exact", ids, match)
	}
}

// The limit counts the nodes served: hidden matches ranked first do not use
// it up, and the strength of the answer is that of the first node served.
func TestFindNodesFillsItsLimitPastHiddenMatches(t *testing.T) {
	db := openReviewBrain(t, "find-limit")
	ctx := context.Background()
	// Among the chunks, ranked: old1 (exact, hidden), then by length the
	// ones that contain the name, live1, live2, live3, then the longer
	// old2 (hidden) and live4.
	docs := []struct{ id, content string }{
		{"old1", "Zephyr Gate"},
		{"live1", "Zephyr Gate shuts"},
		{"live2", "Zephyr Gate closes at dusk."},
		{"live3", "Zephyr Gate closes at dusk again."},
		{"old2", "Zephyr Gate opened at dawn, long before the dusk."},
		{"live4", "Zephyr Gate shuts at night, long after the dusk."},
	}
	for _, d := range docs {
		req := KnowledgeSaveRequest{KnowledgeID: d.id, Content: d.content}
		if strings.HasPrefix(d.id, "old") {
			req.ExpiresAt = strPtr(pastInstant())
		}
		if _, err := db.SaveKnowledge(ctx, req); err != nil {
			t.Fatal(err)
		}
	}
	chunks := func(ids ...string) string {
		out := make([]string, 0, len(ids))
		for _, id := range ids {
			out = append(out, graphChunkNodeID(id, 0))
		}
		return strings.Join(out, ",")
	}
	for limit, want := range map[int]string{1: chunks("live1"), 2: chunks("live1", "live2"), 3: chunks("live1", "live2", "live3")} {
		ids, match := findNodeIDs(t, db.GraphRAGTools(), ctx, "Zephyr Gate", limit, "chunk")
		if strings.Join(ids, ",") != want || match != "contains" {
			t.Errorf("find_nodes limit %d = %v (%s), want %s, contains", limit, ids, match, want)
		}
	}
	// The strength is that of the first node served, not of the last.
	ids, match := findNodeIDs(t, db.GraphRAGTools(), ctx, "Zephyr Gate shuts", 0, "chunk")
	if strings.Join(ids, ",") != chunks("live1", "live4") || match != "exact" {
		t.Errorf("find_nodes = %v (%s), want live1 then live4, exact", ids, match)
	}
	// Every match hidden: no node and no strength, as for a name nothing
	// matches.
	ids, match = findNodeIDs(t, db.GraphRAGTools(), ctx, "Zephyr Gate opened", 0, "chunk", "document")
	if len(ids) != 0 || match != "" {
		t.Errorf("find_nodes on hidden matches only = %v (%q), want none", ids, match)
	}
}

// Each window starts where the last one ended: a node served at the end of
// one window is not served again by the next.
func TestFindNodesServesEachNodeOnceAcrossWindows(t *testing.T) {
	db := openReviewBrain(t, "find-windows")
	ctx := context.Background()
	// Ranked: old1 (exact, hidden), then by length live1, old2 (hidden),
	// live2, live3. With limit 3 the first window holds two hidden matches,
	// so a second window is read, and live2 lies on the window boundary.
	docs := []struct{ id, content string }{
		{"old1", "Yarrow Bend"},
		{"live1", "Yarrow Bend a"},
		{"old2", "Yarrow Bend bb"},
		{"live2", "Yarrow Bend ccc"},
		{"live3", "Yarrow Bend dddd"},
	}
	for _, d := range docs {
		req := KnowledgeSaveRequest{KnowledgeID: d.id, Content: d.content}
		if strings.HasPrefix(d.id, "old") {
			req.ExpiresAt = strPtr(pastInstant())
		}
		if _, err := db.SaveKnowledge(ctx, req); err != nil {
			t.Fatal(err)
		}
	}
	want := strings.Join([]string{graphChunkNodeID("live1", 0), graphChunkNodeID("live2", 0), graphChunkNodeID("live3", 0)}, ",")
	if ids, _ := findNodeIDs(t, db.GraphRAGTools(), ctx, "Yarrow Bend", 3, "chunk"); strings.Join(ids, ",") != want {
		t.Errorf("find_nodes limit 3 = %v, want %s", ids, want)
	}
}

// As many windows are read as hidden matches fill: with limit 2 and four
// hidden matches ranked first, the third window serves the live ones.
func TestFindNodesReadsAsManyWindowsAsHiddenMatchesFill(t *testing.T) {
	db := openReviewBrain(t, "find-many-windows")
	ctx := context.Background()
	docs := []struct{ id, content string }{
		{"old1", "Quill Ford"},
		{"old2", "Quill Ford a"},
		{"old3", "Quill Ford bb"},
		{"old4", "Quill Ford ccc"},
		{"live1", "Quill Ford dddd"},
		{"live2", "Quill Ford eeeee"},
	}
	for _, d := range docs {
		req := KnowledgeSaveRequest{KnowledgeID: d.id, Content: d.content}
		if strings.HasPrefix(d.id, "old") {
			req.ExpiresAt = strPtr(pastInstant())
		}
		if _, err := db.SaveKnowledge(ctx, req); err != nil {
			t.Fatal(err)
		}
	}
	want := graphChunkNodeID("live1", 0) + "," + graphChunkNodeID("live2", 0)
	if ids, match := findNodeIDs(t, db.GraphRAGTools(), ctx, "Quill Ford", 2, "chunk"); strings.Join(ids, ",") != want || match != "contains" {
		t.Errorf("find_nodes limit 2 = %v (%s), want %s, contains", ids, match, want)
	}
}

// An exact match served first and a fold match served after it: the
// answer is as strong as the first, not the last.
func TestFindNodesReportsTheStrengthOfTheFirstNodeServed(t *testing.T) {
	db := openReviewBrain(t, "find-strength")
	ctx := context.Background()
	for _, d := range []struct{ id, content string }{{"exact", "Sorrel Ford"}, {"fold", "sorrel-ford"}} {
		if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: d.id, Content: d.content}); err != nil {
			t.Fatal(err)
		}
	}
	ids, match := findNodeIDs(t, db.GraphRAGTools(), ctx, "Sorrel Ford", 2, "chunk")
	if want := graphChunkNodeID("exact", 0) + "," + graphChunkNodeID("fold", 0); strings.Join(ids, ",") != want || match != "exact" {
		t.Errorf("find_nodes = %v %s, want %s exact", ids, match, want)
	}
}
