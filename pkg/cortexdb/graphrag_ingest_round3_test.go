package cortexdb

import (
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const threeChunks = "one two three four five six seven eight nine ten eleven twelve"

func chunkRows(t *testing.T, db *DB, docID string) int {
	t.Helper()
	var n int
	if err := db.SQL().QueryRow(`SELECT COUNT(*) FROM embeddings WHERE doc_id = ?`, docID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// "Unchanged" needs every chunk row, not only some: with one row of three
// gone, the same ingest rebuilds the chunks.
func TestUnchangedNeedsEveryChunkRow(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "rows")
	req := ToolIngestDocumentRequest{DocumentID: "d1", Content: threeChunks, ChunkSize: 4}
	if _, err := tools.IngestDocument(ctx, req); err != nil {
		t.Fatal(err)
	}
	if n := chunkRows(t, db, "d1"); n != 3 {
		t.Fatalf("%d chunk rows, want 3", n)
	}
	if _, err := db.SQL().Exec(`DELETE FROM embeddings WHERE id = (SELECT MAX(id) FROM embeddings WHERE doc_id = 'd1')`); err != nil {
		t.Fatal(err)
	}
	again, err := tools.IngestDocument(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if again.Unchanged {
		t.Fatal("unchanged was said with a chunk row missing")
	}
	if n := chunkRows(t, db, "d1"); n != 3 {
		t.Fatalf("%d chunk rows after the ingest, want 3", n)
	}
}

// The marks describe the chunks that are stored: knowledge_save and
// knowledge_update write the chunking they used and the hash of the content,
// on the record and on the document node.
func TestTheMarksDescribeTheStoredChunks(t *testing.T) {
	db, _, ctx := openIngestBrain(t, "marks-chunks")
	check := func(what, chunkSize, chunkOverlap, content string) {
		t.Helper()
		doc, err := db.store.GetDocument(ctx, "k1")
		if err != nil {
			t.Fatal(err)
		}
		if doc.Metadata[chunkSizeKey] != chunkSize || doc.Metadata[chunkOverlapKey] != chunkOverlap || doc.Metadata[contentHashKey] != contentSHA256(content) {
			t.Errorf("%s: the record's marks are %v, want chunk size %s, overlap %s and the hash of the content", what, doc.Metadata, chunkSize, chunkOverlap)
		}
		nodes, err := db.graph.GetNodesBatch(ctx, []string{graphDocumentNodeID("k1")})
		if err != nil || len(nodes) != 1 {
			t.Fatalf("%s: document node %v %v", what, nodes, err)
		}
		if nodes[0].Properties[contentHashKey] != contentSHA256(content) {
			t.Errorf("%s: the document node's hash is %v, want the content's", what, nodes[0].Properties[contentHashKey])
		}
	}
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "k1", Content: threeChunks, ChunkSize: 4}); err != nil {
		t.Fatal(err)
	}
	check("knowledge_save", "4", "0", threeChunks)
	if n := chunkRows(t, db, "k1"); n != 3 {
		t.Fatalf("%d chunk rows, want 3", n)
	}
	shorter, five := "one two three four five six", 5
	if _, err := db.UpdateKnowledge(ctx, KnowledgeUpdateRequest{KnowledgeID: "k1", Content: &shorter, ChunkSize: &five}); err != nil {
		t.Fatal(err)
	}
	check("knowledge_update", "5", "0", shorter)
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "k1", Content: threeChunks, ChunkSize: 5, ChunkOverlap: 1}); err != nil {
		t.Fatal(err)
	}
	check("knowledge_save with an overlap", "5", "1", threeChunks)
	two := 2
	if _, err := db.UpdateKnowledge(ctx, KnowledgeUpdateRequest{KnowledgeID: "k1", Content: &shorter, ChunkSize: &five, ChunkOverlap: &two}); err != nil {
		t.Fatal(err)
	}
	check("knowledge_update with an overlap", "5", "2", shorter)
}

// Another chunking of the same content is not "unchanged", even when it
// gives as many chunks: 12 words in chunks of 4 and in chunks of 5 with an
// overlap of 1 are three chunks both times, cut in other places.
func TestAnotherChunkingOfTheSameCountIsNotUnchanged(t *testing.T) {
	// Each case changes one chunking mark and keeps three chunks, so only
	// the mark that changed can tell the two ingests apart.
	for _, c := range []struct {
		name               string
		size, overlap      int
		againSize, againOv int
	}{
		{"size only", 4, 0, 5, 0},
		{"overlap only", 5, 0, 5, 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			db, tools, ctx := openIngestBrain(t, "same-count")
			if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: threeChunks, ChunkSize: c.size, ChunkOverlap: c.overlap}); err != nil {
				t.Fatal(err)
			}
			again, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: threeChunks, ChunkSize: c.againSize, ChunkOverlap: c.againOv})
			if err != nil {
				t.Fatal(err)
			}
			if again.Unchanged || len(again.ChunkNodeIDs) != 3 {
				t.Fatalf("unchanged %v with %d chunks, want a rewrite of 3 chunks", again.Unchanged, len(again.ChunkNodeIDs))
			}
			doc, err := db.store.GetDocument(ctx, "d1")
			if err != nil {
				t.Fatal(err)
			}
			if doc.Metadata[chunkSizeKey] != strconv.Itoa(c.againSize) || doc.Metadata[chunkOverlapKey] != strconv.Itoa(c.againOv) {
				t.Errorf("the marks say chunk size %v, overlap %v, want %d and %d", doc.Metadata[chunkSizeKey], doc.Metadata[chunkOverlapKey], c.againSize, c.againOv)
			}
		})
	}
}

// Dedupe names the smallest id among the holders of the collection, so the
// answer does not depend on the order the rows come in.
func TestDedupeNamesTheSmallestHolder(t *testing.T) {
	_, tools, ctx := openIngestBrain(t, "smallest")
	const content = "Two copies of the same page."
	for _, id := range []string{"m-b", "m-a", "m-c"} {
		if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: id, Content: content}); err != nil {
			t.Fatal(err)
		}
	}
	dup, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "z", Content: content, Dedupe: true})
	if err != nil || dup.DuplicateOf != "m-a" {
		t.Fatalf("duplicate_of %q (%v), want m-a", dup.DuplicateOf, err)
	}
	if body, _ := json.Marshal(dup); !strings.Contains(string(body), `"chunk_node_ids":[]`) {
		t.Errorf("a duplicate stores no chunk and says so with an empty list: %s", body)
	}
}

// A chunk retired by a shorter re-ingest goes to history as a re-ingest, not
// as a document cleanup.
func TestARetiredChunkIsRecordedAsAReingest(t *testing.T) {
	_, tools, ctx := openIngestBrain(t, "producer")
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: threeChunks, ChunkSize: 4}); err != nil {
		t.Fatal(err)
	}
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: "one two three four", ChunkSize: 4}); err != nil {
		t.Fatal(err)
	}
	out, err := tools.Call(ctx, "changes_since", json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(out)
	var feed struct {
		Events []struct {
			ID       string `json:"id"`
			Op       string `json:"op"`
			Producer string `json:"producer"`
		} `json:"events"`
	}
	if err := json.Unmarshal(raw, &feed); err != nil {
		t.Fatal(err)
	}
	retired := graphChunkNodeID("d1", 2)
	seen := 0
	for _, ev := range feed.Events {
		if ev.ID != retired || ev.Op == "insert" {
			continue
		}
		seen++
		if ev.Producer != producerDocumentReingest {
			t.Errorf("the retired chunk %s is recorded by %s, want %s", retired, ev.Producer, producerDocumentReingest)
		}
	}
	if seen == 0 {
		t.Fatalf("the retired chunk %s is not in history", retired)
	}
}

// EngineDocumentKeys names every key the engine writes on a document record
// from its content, its chunking and its fields (the expiry here), so a port that hands them
// to the facade hands all of them and no more.
func TestEngineDocumentKeysNameEveryKeyTheEngineWrites(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "engine-keys")
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d1", Content: threeChunks, Metadata: map[string]string{"team": "a"}, ExpiresAt: strPtr("2031-10-18T00:00:00Z")}); err != nil {
		t.Fatal(err)
	}
	doc, err := db.store.GetDocument(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	var written []string
	for k := range doc.Metadata {
		if strings.HasPrefix(k, "_") {
			written = append(written, k)
		}
	}
	slices.Sort(written)
	want := slices.Sorted(slices.Values(EngineDocumentKeys()))
	if !slices.Equal(written, want) {
		t.Fatalf("the engine wrote %v, EngineDocumentKeys names %v", written, want)
	}
}
