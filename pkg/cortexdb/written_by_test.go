package cortexdb

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestSavesRecordTheirWriterAndAClientCannotClaimAnother(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "writer.db")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	hermes := WithWriter(context.Background(), "hermes")
	zeus := WithWriter(context.Background(), "zeus")

	saved, err := db.SaveMemory(hermes, MemorySaveRequest{
		MemoryID: "m1",
		Scope:    "global",
		Content:  "the gateway fronts the store",
		Metadata: map[string]any{metadataWrittenBy: "zeus", metadataUpdatedBy: "zeus"},
	})
	if err != nil {
		t.Fatalf("save memory: %v", err)
	}
	if got := saved.Memory.Metadata[metadataWrittenBy]; got != "hermes" {
		t.Fatalf("written_by = %v, want hermes", got)
	}
	if _, ok := saved.Memory.Metadata[metadataUpdatedBy]; ok {
		t.Fatalf("a save carried the updated_by its caller sent: %v", saved.Memory.Metadata)
	}

	content := "the gateway fronts the store and the cache"
	updated, err := db.UpdateMemory(zeus, MemoryUpdateRequest{
		MemoryID: "m1",
		Content:  &content,
		Metadata: map[string]any{metadataWrittenBy: "zeus"},
	})
	if err != nil {
		t.Fatalf("update memory: %v", err)
	}
	if got := updated.Memory.Metadata[metadataWrittenBy]; got != "hermes" {
		t.Fatalf("after update written_by = %v, want hermes kept", got)
	}
	if got := updated.Memory.Metadata[metadataUpdatedBy]; got != "zeus" {
		t.Fatalf("after update updated_by = %v, want zeus", got)
	}

	if _, err := db.SaveKnowledge(hermes, KnowledgeSaveRequest{
		KnowledgeID: "k1",
		Title:       "Gateway",
		Content:     "The gateway fronts the store.",
	}); err != nil {
		t.Fatalf("save knowledge: %v", err)
	}
	title := "Gateway, revised"
	if _, err := db.UpdateKnowledge(zeus, KnowledgeUpdateRequest{
		KnowledgeID: "k1",
		Title:       &title,
		Metadata:    map[string]string{metadataWrittenBy: "zeus"},
	}); err != nil {
		t.Fatalf("update knowledge: %v", err)
	}
	got, err := db.GetKnowledge(context.Background(), KnowledgeGetRequest{KnowledgeID: "k1"})
	if err != nil {
		t.Fatalf("get knowledge: %v", err)
	}
	if w, u := got.Knowledge.Metadata[metadataWrittenBy], got.Knowledge.Metadata[metadataUpdatedBy]; w != "hermes" || u != "zeus" {
		t.Fatalf("knowledge written_by=%v updated_by=%v, want hermes and zeus", w, u)
	}
}

func TestWithoutAWriterMetadataIsStoredAsGiven(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "nowriter.db")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()

	saved, err := db.SaveMemory(context.Background(), MemorySaveRequest{
		MemoryID: "m1",
		Scope:    "global",
		Content:  "imported from an export",
		Metadata: map[string]any{metadataWrittenBy: "hermes"},
	})
	if err != nil {
		t.Fatalf("save: %v", err)
	}
	if got := saved.Memory.Metadata[metadataWrittenBy]; got != "hermes" {
		t.Fatalf("written_by = %v, want the given hermes kept", got)
	}
}

func TestAMemoryHitSaysWhichRetrieverFoundItAndHowCloseItIs(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "via.db")), WithEmbedder(viaEmbedder{}))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()
	for id, content := range map[string]string{
		"close": "alpha bravo",
		"word":  "charlie zulu",
	} {
		if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: id, Scope: "global", Content: content}); err != nil {
			t.Fatalf("save %s: %v", id, err)
		}
	}

	resp, err := db.SearchMemory(ctx, MemorySearchRequest{Query: "alpha zulu", Scope: "global", TopK: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	byID := map[string]MemorySearchHit{}
	for _, hit := range resp.Results {
		byID[hit.Memory.ID] = hit
	}
	close, word := byID["close"], byID["word"]
	if !slices.Contains(close.Via, memoryViaSemantic) || close.Similarity <= 0 {
		t.Fatalf("close hit via=%v similarity=%v, want semantic with a similarity", close.Via, close.Similarity)
	}
	if close.Similarity == close.Score {
		t.Fatalf("similarity %v equals the boosted score; it must be the raw one", close.Similarity)
	}
	if !slices.Contains(word.Via, memoryViaKeyword) {
		t.Fatalf("word hit via=%v, want keyword", word.Via)
	}
}

// viaEmbedder puts texts that mention alpha near each other and everything
// else far away, so one memory is close in meaning and the other only shares
// a word.
type viaEmbedder struct{}

func (viaEmbedder) Dim() int { return 2 }

func (viaEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if strings.Contains(text, "alpha") {
		return []float32{1, 0}, nil
	}
	return []float32{0, 1}, nil
}

func (e viaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		out[i], _ = e.Embed(ctx, text)
	}
	return out, nil
}
