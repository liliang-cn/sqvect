package cortexdb

import (
	"context"
	"path/filepath"
	"testing"
)

// openSQLiteLexical is a no-embedder SQLite store. The accent tests run on
// SQLite only: its word index folds accents (FTS5 unicode61), so a row found
// there without them must not be dropped by the coverage floor. PostgreSQL's
// 'simple' configuration keeps them, so the index there does not find the row
// in the first place.
func openSQLiteLexical(t *testing.T) *DB {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "accents.db")))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if db.HasEmbedder() {
		t.Fatal("a lexical test must run without an embedder")
	}
	return db
}

func saveKnowledgeDocs(t *testing.T, db *DB, docs map[string]string) {
	t.Helper()
	for id, body := range docs {
		if _, err := db.SaveKnowledge(context.Background(), KnowledgeSaveRequest{KnowledgeID: id, Content: body}); err != nil {
			t.Fatalf("save %s: %v", id, err)
		}
	}
}

func lexicalKnowledgeIDs(t *testing.T, db *DB, req KnowledgeSearchRequest) []string {
	t.Helper()
	req.RetrievalMode = RetrievalModeLexical
	if req.TopK == 0 {
		req.TopK = 5
	}
	resp, err := db.SearchKnowledge(context.Background(), req)
	if err != nil {
		t.Fatalf("search %q: %v", req.Query, err)
	}
	var ids []string
	for _, h := range resp.Results {
		ids = append(ids, h.KnowledgeID)
	}
	return ids
}

var accentDocs = map[string]string{
	"uni":    "Die Universität Zürich untersteht dem kantonalen Recht und der Aufsicht des Regierungsrates.",
	"etat":   "Le Conseil d'Etat exerce le pouvoir executif et dirige l'administration du canton.",
	"budget": "Le parlement cantonal vote le budget annuel et contrôle les comptes de l'administration.",
}

// "universitat zurich recht" is how the question is typed on a keyboard
// without umlauts. The word index finds "Universität Zürich" for it; the
// coverage floor compared the words with their accents, found none of the
// two, and dropped the row.
func TestKnowledgeKeywordFloorComparesWordsWithoutAccents(t *testing.T) {
	db := openSQLiteLexical(t)
	saveKnowledgeDocs(t, db, accentDocs)
	if got := lexicalKnowledgeIDs(t, db, KnowledgeSearchRequest{Query: "universitat zurich recht"}); len(got) == 0 || got[0] != "uni" {
		t.Fatalf("results %v, want the Universität document first", got)
	}
	if got := lexicalKnowledgeIDs(t, db, KnowledgeSearchRequest{Query: "universitat"}); len(got) != 1 || got[0] != "uni" {
		t.Fatalf("one word without its umlaut: results %v, want the Universität document", got)
	}
}

// The other way round: the question has its accents, the stored text does not.
func TestKnowledgeKeywordFloorMatchesAccentedQueryOnUnaccentedText(t *testing.T) {
	db := openSQLiteLexical(t)
	saveKnowledgeDocs(t, db, accentDocs)
	if got := lexicalKnowledgeIDs(t, db, KnowledgeSearchRequest{Query: "état exécutif"}); len(got) == 0 || got[0] != "etat" {
		t.Fatalf("results %v, want the Conseil d'Etat document first", got)
	}
}

// A keyword typed without its accents counts toward coverage like one typed
// with them.
func TestKnowledgeKeywordFloorComparesKeywordsWithoutAccents(t *testing.T) {
	db := openSQLiteLexical(t)
	saveKnowledgeDocs(t, db, accentDocs)
	got := lexicalKnowledgeIDs(t, db, KnowledgeSearchRequest{Query: "Hochschule Zürich", Keywords: []string{"universitat"}})
	if len(got) == 0 || got[0] != "uni" {
		t.Fatalf("results %v, want the Universität document first", got)
	}
}

// The memory path counts coverage the same way.
func TestMemoryKeywordFloorComparesWordsWithoutAccents(t *testing.T) {
	db := openSQLiteLexical(t)
	saveGlobalMemories(t, db, map[string]string{
		"uni":   "Die Universität Zürich untersteht dem kantonalen Recht.",
		"other": "Der Regierungsrat legt das Budget dem Kantonsrat vor.",
	})
	resp, err := db.SearchMemory(context.Background(), MemorySearchRequest{Query: "universitat zurich", Scope: "global", TopK: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if got := memoryIDs(resp.Results); len(got) == 0 || got[0] != "uni" {
		t.Fatalf("results %v, want the Universität memory first", got)
	}
}

// Comparing without accents does not make the floor let through what it
// refused: a row that shares one word of three, accented or not, is still not
// an answer, and a question about nothing stored still returns nothing.
func TestKnowledgeKeywordFloorStillRefusesOneSharedWordWithoutAccents(t *testing.T) {
	db := openSQLiteLexical(t)
	saveKnowledgeDocs(t, db, accentDocs)
	for _, q := range []string{"universitat basel fussball", "universität basel fussball", "budget fondue raclette"} {
		if got := lexicalKnowledgeIDs(t, db, KnowledgeSearchRequest{Query: q}); len(got) != 0 {
			t.Errorf("%q returned %v; the row shares one word of three", q, got)
		}
	}
}

// One word typed two ways is one synonym. A row that holds it covers one word
// of the question's three, and is not an answer.
func TestKnowledgeKeywordFloorCountsAKeywordTypedTwoWaysOnce(t *testing.T) {
	db := openSQLiteLexical(t)
	saveKnowledgeDocs(t, db, accentDocs)
	if got := lexicalKnowledgeIDs(t, db, KnowledgeSearchRequest{Query: "hochschule basel fussball", Keywords: []string{"universität", "universitat"}}); len(got) != 0 {
		t.Fatalf("results %v, want none", got)
	}
}

func TestMemoryKeywordFloorCountsAKeywordTypedTwoWaysOnce(t *testing.T) {
	db := openSQLiteLexical(t)
	saveGlobalMemories(t, db, map[string]string{
		"uni":   "Die Universität Zürich untersteht dem kantonalen Recht.",
		"other": "Der Regierungsrat legt das Budget dem Kantonsrat vor.",
	})
	resp, err := db.SearchMemory(context.Background(), MemorySearchRequest{Query: "hochschule basel fussball", Keywords: []string{"universität", "universitat"}, Scope: "global", TopK: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if got := memoryIDs(resp.Results); len(got) != 0 {
		t.Fatalf("results %v, want none", got)
	}
}

// A question word typed two ways is one word of the question, as a keyword
// typed two ways is one synonym. "universität universitat basel" asks two
// words; a row that holds only the first is not an answer.
func TestKnowledgeKeywordFloorCountsAQuestionWordTypedTwoWaysOnce(t *testing.T) {
	db := openSQLiteLexical(t)
	saveKnowledgeDocs(t, db, accentDocs)
	if got := lexicalKnowledgeIDs(t, db, KnowledgeSearchRequest{Query: "universität universitat basel"}); len(got) != 0 {
		t.Fatalf("results %v, want none", got)
	}
}

func TestMemoryKeywordFloorCountsAQuestionWordTypedTwoWaysOnce(t *testing.T) {
	db := openSQLiteLexical(t)
	saveGlobalMemories(t, db, map[string]string{
		"uni":   "Die Universität Zürich untersteht dem kantonalen Recht.",
		"other": "Der Regierungsrat legt das Budget dem Kantonsrat vor.",
	})
	resp, err := db.SearchMemory(context.Background(), MemorySearchRequest{Query: "universität universitat basel", Scope: "global", TopK: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if got := memoryIDs(resp.Results); len(got) != 0 {
		t.Fatalf("results %v, want none", got)
	}
}

// Romanian: the index folds ș and ț (U+0219, U+021B), and so does coverage.
func TestKnowledgeKeywordFloorComparesRomanianWordsWithoutAccents(t *testing.T) {
	db := openSQLiteLexical(t)
	saveKnowledgeDocs(t, db, map[string]string{"ro": "Constituția României garantează dreptul la învățătură."})
	if got := lexicalKnowledgeIDs(t, db, KnowledgeSearchRequest{Query: "constitutia romaniei invatatura"}); len(got) != 1 || got[0] != "ro" {
		t.Fatalf("results %v, want the Romanian document", got)
	}
}
