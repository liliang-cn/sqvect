package eval_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/internal/testname"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/liliang-cn/cortexdb/v2/pkg/eval"
)

// German, French and Italian queries with no embedder, against the Swiss
// Federal Constitution (SR 101, Fedlex, version of 3 March 2024) in its four
// official languages. Each query is asked against the article in its own
// language, so this measures the word path within a language, not across
// languages: as written, without its accents (as people type "universitat"
// for "Universität"; the set never writes "ue" for "ü"), and with a German
// word the article holds only inside a compound ("Versicherung" in
// "Unfallversicherung"). The Romansh texts are distractors.
//
// A heading query is the article's title without "Art. N". The lexical path
// does not search the title, so it measures whether the heading's words
// occur in the article's text.
//
// Each category holds its own floor, a little under what it measures today.
// The negatives return nothing. One of them shares a single word with one
// article ("Gesicht"), which is not an answer; another shares the same word
// and a second one only inside a longer word ("heimisch" in
// "einheimischen"), which is not a second shared word; a third does the same
// with a word of three letters ("Ort" in "Orten").
func TestSwissLexicalRetrievalQuality(t *testing.T) {
	ds, err := eval.BuiltinSwiss()
	if err != nil {
		t.Fatalf("load dataset: %v", err)
	}

	dbPath := filepath.Join(t.TempDir(), fmt.Sprintf("test_eval_ch_%d.db", testname.Nano()))
	db, err := cortexdb.Open(cortexdb.DefaultConfig(dbPath))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() {
		_ = db.Close()
		for _, s := range []string{"", "-wal", "-shm"} {
			_ = os.Remove(dbPath + s)
		}
	})
	if db.HasEmbedder() {
		t.Fatal("this measures the no-embedder path")
	}

	ctx := context.Background()
	for _, d := range ds.Documents {
		if _, err := db.SaveKnowledge(ctx, cortexdb.KnowledgeSaveRequest{
			KnowledgeID: d.ID,
			Title:       d.Title,
			Content:     d.Content,
		}); err != nil {
			t.Fatalf("save %q: %v", d.ID, err)
		}
	}

	search := func(ctx context.Context, query string, k int) ([]string, error) {
		resp, err := db.SearchKnowledge(ctx, cortexdb.KnowledgeSearchRequest{
			Query:         query,
			RetrievalMode: cortexdb.RetrievalModeLexical,
			TopK:          k,
		})
		if err != nil {
			return nil, err
		}
		ids := make([]string, 0, len(resp.Results))
		for _, hit := range resp.Results {
			ids = append(ids, hit.KnowledgeID)
		}
		return ids, nil
	}

	for _, floor := range swissFloors {
		sub := *ds
		sub.Name = ds.Name + "/" + floor.category
		sub.Queries = nil
		for _, q := range ds.Queries {
			if q.Category == floor.category {
				sub.Queries = append(sub.Queries, q)
			}
		}
		if len(sub.Queries) == 0 {
			t.Fatalf("no %s queries in the dataset", floor.category)
		}
		rep, err := eval.Run(ctx, &sub, eval.RetrieverFunc(search), 1, 5, 10)
		if err != nil {
			t.Fatalf("run eval: %v", err)
		}
		t.Logf("swiss lexical retrieval quality:\n%s", rep.Summary())
		for _, q := range rep.PerQuery {
			if q.RR == 0 {
				t.Logf("miss %s: %v", q.QueryID, q.Retrieved)
			}
		}
		if rep.RecallAtK[10] < floor.recall10 {
			t.Errorf("%s: recall@10 = %.3f, below floor %.2f", floor.category, rep.RecallAtK[10], floor.recall10)
		}
		if rep.MRR < floor.mrr {
			t.Errorf("%s: MRR = %.3f, below floor %.2f", floor.category, rep.MRR, floor.mrr)
		}
	}

	for _, n := range ds.Negatives {
		got, err := search(ctx, n.Text, 5)
		if err != nil {
			t.Fatalf("negative %s: %v", n.ID, err)
		}
		if len(got) != 0 {
			t.Errorf("negative %s (%s) returned %v; the corpus holds nothing about it", n.ID, n.Text, got)
		}
	}
}

// swissFloors are a little under what each category measures today (recall@10,
// MRR): plain 0.986, 0.926; unaccented 1.000, 0.925; heading 0.764, 0.691;
// heading_unaccented 0.806, 0.769; compound 0.833, 0.524.
//
// unaccented and heading_unaccented hold the coverage floor to comparing
// words as the word index reads them, without their accents: "Liberte
// economique" finds "Liberté économique" in the index, and when the floor
// compared the words with their accents it dropped the row (0.980, 0.908 and
// 0.387, 0.349).
var swissFloors = []struct {
	category      string
	recall10, mrr float64
}{
	{"plain", 0.95, 0.90},
	{"unaccented", 0.99, 0.91},
	{"heading", 0.74, 0.67},
	{"heading_unaccented", 0.78, 0.74},
	{"compound", 0.80, 0.50},
}
