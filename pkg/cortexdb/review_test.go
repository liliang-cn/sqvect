package cortexdb

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func ctx() context.Context { return context.Background() }

// openReviewBrain opens a fresh brain, with the review gate when asked.
func openReviewBrain(t *testing.T, name string, opts ...Option) *DB {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), name+".db")), opts...)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func searchIDs(t *testing.T, db *DB, query string, filters *RetrievalFilters) string {
	t.Helper()
	req := KnowledgeSearchRequest{Query: query, TopK: 10}
	if filters != nil {
		req.Plan = &RetrievalPlan{Filters: filters}
	}
	resp, err := db.SearchKnowledge(ctx(), req)
	if err != nil {
		t.Fatalf("search %q: %v", query, err)
	}
	return strings.Join(hitIDs(resp), ",")
}

// saveReviewed writes four documents with the same words and four statuses.
func saveReviewed(t *testing.T, db *DB) {
	t.Helper()
	content := "The heron protocol governs the nightly ledger export."
	save := func(id, status, by string) {
		if _, err := db.SaveKnowledge(ctx(), KnowledgeSaveRequest{KnowledgeID: id, Content: content, ReviewStatus: status, ReviewedBy: by}); err != nil {
			t.Fatalf("save %s: %v", id, err)
		}
	}
	save("k-plain", "", "")
	save("k-prop", ReviewProposed, "")
	save("k-rej", ReviewRejected, "mdupont")
	save("k-acc", ReviewAccepted, "mdupont")
}

func recallIDs(t *testing.T, db *DB) string {
	t.Helper()
	resp, err := db.SearchMemory(ctx(), MemorySearchRequest{UserID: "u", Scope: "user", Query: "osprey job", TopK: 10})
	if err != nil {
		t.Fatalf("search memory: %v", err)
	}
	ids := make([]string, 0, len(resp.Results))
	for _, h := range resp.Results {
		ids = append(ids, h.Memory.ID)
	}
	return strings.Join(ids, ",")
}

func saveReviewedMemories(t *testing.T, db *DB) {
	t.Helper()
	if _, err := db.SaveMemory(ctx(), MemorySaveRequest{MemoryID: "m-ok", UserID: "u", Scope: "user", Content: "The osprey job runs at dawn."}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveMemory(ctx(), MemorySaveRequest{MemoryID: "m-prop", UserID: "u", Scope: "user", Content: "The osprey job runs at dusk.",
		ReviewStatus: ReviewProposed}); err != nil {
		t.Fatal(err)
	}
}

// Without the gate, the status is kept and checked but changes nothing: a
// brain that never turns it on searches and recalls exactly as before. The
// filter review_status still lists rows by status, which is how a curator
// finds what waits.
func TestWithoutTheGateEveryReviewedRowIsServed(t *testing.T) {
	db := openReviewBrain(t, "review-off")
	saveReviewed(t, db)
	saveReviewedMemories(t, db)

	got := searchIDs(t, db, "heron ledger", nil)
	for _, id := range []string{"k-plain", "k-prop", "k-rej", "k-acc"} {
		if !strings.Contains(got, id) {
			t.Fatalf("without the gate %s was hidden: %s", id, got)
		}
	}
	if got := searchIDs(t, db, "heron ledger", &RetrievalFilters{ReviewStatus: ReviewProposed}); got != "k-prop" {
		t.Fatalf("review_status=proposed listed %q", got)
	}
	if got := searchIDs(t, db, "heron ledger", &RetrievalFilters{ReviewStatus: ReviewAccepted}); strings.Contains(got, "k-rej") || strings.Contains(got, "k-prop") || !strings.Contains(got, "k-plain") || !strings.Contains(got, "k-acc") {
		t.Fatalf("review_status=accepted listed %q", got)
	}
	if got := recallIDs(t, db); !strings.Contains(got, "m-ok") || !strings.Contains(got, "m-prop") {
		t.Fatalf("without the gate recall served %q, want both", got)
	}
	doc, err := db.store.GetDocument(ctx(), "k-rej")
	if err != nil {
		t.Fatalf("record of k-rej: %v", err)
	}
	if doc.Metadata[reviewStatusKey] != ReviewRejected {
		t.Fatalf("the record does not carry the review status: %v", doc.Metadata)
	}
}

// With the gate on, a proposed document is left out of search until a
// reviewer accepts it, and a rejected one stays out. The filter
// review_status=proposed still lists what waits.
func TestTheReviewGateHidesAProposedDocumentUntilAccepted(t *testing.T) {
	db := openReviewBrain(t, "review-doc", WithReviewGate())
	saveReviewed(t, db)
	c := ctx()

	got := searchIDs(t, db, "heron ledger", nil)
	if strings.Contains(got, "k-prop") || strings.Contains(got, "k-rej") {
		t.Fatalf("an unreviewed document was served: %s", got)
	}
	if !strings.Contains(got, "k-plain") || !strings.Contains(got, "k-acc") {
		t.Fatalf("accepted documents missing from %q", got)
	}
	if got := searchIDs(t, db, "heron ledger", &RetrievalFilters{ReviewStatus: ReviewProposed}); got != "k-prop" {
		t.Fatalf("review_status=proposed listed %q", got)
	}
	if got := searchIDs(t, db, "heron ledger", &RetrievalFilters{ReviewStatus: ReviewAccepted}); strings.Contains(got, "k-rej") || !strings.Contains(got, "k-plain") {
		t.Fatalf("review_status=accepted listed %q", got)
	}

	// The reviewer accepts: the document is served, and the decision is dated.
	if _, err := db.UpdateKnowledge(c, KnowledgeUpdateRequest{KnowledgeID: "k-prop", ReviewStatus: ReviewAccepted, ReviewedBy: "mdupont"}); err != nil {
		t.Fatalf("accept: %v", err)
	}
	if got := searchIDs(t, db, "heron ledger", nil); !strings.Contains(got, "k-prop") {
		t.Fatalf("an accepted document is still hidden: %s", got)
	}
	doc, err := db.store.GetDocument(c, "k-prop")
	if err != nil {
		t.Fatal(err)
	}
	if at, _ := doc.Metadata[reviewedAtKey].(string); at == "" {
		t.Fatalf("reviewed_at not set on acceptance: %v", doc.Metadata)
	}
}

// A decision is checked on every write tool: an acceptance or a rejection
// names who made it, the status is one of three, reviewed_at is an instant.
// A reviewer or an instant named without a status is not a decision and is
// not stored.
func TestADecisionNamesItsReviewer(t *testing.T) {
	db, tools, _ := openIngestBrain(t, "review-bad")
	c := ctx()
	_, err := tools.IngestDocument(c, ToolIngestDocumentRequest{DocumentID: "d", Content: "x", ReviewStatus: ReviewAccepted})
	if err == nil || !strings.Contains(err.Error(), "reviewed_by") {
		t.Fatalf("an acceptance without reviewer was taken by ingest: %v", err)
	}
	_, err = db.SaveKnowledge(c, KnowledgeSaveRequest{KnowledgeID: "k", Content: "x", ReviewStatus: "maybe"})
	if err == nil || !strings.Contains(err.Error(), "proposed, accepted or rejected") {
		t.Fatalf("a made-up status was taken by knowledge_save: %v", err)
	}
	_, err = db.SaveMemory(c, MemorySaveRequest{MemoryID: "m", UserID: "u", Scope: "user", Content: "x", ReviewStatus: ReviewRejected})
	if err == nil || !strings.Contains(err.Error(), "reviewed_by") {
		t.Fatalf("a rejection without reviewer was taken by memory_save: %v", err)
	}
	_, err = db.SaveMemory(c, MemorySaveRequest{MemoryID: "m", UserID: "u", Scope: "user", Content: "x", ReviewStatus: ReviewAccepted, ReviewedBy: "r", ReviewedAt: "yesterday"})
	if err == nil || !strings.Contains(err.Error(), "RFC 3339") {
		t.Fatalf("a bad reviewed_at was taken by memory_save: %v", err)
	}
	if _, err := db.SaveMemory(c, MemorySaveRequest{MemoryID: "m", UserID: "u", Scope: "user", Content: "x"}); err != nil {
		t.Fatal(err)
	}
	_, err = db.UpdateMemory(c, MemoryUpdateRequest{MemoryID: "m", ReviewStatus: "maybe"})
	if err == nil || !strings.Contains(err.Error(), "proposed, accepted or rejected") {
		t.Fatalf("a made-up status was taken by memory_update: %v", err)
	}
	if _, err := db.SaveKnowledge(c, KnowledgeSaveRequest{KnowledgeID: "k2", Content: "x", ReviewedBy: "r", ReviewedAt: "2026-10-18T10:00:00Z"}); err != nil {
		t.Fatalf("a reviewer named without a status was refused: %v", err)
	}
	if doc, _ := db.store.GetDocument(c, "k2"); doc.Metadata[reviewedByKey] != nil || doc.Metadata[reviewedAtKey] != nil {
		t.Fatalf("a reviewer named without a status was stored: %v", doc.Metadata)
	}
	for _, tc := range []struct{ id, given string }{{"m-at", "2026-10-18T10:00:00Z"}, {"m-now", ""}} {
		if _, err := db.SaveMemory(c, MemorySaveRequest{MemoryID: tc.id, UserID: "u", Scope: "user", Content: "x", ReviewStatus: ReviewAccepted, ReviewedBy: "r", ReviewedAt: tc.given}); err != nil {
			t.Fatal(err)
		}
		got, err := db.GetMemory(c, MemoryGetRequest{MemoryID: tc.id})
		if err != nil {
			t.Fatal(err)
		}
		at, _ := got.Memory.Metadata[reviewedAtKey].(string)
		if tc.given != "" && at != tc.given {
			t.Fatalf("%s: reviewed_at %q, want the given %q", tc.id, at, tc.given)
		}
		if _, perr := time.Parse(time.RFC3339, at); perr != nil {
			t.Fatalf("%s: reviewed_at %q is not an RFC 3339 instant", tc.id, at)
		}
	}
	if _, err := db.SaveKnowledge(c, KnowledgeSaveRequest{KnowledgeID: "k3", Content: "x", SourceURL: "https://wiki/k3", Author: "mdupont",
		ReviewStatus: ReviewAccepted, ReviewedBy: "r", ReviewedAt: "2026-10-18T10:00:00Z"}); err != nil {
		t.Fatal(err)
	}
	doc, err := db.store.GetDocument(c, "k3")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Metadata[reviewedAtKey] != "2026-10-18T10:00:00Z" {
		t.Fatalf("a given reviewed_at was not kept: %v", doc.Metadata[reviewedAtKey])
	}
	if doc.Metadata[reviewedByKey] != "r" || doc.Metadata[reviewStatusKey] != ReviewAccepted {
		t.Fatalf("record of k3: %v", doc.Metadata)
	}
}

// With the gate on, a proposed memory is left out of recall but still
// listed, with its status, so the reviewer can find it; accepting it brings
// it into recall.
func TestTheReviewGateLeavesAProposedMemoryOutOfRecall(t *testing.T) {
	db := openReviewBrain(t, "review-mem", WithReviewGate())
	saveReviewedMemories(t, db)
	c := ctx()
	if got := recallIDs(t, db); got != "m-ok" {
		t.Fatalf("recall served %q, want only m-ok", got)
	}

	all, err := db.ListAllMemories(c)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]string{}
	for _, m := range all {
		seen[m.ID] = m.ReviewStatus
	}
	if seen["m-prop"] != ReviewProposed || len(seen) != 2 {
		t.Fatalf("listing does not show the proposed memory with its status: %v", seen)
	}

	if _, err := db.UpdateMemory(c, MemoryUpdateRequest{MemoryID: "m-prop", ReviewStatus: ReviewAccepted, ReviewedBy: "mdupont"}); err != nil {
		t.Fatalf("accept memory: %v", err)
	}
	if got := recallIDs(t, db); !strings.Contains(got, "m-prop") {
		t.Fatalf("an accepted memory is still out of recall: %s", got)
	}
}

// wordBagEmbedder is the smallest embedder that makes two texts with the
// same words close: each word lands in one of dim buckets. Enough to drive
// the semantic memory path in a test without a model.
type wordBagEmbedder struct{ dim int }

func (e wordBagEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	v := make([]float32, e.dim)
	for _, w := range strings.Fields(strings.ToLower(text)) {
		h := 0
		for _, r := range w {
			h = h*31 + int(r)
		}
		if h < 0 {
			h = -h
		}
		v[h%e.dim]++
	}
	var norm float64
	for _, x := range v {
		norm += float64(x * x)
	}
	if norm > 0 {
		for i := range v {
			v[i] = float32(float64(v[i]) / math.Sqrt(norm))
		}
	}
	return v, nil
}

func (e wordBagEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i], _ = e.Embed(ctx, t)
	}
	return out, nil
}

func (e wordBagEmbedder) Dim() int { return e.dim }

// With the gate on, a proposed memory is out of every recall path, not only
// the lexical one: semantic, CJK lexical, graph by entity, and the PPR walk.
// The accepted twin is served by each.
func TestTheReviewGateHoldsOnEveryMemoryPath(t *testing.T) {
	db := openReviewBrain(t, "review-mem-paths", WithReviewGate(), WithEmbedder(wordBagEmbedder{dim: 64}), WithMemorySemanticFloor(0.2))
	c := ctx()
	save := func(id, content string, proposed bool) {
		req := MemorySaveRequest{MemoryID: id, Scope: MemoryScopeGlobal, Content: content,
			Entities: []ToolEntityInput{{Name: "Arvid Lund"}}}
		if proposed {
			req.ReviewStatus = ReviewProposed
		}
		if _, err := db.SaveMemory(c, req); err != nil {
			t.Fatalf("save %s: %v", id, err)
		}
	}
	save("m-ok", "Arvid Lund grew up in Malmo, a coastal city.", false)
	save("m-prop", "Arvid Lund grew up in Lund, an old university town.", true)
	save("m-cjk-ok", "鱼鹰任务在黎明运行。", false)
	save("m-cjk-prop", "鱼鹰任务在黄昏运行。", true)

	for name, req := range map[string]MemorySearchRequest{
		"semantic": {Query: "Arvid Lund grew up"},
		"lexical":  {Query: "Arvid Lund grew up", RetrievalMode: RetrievalModeLexical},
		"cjk":      {Query: "鱼鹰任务", RetrievalMode: RetrievalModeLexical},
		"graph":    {Query: "where did he grow up", EntityNames: []string{"Arvid Lund"}, RetrievalMode: RetrievalModeGraph},
		"ppr":      {Query: "Where was Arvid Lund raised?", RetrievalMode: RetrievalModePPR},
	} {
		req.Scope = MemoryScopeGlobal
		req.TopK = 10
		resp, err := db.SearchMemory(c, req)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		ids := make([]string, 0, len(resp.Results))
		for _, h := range resp.Results {
			ids = append(ids, h.Memory.ID)
		}
		got := strings.Join(ids, ",")
		if strings.Contains(got, "prop") {
			t.Fatalf("%s served a proposed memory: %s", name, got)
		}
		if !strings.Contains(got, "-ok") {
			t.Fatalf("%s lost the accepted memory: %s (mode %s)", name, got, resp.Decision.EffectiveMode)
		}
	}
}

// The filters a caller passes with a plan still apply next to the gate: a
// document_ids filter keeps only those documents, and a review_status given
// at request level overrides the one in the plan.
func TestReviewFiltersMergeWithTheRestOfThePlan(t *testing.T) {
	db := openReviewBrain(t, "review-filters")
	saveReviewed(t, db)
	if got := searchIDs(t, db, "heron protocol", &RetrievalFilters{DocumentIDs: []string{"k-acc"}}); got != "k-acc" {
		t.Fatalf("document_ids filter served %q, want k-acc", got)
	}
	res := resolveRetrievalPlan(retrievalPlanInput{
		Query:   "q",
		Plan:    &RetrievalPlan{Filters: &RetrievalFilters{ReviewStatus: ReviewAccepted}},
		Filters: &RetrievalFilters{ReviewStatus: ReviewProposed},
	})
	if res.Plan.Filters == nil || res.Plan.Filters.ReviewStatus != ReviewProposed {
		t.Fatalf("request-level review_status did not override the plan's: %+v", res.Plan.Filters)
	}
}
