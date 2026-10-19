package cortexdb

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

func hitIDs(resp *KnowledgeSearchResponse) []string {
	ids := make([]string, 0, len(resp.Results))
	for _, h := range resp.Results {
		ids = append(ids, h.KnowledgeID)
	}
	return ids
}

// A document past its expires_at is left out of search, a document before it
// is not. The expiry is kept as metadata, so a reader of the record sees it.
func TestAnExpiredDocumentIsLeftOutOfSearch(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "expiry-search")
	content := "The zebra ledger reconciles the quarterly accounts."
	past := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	future := time.Now().UTC().Add(time.Hour).Format(time.RFC3339)

	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "old", Content: content, ExpiresAt: strPtr(past)}); err != nil {
		t.Fatalf("ingest old: %v", err)
	}
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "soon", Content: content, ExpiresAt: strPtr(future)}); err != nil {
		t.Fatalf("ingest soon: %v", err)
	}
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "forever", Content: content}); err != nil {
		t.Fatalf("ingest forever: %v", err)
	}

	resp, err := db.SearchKnowledge(ctx, KnowledgeSearchRequest{Query: "zebra ledger", TopK: 10})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	got := strings.Join(hitIDs(resp), ",")
	if strings.Contains(got, "old") {
		t.Fatalf("an expired document was returned: %s", got)
	}
	if !strings.Contains(got, "soon") || !strings.Contains(got, "forever") {
		t.Fatalf("live documents missing from %q", got)
	}

	doc, err := db.store.GetDocument(ctx, "old")
	if err != nil {
		t.Fatalf("get old: %v", err)
	}
	if doc.Metadata[expiresAtKey] != past {
		t.Fatalf("record metadata expires_at %v, want %s", doc.Metadata[expiresAtKey], past)
	}
}

func TestABadExpiresAtIsRefused(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "expiry-bad")
	_, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d", Content: "x", ExpiresAt: strPtr("next year")})
	if err == nil || !strings.Contains(err.Error(), "RFC 3339") {
		t.Fatalf("a bad expires_at was accepted: %v", err)
	}
	// The field is checked the same way on save.
	_, err = db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "k", Content: "x", ExpiresAt: strPtr("2031-13-01")})
	if err == nil {
		t.Fatal("a bad expires_at was accepted by knowledge_save")
	}
	// A zoned instant is accepted and stored as UTC.
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "k", Content: "x", ExpiresAt: strPtr("2031-10-18T02:00:00+02:00")}); err != nil {
		t.Fatalf("a zoned expires_at was refused: %v", err)
	}
	doc, err := db.store.GetDocument(ctx, "k")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Metadata[expiresAtKey] != "2031-10-18T00:00:00Z" {
		t.Fatalf("expires_at stored as %v", doc.Metadata[expiresAtKey])
	}
}

// The sweep removes expired documents (record, chunks, graph node) and expired
// memories, and nothing else. A dry run lists the same ids and removes nothing.
func TestSweepExpiredRemovesExpiredDocumentsAndMemories(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "expiry-sweep")
	past := time.Now().UTC().Add(-time.Minute).Format(time.RFC3339)

	old, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "old", Content: "Old content about the ledger.", ExpiresAt: strPtr(past)})
	if err != nil {
		t.Fatalf("ingest old: %v", err)
	}
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "live", Content: "Live content about the ledger."}); err != nil {
		t.Fatalf("ingest live: %v", err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "m-old", UserID: "u", Scope: "user", Content: "short-lived", TTLSeconds: 1}); err != nil {
		t.Fatalf("save m-old: %v", err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "m-live", UserID: "u", Scope: "user", Content: "kept"}); err != nil {
		t.Fatalf("save m-live: %v", err)
	}
	// The memory's ttl of one second has passed at an instant two seconds
	// on: the sweep takes its instant as an argument, so nothing waits.
	now := time.Now().Add(2 * time.Second)

	dry, err := db.SweepExpired(ctx, now, true)
	if err != nil {
		t.Fatalf("dry run: %v", err)
	}
	if strings.Join(dry.Documents, ",") != "old" || strings.Join(dry.Memories, ",") != "m-old" {
		t.Fatalf("dry run listed %v / %v", dry.Documents, dry.Memories)
	}
	if _, err := db.store.GetDocument(ctx, "old"); err != nil {
		t.Fatalf("dry run removed the document: %v", err)
	}
	if _, err := db.loadMemoryRow(ctx, "m-old"); err != nil {
		t.Fatalf("dry run removed the memory: %v", err)
	}

	report, err := db.SweepExpired(ctx, now, false)
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if strings.Join(report.Documents, ",") != "old" || strings.Join(report.Memories, ",") != "m-old" {
		t.Fatalf("sweep removed %v / %v", report.Documents, report.Memories)
	}
	if _, err := db.store.GetDocument(ctx, "old"); !errors.Is(err, core.ErrNotFound) {
		t.Fatalf("expired document still stored: %v", err)
	}
	if chunks, _ := db.store.GetByDocID(ctx, "old"); len(chunks) != 0 {
		t.Fatalf("expired document kept %d chunks", len(chunks))
	}
	if node, err := db.graph.GetNode(ctx, old.DocumentNodeID); err == nil && node != nil {
		t.Fatal("expired document kept its graph node")
	}
	if _, err := db.loadMemoryRow(ctx, "m-old"); err == nil {
		t.Fatal("expired memory still stored")
	}
	if _, err := db.store.GetDocument(ctx, "live"); err != nil {
		t.Fatalf("live document was swept: %v", err)
	}
	if _, err := db.loadMemoryRow(ctx, "m-live"); err != nil {
		t.Fatalf("live memory was swept: %v", err)
	}

	// Nothing left to sweep.
	again, err := db.SweepExpired(ctx, time.Now(), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(again.Documents)+len(again.Memories) != 0 {
		t.Fatalf("second sweep found %v / %v", again.Documents, again.Memories)
	}
}

// An empty expires_at, on ingest or in metadata, removes the expiry: the key
// is gone from the record and the document is served again.
func TestAnEmptyExpiresAtClearsTheExpiry(t *testing.T) {
	db, tools, ctx := openIngestBrain(t, "expiry-clear")
	past := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	content := "The zebra ledger reconciles the quarterly accounts."
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d", Content: content, ExpiresAt: strPtr(past)}); err != nil {
		t.Fatal(err)
	}
	if resp, _ := db.SearchKnowledge(ctx, KnowledgeSearchRequest{Query: "zebra ledger", TopK: 5}); len(hitIDs(resp)) != 0 {
		t.Fatalf("expired document served: %v", hitIDs(resp))
	}
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "d", Content: content, ExpiresAt: strPtr("")}); err != nil {
		t.Fatalf("clearing the expiry: %v", err)
	}
	doc, err := db.store.GetDocument(ctx, "d")
	if err != nil {
		t.Fatal(err)
	}
	if _, still := doc.Metadata[expiresAtKey]; still {
		t.Fatalf("expires_at still on the record: %v", doc.Metadata[expiresAtKey])
	}
	if resp, _ := db.SearchKnowledge(ctx, KnowledgeSearchRequest{Query: "zebra ledger", TopK: 5}); strings.Join(hitIDs(resp), ",") != "d" {
		t.Fatalf("document not served after the expiry was cleared: %v", hitIDs(resp))
	}
}

// The graph walk (PPR) leaves an expired passage out too, both as a first-stage
// seed and as a chunk the walk reaches through an entity.
func TestTheGraphWalkLeavesAnExpiredPassageOut(t *testing.T) {
	for name, db := range lexicalBackends(t) {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			seedPPRKnowledge(t, ctx, db)
			past := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
			if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{
				KnowledgeID: "lund-old",
				Content:     "Arvid Lund was raised in Lund, an old university town.",
				Entities:    []ToolEntityInput{{Name: "Arvid Lund", ChunkIDs: []string{graphChunkNodeID("lund-old", 0)}}},
				ExpiresAt:   strPtr(past),
			}); err != nil {
				t.Fatal(err)
			}
			resp, err := db.SearchKnowledge(ctx, KnowledgeSearchRequest{Query: pprTwoHopQuery, TopK: 5, RetrievalMode: RetrievalModePPR})
			if err != nil {
				t.Fatalf("ppr search: %v", err)
			}
			if resp.Decision.EffectiveMode != RetrievalModePPR {
				t.Fatalf("effective mode %q, want PPR", resp.Decision.EffectiveMode)
			}
			if knowledgeRank(resp, "lund-old") != 0 {
				t.Fatalf("the walk returned an expired passage: %+v", resp.Chunks)
			}
			if knowledgeRank(resp, "lund") == 0 {
				t.Fatalf("the walk lost the live passage: %+v", resp.Chunks)
			}
		})
	}
}

// A brain that cannot be read: the expiry filter lets rows through, because
// it is advisory and hiding everything on an outage would look like an empty
// brain; the sweep, which is the guarantee, reports the failure instead.
func TestAnUnreadableBrainIsNotAnEmptyBrain(t *testing.T) {
	db, _, c := openIngestBrain(t, "expiry-unreadable")
	if hidden := db.hiddenAmong(c, nil, []string{""}); len(hidden) != 0 {
		t.Fatal("a chunk without a document was treated as expired")
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if hits := db.filterScoredEmbeddings(c, []core.ScoredEmbedding{{Embedding: core.Embedding{DocID: "any"}}}, nil); len(hits) != 1 {
		t.Fatal("the expiry filter hid a document it could not check")
	}
	// The aggregates cannot be advisory: the exclusion runs in the same
	// statement as the count, so a brain that cannot be read is an error,
	// not a count.
	if _, err := db.GraphRAGTools().AggregateMetadata(c, ToolAggregateMetadataRequest{Type: "count"}); err == nil {
		t.Fatalf("aggregate on a closed brain: %v", err)
	}
	if _, err := db.GraphRAGTools().RepresentativeRecords(c, ToolRepresentativeRecordsRequest{}); err == nil {
		t.Fatalf("representative records on a closed brain: %v", err)
	}
	if _, err := db.SweepExpired(c, time.Now(), false); err == nil {
		t.Fatal("a sweep on a closed brain reported success")
	}
}

// dropChangeLogTriggers removes the audit triggers of a table, so a test can
// write a row the application never would.
func dropChangeLogTriggers(t *testing.T, db *DB, table string) {
	t.Helper()
	rows, err := db.SQL().Query(`SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = ? AND sql LIKE '%change_log%'`, table)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		names = append(names, name)
	}
	_ = rows.Close()
	for _, name := range names {
		if _, err := db.SQL().Exec(`DROP TRIGGER ` + name); err != nil {
			t.Fatal(err)
		}
	}
}

// A row the sweep cannot read is an error, not a row silently skipped: a
// document or memory row without an id fails the listing. A memory row whose
// metadata is not JSON is not a memory record and is left alone.
func TestARowTheSweepCannotReadIsReported(t *testing.T) {
	db, _, c := openIngestBrain(t, "expiry-unreadable-row")
	past := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	dropChangeLogTriggers(t, db, "documents")
	if _, err := db.SQL().Exec(`INSERT INTO documents (id, content, metadata) VALUES (NULL, 'x', ?)`, `{"_expires_at":"`+past+`"}`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SweepExpired(c, time.Now(), true); err == nil || !strings.Contains(err.Error(), "scan document expiry") {
		t.Fatalf("a document row without an id: %v", err)
	}
	if _, err := db.SQL().Exec(`DELETE FROM documents WHERE id IS NULL`); err != nil {
		t.Fatal(err)
	}

	if _, err := db.SaveMemory(c, MemorySaveRequest{MemoryID: "m-seed", UserID: "u", Scope: "user", Content: "seed"}); err != nil {
		t.Fatal(err)
	}
	var session string
	if err := db.SQL().QueryRow(`SELECT session_id FROM messages WHERE id = 'm-seed'`).Scan(&session); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SQL().Exec(`INSERT INTO messages (id, session_id, role, content, metadata) VALUES ('m-text', ?, 'user', 'x', 'not json')`, session); err != nil {
		t.Fatal(err)
	}
	report, err := db.SweepExpired(c, time.Now(), true)
	if err != nil || len(report.Memories) != 0 {
		t.Fatalf("a memory row whose metadata is not JSON: %+v, %v", report, err)
	}
	dropChangeLogTriggers(t, db, "messages")
	if _, err := db.SQL().Exec(`INSERT INTO messages (id, session_id, role, content, metadata) VALUES (NULL, ?, 'user', 'x', '{}')`, session); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SweepExpired(c, time.Now(), true); err == nil || !strings.Contains(err.Error(), "scan memory expiry") {
		t.Fatalf("a memory row without an id: %v", err)
	}
}

// A listing that fails fails the sweep, and a delete that fails stops it:
// the report then names what was removed before the failure, and nothing
// else.
func TestASweepStopsAtTheFirstFailure(t *testing.T) {
	db, tools, c := openIngestBrain(t, "expiry-sweep-fails")
	past := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	if _, err := tools.IngestDocument(c, ToolIngestDocumentRequest{DocumentID: "k-old", Content: "An expired note.", ExpiresAt: strPtr(past)}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveMemory(c, MemorySaveRequest{MemoryID: "m-old", UserID: "u", Scope: "user", Content: "old", TTLSeconds: 1}); err != nil {
		t.Fatal(err)
	}
	// The sweep's instant is two seconds on: the memory's ttl of one second
	// has passed there, and nothing waits.
	later := time.Now().Add(2 * time.Second)

	if _, err := db.SQL().Exec(`ALTER TABLE messages RENAME COLUMN metadata TO metadata_hidden`); err != nil {
		t.Fatal(err)
	}
	if report, err := db.SweepExpired(c, later, true); err == nil || report != nil || !strings.Contains(err.Error(), "list memory expiries") {
		t.Fatalf("an unreadable memory table: %+v, %v", report, err)
	}
	if _, err := db.SQL().Exec(`ALTER TABLE messages RENAME COLUMN metadata_hidden TO metadata`); err != nil {
		t.Fatal(err)
	}

	block := func(table string) {
		if _, err := db.SQL().Exec(`CREATE TRIGGER block_` + table + ` BEFORE DELETE ON ` + table + ` BEGIN SELECT RAISE(ABORT, 'blocked'); END`); err != nil {
			t.Fatal(err)
		}
	}
	unblock := func(table string) {
		if _, err := db.SQL().Exec(`DROP TRIGGER block_` + table); err != nil {
			t.Fatal(err)
		}
	}
	block("documents")
	report, err := db.SweepExpired(c, later, false)
	if err == nil || !strings.Contains(err.Error(), "sweep document k-old") || report == nil || len(report.Documents) != 0 || report.Memories != nil {
		t.Fatalf("a document delete that fails: %+v, %v", report, err)
	}
	unblock("documents")
	block("messages")
	report, err = db.SweepExpired(c, later, false)
	if err == nil || !strings.Contains(err.Error(), "sweep memory m-old") || report == nil || strings.Join(report.Documents, ",") != "k-old" || len(report.Memories) != 0 {
		t.Fatalf("a memory delete that fails: %+v, %v", report, err)
	}
	unblock("messages")
	report, err = db.SweepExpired(c, later, false)
	if err != nil || strings.Join(report.Memories, ",") != "m-old" || len(report.Documents) != 0 {
		t.Fatalf("the sweep after the failures: %+v, %v", report, err)
	}
}
