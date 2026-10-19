package cortexdb

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

// The expiry on PostgreSQL, through the facade: the search leaves the
// expired document out, the aggregate does not count it, dedupe does not
// take it as a holder, and the sweep removes it and only it.
func TestTheExpiryHoldsOnPostgres(t *testing.T) {
	db := openPostgresBrain(t, 4)
	tools := db.GraphRAGTools()
	ctx := context.Background()
	past := pastInstant()
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "old", Content: "The heron ledger, kept on PostgreSQL.", ExpiresAt: &past}); err != nil {
		t.Fatal(err)
	}
	if _, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "live", Content: "The live ledger, kept on PostgreSQL."}); err != nil {
		t.Fatal(err)
	}
	hits, err := tools.SearchText(ctx, ToolSearchTextRequest{Query: "ledger PostgreSQL", TopK: 10})
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Join(func() []string {
		var ids []string
		for _, h := range hits.Chunks {
			ids = append(ids, h.DocumentID)
		}
		return ids
	}(), ",")
	if strings.Contains(body, "old") || !strings.Contains(body, "live") {
		t.Errorf("search_text served the documents %q, want live only", body)
	}
	agg, err := db.Aggregate(ctx, core.AggregationRequest{Type: core.AggregationCount})
	if err != nil {
		t.Fatal(err)
	}
	if len(agg.Results) != 1 || agg.Results[0].Count != 1 {
		t.Errorf("aggregate %+v, want one row counted", agg.Results)
	}
	dup, err := tools.IngestDocument(ctx, ToolIngestDocumentRequest{DocumentID: "copy", Content: "The heron ledger, kept on PostgreSQL.", Dedupe: true})
	if err != nil {
		t.Fatal(err)
	}
	if dup.DuplicateOf != "" {
		t.Errorf("dedupe took the expired document as a holder: %q", dup.DuplicateOf)
	}
	report, err := db.SweepExpired(ctx, time.Now(), false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(report.Documents, ",") != "old" {
		t.Errorf("the sweep removed %v, want old", report.Documents)
	}
}
