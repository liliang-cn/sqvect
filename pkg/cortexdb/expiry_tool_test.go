package cortexdb

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// The sweep is a tool of the toolbox, so a client of a shared brain can run
// it where the brain is: a dry run lists the expired document and removes
// nothing, the real run removes it and leaves the live one, and the tool is
// declared as one that writes.
func TestTheSweepIsAToolOfTheSharedBrain(t *testing.T) {
	db, tools, c := openIngestBrain(t, "sweep-tool")
	past := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	for id, until := range map[string]string{"old": past, "live": ""} {
		if _, err := tools.IngestDocument(c, ToolIngestDocumentRequest{DocumentID: id, Content: "A note on the " + id + " ledger.", ExpiresAt: strPtr(until)}); err != nil {
			t.Fatal(err)
		}
	}

	var def ToolDefinition
	for _, d := range ToolDefinitions() {
		if d.Name == "sweep_expired" {
			def = d
		}
	}
	if def.Name == "" || !def.Mutates || !strings.Contains(def.Description, "expire") {
		t.Fatalf("sweep_expired is not declared as a writing tool: %+v", def)
	}

	result, err := tools.Call(c, "sweep_expired", json.RawMessage(`{"dry_run":true}`))
	if err != nil {
		t.Fatal(err)
	}
	report, ok := result.(*ExpiredSweepReport)
	if !ok || !report.DryRun || strings.Join(report.Documents, ",") != "old" {
		t.Fatalf("dry run: %#v", result)
	}
	if _, err := db.GetKnowledge(c, KnowledgeGetRequest{KnowledgeID: "old"}); err != nil {
		t.Fatalf("the dry run removed the document: %v", err)
	}

	// The same dry run over MCP, the way a host calls the tool.
	if got := sweepOverMCP(t, db); !got.DryRun || strings.Join(got.Documents, ",") != "old" {
		t.Fatalf("dry run over MCP: %+v", got)
	}

	if _, err := tools.Call(c, "sweep_expired", json.RawMessage(`{"dry_run":"yes"}`)); err == nil || !strings.Contains(err.Error(), "decode sweep_expired") {
		t.Fatalf("a bad argument was not refused: %v", err)
	}

	result, err = tools.Call(c, "sweep_expired", json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if report = result.(*ExpiredSweepReport); report.DryRun || strings.Join(report.Documents, ",") != "old" {
		t.Fatalf("sweep: %+v", report)
	}
	if _, err := db.GetKnowledge(c, KnowledgeGetRequest{KnowledgeID: "old"}); err == nil {
		t.Fatal("the expired document is still there")
	}
	if _, err := db.GetKnowledge(c, KnowledgeGetRequest{KnowledgeID: "live"}); err != nil {
		t.Fatalf("the live document went with it: %v", err)
	}
}

// sweepOverMCP calls sweep_expired with dry_run through an in-memory MCP
// session and decodes the report the host receives.
func sweepOverMCP(t *testing.T, db *DB) ExpiredSweepReport {
	t.Helper()
	ctx := context.Background()
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	serverSession, err := db.NewMCPServer(MCPServerOptions{}).Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = serverSession.Close() }()
	session, err := mcp.NewClient(&mcp.Implementation{Name: "sweep-test"}, nil).Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	result, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "sweep_expired", Arguments: map[string]any{"dry_run": true}})
	if err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(result.StructuredContent)
	if err != nil {
		t.Fatal(err)
	}
	var report ExpiredSweepReport
	if err := json.Unmarshal(payload, &report); err != nil {
		t.Fatalf("decode %s: %v", payload, err)
	}
	return report
}
