package cortexdb

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// callOverMCP calls a tool through the embedded MCP server, the one
// cortexdb-mcp-stdio serves, and returns the structured answer.
func callOverMCP(t *testing.T, db *DB, name string, args map[string]any) (map[string]any, error) {
	t.Helper()
	ctx := context.Background()
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	serverSession, err := db.NewMCPServer(MCPServerOptions{}).Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = serverSession.Close() }()
	session, err := mcp.NewClient(&mcp.Implementation{Name: "list-test"}, nil).Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	result, err := session.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		return nil, err
	}
	if result.IsError {
		raw, _ := json.Marshal(result.Content)
		return nil, &mcpToolError{string(raw)}
	}
	raw, _ := json.Marshal(result.StructuredContent)
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out, nil
}

type mcpToolError struct{ text string }

func (e *mcpToolError) Error() string { return e.text }

// idsUnder reads the ids of the records listed under key.
func idsUnder(out map[string]any, key string) []string {
	var ids []string
	list, _ := out[key].([]any)
	for _, item := range list {
		record, _ := item.(map[string]any)
		ids = append(ids, record["id"].(string))
	}
	return ids
}

func stringsUnder(out map[string]any, key string) []string {
	var got []string
	list, _ := out[key].([]any)
	for _, item := range list {
		got = append(got, item.(string))
	}
	return got
}

// The list forms of knowledge_get and memory_get work over the embedded MCP
// server too. It infers each tool's input schema from the request struct, so
// a single-form field without omitempty was required there, and a call with
// the list alone was refused before it reached the facade.
func TestKnowledgeGetAndMemoryGetTakeTheirListOverMCP(t *testing.T) {
	db := openListBrain(t)
	ctx := context.Background()
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "k1", Content: "The lyre of Orpheus."}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "m1", UserID: "hermes", Content: "Hermes carries the messages."}); err != nil {
		t.Fatal(err)
	}
	out, err := callOverMCP(t, db, "knowledge_get", map[string]any{"knowledge_ids": []string{"k1", "nope"}})
	if err != nil {
		t.Fatalf("knowledge_get with knowledge_ids over MCP: %v", err)
	}
	if got, missing := idsUnder(out, "knowledges"), stringsUnder(out, "missing"); !reflect.DeepEqual(got, []string{"k1"}) || !reflect.DeepEqual(missing, []string{"nope"}) {
		t.Errorf("knowledge_get over MCP: served %v, missing %v; want [k1], [nope]", got, missing)
	}
	out, err = callOverMCP(t, db, "memory_get", map[string]any{"memory_ids": []string{"nope", "m1"}})
	if err != nil {
		t.Fatalf("memory_get with memory_ids over MCP: %v", err)
	}
	if got, missing := idsUnder(out, "memories"), stringsUnder(out, "missing"); !reflect.DeepEqual(got, []string{"m1"}) || !reflect.DeepEqual(missing, []string{"nope"}) {
		t.Errorf("memory_get over MCP: served %v, missing %v; want [m1], [nope]", got, missing)
	}
	// The single forms are answered as before.
	if out, err := callOverMCP(t, db, "knowledge_get", map[string]any{"knowledge_id": "k1"}); err != nil || out["knowledge"].(map[string]any)["id"] != "k1" {
		t.Errorf("knowledge_get single form over MCP: %v %v", out, err)
	}
	if out, err := callOverMCP(t, db, "memory_get", map[string]any{"memory_id": "m1"}); err != nil || out["memory"].(map[string]any)["id"] != "m1" {
		t.Errorf("memory_get single form over MCP: %v %v", out, err)
	}
}

// The schema a remote client reads for knowledge_get requires neither form,
// as memory_get's requires neither: a call may send the list alone.
func TestTheKnowledgeGetSchemaRequiresNeitherForm(t *testing.T) {
	for _, def := range KnowledgeMemoryToolDefinitions() {
		if def.Name != "knowledge_get" && def.Name != "memory_get" {
			continue
		}
		schema := def.InputSchema
		if required, ok := schema["required"]; ok {
			t.Errorf("%s requires %v; a call with the list alone must validate", def.Name, required)
		}
		props, _ := schema["properties"].(map[string]any)
		if len(props) != 2 {
			t.Errorf("%s properties %v, want the single and the list form", def.Name, props)
		}
		// Remote clients install this schema as the tool's input schema
		// (cortexdb-mcp-stdio): the list must be an array of strings.
		list := map[string]string{"knowledge_get": "knowledge_ids", "memory_get": "memory_ids"}[def.Name]
		prop, _ := props[list].(map[string]any)
		items, _ := prop["items"].(map[string]any)
		if prop["type"] != "array" || items["type"] != "string" {
			t.Errorf("%s.%s is %v, want an array of strings", def.Name, list, prop)
		}
	}
}

// The texts an agent reads when it picks ids name the list form of both
// tools: the index a recall returns, the detail argument's schema, and the
// description of each get tool.
func TestTheAgentTextsNameTheListForms(t *testing.T) {
	for _, def := range KnowledgeMemoryToolDefinitions() {
		list := map[string]string{"knowledge_get": "knowledge_ids", "memory_get": "memory_ids"}[def.Name]
		if list != "" && !strings.Contains(def.Description, "with "+list) {
			t.Errorf("the %s description does not name %s: %q", def.Name, list, def.Description)
		}
	}
	resp := &KnowledgeMemoryRecallResponse{
		Memories:  []MemorySearchHit{{Memory: MemoryRecord{ID: "m1", Content: "A memory."}}},
		Knowledge: []KnowledgeSearchHit{{KnowledgeID: "k1", Title: "A document."}},
	}
	indexRecall(resp)
	for _, want := range []string{"memory_get memory_ids", "knowledge_get knowledge_ids"} {
		if !strings.Contains(resp.ContextPack.Text, want) {
			t.Errorf("the index does not say %q: %s", want, resp.ContextPack.Text)
		}
	}
	found := false
	for _, def := range KnowledgeMemoryFacadeToolDefinitions() {
		raw, _ := json.Marshal(def.InputSchema)
		if def.Name == "knowledge_memory_recall" {
			found = true
			if !strings.Contains(string(raw), "knowledge_get knowledge_ids") {
				t.Errorf("knowledge_memory_recall's detail does not name knowledge_get knowledge_ids: %s", raw)
			}
		}
	}
	if !found {
		t.Fatal("knowledge_memory_recall is not defined")
	}
}
