package cortexdb

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"

	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

// The changes_since output schema must accept events whose before/after row
// summaries are JSON objects, as the change feed writes them.
func TestChangesSinceMCPOutputSchemaAcceptsObjectSummaries(t *testing.T) {
	resp := &ChangesSinceResponse{
		Events: []graph.ChangeEvent{{
			Seq: 1, At: time.Unix(0, 0).UTC(), Kind: "memory", Op: "insert", ID: "m1",
			After: json.RawMessage(`{"title":"t","metadata":{"type":"project"},"version":1}`),
		}},
		NextCursor: 1, Head: 1,
	}
	schema, err := jsonschema.For[changesSinceMCPResponse](nil)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := schema.Resolve(nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(newChangesSinceMCPResponse(resp))
	if err != nil {
		t.Fatal(err)
	}
	var instance map[string]any
	if err := json.Unmarshal(raw, &instance); err != nil {
		t.Fatal(err)
	}
	if err := resolved.Validate(instance); err != nil {
		t.Fatalf("output schema rejects a change event: %v\n%s", err, raw)
	}
	after := instance["events"].([]any)[0].(map[string]any)["after"]
	if _, ok := after.(map[string]any); !ok {
		t.Fatalf("after should stay embedded JSON, got %T", after)
	}
}
