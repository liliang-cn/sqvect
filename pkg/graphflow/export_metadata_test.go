package graphflow

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

func TestExportPreservesExtractionMetadata(t *testing.T) {
	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(t.TempDir(), "graph.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx := context.Background()
	nodeMetadata := map[string]string{"owner": "Ada", "empty": "", "meta_nested": "value"}
	edgeMetadata := map[string]string{"evidence": "document:2", "empty": ""}
	extraction := ExtractionResult{
		SourceID: "document",
		Nodes: []ExtractionNode{
			{ID: "alice", Label: "Alice", Metadata: nodeMetadata},
			{ID: "apollo", Label: "Apollo"},
		},
		Edges: []ExtractionEdge{{Source: "alice", Target: "apollo", Relation: "owns", Confidence: ConfidenceExtracted, Metadata: edgeMetadata}},
	}
	if _, err := Build(ctx, db, []ExtractionResult{extraction}, BuildOptions{}); err != nil {
		t.Fatal(err)
	}
	result, err := Export(ctx, db, ExportRequest{OutputDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	payload, err := os.ReadFile(result.GraphJSON)
	if err != nil {
		t.Fatal(err)
	}
	var bundle exportBundle
	if err := json.Unmarshal(payload, &bundle); err != nil {
		t.Fatal(err)
	}
	for _, node := range bundle.Nodes {
		if node.ID == "alice" && !reflect.DeepEqual(node.Metadata, nodeMetadata) {
			t.Errorf("node metadata = %#v, want %#v", node.Metadata, nodeMetadata)
		}
		if node.ID == "apollo" && len(node.Metadata) != 0 {
			t.Errorf("internal properties leaked as metadata: %#v", node.Metadata)
		}
	}
	if len(bundle.Edges) != 1 || !reflect.DeepEqual(bundle.Edges[0].Metadata, edgeMetadata) {
		t.Fatalf("edge metadata lost: %+v", bundle.Edges)
	}
}
