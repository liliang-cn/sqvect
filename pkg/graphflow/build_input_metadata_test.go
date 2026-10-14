package graphflow

import (
	"context"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

func TestBuildDoesNotMutateExtractionMetadataWhileMerging(t *testing.T) {
	db, err := cortexdb.Open(cortexdb.DefaultConfig(filepath.Join(t.TempDir(), "graph.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx := context.Background()
	extractions := []ExtractionResult{
		{SourceID: "first", Nodes: []ExtractionNode{{ID: "shared", Label: "Shared", Metadata: map[string]string{"first": "1"}}}},
		{SourceID: "second", Nodes: []ExtractionNode{{ID: "shared", Label: "Shared", Metadata: map[string]string{"second": "2"}}}},
	}
	if _, err := Build(ctx, db, extractions, BuildOptions{}); err != nil {
		t.Fatal(err)
	}
	for i, want := range []map[string]string{{"first": "1"}, {"second": "2"}} {
		if got := extractions[i].Nodes[0].Metadata; !reflect.DeepEqual(got, want) {
			t.Errorf("extraction %d metadata mutated: %#v, want %#v", i, got, want)
		}
	}
	node, err := db.Graph().GetNode(ctx, "graphflow:node:shared")
	if err != nil {
		t.Fatal(err)
	}
	if node.Properties["meta_first"] != "1" || node.Properties["meta_second"] != "2" {
		t.Fatalf("merged metadata was not persisted: %#v", node.Properties)
	}
}
