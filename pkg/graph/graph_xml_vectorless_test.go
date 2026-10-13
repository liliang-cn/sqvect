package graph

import (
	"bytes"
	"context"
	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"path/filepath"
	"testing"
)

func TestXMLGraphFormatsPreserveVectorlessNodes(t *testing.T) {
	for _, format := range []ExportFormat{FormatGraphML, FormatGEXF} {
		for _, dim := range []int{3, 4} {
			t.Run(string(format)+"/"+string(rune('0'+dim)), func(t *testing.T) {
				ctx := context.Background()
				source, err := core.New(filepath.Join(t.TempDir(), "source.db"), dim)
				if err != nil {
					t.Fatal(err)
				}
				defer source.Close()
				if err = source.Init(ctx); err != nil {
					t.Fatal(err)
				}
				src := NewGraphStore(source)
				if err = src.InitGraphSchema(ctx); err != nil {
					t.Fatal(err)
				}
				if err = src.UpsertNode(ctx, &GraphNode{ID: "plain", NodeType: "entity", Content: "Vectorless node"}); err != nil {
					t.Fatal(err)
				}
				vector := make([]float32, dim)
				vector[0] = 1
				if err = src.UpsertNode(ctx, &GraphNode{ID: "embedded", NodeType: "entity", Vector: vector}); err != nil {
					t.Fatal(err)
				}
				var buf bytes.Buffer
				if err = src.Export(ctx, &buf, format); err != nil {
					t.Fatal(err)
				}
				destination, err := core.New(filepath.Join(t.TempDir(), "destination.db"), dim)
				if err != nil {
					t.Fatal(err)
				}
				defer destination.Close()
				if err = destination.Init(ctx); err != nil {
					t.Fatal(err)
				}
				dst := NewGraphStore(destination)
				if err = dst.InitGraphSchema(ctx); err != nil {
					t.Fatal(err)
				}
				if err = dst.Import(ctx, &buf, format); err != nil {
					t.Fatal(err)
				}
				plain, err := dst.GetNode(ctx, "plain")
				if err != nil {
					t.Fatal(err)
				}
				if len(plain.Vector) != 0 {
					t.Fatalf("invented vector for a vectorless node: %v", plain.Vector)
				}
				embedded, err := dst.GetNode(ctx, "embedded")
				if err != nil {
					t.Fatal(err)
				}
				if len(embedded.Vector) != dim || embedded.Vector[0] != 1 {
					t.Fatalf("embedded vector changed: %v", embedded.Vector)
				}
			})
		}
	}
}

func TestXMLGraphImportsRejectMalformedVectorsBeforeWritingNodes(t *testing.T) {
	docs := map[ExportFormat]string{
		FormatGraphML: `<graphml xmlns="http://graphml.graphdrawing.org/xmlns"><key id="v" for="node" attr.name="vector" attr.type="string"/><graph id="G" edgedefault="directed"><node id="bad"><data key="v">not-json</data></node></graph></graphml>`,
		FormatGEXF:    `<gexf xmlns="http://www.gexf.net/1.3" version="1.3"><graph mode="static" defaultedgetype="directed"><attributes class="node"><attribute id="v" title="vector" type="string"/></attributes><nodes><node id="bad"><attvalues><attvalue for="v" value="not-json"/></attvalues></node></nodes></graph></gexf>`,
	}
	for format, doc := range docs {
		t.Run(string(format), func(t *testing.T) {
			_, g, cleanup := setupTestGraph(t)
			defer cleanup()
			if err := g.Import(context.Background(), bytes.NewBufferString(doc), format); err == nil {
				t.Fatal("invalid vector accepted")
			}
			nodes, err := g.GetAllNodes(context.Background(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(nodes) != 0 {
				t.Fatalf("malformed input wrote %d nodes", len(nodes))
			}
		})
	}
}
