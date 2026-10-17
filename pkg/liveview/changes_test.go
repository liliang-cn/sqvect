package liveview

import (
	"context"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

func TestGraphChangesReadsTheFeed(t *testing.T) {
	ctx := context.Background()
	pages := map[int64]cortexdb.ChangesSinceResponse{
		// Memories recalled fill the log: no node or edge, more to read.
		10:   {Events: []graph.ChangeEvent{}, NextCursor: 1010, Head: 1500, More: true},
		1010: {Events: []graph.ChangeEvent{}, NextCursor: 1500, Head: 1500},
		// A node changed.
		1500: {Events: []graph.ChangeEvent{{Seq: 1501, Kind: graph.ChangeKindNode}}, NextCursor: 1501, Head: 1502},
		// Retention dropped what the view had not read.
		2: {Pruned: true, PrunedThrough: 40, NextCursor: 40, Head: 90},
	}
	var kinds []string
	changes := graphChanges(func(_ context.Context, req cortexdb.ChangesSinceRequest) (*cortexdb.ChangesSinceResponse, error) {
		if req.After < 0 {
			return &cortexdb.ChangesSinceResponse{NextCursor: 10, Head: 10}, nil
		}
		kinds = req.Kinds
		r := pages[req.After]
		return &r, nil
	})

	if c, changed, err := changes(ctx, -1); err != nil || changed || c != 10 {
		t.Fatalf("head: cursor %d changed %v err %v, want 10 false nil", c, changed, err)
	}
	if c, changed, _ := changes(ctx, 10); changed || c != 1500 {
		t.Fatalf("other kinds only: cursor %d changed %v, want 1500 false", c, changed)
	}
	if len(kinds) != 2 || kinds[0] != graph.ChangeKindNode || kinds[1] != graph.ChangeKindEdge {
		t.Fatalf("asked for kinds %v, want node and edge", kinds)
	}
	if c, changed, _ := changes(ctx, 1500); !changed || c != 1502 {
		t.Fatalf("a node changed: cursor %d changed %v, want 1502 true", c, changed)
	}
	if c, changed, _ := changes(ctx, 2); !changed || c != 90 {
		t.Fatalf("pruned: cursor %d changed %v, want 90 true", c, changed)
	}
}
