package core

import (
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"io"
	"path/filepath"
	"testing"
)

type failingCSVWriter struct{}

func (failingCSVWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestCSVDumpReportsBufferedWriteFailure(t *testing.T) {
	for _, insert := range []bool{false, true} {
		t.Run(map[bool]string{false: "header only", true: "with row"}[insert], func(t *testing.T) {
			store, err := New(filepath.Join(t.TempDir(), "dump.db"), 2)
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			ctx := context.Background()
			if err := store.Init(ctx); err != nil {
				t.Fatal(err)
			}
			if insert {
				if err := store.Upsert(ctx, &Embedding{ID: "row", Vector: []float32{1, 0}, Content: "content"}); err != nil {
					t.Fatal(err)
				}
			}
			opts := DefaultDumpOptions()
			opts.Format = DumpFormatCSV
			if _, err := store.Dump(ctx, failingCSVWriter{}, opts); !errors.Is(err, io.ErrClosedPipe) {
				t.Fatalf("got %v, want buffered writer error", err)
			}
			var buf bytes.Buffer
			if _, err := store.Dump(ctx, &buf, opts); err != nil {
				t.Fatal(err)
			}
			rows, err := csv.NewReader(&buf).ReadAll()
			wantRows := 1
			if insert {
				wantRows++
			}
			if err != nil || len(rows) != wantRows {
				t.Fatalf("successful dump: rows=%v err=%v", rows, err)
			}
		})
	}
}
