package importflow

import (
	"context"
	"strconv"
	"strings"
	"testing"
)

func TestCSVSampleSizeDefaultsAndBounds(t *testing.T) {
	for _, tc := range []struct{ size, want int }{{-5, 5}, {-1, 5}, {0, 5}, {1, 1}, {2, 2}, {5, 5}, {99, 6}} {
		t.Run(strconv.Itoa(tc.size), func(t *testing.T) {
			defer func() {
				if value := recover(); value != nil {
					t.Fatalf("Schemas panicked for sample size %d: %v", tc.size, value)
				}
			}()
			src, err := NewCSVSource(strings.NewReader("id\n1\n2\n3\n4\n5\n6\n"), CSVOptions{SampleSize: tc.size})
			if err != nil {
				t.Fatal(err)
			}
			defer src.Close()
			schemas, err := src.Schemas(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if len(schemas) != 1 || len(schemas[0].Sample) != tc.want {
				t.Fatalf("schemas = %+v; want %d sample rows", schemas, tc.want)
			}
		})
	}
}
