package importflow

import (
	"context"
	"strings"
	"testing"
)

func TestCSVSourceRejectsAmbiguousHeaders(t *testing.T) {
	for _, header := range []string{"id,id", "id, id ", "id,", "id,   "} {
		t.Run(header, func(t *testing.T) {
			if _, err := NewCSVSource(strings.NewReader(header+"\n1,Ada\n"), CSVOptions{}); err == nil {
				t.Fatal("ambiguous header accepted; map keys would discard or misname a column")
			}
		})
	}
}

func TestCSVSourceKeepsDistinctTrimmedHeaders(t *testing.T) {
	src, err := NewCSVSource(strings.NewReader(" id , name \n1,Ada\n"), CSVOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if err := src.Records(context.Background(), func(r Record) error {
		if r.Values["id"] != "1" || r.Values["name"] != "Ada" || len(r.Values) != 2 {
			t.Fatalf("record changed: %#v", r.Values)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
