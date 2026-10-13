package connector

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/importflow"
)

func TestDesensitizerPreservesOnlyAllowedNullColumns(t *testing.T) {
	plan := MaskingPlan{
		Columns: []ColumnRule{
			{Table: "users", Column: "id", Action: ActionKeep},
			{Table: "users", Column: "name", Action: ActionMask, PiiKind: PiiName},
			{Table: "users", Column: "secret", Action: ActionDrop},
		},
		TextScan: []TextScanRule{{Table: "users", Column: "notes"}},
	}
	plan.Sign("tester", time.Unix(1, 0))
	d, err := NewDesensitizer(plan, DesensitizerOptions{})
	if err != nil {
		t.Fatal(err)
	}
	row := importflow.Record{Table: "users", Values: map[string]string{"id": "1"}, Nulls: map[string]bool{
		"name": true, "notes": true, "secret": true, "unlisted": true, "non_null": false,
	}}

	got, err := d.Apply(context.Background(), row)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Nulls, map[string]bool{"name": true, "notes": true}) {
		t.Fatalf("NULL columns = %#v; want allowed name/notes without dropped or unlisted fields", got.Nulls)
	}
	if !reflect.DeepEqual(got.Values, map[string]string{"id": "1"}) {
		t.Fatalf("values = %#v", got.Values)
	}
	if len(row.Nulls) != 5 {
		t.Fatal("input NULL map was modified")
	}
}
