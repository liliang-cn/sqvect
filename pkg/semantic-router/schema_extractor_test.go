package semanticrouter

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type schemaExtractorFixture struct {
	params  map[string]interface{}
	missing []string
	err     error
	calls   int
}

func (e *schemaExtractorFixture) Extract(_ context.Context, _ string, _ *FunctionSchema) (map[string]interface{}, []string, error) {
	e.calls++
	return e.params, e.missing, e.err
}

func TestFunctionRouterUsesPerSchemaExtractor(t *testing.T) {
	sentinel := errors.New("custom extraction failed")
	for _, tc := range []struct {
		name    string
		missing []string
		err     error
	}{
		{"complete", nil, nil}, {"incomplete", []string{"required"}, nil}, {"error", nil, sentinel},
	} {
		t.Run(tc.name, func(t *testing.T) {
			extractor := &schemaExtractorFixture{params: map[string]interface{}{"answer": "CustomValue"}, missing: tc.missing, err: tc.err}
			base, _ := NewRouter(NewMockEmbedder(8))
			route := CreateRouteWithFunction("custom", []string{"custom query"}, &FunctionSchema{Name: "custom", Extractor: extractor})
			if err := base.Add(route.Route); err != nil {
				t.Fatal(err)
			}
			router := NewFunctionRouter(base)
			if err := router.Add(route); err != nil {
				t.Fatal(err)
			}
			got, err := router.Route(context.Background(), "custom query")
			if extractor.calls != 1 {
				t.Fatalf("custom extractor called %d times", extractor.calls)
			}
			if tc.err != nil {
				if !errors.Is(err, tc.err) {
					t.Fatalf("got %v want wrapped %v", err, tc.err)
				}
				return
			}
			if err != nil || got == nil {
				t.Fatalf("got %+v err %v", got, err)
			}
			if !reflect.DeepEqual(got.Arguments, extractor.params) || !reflect.DeepEqual(got.Missing, tc.missing) || got.Complete != (len(tc.missing) == 0) {
				t.Fatalf("custom extraction ignored: %+v", got)
			}
		})
	}
}
