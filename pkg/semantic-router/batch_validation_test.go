package semanticrouter

import (
	"reflect"
	"testing"
)

func TestRouterAddBatchRejectsInvalidInputBeforeMutation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		routes []*Route
	}{
		{"nil route", []*Route{{Name: "new", Utterances: []string{"new"}}, nil}},
		{"duplicate inside batch", []*Route{{Name: "new", Utterances: []string{"new"}}, {Name: "new", Utterances: []string{"other"}}}},
		{"duplicate existing route", []*Route{{Name: "new", Utterances: []string{"new"}}, {Name: "existing", Utterances: []string{"other"}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, _ := NewRouter(NewMockEmbedder(8))
			if err := router.Add(&Route{Name: "existing", Utterances: []string{"existing"}}); err != nil {
				t.Fatal(err)
			}
			var err error
			func() {
				defer func() {
					if recovered := recover(); recovered != nil {
						t.Errorf("AddBatch panicked: %v", recovered)
					}
				}()
				err = router.AddBatch(tc.routes)
			}()
			if err == nil {
				t.Error("invalid batch must return an error")
			}
			if got := router.List(); !reflect.DeepEqual(got, []string{"existing"}) {
				t.Errorf("invalid batch mutated routes: %v", got)
			}
		})
	}
}

func TestRouterAddBatchAcceptsUniqueRoutes(t *testing.T) {
	router, _ := NewRouter(NewMockEmbedder(8))
	if err := router.AddBatch([]*Route{{Name: "one", Utterances: []string{"one"}}, {Name: "two", Utterances: []string{"two"}}}); err != nil {
		t.Fatal(err)
	}
	if got := router.List(); !reflect.DeepEqual(got, []string{"one", "two"}) {
		t.Fatalf("got %v", got)
	}
}
