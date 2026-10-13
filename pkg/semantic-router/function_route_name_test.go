package semanticrouter

import (
	"context"
	"testing"
)

func TestFunctionRouterSupportsDistinctRouteAndFunctionNames(t *testing.T) {
	base, _ := NewRouter(NewMockEmbedder(8))
	router := NewFunctionRouter(base)
	for _, name := range []string{"lookup weather", "forecast today"} {
		schema := &FunctionSchema{Name: "get_weather", Handler: func(context.Context, map[string]interface{}) (interface{}, error) { return "weather", nil }}
		route := CreateRouteWithFunction(name, []string{name}, schema)
		if err := base.Add(route.Route); err != nil {
			t.Fatal(err)
		}
		if err := router.Add(route); err != nil {
			t.Fatal(err)
		}
	}
	for _, query := range []string{"lookup weather", "forecast today"} {
		call, err := router.Route(context.Background(), query)
		if err != nil || call == nil {
			t.Fatalf("route %q: call=%+v err=%v", query, call, err)
		}
		if call.Name != "get_weather" {
			t.Fatalf("wrong function name: %q", call.Name)
		}
		result, err := call.Execute(context.Background())
		if err != nil || result != "weather" {
			t.Fatalf("execute: %v %v", result, err)
		}
	}
}
