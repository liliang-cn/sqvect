package semanticrouter

import (
	"context"
	"reflect"
	"testing"
)

func TestRegexExtractorHonorsSchemaRequiredList(t *testing.T) {
	schema, err := ParseFunctionSchema(`{"name":"weather","parameters":{"city":{"type":"string"}},"required":["city"]}`)
	if err != nil {
		t.Fatal(err)
	}
	params, missing, err := NewRegexExtractor().Extract(context.Background(), "forecast", schema)
	if err != nil || len(params) != 0 || !reflect.DeepEqual(missing, []string{"city"}) {
		t.Fatalf("params=%v missing=%v err=%v", params, missing, err)
	}
	params, missing, err = NewRegexExtractor().Extract(context.Background(), "city paris", schema)
	if err != nil || len(missing) != 0 || params["city"] != "paris" {
		t.Fatalf("provided city: params=%v missing=%v err=%v", params, missing, err)
	}
	schema.Parameters["city"].Default = "paris"
	params, missing, err = NewRegexExtractor().Extract(context.Background(), "forecast", schema)
	if err != nil || len(missing) != 0 || params["city"] != "paris" {
		t.Fatalf("default city: params=%v missing=%v err=%v", params, missing, err)
	}
}

func TestRequiredListPreventsIncompleteFunctionExecution(t *testing.T) {
	base, _ := NewRouter(NewMockEmbedder(8))
	invoked := false
	schema := &FunctionSchema{Name: "weather", Parameters: map[string]*ParameterSchema{"city": {Type: "string"}}, Required: []string{"city"}, Handler: func(context.Context, map[string]interface{}) (interface{}, error) { invoked = true; return nil, nil }}
	route := CreateRouteWithFunction("weather", []string{"forecast"}, schema)
	if err := base.Add(route.Route); err != nil {
		t.Fatal(err)
	}
	router := NewFunctionRouter(base)
	if err := router.Add(route); err != nil {
		t.Fatal(err)
	}
	call, err := router.Route(context.Background(), "forecast")
	if err != nil || call == nil {
		t.Fatalf("got %+v, %v", call, err)
	}
	if call.Complete {
		t.Error("call missing schema-required city was marked complete")
	}
	if _, err := call.Execute(context.Background()); err == nil || invoked {
		t.Fatalf("incomplete function executed: invoked=%v err=%v", invoked, err)
	}
}
