package semanticrouter

import (
	"context"
	"testing"
)

func TestRegexExtractorPreservesValuesAndLiteralKeywords(t *testing.T) {
	for _, tc := range []struct{ key, text, want string }{
		{"city", "CITY NewYork", "NewYork"},
		{"token", "token AbC123", "AbC123"},
		{"code[", "code[ MiXeD", "MiXeD"},
		{"api.key", "api.key ABC", "ABC"},
	} {
		t.Run(tc.key, func(t *testing.T) {
			defer func() {
				if v := recover(); v != nil {
					t.Errorf("literal keyword caused panic: %v", v)
				}
			}()
			schema := &FunctionSchema{Name: "lookup", Parameters: map[string]*ParameterSchema{tc.key: {Type: "string", Required: true}}}
			params, missing, err := NewRegexExtractor().Extract(context.Background(), tc.text, schema)
			if err != nil || len(missing) != 0 || params[tc.key] != tc.want {
				t.Fatalf("params=%v missing=%v err=%v; want %q", params, missing, err, tc.want)
			}
		})
	}
}
