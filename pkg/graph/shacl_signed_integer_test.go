package graph

import (
	"context"
	"testing"
)

func TestSHACLChecksNegativeIntegerValueSpaces(t *testing.T) {
	_, g, cleanup := setupTestGraph(t)
	defer cleanup()
	cases := []struct {
		datatype, lexical string
		conforms          bool
	}{
		{"negativeInteger", "-1", true}, {"negativeInteger", "-00012", true},
		{"negativeInteger", "-999999999999999999999999999999", true},
		{"negativeInteger", "0", false}, {"negativeInteger", "-0", false},
		{"negativeInteger", "1", false}, {"negativeInteger", "nope", false},
		{"nonPositiveInteger", "0", true}, {"nonPositiveInteger", "-0", true},
		{"nonPositiveInteger", "+000", true}, {"nonPositiveInteger", "-12", true},
		{"nonPositiveInteger", "1", false}, {"nonPositiveInteger", "nope", false},
	}
	for _, c := range cases {
		t.Run(c.datatype+"/"+c.lexical, func(t *testing.T) {
			shapes := nodeShape("SignedIntegerShape",
				tri(ex("SignedIntegerShape"), sh("targetNode"), NewTypedLiteral(c.lexical, xsd(c.datatype))),
				tri(ex("SignedIntegerShape"), sh("datatype"), NewIRI(xsd(c.datatype))))
			report, err := g.ValidateSHACL(context.Background(), shapes)
			if err != nil {
				t.Fatal(err)
			}
			if report.Conforms != c.conforms {
				t.Fatalf("conforms=%v, want %v: %+v", report.Conforms, c.conforms, report.Results)
			}
			if !c.conforms && (len(report.Results) != 1 || report.Results[0].Component != SHACLDatatypeConstraintComponent) {
				t.Fatalf("wrong violation: %+v", report.Results)
			}
		})
	}
}
