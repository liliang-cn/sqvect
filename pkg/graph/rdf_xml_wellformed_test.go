package graph

import (
	"context"
	"encoding/xml"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestRDFXMLRequiresMatchingAndClosedElements(t *testing.T) {
	const head = `<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:ex="http://example.org/">`
	const node = `<rdf:Description rdf:about="http://example.org/s"><ex:p>value</ex:p></rdf:Description>`
	cases := []struct{ name, doc string }{
		{"wrong root end", head + node + `</ex:wrong>`},
		{"wrong property end", head + `<rdf:Description rdf:about="http://example.org/s"><ex:p>value</ex:q></rdf:Description></rdf:RDF>`},
		{"unfinished root", head + node},
		{"unfinished property", head + `<rdf:Description rdf:about="http://example.org/s"><ex:p>value`},
		{"extra end", head + node + `</rdf:RDF></rdf:RDF>`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// Token validates XML nesting independently of CortexDB's RawToken tree.
			d := xml.NewDecoder(strings.NewReader(c.doc))
			var oracleErr error
			for oracleErr == nil {
				_, oracleErr = d.Token()
			}
			if errors.Is(oracleErr, io.EOF) {
				t.Fatal("fixture is well formed")
			}
			_, g, cleanup := setupTestGraph(t)
			defer cleanup()
			n, err := g.ImportRDF(context.Background(), strings.NewReader(c.doc), RDFFormatRDFXML)
			if err == nil {
				t.Fatalf("imported %d triples from malformed XML", n)
			}
			triples, err := g.FindTriples(context.Background(), TriplePattern{})
			if err != nil {
				t.Fatal(err)
			}
			if len(triples) != 0 {
				t.Fatalf("failed import stored %d triples", len(triples))
			}
		})
	}
}
