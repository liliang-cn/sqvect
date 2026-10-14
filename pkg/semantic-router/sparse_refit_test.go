package semanticrouter

import (
	"context"
	"reflect"
	"testing"
)

type fittedSparseEncoder interface {
	SparseEncoder
	Fit(context.Context, []string) error
}

func TestSparseEncoderRefitReplacesCorpusState(t *testing.T) {
	for name, makeEncoder := range map[string]func() fittedSparseEncoder{
		"BM25":  func() fittedSparseEncoder { return NewBM25Encoder() },
		"TFIDF": func() fittedSparseEncoder { return NewTFIDFEncoder() },
	} {
		t.Run(name, func(t *testing.T) {
			encoder := makeEncoder()
			if err := encoder.Fit(context.Background(), []string{"obsolete obsolete", "obsolete", "retired"}); err != nil {
				t.Fatal(err)
			}
			fresh := makeEncoder()
			corpus := []string{"alpha beta", "beta gamma"}
			for _, e := range []fittedSparseEncoder{encoder, fresh} {
				if err := e.Fit(context.Background(), corpus); err != nil {
					t.Fatal(err)
				}
			}
			if encoder.Dimensions() != fresh.Dimensions() || !reflect.DeepEqual(encoder.Vocabulary(), fresh.Vocabulary()) {
				t.Errorf("refit vocabulary %v differs from fresh %v", encoder.Vocabulary(), fresh.Vocabulary())
			}
			if got, want := encoder.EncodeSparse("obsolete alpha"), fresh.EncodeSparse("obsolete alpha"); !reflect.DeepEqual(got, want) {
				t.Errorf("refit encoding %v differs from fresh %v", got, want)
			}
		})
	}
}
