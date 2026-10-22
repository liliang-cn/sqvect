package observability

import "testing"

// A histogram's buckets are part of its shape.
//
// declare exists to refuse a redeclaration that is not identical, and says
// why: the alternatives "produce a scrape that silently reports the wrong
// shape", invisible "until a dashboard has been lying for a month". It
// compared the kind and the label names and ignored the bounds — so a second
// NewHistogram with the same name and different buckets quietly returned the
// first family, and every Observe against the handle the caller believed it
// had just declared landed in somebody else's buckets.
//
// It is the one declaration that cannot be corrected afterwards.
// DefaultLatencyBuckets says so in the file above: "Buckets are the one
// histogram decision that cannot be changed later without discarding recorded
// history."
func TestRedeclaringAHistogramWithDifferentBucketsIsRefused(t *testing.T) {
	for _, tc := range []struct {
		name   string
		first  []float64
		second []float64
		want   bool // want a panic
	}{
		{"different bounds", []float64{0.01, 0.1}, []float64{1, 10}, true},
		{"an extra bound", []float64{0.01, 0.1}, []float64{0.01, 0.1, 1}, true},
		{"explicit default against nil", nil, DefaultLatencyBuckets, false},
		{"nil against explicit default", DefaultLatencyBuckets, nil, false},
		{"the same bounds unsorted and duplicated", []float64{0.01, 0.1}, []float64{0.1, 0.01, 0.1}, false},
		{"identical", []float64{0.01, 0.1}, []float64{0.01, 0.1}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := NewRegistry()
			first := r.NewHistogram("cortexdb_latency_seconds", "Handler latency.", tc.first, "method")

			panicked := func() (p bool) {
				defer func() { p = recover() != nil }()
				second := r.NewHistogram("cortexdb_latency_seconds", "Handler latency.", tc.second, "method")
				if second.f != first.f {
					t.Error("an accepted redeclaration should return the same family")
				}
				return
			}()

			if panicked != tc.want {
				if tc.want {
					t.Fatalf("declaring buckets %v over %v was accepted; observations would land "+
						"in the first family's buckets %v", tc.second, tc.first, first.f.bounds)
				}
				t.Fatalf("declaring buckets %v over %v panicked, but they normalise to the same bounds",
					tc.second, tc.first)
			}
		})
	}
}
