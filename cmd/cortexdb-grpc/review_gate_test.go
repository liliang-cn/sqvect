package main

import (
	"strings"
	"testing"
)

func TestTheReviewGateIsReadFromTheEnvironment(t *testing.T) {
	for _, tc := range []struct {
		value string
		on    bool
	}{{"", false}, {"0", false}, {"false", false}, {"1", true}, {"true", true}} {
		t.Setenv("CORTEXDB_REVIEW_GATE", tc.value)
		on, err := reviewGateFromEnv()
		if err != nil || on != tc.on {
			t.Fatalf("CORTEXDB_REVIEW_GATE=%q: on=%v err=%v, want on=%v", tc.value, on, err, tc.on)
		}
	}
	t.Setenv("CORTEXDB_REVIEW_GATE", "maybe")
	if _, err := reviewGateFromEnv(); err == nil || !strings.Contains(err.Error(), "CORTEXDB_REVIEW_GATE") {
		t.Fatalf("a bad value was accepted: %v", err)
	}
}

func TestTheReviewGateOptionFollowsTheEnvironment(t *testing.T) {
	t.Setenv("CORTEXDB_REVIEW_GATE", "")
	if opts, err := appendReviewGate(nil); err != nil || len(opts) != 0 {
		t.Fatalf("no gate asked: %d options, %v", len(opts), err)
	}
	t.Setenv("CORTEXDB_REVIEW_GATE", "1")
	if opts, err := appendReviewGate(nil); err != nil || len(opts) != 1 {
		t.Fatalf("gate asked: %d options, %v", len(opts), err)
	}
	t.Setenv("CORTEXDB_REVIEW_GATE", "maybe")
	if opts, err := appendReviewGate(nil); err == nil || opts != nil {
		t.Fatalf("a bad value: %v options, %v", opts, err)
	}
}
