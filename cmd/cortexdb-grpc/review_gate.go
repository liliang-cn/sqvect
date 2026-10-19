package main

import (
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// reviewGateFromEnv reads CORTEXDB_REVIEW_GATE. Off unless asked: a brain
// that never reviews anything searches as before. A value that is not a
// boolean is an error, not a silent off.
func reviewGateFromEnv() (bool, error) {
	v := os.Getenv("CORTEXDB_REVIEW_GATE")
	if v == "" {
		return false, nil
	}
	on, err := strconv.ParseBool(v)
	if err != nil {
		return false, fmt.Errorf("invalid CORTEXDB_REVIEW_GATE %q: want 1 or 0", v)
	}
	return on, nil
}

// appendReviewGate adds WithReviewGate to opts when CORTEXDB_REVIEW_GATE asks
// for it, and says so in the log.
func appendReviewGate(opts []cortexdb.Option) ([]cortexdb.Option, error) {
	on, err := reviewGateFromEnv()
	if err != nil {
		return nil, err
	}
	if on {
		opts = append(opts, cortexdb.WithReviewGate())
		log.Printf("review gate: on (proposed and rejected rows are left out of search)")
	}
	return opts, nil
}
