package cortexdb

import (
	"math"
	"testing"
)

// The pool starts at top_k*8 and doubles; past maxChunkWalkPool, overflow
// included, it is no limit (0), so a widened walk is the last one.
func TestTheChunkWalkPoolHasNoLimitPastItsCap(t *testing.T) {
	for _, c := range []struct{ got, want int }{
		{chunkWalkPool(1), 8},
		{chunkWalkPool(maxChunkWalkPool / 8), maxChunkWalkPool},
		{chunkWalkPool(maxChunkWalkPool/8 + 1), 0},
		{chunkWalkPool(1 << 61), 0},
		{chunkWalkPool(math.MaxInt), 0},
		{widerChunkWalkPool(8), 16},
		{widerChunkWalkPool(maxChunkWalkPool / 2), maxChunkWalkPool},
		{widerChunkWalkPool(maxChunkWalkPool/2 + 1), 0},
	} {
		if c.got != c.want {
			t.Errorf("pool %d, want %d", c.got, c.want)
		}
	}
}
