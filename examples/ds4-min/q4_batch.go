package main

import (
	"fmt"
	"math"
)

// q4BatchShape checks the byte and value counts before device slices or
// launch dimensions are formed. Q8_K packs 256 values into 292 bytes.
func q4BatchShape(cols, rows uint64, pairs, slots, inputBytes, outputValues int) error {
	if cols == 0 || cols%256 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 ||
		pairs <= 0 || pairs > math.MaxInt32 || slots <= 0 || pairs%slots != 0 || inputBytes < 0 || outputValues < 0 ||
		uint64(inputBytes) != cols/256*292*uint64(pairs/slots) || uint64(outputValues) != rows*uint64(pairs) {
		return fmt.Errorf("cuda: Q4_K gather: invalid buffer shape")
	}
	return nil
}

type q4BatchCapacity struct {
	norm, input, gate, up, mid, quantMid, down, ids, weights, pairs, profile int
}

func checkQ4BatchCapacity(tokens int, c q4BatchCapacity, profile bool) error {
	if tokens <= 0 || tokens > math.MaxInt/(6*4096) || tokens > math.MaxInt32/6 {
		return fmt.Errorf("cuda: Q4_K batch: invalid token count")
	}
	for _, b := range []struct {
		name       string
		have, want int
	}{
		{"norm", c.norm, tokens * 4096},
		{"input", c.input, tokens * 4096 / 256 * 292},
		{"gate", c.gate, tokens * 6 * 2048},
		{"up", c.up, tokens * 6 * 2048},
		{"mid", c.mid, tokens * 6 * 2048},
		{"quantized mid", c.quantMid, tokens * 6 * 2048 / 256 * 292},
		{"down", c.down, tokens * 6 * 4096},
		{"ids", c.ids, tokens * 6},
		{"weights", c.weights, tokens * 6},
		{"pairs", c.pairs, tokens * 6},
	} {
		if b.have < b.want {
			return fmt.Errorf("cuda: Q4_K batch: %s scratch truncated", b.name)
		}
	}
	if profile && c.profile < tokens*6*2048 {
		return fmt.Errorf("cuda: Q4_K batch: profile scratch truncated")
	}
	return nil
}
