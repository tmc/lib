//go:build darwin && arm64

package main

import (
	"fmt"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestMetalMatmulQ80PreservedBitIdentity(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()

	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	mm, err := newMetalMatmulPreservedQ80(ops)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	// cols and rows are swept because the kernel's block stride is
	// simdgroups*8 = 32 blocks and its row stride is 2. At cols=2048 the 64
	// blocks are exactly two strides and at rows=16 the rows divide evenly,
	// so those alone exercise neither the block tail, the sub-stride case
	// where a simdgroup owns no block at all, nor the odd final row that
	// "if (first_row + r >= rows) continue" exists for.
	var ordered, blind []string
	for _, shape := range []struct{ cols, rows int }{
		{2048, 16}, // even strides both ways
		{1536, 16}, // 48 blocks: one and a half block strides
		{512, 16},  // 16 blocks: fewer than one stride, most simdgroups idle
		{2048, 17}, // odd rows: the last threadgroup carries one row
		{2048, 1},  // a single row
		{1536, 3},  // both tails at once
	} {
		t.Run(fmt.Sprintf("cols=%d/rows=%d", shape.cols, shape.rows), func(t *testing.T) {
			if checkMatmulQ80Shape(t, ops, mm, arena, shape.cols, shape.rows) {
				ordered = append(ordered, fmt.Sprintf("%dx%d", shape.cols, shape.rows))
			} else {
				blind = append(blind, fmt.Sprintf("%dx%d", shape.cols, shape.rows))
			}
		})
	}
	// Bit-identity is asserted at every shape above. Order sensitivity is not
	// available at every shape, and asserting it where it cannot hold is how
	// the F16 test shipped fail-open in the other direction. At cols=512 the
	// 16 blocks are fewer than the 32-block stride, so every active thread
	// accumulates exactly one block and reversal only permutes blocks across
	// lanes -- and reversing 0..31 is XOR with 31, which a butterfly simd_sum
	// pairs identically, so the cross-lane sum does not move either. Measured:
	// forward and reversed agree on all 1024 outputs there. That shape still
	// exercises the sub-stride case; it just cannot testify about order. The
	// suite-level requirement is that at least one shape can, which keeps the
	// whole sweep red if the reversed twin ever stops reversing.
	if len(ordered) == 0 {
		t.Errorf("no shape in the sweep could detect reassociation; the order-preservation claim is untested at every one of %v", blind)
	}
	if len(blind) > 0 {
		t.Logf("order sensitivity established at %v; not available at %v", ordered, blind)
	}
}

// checkMatmulQ80Shape asserts bit-identity against the per-token matvec at one
// shape and reports whether reversing the block accumulation changed anything,
// which is what says the assertion could have failed.
func checkMatmulQ80Shape(t *testing.T, ops *metalPinnedOps, mm *metalMatmulPreservedQ80, arena *metalmem.Arena, cols, rows int) bool {
	t.Helper()
	orderSensitive := false
	blocks := cols / 32

	// Synthetic Q8_0 weights: 34 bytes per 32-element block.
	// We generate pseudo-random bits so all scales and quants have dynamic range.
	weightBytes := metalmem.AllocIn[byte](arena, fmt.Sprintf("q80-weight-%dx%d", cols, rows), rows*blocks*34)
	rng := uint32(0x12345678)
	next := func() uint32 {
		rng = rng*1664525 + 1013904223
		return rng
	}
	wSlice := weightBytes.HostSlice()
	for i := range wSlice {
		wSlice[i] = byte(next())
	}
	// For each row & block, ensure scale half at +0 is a valid non-zero normal float
	for r := 0; r < rows; r++ {
		for b := 0; b < blocks; b++ {
			p := (r*blocks + b) * 34
			d := floatToF16Bits(float32((next()%13)+1) * 0.125)
			wSlice[p+0] = byte(d)
			wSlice[p+1] = byte(d >> 8)
		}
	}

	weightTensor := boundMetalTensor{
		name:       "test.q80",
		typeID:     8,
		dimensions: []uint64{uint64(cols), uint64(rows)},
		data:       weightBytes,
	}

	testTokens := []int{1, 3, 7, 8, 13, 32, 64}

	for _, T := range testTokens {
		t.Run(fmt.Sprintf("T=%d", T), func(t *testing.T) {
			activations := make([]float32, cols*T)
			for i := range activations {
				r := next()
				activations[i] = float32(int32(r>>12)-524288) / float32(int32(1)<<(4+r%9))
			}

			// 1. Run sequential reference ops.MatvecQ80 for each token
			wantOut := make([]float32, rows*T)
			for tok := 0; tok < T; tok++ {
				tokIn := activations[tok*cols : (tok+1)*cols]
				res, err := ops.MatvecQ80(weightTensor, tokIn)
				if err != nil {
					t.Fatalf("ops.MatvecQ80 token %d failed: %v", tok, err)
				}
				if len(res) != rows {
					t.Fatalf("ops.MatvecQ80 token %d returned %d rows, want %d", tok, len(res), rows)
				}
				copy(wantOut[tok*rows:(tok+1)*rows], res)
			}

			// 2. Run batched matmul for all T tokens, counting its buffers.
			before := ops.submissions
			gotOut, err := mm.Mul(weightTensor, activations, T)
			if err != nil {
				t.Fatalf("mm.Mul(T=%d) failed: %v", T, err)
			}
			if len(gotOut) != rows*T {
				t.Fatalf("mm.Mul returned %d elements, want %d", len(gotOut), rows*T)
			}
			// One command buffer whatever T is.
			if got := ops.submissions.sub(before).Buffers; got != 1 {
				t.Errorf("batched matmul over %d tokens used %d command buffers, want 1", T, got)
			}
			if orderSensitiveQ80(t, mm, weightTensor, activations, T, gotOut) {
				orderSensitive = true
			}

			// 3. Bit-exact float32 comparison across all elements using math.Float32bits
			for tok := 0; tok < T; tok++ {
				for r := 0; r < rows; r++ {
					idx := tok*rows + r
					gotBits := math.Float32bits(gotOut[idx])
					wantBits := math.Float32bits(wantOut[idx])
					if gotBits != wantBits {
						t.Fatalf("bit mismatch at token %d row %d: got 0x%08x (%g), want 0x%08x (%g)",
							tok, r, gotBits, gotOut[idx], wantBits, wantOut[idx])
					}
				}
			}
		})
	}
	return orderSensitive
}

// orderSensitiveQ80 reports whether reassociating the block accumulation changes
// the result on this test's own data. When it does, the bit-identity assertion
// above could have failed, so passing it means something. Some shapes are not
// sensitive at all; see the caller, which requires at least one that is.
func orderSensitiveQ80(t *testing.T, mm *metalMatmulPreservedQ80, w boundMetalTensor, in []float32, tokens int, forward []float32) bool {
	t.Helper()
	reversed, err := mm.mulReversed(w, in, tokens)
	if err != nil {
		t.Fatalf("reversed twin: %v", err)
	}
	if len(reversed) != len(forward) {
		t.Fatalf("reversed twin returned %d elements, want %d", len(reversed), len(forward))
	}
	for i := range forward {
		if math.Float32bits(forward[i]) != math.Float32bits(reversed[i]) {
			return true
		}
	}
	return false
}
