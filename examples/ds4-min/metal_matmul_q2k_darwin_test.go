//go:build darwin && arm64

package main

import (
	"context"
	"fmt"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestMetalMatmulQ2KPreservedBitIdentity(t *testing.T) {
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

	mm, err := newMetalMatmulPreservedQ2K(ops)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	const cols = 2048 // Typical Q2_K projection dimension (e.g. expert down/up/gate)
	const rows = 16   // Multiple output rows
	const blocks = cols / 256

	// Synthetic Q2_K weights: 84 bytes per 256-element block.
	// We generate pseudo-random bits so all scales and quants have dynamic range.
	weightBytes := metalmem.AllocIn[byte](arena, "q2k-weight", rows*blocks*84)
	rng := uint32(0x12345678)
	next := func() uint32 {
		rng = rng*1664525 + 1013904223
		return rng
	}
	wSlice := weightBytes.HostSlice()
	for i := range wSlice {
		wSlice[i] = byte(next())
	}
	// For each row & block, ensure scale halfs at +80 and +82 are valid non-zero normal floats
	for r := 0; r < rows; r++ {
		for b := 0; b < blocks; b++ {
			p := (r*blocks + b) * 84
			d := floatToF16Bits(float32((next()%13)+1) * 0.125)
			dm := floatToF16Bits(float32((next()%11)+1) * 0.0625)
			wSlice[p+80] = byte(d)
			wSlice[p+81] = byte(d >> 8)
			wSlice[p+82] = byte(dm)
			wSlice[p+83] = byte(dm >> 8)
		}
	}

	weightTensor := boundMetalTensor{
		name:       "test.q2k",
		typeID:     10,
		dimensions: []uint64{cols, rows},
		data:       weightBytes,
	}

	testTokens := []int{1, 3, 7, 8, 13, 32}

	for _, T := range testTokens {
		t.Run(fmt.Sprintf("T=%d", T), func(t *testing.T) {
			activations := make([]float32, cols*T)
			for i := range activations {
				r := next()
				activations[i] = float32(int32(r>>12)-524288) / float32(int32(1)<<(4+r%9))
			}

			// 1. Run sequential reference ops.MatvecQ2K for each token
			wantOut := make([]float32, rows*T)
			for tok := 0; tok < T; tok++ {
				tokIn := activations[tok*cols : (tok+1)*cols]
				res, err := ops.MatvecQ2K(weightTensor, tokIn)
				if err != nil {
					t.Fatalf("ops.MatvecQ2K token %d failed: %v", tok, err)
				}
				if len(res) != rows {
					t.Fatalf("ops.MatvecQ2K token %d returned %d rows, want %d", tok, len(res), rows)
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
			// The whole point of batching: one command buffer whatever T is.
			// A per-token buffer count is the 178,183-against-5 gap itself.
			if got := ops.submissions.sub(before).Buffers; got != 1 {
				t.Errorf("batched matmul over %d tokens used %d command buffers, want 1", T, got)
			}
			requireOrderSensitiveQ2K(t, mm, weightTensor, activations, T, gotOut)

			// 2b. Run encoded batched matmul: exactly 0 command buffers on encode, bit-identical to standalone.
			// Note: Encoded-vs-standalone bit identity compares two arms that call the same encode* body.
			// A defect in that body moves both arms equally and this check stays green. Its teeth are on
			// wrapper, allocation and pipeline divergence only. What covers the shared body is the reference
			// comparison and the reversed-twin machinery above.
			inBuf, err := metalmem.AllocFromSlice(dev, activations)
			if err != nil {
				t.Fatalf("inBuf alloc failed: %v", err)
			}
			defer inBuf.Free()
			q8Scratch, err := metalmem.Alloc[byte](dev, T*blocks*292)
			if err != nil {
				t.Fatalf("q8Scratch alloc failed: %v", err)
			}
			defer q8Scratch.Free()
			outBuf, err := metalmem.Alloc[float32](dev, rows*T)
			if err != nil {
				t.Fatalf("outBuf alloc failed: %v", err)
			}
			defer outBuf.Free()

			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatalf("newMetalBatch failed: %v", err)
			}
			beforeEnc := ops.submissions
			if err := batch.EncodeMatmulQ2K(weightTensor, inBuf, outBuf, q8Scratch, T); err != nil {
				t.Fatalf("batch.EncodeMatmulQ2K failed: %v", err)
			}
			if got := ops.submissions.sub(beforeEnc).Buffers; got != 0 {
				t.Errorf("EncodeMatmulQ2K used %d command buffers, want 0", got)
			}
			if err := batch.Wait(context.Background()); err != nil {
				t.Fatalf("batch.Wait failed: %v", err)
			}
			encOut := outBuf.HostSlice()
			if len(encOut) != rows*T {
				t.Fatalf("encoded matmul returned %d elements, want %d", len(encOut), rows*T)
			}
			for i := range gotOut {
				if math.Float32bits(encOut[i]) != math.Float32bits(gotOut[i]) {
					t.Fatalf("encoded mismatch with standalone at index %d: got 0x%08x (%g), want 0x%08x (%g)",
						i, math.Float32bits(encOut[i]), encOut[i], math.Float32bits(gotOut[i]), gotOut[i])
				}
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
}

// requireOrderSensitiveQ2K fails unless reassociating the block accumulation
// changes the result on this test's own data.
//
// Without it the bit-identity assertion is fail-open. The F16 version of this
// kernel shipped a test whose weights and activations made every partial sum
// exact in float32; exact sums are order-independent, so the test passed with
// the k loop reversed and could not see the one property it exists to check.
// Here the twin is a second specialization of the same kernel rather than a
// source edit, so the check is permanent instead of a mutation someone has to
// remember to re-run.
func requireOrderSensitiveQ2K(t *testing.T, mm *metalMatmulPreservedQ2K, w boundMetalTensor, in []float32, tokens int, forward []float32) {
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
			return
		}
	}
	t.Fatalf("test inputs are order-insensitive: reversing the block accumulation leaves all %d outputs bit-identical, so the bit-identity assertion cannot detect reassociation", len(forward))
}
