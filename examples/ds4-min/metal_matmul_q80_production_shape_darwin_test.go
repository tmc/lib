//go:build darwin && arm64

package main

import (
	"fmt"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestMetalMatmulQ80AtProductionShapes runs the Q8_0 bit-identity check at the
// shapes the model actually holds.
//
// TestMetalMatmulQ80BitIdentity sweeps cols of 2048, 1536 and 512 with rows no
// greater than 17, chosen to exercise the block tail, the sub-stride case and
// the odd final row. None of them is a shape this model uses. The production
// Q8_0 weights are 4096x1024, 1024x32768, 4096x512, 4096x8192, 8192x4096,
// 4096x2048 and 2048x4096, so the dispatch geometry at 32768 output rows -- two
// thousand times the swept maximum -- had no coverage.
//
// That gap is the same shape as the one TestMetalMatmulF16AtRuntimeShape found:
// a green bit-identity assertion whose every shape misses the kernel or the
// geometry production runs. There the structural argument was wrong. Here it
// looks right -- matmul_q8_0 is a verbatim clone of matvec_q8_0 with a token
// index and a function constant that is false in every runtime pipeline -- but
// the F16 case is why that argument is measured rather than trusted.
//
// Model-free: the weights are synthetic, so this needs no GGUF and no model slot.
func TestMetalMatmulQ80AtProductionShapes(t *testing.T) {
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

	var ordered, blind []string
	for _, shape := range []struct {
		cols, rows int
		tensor     string
	}{
		{4096, 1024, "attn_q_a"},
		{1024, 32768, "attn_q_b"},
		{4096, 512, "attn_kv"},
		{4096, 8192, "attn_output_a"},
		{8192, 4096, "attn_output_b"},
		{4096, 2048, "ffn_gate_shexp"},
		{2048, 4096, "ffn_down_shexp"},
	} {
		t.Run(fmt.Sprintf("%s/%dx%d", shape.tensor, shape.cols, shape.rows), func(t *testing.T) {
			// A fresh arena per shape: the sweep holds tens of megabytes of
			// synthetic weights per entry and there is no reason to keep an
			// earlier shape's alive while a later one runs.
			arena := metalmem.NewArena(dev)
			defer arena.Free()
			name := fmt.Sprintf("%dx%d", shape.cols, shape.rows)
			if checkMatmulQ80Shape(t, ops, mm, arena, shape.cols, shape.rows) {
				ordered = append(ordered, name)
			} else {
				blind = append(blind, name)
			}
		})
	}
	// Same suite-level requirement the shape sweep documents: bit-identity is
	// asserted everywhere, but it only means something where a reversal is
	// actually a reversal. If no production shape can tell, say so.
	//
	// 1024x32768 is the one shape that cannot tell, and the kernel says why: at
	// 1024 columns there are 32 blocks, the stride is simdgroups*8 = 32, so
	// every thread visits exactly one block and the accumulation loop has no
	// second term to reassociate. Reversing the block order there is the
	// identity, not a twin. The bit-identity assertion still runs at that shape
	// -- it is where the 32768-row grid geometry is exercised, which is what
	// this test was written for -- but its negative control is vacuous, and the
	// other six shapes are what establish that a reassociation would be seen.
	if len(ordered) == 0 {
		t.Errorf("no production shape could detect reassociation; the order-preservation claim is untested at every one of %v", blind)
	}
	if len(blind) > 0 {
		t.Logf("order sensitivity established at %v; not available at %v", ordered, blind)
	}
}
