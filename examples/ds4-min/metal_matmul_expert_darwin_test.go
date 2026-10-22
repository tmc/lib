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

func TestMetalMatmulExpertIQ2PreservedBitIdentity(t *testing.T) {
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

	mm, err := newMetalMatmulPreservedExpert(ops)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	// Sweep shapes off the kernel's own boundaries, plus full 2048 shape
	type shapeDef struct {
		cols, rows, experts, slots int
	}
	shapes := []shapeDef{
		{cols: 256, rows: 7, experts: 5, slots: 6},    // 1 block (8 parts, tail in simdgroup), odd rows
		{cols: 512, rows: 11, experts: 8, slots: 6},   // 2 blocks (16 parts), prime rows
		{cols: 1024, rows: 17, experts: 6, slots: 6},  // 4 blocks (32 parts, exactly 1 stride), prime rows
		{cols: 1280, rows: 13, experts: 7, slots: 6},  // 5 blocks (40 parts, off-stride tail)
		{cols: 2048, rows: 16, experts: 8, slots: 6},  // 8 blocks (64 parts, 2 full strides)
		{cols: 768, rows: 13, experts: 6, slots: 6},   // 24 parts: the only parts%32 residue class the rest of the sweep misses
		{cols: 4096, rows: 16, experts: 8, slots: 6},  // the real K, 128 parts
		{cols: 512, rows: 1, experts: 8, slots: 6},    // one row: the grid's Width axis degenerates
		{cols: 512, rows: 11, experts: 256, slots: 6}, // 256 experts; the "high" ID pattern reaches 255
		{cols: 512, rows: 11, experts: 8, slots: 1},   // one slot: MulIQ2 accepts any slot count and production encodes some
		{cols: 512, rows: 11, experts: 8, slots: 3},   // three slots, so the Height axis is neither 1 nor 6

		// The two shapes the model actually holds. ffn_gate_exps and
		// ffn_up_exps are [4096 2048 256] and ffn_down_exps is [2048 4096 256],
		// so every case above runs a row count two orders of magnitude below the
		// dispatched one and none of them exercises the grid at its real Width.
		//
		// These deliberately do not also carry 256 experts: the weight bytes go
		// as cols*rows*experts, and the full product is a gigabyte per tensor to
		// fill synthetically for no added signal. The expert count reaches its
		// production value in the 256-expert case above, where the concern is ID
		// addressing and rows is irrelevant; here rows reaches its production
		// value, where the concern is grid geometry and the expert count is. The
		// sweep covers each axis at its real value rather than their product,
		// which is a choice and not an oversight.
		{cols: 4096, rows: 2048, experts: 8, slots: 6}, // ffn_gate_exps, ffn_up_exps
		{cols: 2048, rows: 4096, experts: 8, slots: 6}, // ffn_down_exps
	}

	testTokens := []int{1, 3, 7, 13} // 1, plus prime token counts that are not multiples of anything

	rng := uint32(0x12345678)
	next := func() uint32 {
		rng = rng*1664525 + 1013904223
		return rng
	}

	var ordered, blind []string

	for _, s := range shapes {
		cols, rows, experts, slots := s.cols, s.rows, s.experts, s.slots
		shapeSensitive := false
		gate := selectedExpertTensor(arena, fmt.Sprintf("gate-%d-%d-%d", cols, rows, experts), 16, cols, rows, experts, 3)
		up := selectedExpertTensor(arena, fmt.Sprintf("up-%d-%d-%d", cols, rows, experts), 16, cols, rows, experts, 11)

		for _, T := range testTokens {
			// Case 1: Shared experts across tokens
			// Case 2: Disjoint experts (when T * slots <= experts, or modulo pattern)
			for _, pattern := range expertIDPatterns {
				name := fmt.Sprintf("cols=%d/rows=%d/experts=%d/slots=%d/T=%d/ids=%s", cols, rows, experts, slots, T, pattern)
				t.Run(name, func(t *testing.T) {
					expertIDs := expertIDsFor(pattern, T, slots, experts)

					inValues := make([]float32, cols*T)
					for i := range inValues {
						r := next()
						inValues[i] = float32(int32(r>>12)-524288) / float32(int32(1)<<(4+r%9))
					}

					// 1. Reference: run per-token using ops.MatvecIQ2Selected
					wantGate := make([]float32, T*slots*rows)
					wantUp := make([]float32, T*slots*rows)
					for tok := range T {
						tokIn := inValues[tok*cols : (tok+1)*cols]
						tokIDs := expertIDs[tok*slots : (tok+1)*slots]
						g, u, err := ops.MatvecIQ2Selected(gate, up, tokIDs, tokIn)
						if err != nil {
							t.Fatalf("token %d reference failed: %v", tok, err)
						}
						copy(wantGate[tok*slots*rows:(tok+1)*slots*rows], g)
						copy(wantUp[tok*slots*rows:(tok+1)*slots*rows], u)
					}

					// 2. Batched execution: exactly 1 command buffer
					before := ops.submissions
					gotGate, gotUp, err := mm.MulIQ2(gate, up, expertIDs, inValues, T, slots)
					if err != nil {
						t.Fatalf("MulIQ2 failed: %v", err)
					}
					if got := ops.submissions.sub(before).Buffers; got != 1 {
						t.Errorf("batched IQ2 matmul over %d tokens used %d command buffers, want 1", T, got)
					}

					// 2b. Run encoded batched matmul: exactly 0 command buffers on encode, bit-identical to standalone.
					dGateOut, err := metalmem.Alloc[float32](ops.dev, T*slots*rows)
					if err != nil {
						t.Fatalf("dGateOut alloc failed: %v", err)
					}
					defer dGateOut.Free()
					dUpOut, err := metalmem.Alloc[float32](ops.dev, T*slots*rows)
					if err != nil {
						t.Fatalf("dUpOut alloc failed: %v", err)
					}
					defer dUpOut.Free()
					dExpertIDs, err := metalmem.AllocFromSlice(ops.dev, expertIDs)
					if err != nil {
						t.Fatalf("dExpertIDs alloc failed: %v", err)
					}
					defer dExpertIDs.Free()
					dIn, err := metalmem.AllocFromSlice(ops.dev, inValues)
					if err != nil {
						t.Fatalf("dIn alloc failed: %v", err)
					}
					defer dIn.Free()

					// 2b. Run encoded batched matmul: exactly 0 command buffers on encode, bit-identical to standalone.
					// Note: Encoded-vs-standalone bit identity compares two arms that call the same encode* body.
					// A defect in that body moves both arms equally and this check stays green. Its teeth are on
					// wrapper, allocation and pipeline divergence only. What covers the shared body is the reference
					// comparison and the reversed-twin machinery.
					batch, err := newMetalBatch(ops)
					if err != nil {
						t.Fatalf("newMetalBatch failed: %v", err)
					}
					beforeEnc := ops.submissions
					if err := batch.EncodeMatmulIQ2Selected(gate, up, dExpertIDs, dIn, dGateOut, dUpOut, T, slots); err != nil {
						t.Fatalf("batch.EncodeMatmulIQ2Selected failed: %v", err)
					}
					if got := ops.submissions.sub(beforeEnc).Buffers; got != 0 {
						t.Errorf("EncodeMatmulIQ2Selected used %d command buffers, want 0", got)
					}
					if err := batch.Wait(context.Background()); err != nil {
						t.Fatalf("batch.Wait failed: %v", err)
					}
					encGate := dGateOut.HostSlice()
					encUp := dUpOut.HostSlice()
					for i := range gotGate {
						if math.Float32bits(encGate[i]) != math.Float32bits(gotGate[i]) {
							t.Fatalf("encoded mismatch with standalone gate at index %d: got 0x%08x (%g), want 0x%08x (%g)",
								i, math.Float32bits(encGate[i]), encGate[i], math.Float32bits(gotGate[i]), gotGate[i])
						}
						if math.Float32bits(encUp[i]) != math.Float32bits(gotUp[i]) {
							t.Fatalf("encoded mismatch with standalone up at index %d: got 0x%08x (%g), want 0x%08x (%g)",
								i, math.Float32bits(encUp[i]), encUp[i], math.Float32bits(gotUp[i]), gotUp[i])
						}
					}

					// 3. Bit-exact float32 comparison
					if len(gotGate) != len(wantGate) || len(gotUp) != len(wantUp) {
						t.Fatalf("length mismatch: got gate %d want %d, got up %d want %d",
							len(gotGate), len(wantGate), len(gotUp), len(wantUp))
					}
					for i := range wantGate {
						if math.Float32bits(gotGate[i]) != math.Float32bits(wantGate[i]) {
							t.Fatalf("gate bit mismatch at index %d: got 0x%08x (%g), want 0x%08x (%g)",
								i, math.Float32bits(gotGate[i]), gotGate[i], math.Float32bits(wantGate[i]), wantGate[i])
						}
						if math.Float32bits(gotUp[i]) != math.Float32bits(wantUp[i]) {
							t.Fatalf("up bit mismatch at index %d: got 0x%08x (%g), want 0x%08x (%g)",
								i, math.Float32bits(gotUp[i]), gotUp[i], math.Float32bits(wantUp[i]), wantUp[i])
						}
					}

					// Check order sensitivity with reversed twin
					revGate, revUp, err := mm.MulIQ2Reversed(gate, up, expertIDs, inValues, T, slots)
					if err != nil {
						t.Fatalf("reversed twin failed: %v", err)
					}
					diff := false
					for i := range gotGate {
						if math.Float32bits(gotGate[i]) != math.Float32bits(revGate[i]) ||
							math.Float32bits(gotUp[i]) != math.Float32bits(revUp[i]) {
							diff = true
							break
						}
					}
					if diff {
						shapeSensitive = true
					}
				})
			}
		}
		if shapeSensitive {
			ordered = append(ordered, fmt.Sprintf("%dx%d", cols, rows))
		} else {
			blind = append(blind, fmt.Sprintf("%dx%d", cols, rows))
		}
	}

	reportOrderSensitivity(t, "IQ2_XXS", ordered, blind)
}

func TestMetalMatmulExpertQ2KPreservedBitIdentity(t *testing.T) {
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

	mm, err := newMetalMatmulPreservedExpert(ops)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	// Sweep shapes off the kernel's own boundaries, plus full 2048 shape
	type shapeDef struct {
		cols, rows, experts, slots int
	}
	shapes := []shapeDef{
		{cols: 256, rows: 7, experts: 5, slots: 6},    // 1 block (256 elements), odd rows
		{cols: 512, rows: 11, experts: 8, slots: 6},   // 2 blocks (512 elements), prime rows
		{cols: 768, rows: 13, experts: 6, slots: 6},   // 3 blocks (768 elements, not power of 2)
		{cols: 1024, rows: 17, experts: 7, slots: 6},  // 4 blocks (1024 elements)
		{cols: 2048, rows: 16, experts: 8, slots: 6},  // 8 blocks (2048 elements)
		{cols: 768, rows: 300, experts: 6, slots: 6},  // rows > 256: two threadgroups, the second partial
		{cols: 512, rows: 512, experts: 5, slots: 6},  // rows an exact multiple of the 256-wide threadgroup
		{cols: 512, rows: 1100, experts: 5, slots: 6}, // rows past the pipeline's maximum threadgroup width, which is what min(rows, 256) is for
		{cols: 512, rows: 1, experts: 8, slots: 6},    // one row
		{cols: 512, rows: 11, experts: 256, slots: 6}, // 256 experts; the "high" ID pattern reaches 255
	}
	// slots stays 6 throughout. MulQ2K accepts any positive slot count, but the
	// production encoder it stands for requires exactly 6
	// (metal_batch_darwin.go, EncodeMatvecQ2KSelected), so the reference cannot
	// be run at any other count and the extra generality is unprovable here.

	testTokens := []int{1, 3, 7, 13}

	rng := uint32(0x1337beef)
	next := func() uint32 {
		rng = rng*1664525 + 1013904223
		return rng
	}

	var ordered, blind []string

	for _, s := range shapes {
		cols, rows, experts, slots := s.cols, s.rows, s.experts, s.slots
		shapeSensitive := false
		down := selectedExpertTensor(arena, fmt.Sprintf("down-%d-%d-%d", cols, rows, experts), 10, cols, rows, experts, 7)

		for _, T := range testTokens {
			for _, pattern := range expertIDPatterns {
				name := fmt.Sprintf("cols=%d/rows=%d/experts=%d/slots=%d/T=%d/ids=%s", cols, rows, experts, slots, T, pattern)
				t.Run(name, func(t *testing.T) {
					expertIDs := expertIDsFor(pattern, T, slots, experts)

					inValues := make([]float32, cols*T*slots)
					for i := range inValues {
						r := next()
						inValues[i] = float32(int32(r>>12)-524288) / float32(int32(1)<<(4+r%9))
					}

					// 1. Reference: run per-token using ops.MatvecQ2KSelected
					wantOut := make([]float32, T*slots*rows)
					for tok := range T {
						tokIn := inValues[tok*slots*cols : (tok+1)*slots*cols]
						tokIDs := expertIDs[tok*slots : (tok+1)*slots]
						out, err := ops.MatvecQ2KSelected(down, tokIDs, tokIn)
						if err != nil {
							t.Fatalf("token %d reference failed: %v", tok, err)
						}
						copy(wantOut[tok*slots*rows:(tok+1)*slots*rows], out)
					}

					// 2. Batched execution: exactly 1 command buffer
					before := ops.submissions
					gotOut, err := mm.MulQ2K(down, expertIDs, inValues, T, slots)
					if err != nil {
						t.Fatalf("MulQ2K failed: %v", err)
					}
					if got := ops.submissions.sub(before).Buffers; got != 1 {
						t.Errorf("batched Q2_K matmul over %d tokens used %d command buffers, want 1", T, got)
					}

					// 2b. Run encoded batched matmul: exactly 0 command buffers on encode, bit-identical to standalone.
					dOut, err := metalmem.Alloc[float32](ops.dev, T*slots*rows)
					if err != nil {
						t.Fatalf("dOut alloc failed: %v", err)
					}
					defer dOut.Free()
					dExpertIDs, err := metalmem.AllocFromSlice(ops.dev, expertIDs)
					if err != nil {
						t.Fatalf("dExpertIDs alloc failed: %v", err)
					}
					defer dExpertIDs.Free()
					dIn, err := metalmem.AllocFromSlice(ops.dev, inValues)
					if err != nil {
						t.Fatalf("dIn alloc failed: %v", err)
					}
					defer dIn.Free()

					// 2b. Run encoded batched matmul: exactly 0 command buffers on encode, bit-identical to standalone.
					// Note: Encoded-vs-standalone bit identity compares two arms that call the same encode* body.
					// A defect in that body moves both arms equally and this check stays green. Its teeth are on
					// wrapper, allocation and pipeline divergence only. What covers the shared body is the reference
					// comparison and the reversed-twin machinery.
					batch, err := newMetalBatch(ops)
					if err != nil {
						t.Fatalf("newMetalBatch failed: %v", err)
					}
					beforeEnc := ops.submissions
					if err := batch.EncodeMatmulQ2KSelected(down, dExpertIDs, dIn, dOut, T, slots); err != nil {
						t.Fatalf("batch.EncodeMatmulQ2KSelected failed: %v", err)
					}
					if got := ops.submissions.sub(beforeEnc).Buffers; got != 0 {
						t.Errorf("EncodeMatmulQ2KSelected used %d command buffers, want 0", got)
					}
					if err := batch.Wait(context.Background()); err != nil {
						t.Fatalf("batch.Wait failed: %v", err)
					}
					encOut := dOut.HostSlice()
					for i := range gotOut {
						if math.Float32bits(encOut[i]) != math.Float32bits(gotOut[i]) {
							t.Fatalf("encoded mismatch with standalone at index %d: got 0x%08x (%g), want 0x%08x (%g)",
								i, math.Float32bits(encOut[i]), encOut[i], math.Float32bits(gotOut[i]), gotOut[i])
						}
					}

					// 3. Bit-exact float32 comparison
					if len(gotOut) != len(wantOut) {
						t.Fatalf("length mismatch: got %d, want %d", len(gotOut), len(wantOut))
					}
					for i := range wantOut {
						if math.Float32bits(gotOut[i]) != math.Float32bits(wantOut[i]) {
							t.Fatalf("bit mismatch at index %d: got 0x%08x (%g), want 0x%08x (%g)",
								i, math.Float32bits(gotOut[i]), gotOut[i], math.Float32bits(wantOut[i]), wantOut[i])
						}
					}

					// Check order sensitivity with reversed twin
					revOut, err := mm.MulQ2KReversed(down, expertIDs, inValues, T, slots)
					if err != nil {
						t.Fatalf("reversed twin failed: %v", err)
					}
					diff := false
					for i := range gotOut {
						if math.Float32bits(gotOut[i]) != math.Float32bits(revOut[i]) {
							diff = true
							break
						}
					}
					if diff {
						shapeSensitive = true
					}
				})
			}
		}
		if shapeSensitive {
			ordered = append(ordered, fmt.Sprintf("%dx%d", cols, rows))
		} else {
			blind = append(blind, fmt.Sprintf("%dx%d", cols, rows))
		}
	}

	reportOrderSensitivity(t, "Q2_K", ordered, blind)
}

// expertIDPatterns names the routing shapes the sweeps run. "shared" makes
// neighbouring tokens select the same expert, which is what the router does in
// practice; "disjoint" walks the ids so no two slots agree; "high" counts down
// from experts-1, which is the only pattern that reaches the top of the id
// range when experts is large.
var expertIDPatterns = []string{"shared", "disjoint", "high"}

func expertIDsFor(pattern string, tokens, slots, experts int) []int32 {
	ids := make([]int32, tokens*slots)
	for i := range ids {
		switch pattern {
		case "shared":
			ids[i] = int32((i%slots + (i/slots)%2) % experts)
		case "disjoint":
			ids[i] = int32(i % experts)
		case "high":
			ids[i] = int32(experts - 1 - i%experts)
		default:
			panic("unknown expert id pattern " + pattern)
		}
	}
	return ids
}

// TestMetalMatmulExpertOutOfRangeID covers the two guards the sweeps never
// reach, because every id in a sweep is taken modulo the expert count.
//
// The guards return before writing, so what the test can see is a slot the
// kernel left alone: a freshly allocated Metal buffer reads back zero, and an
// unguarded kernel would index the weights past the end of the tensor and store
// whatever came back. Requiring exact zeroes at the bad slot -- and at least one
// non-zero at the same slot when the id is valid, so the zero means something --
// is what gives this teeth. If Metal ever stops zeroing fresh buffers the test
// goes red rather than quiet, which is the direction it should fail in.
func TestMetalMatmulExpertOutOfRangeID(t *testing.T) {
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

	mm, err := newMetalMatmulPreservedExpert(ops)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	const cols, rows, experts, slots, tokens = 512, 11, 8, 6, 3
	const badTok, badSlot = 1, 2
	badIdx := badTok*slots + badSlot

	inValues := make([]float32, cols*tokens*slots)
	rng := uint32(0xfeed5eed)
	for i := range inValues {
		rng = rng*1664525 + 1013904223
		inValues[i] = float32(int32(rng>>12)-524288) / float32(int32(1)<<(4+rng%9))
	}

	gate := selectedExpertTensor(arena, "oor-gate", 16, cols, rows, experts, 3)
	up := selectedExpertTensor(arena, "oor-up", 16, cols, rows, experts, 11)
	down := selectedExpertTensor(arena, "oor-down", 10, cols, rows, experts, 7)

	// The IQ2 kernel reads one activation vector per token; the Q2_K kernel
	// reads one per slot. Both take the same id array.
	iq2In := inValues[:cols*tokens]

	for _, bad := range []int32{-1, int32(experts), int32(experts) + 5, 1 << 30} {
		t.Run(fmt.Sprintf("id=%d", bad), func(t *testing.T) {
			good := expertIDsFor("disjoint", tokens, slots, experts)
			baseGate, baseUp, err := mm.MulIQ2(gate, up, good, iq2In, tokens, slots)
			if err != nil {
				t.Fatalf("MulIQ2 baseline: %v", err)
			}
			baseDown, err := mm.MulQ2K(down, good, inValues, tokens, slots)
			if err != nil {
				t.Fatalf("MulQ2K baseline: %v", err)
			}
			// The zero check below only means something if a valid id at this
			// slot writes something other than zero.
			if allZero(baseGate[badIdx*rows:(badIdx+1)*rows]) || allZero(baseUp[badIdx*rows:(badIdx+1)*rows]) || allZero(baseDown[badIdx*rows:(badIdx+1)*rows]) {
				t.Fatalf("slot %d is all zeroes even with a valid expert id; the out-of-range check below would pass vacuously", badIdx)
			}

			ids := append([]int32(nil), good...)
			ids[badIdx] = bad

			gotGate, gotUp, err := mm.MulIQ2(gate, up, ids, iq2In, tokens, slots)
			if err != nil {
				t.Fatalf("MulIQ2 with out-of-range id: %v", err)
			}
			gotDown, err := mm.MulQ2K(down, ids, inValues, tokens, slots)
			if err != nil {
				t.Fatalf("MulQ2K with out-of-range id: %v", err)
			}

			for _, c := range []struct {
				what      string
				base, got []float32
			}{
				{"gate", baseGate, gotGate},
				{"up", baseUp, gotUp},
				{"down", baseDown, gotDown},
			} {
				if len(c.got) != len(c.base) {
					t.Fatalf("%s: got %d elements, want %d", c.what, len(c.got), len(c.base))
				}
				for i := range c.got {
					slot := i / rows
					if slot == badIdx {
						if c.got[i] != 0 {
							t.Fatalf("%s: expert id %d is out of range but slot %d row %d was written: 0x%08x (%g)",
								c.what, bad, slot, i%rows, math.Float32bits(c.got[i]), c.got[i])
						}
						continue
					}
					if math.Float32bits(c.got[i]) != math.Float32bits(c.base[i]) {
						t.Fatalf("%s: out-of-range id at slot %d disturbed slot %d row %d: got 0x%08x, want 0x%08x",
							c.what, badIdx, slot, i%rows, math.Float32bits(c.got[i]), math.Float32bits(c.base[i]))
					}
				}
			}
		})
	}
}

func allZero(v []float32) bool {
	for _, f := range v {
		if f != 0 {
			return false
		}
	}
	return true
}

// TestMetalMatmulExpertRejectsMalformedCalls pins the host-side validation.
// Every one of these would otherwise reach a dispatch whose grid or buffer
// bounds disagree with the arguments, which on a GPU is not an error but a
// wrong answer.
func TestMetalMatmulExpertRejectsMalformedCalls(t *testing.T) {
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

	mm, err := newMetalMatmulPreservedExpert(ops)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	const cols, rows, experts, slots, tokens = 512, 11, 8, 6, 3
	gate := selectedExpertTensor(arena, "bad-gate", 16, cols, rows, experts, 3)
	up := selectedExpertTensor(arena, "bad-up", 16, cols, rows, experts, 11)
	down := selectedExpertTensor(arena, "bad-down", 10, cols, rows, experts, 7)
	ids := expertIDsFor("disjoint", tokens, slots, experts)
	iq2In := make([]float32, cols*tokens)
	q2kIn := make([]float32, cols*tokens*slots)

	ragged := gate
	ragged.dimensions = []uint64{cols + 128, rows, experts}
	flat := gate
	flat.dimensions = []uint64{cols, rows}
	wrongType := gate
	wrongType.typeID = 8
	short := selectedExpertTensor(arena, "bad-short", 16, cols, rows, experts-1, 3)
	short.dimensions = []uint64{cols, rows, experts}
	shortDown := selectedExpertTensor(arena, "bad-short-down", 10, cols, rows, experts-1, 7)
	shortDown.dimensions = []uint64{cols, rows, experts}
	mismatched := selectedExpertTensor(arena, "bad-mismatch", 16, cols, rows+1, experts, 5)

	for _, c := range []struct {
		name string
		run  func() error
	}{
		{"iq2/cols not a multiple of 256", func() error {
			_, _, err := mm.MulIQ2(ragged, ragged, ids, make([]float32, (cols+128)*tokens), tokens, slots)
			return err
		}},
		{"iq2/two dimensions", func() error {
			_, _, err := mm.MulIQ2(flat, flat, ids, iq2In, tokens, slots)
			return err
		}},
		{"iq2/wrong quantization", func() error {
			_, _, err := mm.MulIQ2(wrongType, up, ids, iq2In, tokens, slots)
			return err
		}},
		{"iq2/gate and up disagree", func() error {
			_, _, err := mm.MulIQ2(gate, mismatched, ids, iq2In, tokens, slots)
			return err
		}},
		{"iq2/zero tokens", func() error {
			_, _, err := mm.MulIQ2(gate, up, nil, nil, 0, slots)
			return err
		}},
		{"iq2/zero slots", func() error {
			_, _, err := mm.MulIQ2(gate, up, nil, iq2In, tokens, 0)
			return err
		}},
		{"iq2/short id array", func() error {
			_, _, err := mm.MulIQ2(gate, up, ids[:len(ids)-1], iq2In, tokens, slots)
			return err
		}},
		{"iq2/short input", func() error {
			_, _, err := mm.MulIQ2(gate, up, ids, iq2In[:len(iq2In)-1], tokens, slots)
			return err
		}},
		{"iq2/weights too small for the declared expert count", func() error {
			_, _, err := mm.MulIQ2(short, short, ids, iq2In, tokens, slots)
			return err
		}},
		{"q2k/cols not a multiple of 256", func() error {
			bad := down
			bad.dimensions = []uint64{cols + 128, rows, experts}
			_, err := mm.MulQ2K(bad, ids, make([]float32, (cols+128)*tokens*slots), tokens, slots)
			return err
		}},
		{"q2k/two dimensions", func() error {
			bad := down
			bad.dimensions = []uint64{cols, rows}
			_, err := mm.MulQ2K(bad, ids, q2kIn, tokens, slots)
			return err
		}},
		{"q2k/wrong quantization", func() error {
			bad := down
			bad.typeID = 16
			_, err := mm.MulQ2K(bad, ids, q2kIn, tokens, slots)
			return err
		}},
		{"q2k/zero tokens", func() error {
			_, err := mm.MulQ2K(down, nil, nil, 0, slots)
			return err
		}},
		{"q2k/zero slots", func() error {
			_, err := mm.MulQ2K(down, nil, q2kIn, tokens, 0)
			return err
		}},
		{"q2k/short id array", func() error {
			_, err := mm.MulQ2K(down, ids[:len(ids)-1], q2kIn, tokens, slots)
			return err
		}},
		{"q2k/input sized per token rather than per slot", func() error {
			_, err := mm.MulQ2K(down, ids, q2kIn[:cols*tokens], tokens, slots)
			return err
		}},
		{"q2k/weights too small for the declared expert count", func() error {
			_, err := mm.MulQ2K(shortDown, ids, q2kIn, tokens, slots)
			return err
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			before := ops.submissions
			if err := c.run(); err == nil {
				t.Fatal("call was accepted; it should have been rejected before any dispatch")
			}
			if got := ops.submissions.sub(before).Buffers; got != 0 {
				t.Errorf("rejected call still created %d command buffers, want 0", got)
			}
		})
	}
}

// reportOrderSensitivity keeps the sweeps' order-preservation claim honest
// about where it is actually tested.
//
// Bit-identity against the per-token reference is asserted at every shape. The
// reversed twin is what says that assertion could have failed, and it cannot
// say so everywhere: where a reversal is provably not a reversal, the two
// pipelines are the same computation and their agreement is empty. Q2_K at
// cols=256 has nb=1, so `block = kReverseBlocks ? (nb-1-bi) : bi` is the same
// instruction on both branches. IQ2 at cols<=1024 has parts<=32, so every
// active lane holds one part and reversal only permutes parts across lanes --
// and for a power-of-two count that permutation is XOR with parts-1, which a
// butterfly simd_sum pairs identically at every level. Same mechanism the Q8_0
// sweep documents and measured at cols=512.
//
// So the requirement is the suite-level one: at least one shape must be able to
// tell, or a twin that quietly stopped reversing would leave the whole sweep
// green. The blind shapes are named rather than dropped -- they still exercise
// their own dispatch geometry -- and the log says which is which, so nobody has
// to re-derive it from the kernel.
func reportOrderSensitivity(t *testing.T, kind string, ordered, blind []string) {
	t.Helper()
	if len(ordered) == 0 {
		t.Errorf("%s: no shape in the sweep could detect reassociation; the order-preservation claim is untested at every one of %v", kind, blind)
	}
	if len(blind) > 0 {
		t.Logf("%s: order sensitivity established at %v; not available at %v", kind, ordered, blind)
	}
}
