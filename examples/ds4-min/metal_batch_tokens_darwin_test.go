//go:build darwin && arm64

package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// tokenTestCounts are the token counts every token-major test runs.
// One is the numerics bar: a one-token call through the token-major path must
// bind what the single-token form binds. The rest are odd, even and larger
// than a threadgroup.
var tokenTestCounts = []int{1, 2, 3, 7, 16}

// TestMetalRouterTokensBitIdentity pins the N-token router encodings against N
// separate single-token calls, bit for bit.
//
// The comparison is on raw float32 bits, not a tolerance: the routers
// normalize by a sum over six selected slots, and a reordering or a widened
// kernel that reassociated that sum would move the low bits while staying
// within any tolerance worth writing.
func TestMetalRouterTokensBitIdentity(t *testing.T) {
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

	const (
		width = 256
		slots = 6
		vocab = 19
	)

	for _, test := range []struct{ name, kind string }{
		{name: "hash", kind: "hash"},
		{name: "biased", kind: "biased"},
	} {
		for _, tokens := range tokenTestCounts {
			t.Run(fmt.Sprintf("%s/T=%d", test.name, tokens), func(t *testing.T) {
				arena := metalmem.NewArena(dev)
				defer arena.Free()

				// Distinct logits per token. The scale is wide enough that
				// softplus does not saturate into the x>20 branch for every
				// element, so the six selected scores differ per token.
				logits := make([]float32, width*tokens)
				rng := uint32(0x1234567)
				next := func() uint32 {
					rng = rng*1664525 + 1013904223
					return rng
				}
				for i := range logits {
					logits[i] = float32(int32(next()>>8)-(1<<23)) / float32(1<<21)
				}
				biasValues := make([]float32, width)
				for i := range biasValues {
					biasValues[i] = float32((i*19)%29-14) / 31
				}
				bias := testBatchF32Tensor(arena, "router-bias", biasValues)

				// A hash table with one distinct six-slot row per vocab entry,
				// so two tokens with different ids select different experts.
				tableData := metalmem.AllocIn[byte](arena, "router-table", slots*vocab*4)
				for v := 0; v < vocab; v++ {
					for s := 0; s < slots; s++ {
						id := int32((v*37 + s*53) % width)
						binary.LittleEndian.PutUint32(tableData.HostSlice()[4*(v*slots+s):], uint32(id))
					}
				}
				table := boundMetalTensor{name: "router-table", typeID: 26, dimensions: []uint64{slots, vocab}, data: tableData}

				tokenIDs := metalmem.AllocIn[uint32](arena, "router-token-ids", tokens)
				for i := range tokenIDs.HostSlice() {
					tokenIDs.HostSlice()[i] = uint32((i*7 + 3) % vocab)
				}
				if err := arena.Err(); err != nil {
					t.Fatal(err)
				}
				ids := append([]uint32(nil), tokenIDs.HostSlice()...)

				// Arm A: N separate single-token calls, each its own batch,
				// each with its own single-token output storage.
				wantProbs := make([]float32, width*tokens)
				wantIDs := make([]int32, slots*tokens)
				wantW := make([]float32, slots*tokens)
				for token := 0; token < tokens; token++ {
					oneLogits := metalmem.UploadIn(arena, fmt.Sprintf("one-logits-%d", token), logits[token*width:(token+1)*width])
					oneProbs := metalmem.AllocIn[float32](arena, fmt.Sprintf("one-probs-%d", token), width)
					oneIDs := metalmem.AllocIn[int32](arena, fmt.Sprintf("one-ids-%d", token), slots)
					oneW := metalmem.AllocIn[float32](arena, fmt.Sprintf("one-w-%d", token), slots)
					if err := arena.Err(); err != nil {
						t.Fatal(err)
					}
					batch, err := newMetalBatch(ops)
					if err != nil {
						t.Fatal(err)
					}
					if test.kind == "hash" {
						err = batch.EncodeRouterHash(oneLogits, table, ids[token], oneProbs, oneIDs, oneW)
					} else {
						err = batch.EncodeRouterBiased(oneLogits, bias, oneProbs, oneIDs, oneW)
					}
					if err != nil {
						t.Fatalf("single-token encode token %d: %v", token, err)
					}
					if err := batch.Wait(context.Background()); err != nil {
						t.Fatalf("single-token wait token %d: %v", token, err)
					}
					copy(wantProbs[token*width:], oneProbs.HostSlice())
					copy(wantIDs[token*slots:], oneIDs.HostSlice())
					copy(wantW[token*slots:], oneW.HostSlice())
				}
				assertNonzeroFinite(t, "single-token probabilities", wantProbs)
				assertNonzeroFinite(t, "single-token weights", wantW)
				requireRouterTokensDistinct(t, wantIDs, wantW, slots, tokens)

				// Arm B: one batch, one token-major call.
				allLogits := metalmem.UploadIn(arena, "all-logits", logits)
				allProbs := metalmem.AllocIn[float32](arena, "all-probs", width*tokens)
				allIDs := metalmem.AllocIn[int32](arena, "all-ids", slots*tokens)
				allW := metalmem.AllocIn[float32](arena, "all-w", slots*tokens)
				if err := arena.Err(); err != nil {
					t.Fatal(err)
				}
				// newMetalBatch creates the command buffer, and
				// metalPinnedOps counts it there, so the batch-creation
				// snapshot and the encode snapshot are separate.
				beforeBatch := ops.submissions
				batch, err := newMetalBatch(ops)
				if err != nil {
					t.Fatal(err)
				}
				beforeEncode := ops.submissions
				if test.kind == "hash" {
					err = batch.EncodeRouterHashTokens(allLogits, table, tokenIDs, allProbs, allIDs, allW, tokens)
				} else {
					err = batch.EncodeRouterBiasedTokens(allLogits, bias, allProbs, allIDs, allW, tokens)
				}
				if err != nil {
					t.Fatalf("token-major encode: %v", err)
				}
				if got := ops.submissions.sub(beforeEncode).Buffers; got != 0 {
					t.Errorf("token-major encode used %d command buffers, want 0", got)
				}
				if err := batch.Wait(context.Background()); err != nil {
					t.Fatalf("token-major wait: %v", err)
				}
				// One command buffer for all tokens is the point of the
				// widening; measure it rather than assert a constant.
				if got := ops.submissions.sub(beforeBatch).Buffers; got != 1 {
					t.Errorf("token-major path over %d tokens used %d command buffers, want 1", tokens, got)
				}

				assertBitIdentical(t, "probabilities", allProbs.HostSlice(), wantProbs)
				assertBitIdentical(t, "expert weights", allW.HostSlice(), wantW)
				gotIDs := allIDs.HostSlice()
				for i := range wantIDs {
					if gotIDs[i] != wantIDs[i] {
						t.Fatalf("expert ID mismatch at slot %d: got %d, want %d", i, gotIDs[i], wantIDs[i])
					}
				}
			})
		}
	}
}

// TestMetalSumVectorsTokensBitIdentity pins the token-major slot reduction
// against N separate single-token calls, bit for bit, at both counts the
// runtime uses: six routed experts and two FFN blocks.
func TestMetalSumVectorsTokensBitIdentity(t *testing.T) {
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

	// 6 is the routed-expert sum at metal_runtime_darwin.go:839 and 2 the FFN
	// block sum at :842. The widths are not multiples of the 256-thread
	// threadgroup, so the tail lane is covered too.
	for _, count := range []int{6, 2} {
		for _, width := range []int{13, 257} {
			for _, tokens := range tokenTestCounts {
				t.Run(fmt.Sprintf("count=%d/width=%d/T=%d", count, width, tokens), func(t *testing.T) {
					arena := metalmem.NewArena(dev)
					defer arena.Free()

					// Values spanning a wide exponent range so the partial
					// sums do not all land exactly; requireSlotOrderSensitive
					// checks that on this data rather than assuming it.
					values := make([]float32, width*count*tokens)
					rng := uint32(0x9e3779b9)
					next := func() uint32 {
						rng = rng*1664525 + 1013904223
						return rng
					}
					for i := range values {
						r := next()
						values[i] = float32(int32(r>>12)-524288) / float32(int32(1)<<(4+r%9))
					}
					requireSlotOrderSensitive(t, values, width, count, tokens)
					requireAxisDistinguishable(t, values, width, count, tokens)

					// Arm A: N separate single-token calls.
					want := make([]float32, width*tokens)
					for token := 0; token < tokens; token++ {
						row := values[token*width*count : (token+1)*width*count]
						oneIn := metalmem.UploadIn(arena, fmt.Sprintf("one-in-%d", token), row)
						oneOut := metalmem.AllocIn[float32](arena, fmt.Sprintf("one-out-%d", token), width)
						if err := arena.Err(); err != nil {
							t.Fatal(err)
						}
						batch, err := newMetalBatch(ops)
						if err != nil {
							t.Fatal(err)
						}
						if err := batch.EncodeSumVectors(oneIn, oneOut, count); err != nil {
							t.Fatalf("single-token encode token %d: %v", token, err)
						}
						if err := batch.Wait(context.Background()); err != nil {
							t.Fatalf("single-token wait token %d: %v", token, err)
						}
						copy(want[token*width:], oneOut.HostSlice())
					}
					assertNonzeroFinite(t, "single-token sums", want)

					// Arm B: one batch, one token-major call.
					allIn := metalmem.UploadIn(arena, "all-in", values)
					allOut := metalmem.AllocIn[float32](arena, "all-out", width*tokens)
					if err := arena.Err(); err != nil {
						t.Fatal(err)
					}
					beforeBatch := ops.submissions
					batch, err := newMetalBatch(ops)
					if err != nil {
						t.Fatal(err)
					}
					beforeEncode := ops.submissions
					if err := batch.EncodeSumVectorsTokens(allIn, allOut, count, tokens); err != nil {
						t.Fatalf("token-major encode: %v", err)
					}
					if got := ops.submissions.sub(beforeEncode).Buffers; got != 0 {
						t.Errorf("token-major encode used %d command buffers, want 0", got)
					}
					if err := batch.Wait(context.Background()); err != nil {
						t.Fatalf("token-major wait: %v", err)
					}
					if got := ops.submissions.sub(beforeBatch).Buffers; got != 1 {
						t.Errorf("token-major path over %d tokens used %d command buffers, want 1", tokens, got)
					}
					assertBitIdentical(t, "slot sums", allOut.HostSlice(), want)
				})
			}
		}
	}
}

// TestMetalSumVectorsTokensRejectsBadShape pins that the widened validation
// still rejects the shapes the single-token form rejected.
func TestMetalSumVectorsTokensRejectsBadShape(t *testing.T) {
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
	arena := metalmem.NewArena(dev)
	defer arena.Free()

	in := metalmem.AllocIn[float32](arena, "in", 6*8)
	out := metalmem.AllocIn[float32](arena, "out", 8)
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name          string
		count, tokens int
	}{
		{name: "zero count", count: 0, tokens: 1},
		{name: "negative count", count: -6, tokens: 1},
		{name: "zero tokens", count: 6, tokens: 0},
		{name: "wrong count", count: 5, tokens: 1},
		// out is 8 wide, which no three-token layout divides; this is the
		// shape a caller hits when only one of the two slices was widened.
		{name: "output not widened", count: 6, tokens: 3},
	} {
		t.Run(test.name, func(t *testing.T) {
			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.EncodeSumVectorsTokens(in, out, test.count, test.tokens); err == nil {
				t.Fatalf("EncodeSumVectorsTokens(count=%d, tokens=%d) returned nil error", test.count, test.tokens)
			}
		})
	}
}

// assertBitIdentical fails on the first element whose float32 bits differ.
func assertBitIdentical(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d elements, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: bit mismatch at index %d: got 0x%08x (%g), want 0x%08x (%g)",
				name, i, math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}

// requireRouterTokensDistinct fails unless every token after the first
// produced a selection or a weight that differs from token zero's.
//
// Without it the bit-identity assertion is fail-open on the axis that
// matters: an N-token encoding that bound token zero's logits, or token
// zero's id, for every row would agree with the single-token arm everywhere
// and the test would stay green. The guard runs on the single-token arm's own
// output, so it measures this data rather than asserting a property of it.
func requireRouterTokensDistinct(t *testing.T, ids []int32, weights []float32, slots, tokens int) {
	t.Helper()
	if tokens == 1 {
		return
	}
	for token := 1; token < tokens; token++ {
		differs := false
		for s := 0; s < slots; s++ {
			if ids[token*slots+s] != ids[s] {
				differs = true
			}
			if math.Float32bits(weights[token*slots+s]) != math.Float32bits(weights[s]) {
				differs = true
			}
		}
		if !differs {
			t.Fatalf("token %d routed identically to token 0; the bit-identity check cannot see a token-index defect", token)
		}
	}
}

// requireSlotOrderSensitive fails unless summing some token's slots in reverse
// changes the float32 result for at least one output element.
//
// Without it the bit-identity assertion is fail-open on reassociation: inputs
// whose partial sums are all exact make every association agree, so a widened
// kernel that reordered the slot accumulation would pass.
func requireSlotOrderSensitive(t *testing.T, values []float32, width, count, tokens int) {
	t.Helper()
	if count < 3 {
		// Two terms have one sum. IEEE addition is commutative, so a+b and
		// b+a carry the same bits and no data can make a two-slot reduction
		// order-sensitive; there is no reassociation for the guard to catch.
		// requireAxisDistinguishable still has teeth at this count.
		return
	}
	for token := 0; token < tokens; token++ {
		base := token * width * count
		for i := 0; i < width; i++ {
			var fwd, rev float32
			for row := 0; row < count; row++ {
				fwd += values[base+row*width+i]
			}
			for row := count - 1; row >= 0; row-- {
				rev += values[base+row*width+i]
			}
			if math.Float32bits(fwd) != math.Float32bits(rev) {
				return
			}
		}
	}
	t.Fatalf("no element's slot sum depends on accumulation order; the bit-identity check cannot see a reassociation")
}

// requireAxisDistinguishable fails unless reducing down the token axis --
// what deriving the width from in.Len()/count does -- disagrees bitwise with
// reducing across a token's slots.
//
// This is the negative control for the defect the token-major form exists to
// fix. If the two axes agreed on this data, a call that summed across tokens
// would pass the bit-identity assertion.
func requireAxisDistinguishable(t *testing.T, values []float32, width, count, tokens int) {
	t.Helper()
	if tokens == 1 {
		return // One token: the two axes are the same reduction.
	}
	// The wrong axis: width' = len(values)/count = width*tokens, so element i
	// of the output sums values[row*width*tokens+i] over row < count.
	wrongWidth := width * tokens
	for i := 0; i < width; i++ {
		var right, wrong float32
		for row := 0; row < count; row++ {
			right += values[row*width+i]
			wrong += values[row*wrongWidth+i]
		}
		if math.Float32bits(right) != math.Float32bits(wrong) {
			return
		}
	}
	t.Fatalf("the slot axis and the token axis agree bitwise on this data; the bit-identity check cannot see a wrong-axis reduction")
}
