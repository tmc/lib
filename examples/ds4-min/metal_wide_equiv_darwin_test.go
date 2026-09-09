//go:build darwin && arm64

package main

import (
	"fmt"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// ulpDiff returns the distance between two float32 in representable steps.
func ulpDiff(a, b float32) uint32 {
	ai, bi := int32(math.Float32bits(a)), int32(math.Float32bits(b))
	if ai < 0 {
		ai = math.MinInt32 - ai
	}
	if bi < 0 {
		bi = math.MinInt32 - bi
	}
	if ai > bi {
		return uint32(ai - bi)
	}
	return uint32(bi - ai)
}

// exactSlice is the gate: zero changed float32 bits against the reference.
//
// That is the project's bar for a production numerics change in this area --
// swiglu_slots_identity_darwin_test.go, metal_hc_integration_darwin_test.go,
// metal_hc_layer13_regression_darwin_test.go and exactFloat32Bits in
// cuda_compressed_linux_test.go all compare Float32bits directly, and a 1-2 ULP
// intermediate has been measured elsewhere in this model to amplify into large
// logit differences. A ULP bound is not a substitute for it.
//
// Reaching it took keeping both rotations in one-thread-per-head kernels that
// preserve the reference's constant-bound r loop. compressed_rope_theta's
// pow(160000,-r/64) is folded at compile time only for a constant-bound
// induction variable; a strided p=tid loop evaluates it on the GPU instead, and
// the two disagree by 1 ULP, which propagates through sin. Widening the score
// and value passes -- the work that is actually 512-wide -- does not disturb
// that, so the wide path holds only those. Textual identity is what worked here
// against this compiler; it is not established as universally necessary or
// sufficient, and it is the exact gate rather than the technique that keeps
// this honest. Receipt:
// evidence/worker-DF6906F6/metal-split-rotation-candidate-20260908T00/.
func exactSlice(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("%s[%d] = %g is not finite", name, i, got[i])
		}
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] = %g (0x%08x), want %g (0x%08x): %d ULP; the bar is zero changed bits",
				name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]),
				ulpDiff(got[i], want[i]))
		}
	}
}

// TestRopeAttentionMixedWideMatchesNarrow pins the four-kernel wide dispatch to
// the single-threadgroup kernel it replaces. The wide path splits preparation,
// KV quantisation and the inverse rotation into their own dispatches and gives
// each head a threadgroup for the score and value passes. Nothing is
// reassociated: each score sum runs in one thread over a whole row, the
// denominator is summed by one thread in the reference's order, and both
// rotations run in head-mapped kernels reproducing the reference's loops. The
// gate is therefore equality of bits; no tolerance is consulted here, and the
// frozen 1e-4 one is not in play.
//
// The inputs matter. TestMetalBatchCompressedMixedAttention drives this path
// with q = (i%17-8)/64, whose mean is ~0, so its 512-wide dot products cancel
// to near-zero scores and the softmax is nearly uniform. That test passes even
// when the score scale inside the kernel is perturbed by 1%. The ramps here are
// biased away from zero so scores are large enough for the softmax to be
// sensitive to them.
func TestRopeAttentionMixedWideMatchesNarrow(t *testing.T) {
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

	const rowCap = 96
	q := make([]float32, 64*512)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.031)) + 0.35
	}
	kv := make([]float32, 512)
	for i := range kv {
		kv[i] = float32(math.Cos(float64(i)*0.017)) + 0.42
	}
	sinkValues := make([]float32, 64)
	for i := range sinkValues {
		sinkValues[i] = float32(i%7-3) / 8
	}
	sinks := testBatchF32Tensor(arena, "sinks", sinkValues)
	compValues := make([]float32, 3*512)
	for i := range compValues {
		compValues[i] = float32(math.Sin(float64(i)*0.011)) - 0.28
	}
	comp := metalmem.UploadIn(arena, "comp", compValues)
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		position int
		nComp    int
		allowed  []uint32
	}{
		// 95, 96 and 97 straddle the rowCap=96 wrap, where position%raw_cap
		// returns to 0 and first stops being 0; 127, 128, 129, 511 and 2047 are
		// past it, deep enough that every raw row has been overwritten.
		{position: 0, nComp: 0},
		{position: 1, nComp: 1, allowed: []uint32{1}},
		{position: 17, nComp: 3, allowed: []uint32{1, 0, 1}},
		{position: 63, nComp: 2, allowed: []uint32{1, 1}},
		{position: 95, nComp: 2, allowed: []uint32{1, 1}},
		{position: 96, nComp: 0},
		{position: 97, nComp: 3, allowed: []uint32{1, 0, 1}},
		{position: 127, nComp: 3, allowed: []uint32{1, 1, 0}},
		{position: 128, nComp: 0},
		{position: 129, nComp: 3, allowed: []uint32{0, 1, 1}},
		{position: 200, nComp: 3, allowed: []uint32{0, 1, 1}},
		{position: 511, nComp: 2, allowed: []uint32{1, 1}},
		{position: 2047, nComp: 3, allowed: []uint32{1, 0, 1}},
	} {
		t.Run(fmt.Sprintf("pos%d_ncomp%d", tc.position, tc.nComp), func(t *testing.T) {
			// Separate raw caches, filled identically, so neither path observes
			// the other's writes.
			narrowRaw := metalmem.AllocIn[float32](arena, fmt.Sprintf("narrow-raw-%d", tc.position), rowCap*512)
			wideRaw := metalmem.AllocIn[float32](arena, fmt.Sprintf("wide-raw-%d", tc.position), rowCap*512)
			if err := arena.Err(); err != nil {
				t.Fatal(err)
			}
			nRaw := min(tc.position+1, rowCap)
			for row := range nRaw {
				for i := range 512 {
					v := float32(math.Sin(float64(row*512+i)*0.007)) + 0.31
					narrowRaw.HostSlice()[row*512+i] = v
					wideRaw.HostSlice()[row*512+i] = v
				}
			}
			wantQ, wantKV, wantHeads, err := ops.ropeAttentionMixed(q, kv, sinks, narrowRaw, comp, tc.nComp, tc.allowed, tc.position, 2, false)
			if err != nil {
				t.Fatal(err)
			}
			gotQ, gotKV, gotHeads, err := ops.ropeAttentionMixed(q, kv, sinks, wideRaw, comp, tc.nComp, tc.allowed, tc.position, 2, true)
			if err != nil {
				t.Fatal(err)
			}
			exactSlice(t, "roped q", gotQ, wantQ)
			exactSlice(t, "roped kv", gotKV, wantKV)
			exactSlice(t, "heads", gotHeads, wantHeads)
			exactSlice(t, "raw cache", wideRaw.HostSlice(), narrowRaw.HostSlice())

			// Guard against a degenerate comparison: if the heads were all
			// near zero, agreement would say nothing.
			var peak float64
			for _, v := range wantHeads {
				peak = math.Max(peak, math.Abs(float64(v)))
			}
			if peak < 1e-3 {
				t.Fatalf("reference output is degenerate (peak %g); the comparison would be vacuous", peak)
			}
		})
	}
}

// TestBatchRopeAttentionMixedWideMatchesNarrow anchors the batch encoder -- the
// path real generation takes -- against the original single-threadgroup kernel.
//
// TestMetalBatchCompressedMixedAttention compares the batch encoder against
// metalPinnedOps.RopeAttentionMixed, which is a useful cross-check only while
// the two differ. Both are now wide, so that test can no longer catch an error
// they share. The narrow kernel is the anchor, and this test holds the batch
// path to it.
func TestBatchRopeAttentionMixedWideMatchesNarrow(t *testing.T) {
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

	const rowCap = 96
	qValues := make([]float32, 64*512)
	for i := range qValues {
		qValues[i] = float32(math.Sin(float64(i)*0.031)) + 0.35
	}
	kvValues := make([]float32, 512)
	for i := range kvValues {
		kvValues[i] = float32(math.Cos(float64(i)*0.017)) + 0.42
	}
	sinkValues := make([]float32, 64)
	for i := range sinkValues {
		sinkValues[i] = float32(i%7-3) / 8
	}
	compValues := make([]float32, 3*512)
	for i := range compValues {
		compValues[i] = float32(math.Sin(float64(i)*0.011)) - 0.28
	}
	sinks := testBatchF32Tensor(arena, "b-sinks", sinkValues)
	comp := metalmem.UploadIn(arena, "b-comp", compValues)
	q := metalmem.UploadIn(arena, "b-q", qValues)
	kv := metalmem.UploadIn(arena, "b-kv", kvValues)
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		position int
		nComp    int
		allowed  []uint32
	}{
		{position: 0, nComp: 0, allowed: []uint32{1}},
		{position: 17, nComp: 3, allowed: []uint32{1, 0, 1}},
		{position: 95, nComp: 2, allowed: []uint32{1, 1}},
		{position: 96, nComp: 0, allowed: []uint32{1}},
		{position: 97, nComp: 3, allowed: []uint32{1, 0, 1}},
		{position: 127, nComp: 3, allowed: []uint32{1, 1, 0}},
		{position: 128, nComp: 0, allowed: []uint32{1}},
		{position: 129, nComp: 3, allowed: []uint32{0, 1, 1}},
		{position: 200, nComp: 2, allowed: []uint32{1, 1}},
		{position: 511, nComp: 2, allowed: []uint32{1, 1}},
		{position: 2047, nComp: 3, allowed: []uint32{1, 0, 1}},
	} {
		t.Run(fmt.Sprintf("pos%d_ncomp%d", tc.position, tc.nComp), func(t *testing.T) {
			narrowRaw := metalmem.AllocIn[float32](arena, fmt.Sprintf("b-narrow-raw-%d", tc.position), rowCap*512)
			batchRaw := metalmem.AllocIn[float32](arena, fmt.Sprintf("b-wide-raw-%d", tc.position), rowCap*512)
			nRaw := min(tc.position+1, rowCap)
			scores := metalmem.AllocIn[float32](arena, fmt.Sprintf("b-scores-%d", tc.position), 64*(nRaw+tc.nComp))
			out := metalmem.AllocIn[float32](arena, fmt.Sprintf("b-out-%d", tc.position), 32768+512+32768)
			allowed := metalmem.UploadIn(arena, fmt.Sprintf("b-allowed-%d", tc.position), tc.allowed)
			if err := arena.Err(); err != nil {
				t.Fatal(err)
			}
			for row := range nRaw {
				for i := range 512 {
					v := float32(math.Sin(float64(row*512+i)*0.007)) + 0.31
					narrowRaw.HostSlice()[row*512+i] = v
					batchRaw.HostSlice()[row*512+i] = v
				}
			}
			_, _, wantHeads, err := ops.ropeAttentionMixed(qValues, kvValues, sinks, narrowRaw, comp, tc.nComp, tc.allowed, tc.position, 2, false)
			if err != nil {
				t.Fatal(err)
			}
			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.EncodeRopeAttentionMixed(q, kv, sinks, batchRaw, comp, tc.nComp, allowed, scores, out, tc.position, 2); err != nil {
				t.Fatal(err)
			}
			if err := batch.Wait(t.Context()); err != nil {
				t.Fatal(err)
			}
			if got := batch.Stats().Encoded; got != 1 {
				t.Fatalf("Encoded = %d, want 1; the four dispatches are one operation at this API level", got)
			}
			exactSlice(t, "heads", out.HostSlice()[32768+512:], wantHeads)
			exactSlice(t, "raw cache", batchRaw.HostSlice(), narrowRaw.HostSlice())
		})
	}
}
