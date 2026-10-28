// SPDX-License-Identifier: Apache-2.0

package cudaagreement_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaagreement"
)

func TestCheckBatchInvariance(t *testing.T) {
	base := []float32{1.25, -3.5, 0.0, 42.125}
	ulpShift := math.Float32frombits(math.Float32bits(base[1]) + 1)
	negZero := float32(math.Copysign(0, -1))

	// 1. All bit-identical cases pass.
	repOK, err := cudaagreement.CheckBatchInvariance([]cudaagreement.BatchInvarianceCase{
		{
			SeqID:         "seq-0",
			Input:         base,
			SoloOutput:    base,
			BatchedOutput: []float32{1.25, -3.5, 0.0, 42.125},
			BatchSize:     4,
			RowIndex:      2,
		},
	})
	if err != nil {
		t.Fatalf("CheckBatchInvariance(identical): %v", err)
	}
	if !repOK.Passed() || repOK.TotalCases != 1 || repOK.BitIdenticalCases != 1 || repOK.MaxAbsDiff != 0 {
		t.Fatalf("unexpected report for identical case: %+v", repOK)
	}

	// 2. Single-ULP drift and larger later drift in the same case: records first mismatch and overall MaxAbsDiff.
	batchedDrift := []float32{1.25, ulpShift, 0.0, 42.625}
	repULP, err := cudaagreement.CheckBatchInvariance([]cudaagreement.BatchInvarianceCase{
		{
			SeqID:         "seq-ulp",
			Input:         base,
			SoloOutput:    base,
			BatchedOutput: batchedDrift,
			BatchSize:     8,
			RowIndex:      5,
		},
	})
	if err != nil {
		t.Fatalf("CheckBatchInvariance(ulp): %v", err)
	}
	if repULP.Passed() || repULP.BitIdenticalCases != 0 || len(repULP.Mismatches) != 1 {
		t.Fatalf("expected 1 mismatch on ULP drift, got %+v", repULP)
	}
	mm := repULP.Mismatches[0]
	if mm.SeqID != "seq-ulp" || mm.BatchSize != 8 || mm.RowIndex != 5 || mm.ElemIndex != 1 {
		t.Fatalf("unexpected mismatch metadata: %+v", mm)
	}
	if mm.SoloBits != math.Float32bits(base[1]) || mm.BatchedBits != math.Float32bits(ulpShift) {
		t.Fatalf("unexpected mismatch bits: %#x vs %#x", mm.SoloBits, mm.BatchedBits)
	}
	if math.Abs(repULP.MaxAbsDiff-0.5) > 1e-6 {
		t.Fatalf("MaxAbsDiff = %v, want 0.5", repULP.MaxAbsDiff)
	}

	// 3. Signed-zero difference (-0.0 vs +0.0) has AbsDiff == 0 but different IEEE-754 bits and must fail!
	repZero, err := cudaagreement.CheckBatchInvariance([]cudaagreement.BatchInvarianceCase{
		{
			SeqID:         "seq-negzero",
			Input:         base,
			SoloOutput:    []float32{0.0},
			BatchedOutput: []float32{negZero},
			BatchSize:     2,
			RowIndex:      1,
		},
	})
	if err != nil {
		t.Fatalf("CheckBatchInvariance(negzero): %v", err)
	}
	if repZero.Passed() || len(repZero.Mismatches) != 1 {
		t.Fatalf("expected CheckBatchInvariance to reject -0.0 vs +0.0, got %+v", repZero)
	}

	// 4. Validation errors.
	if _, err := cudaagreement.CheckBatchInvariance(nil); err == nil {
		t.Fatal("expected error for empty cases")
	}
	if _, err := cudaagreement.CheckBatchInvariance([]cudaagreement.BatchInvarianceCase{
		{SeqID: "bad-bs", SoloOutput: []float32{1}, BatchedOutput: []float32{1}, BatchSize: 0, RowIndex: 0},
	}); err == nil {
		t.Fatal("expected error for BatchSize <= 0")
	}
	if _, err := cudaagreement.CheckBatchInvariance([]cudaagreement.BatchInvarianceCase{
		{SeqID: "bad-row", SoloOutput: []float32{1}, BatchedOutput: []float32{1}, BatchSize: 2, RowIndex: 2},
	}); err == nil {
		t.Fatal("expected error for RowIndex >= BatchSize")
	}
	if _, err := cudaagreement.CheckBatchInvariance([]cudaagreement.BatchInvarianceCase{
		{SeqID: "bad-len", SoloOutput: []float32{1, 2}, BatchedOutput: []float32{1}, BatchSize: 2, RowIndex: 0},
	}); err == nil {
		t.Fatal("expected error for length mismatch")
	}
}

func TestVerifyBatchInvarianceAndFixedSplitKReductionCPU(t *testing.T) {
	const numInputs = 6
	const rowLen = 128
	inputs := make([][]float32, numInputs)
	for i := 0; i < numInputs; i++ {
		row := make([]float32, rowLen)
		for j := 0; j < rowLen; j++ {
			// Alternating magnitudes and signs sensitive to floating-point summation order.
			sign := float32(1.0)
			if (i+j)%3 == 1 {
				sign = -1.0
			}
			row[j] = sign * (float32(int(1)<<(uint(j)%7))*0.1234567 + float32(i+1)*1e-4)
		}
		inputs[i] = row
	}
	batchSizes := []int{1, 2, 4, 8, 16}

	// Positive control: FixedSplitKReductionCPU uses a fixed tile size (16) independent of batch size.
	fixedFn := func(batch [][]float32) ([][]float32, error) {
		out := make([][]float32, len(batch))
		for r, row := range batch {
			out[r] = []float32{cudaagreement.FixedSplitKReductionCPU(row, 16)}
		}
		return out, nil
	}
	repFixed, err := cudaagreement.VerifyBatchInvariance(inputs, batchSizes, fixedFn)
	if err != nil {
		t.Fatalf("VerifyBatchInvariance(FixedSplitKReductionCPU): %v", err)
	}
	if !repFixed.Passed() {
		t.Fatalf("FixedSplitKReductionCPU failed batch invariance: %+v", repFixed)
	}
	if repFixed.TotalCases < 60 || repFixed.BitIdenticalCases != repFixed.TotalCases {
		t.Fatalf("unexpected case counts: %+v", repFixed)
	}

	// Negative control: a dynamic split-K reduction whose tile size depends on len(batch)
	// alters the float32 addition association order and must be caught by VerifyBatchInvariance.
	nonInvariantFn := func(batch [][]float32) ([][]float32, error) {
		tileSize := 64 / len(batch)
		if tileSize < 2 {
			tileSize = 2
		}
		out := make([][]float32, len(batch))
		for r, row := range batch {
			out[r] = []float32{cudaagreement.FixedSplitKReductionCPU(row, tileSize)}
		}
		return out, nil
	}
	repBad, err := cudaagreement.VerifyBatchInvariance(inputs, batchSizes, nonInvariantFn)
	if err != nil {
		t.Fatalf("VerifyBatchInvariance(nonInvariantFn): %v", err)
	}
	if repBad.Passed() || len(repBad.Mismatches) == 0 {
		t.Fatalf("expected non-invariant reduction to fail VerifyBatchInvariance, got %+v", repBad)
	}
}
