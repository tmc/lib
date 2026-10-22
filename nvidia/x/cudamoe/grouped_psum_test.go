// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"math"
	"reflect"
	"testing"
)

func TestGroupedGEMMPsumCPU(t *testing.T) {
	a := [][]float32{
		{1, 2, 3, 4, 5, 6}, // group 0: M=2, K=3
		{2, 0, 1},          // group 1: M=1, K=3
	}
	b := [][]float32{
		{1, 0, 0, 1, 1, 1}, // group 0: K=3, N=2
		{3, 2, 1, 4, 0, 5}, // group 1: K=3, N=2
	}
	psum := [][]float32{
		{10, 20, 30, 40}, // group 0: M=2, N=2
		{5, 6},           // group 1: M=1, N=2
	}

	cfg := GroupedPsumGEMMConfig{
		Groups:         2,
		M:              []int{2, 1},
		N:              2,
		K:              3,
		AccumulatePsum: true,
		ScaleD:         0.5,
	}
	got, err := GroupedGEMMPsumCPU(a, b, psum, cfg)
	if err != nil {
		t.Fatalf("GroupedGEMMPsumCPU unexpected error: %v", err)
	}

	// Group 0:
	// A_0 * B_0 = [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]] = [[4, 5], [10, 11]]
	// D_0 = 0.5 * [[4, 5], [10, 11]] + [[10, 20], [30, 40]] = [[12, 22.5], [35, 45.5]]
	// Group 1:
	// A_1 * B_1 = [[2*3+0*1+1*0, 2*2+0*4+1*5]] = [[6, 9]]
	// D_1 = 0.5 * [[6, 9]] + [[5, 6]] = [[8, 10.5]]
	want := [][]float32{
		{12, 22.5, 35, 45.5},
		{8, 10.5},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GroupedGEMMPsumCPU = %v, want %v", got, want)
	}

	// Without AccumulatePsum:
	cfgNoPsum := cfg
	cfgNoPsum.AccumulatePsum = false
	cfgNoPsum.ScaleD = 1.0
	gotNoPsum, err := GroupedGEMMPsumCPU(a, b, nil, cfgNoPsum)
	if err != nil {
		t.Fatalf("GroupedGEMMPsumCPU (no psum) error: %v", err)
	}
	wantNoPsum := [][]float32{
		{4, 5, 10, 11},
		{6, 9},
	}
	if !reflect.DeepEqual(gotNoPsum, wantNoPsum) {
		t.Fatalf("GroupedGEMMPsumCPU (no psum) = %v, want %v", gotNoPsum, wantNoPsum)
	}
}

func TestComputeCommSMPartition(t *testing.T) {
	tests := []struct {
		name       string
		totalSMs   int
		commRatio  float64
		minCommSMs int
		wantComm   int
		wantComp   int
		wantErr    bool
	}{
		{name: "b200_148_sms_24_comm", totalSMs: 148, commRatio: 24.0 / 148.0, minCommSMs: 4, wantComm: 24, wantComp: 124},
		{name: "odd_ratio_rounds_to_even_cta_pair", totalSMs: 148, commRatio: 25.0 / 148.0, minCommSMs: 4, wantComm: 24, wantComp: 124},
		{name: "clamp_to_min_comm_sms", totalSMs: 132, commRatio: 0.01, minCommSMs: 8, wantComm: 8, wantComp: 124},
		{name: "clamp_to_max_comm_sms", totalSMs: 64, commRatio: 0.99, minCommSMs: 4, wantComm: 62, wantComp: 2},
		{name: "invalid_total_sms", totalSMs: 2, commRatio: 0.5, minCommSMs: 2, wantErr: true},
		{name: "invalid_ratio", totalSMs: 148, commRatio: 1.5, minCommSMs: 4, wantErr: true},
		{name: "invalid_min_comm", totalSMs: 148, commRatio: 0.2, minCommSMs: 0, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			comm, comp, err := ComputeCommSMPartition(tc.totalSMs, tc.commRatio, tc.minCommSMs)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got comm=%d comp=%d", comm, comp)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if comm != tc.wantComm || comp != tc.wantComp {
				t.Fatalf("ComputeCommSMPartition(%d, %v, %d) = (%d, %d), want (%d, %d)",
					tc.totalSMs, tc.commRatio, tc.minCommSMs, comm, comp, tc.wantComm, tc.wantComp)
			}
			if comm%2 != 0 {
				t.Fatalf("commSMs %d must be an even CTA pair", comm)
			}
			if comm+comp != tc.totalSMs {
				t.Fatalf("comm (%d) + compute (%d) != totalSMs (%d)", comm, comp, tc.totalSMs)
			}
		})
	}
}

func TestSqrtSoftplus(t *testing.T) {
	// Normal range: sqrt(log(1 + exp(0))) = sqrt(ln(2))
	got0 := SqrtSoftplus(0)
	want0 := float32(math.Sqrt(math.Ln2))
	if math.Abs(float64(got0-want0)) > 1e-6 {
		t.Fatalf("SqrtSoftplus(0) = %v, want %v", got0, want0)
	}

	// Large positive x > 20 branch: sqrt(x) without overflow
	gotLarge := SqrtSoftplus(100)
	if math.Abs(float64(gotLarge-10.0)) > 1e-6 {
		t.Fatalf("SqrtSoftplus(100) = %v, want 10", gotLarge)
	}
}

func TestMoEGateNoAuxTCCPU(t *testing.T) {
	// 8 experts in 4 groups (2 experts/group):
	// Group 0: experts 0, 1
	// Group 1: experts 2, 3
	// Group 2: experts 4, 5
	// Group 3: experts 6, 7
	// Token 0 (text): bias steers toward groups 0 and 2.
	// Token 1 (vision): biasVL steers toward groups 1 and 3.
	logits := []float32{
		// Token 0
		1.0, 0.8, 0.9, 0.7, 1.1, 0.6, 0.5, 0.4,
		// Token 1
		1.0, 0.8, 0.9, 0.7, 1.1, 0.6, 0.5, 0.4,
	}
	correctionBias := []float32{0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0}
	biasVL := []float32{0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0}
	visionMask := []bool{false, true}

	cfg := NoAuxTCGateConfig{
		NumExperts:          8,
		NumGroups:           4,
		TopKGroup:           2,
		TopK:                3,
		ScoringFunc:         "sqrtsoftplus",
		RoutedScalingFactor: 2.5,
		UseBiasVL:           true,
	}

	ids, weights, err := MoEGateNoAuxTCCPU(logits, correctionBias, biasVL, visionMask, cfg)
	if err != nil {
		t.Fatalf("MoEGateNoAuxTCCPU unexpected error: %v", err)
	}
	if len(ids) != 2 || len(weights) != 2 {
		t.Fatalf("got %d tokens, want 2", len(ids))
	}

	// Token 0 selects groups 0 and 2 (experts {0,1,4,5}); top-3 choiceScores are 4, 0, 1.
	wantIds0 := []int32{4, 0, 1}
	if !reflect.DeepEqual(ids[0], wantIds0) {
		t.Fatalf("token 0 topkIds = %v, want %v", ids[0], wantIds0)
	}

	// Token 1 (vision) uses biasVL, selecting groups 1 and 3 (experts {2,3,6,7}); top-3 are 2, 3, 6.
	wantIds1 := []int32{2, 3, 6}
	if !reflect.DeepEqual(ids[1], wantIds1) {
		t.Fatalf("token 1 topkIds = %v, want %v", ids[1], wantIds1)
	}

	// Check normalized weight sum equals RoutedScalingFactor (2.5) for both tokens.
	for tok := 0; tok < 2; tok++ {
		var sum float32
		for _, w := range weights[tok] {
			sum += w
		}
		if math.Abs(float64(sum-cfg.RoutedScalingFactor)) > 1e-5 {
			t.Fatalf("token %d weight sum = %v, want %v", tok, sum, cfg.RoutedScalingFactor)
		}
	}
}
