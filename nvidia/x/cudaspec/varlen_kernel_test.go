// SPDX-License-Identifier: Apache-2.0

package cudaspec

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestComputeSeqlensQByConfidenceCPU(t *testing.T) {
	tests := []struct {
		name           string
		prob           [][]float32
		remap          []int32
		targetTotalBsz int
		maxStep        int
		wantSeqlensQ   []int32
		wantCuSeqlensQ []int32
		wantErr        bool
	}{
		{
			name: "basic_ranking_and_cumprod",
			prob: [][]float32{
				{0.9, 0.8, 0.5}, // cum: 0.90, 0.72, 0.36
				{0.8, 0.5, 0.5}, // cum: 0.80, 0.40, 0.20
				{0.7, 0.2, 0.1}, // cum: 0.70, 0.14, 0.014
			},
			remap:          nil,
			targetTotalBsz: 6, // bsz=3, extraBudget=3 -> top 3 are 0.90 (seq0), 0.80 (seq1), 0.72 (seq0)
			maxStep:        3,
			wantSeqlensQ:   []int32{3, 2, 1},
			wantCuSeqlensQ: []int32{0, 3, 5, 6},
		},
		{
			name: "tie_breaking_by_flat_index",
			prob: [][]float32{
				{0.5, 1.0, 0.2}, // cum: 0.5, 0.5, 0.1 (flat indices 0, 1, 2)
				{0.5, 1.0, 0.2}, // cum: 0.5, 0.5, 0.1 (flat indices 3, 4, 5)
			},
			remap:          nil,
			targetTotalBsz: 5, // bsz=2, extraBudget=3 -> flat indices 0, 1, 3 win ties over 4
			maxStep:        3,
			wantSeqlensQ:   []int32{3, 2},
			wantCuSeqlensQ: []int32{0, 3, 5},
		},
		{
			name: "with_remap_and_fresh_sequence",
			prob: [][]float32{
				{0.4, 0.5}, // row 0 -> cum: 0.4, 0.2
				{0.9, 0.9}, // row 1 -> cum: 0.9, 0.81
			},
			remap:          []int32{1, -1, 0}, // seq0->row1, seq1->fresh(-1 -> 0,0), seq2->row0
			targetTotalBsz: 6,                 // bsz=3, extraBudget=3 -> top 3: seq0(0.9, 0.81), seq2(0.4)
			maxStep:        2,
			wantSeqlensQ:   []int32{3, 1, 2},
			wantCuSeqlensQ: []int32{0, 3, 4, 6},
		},
		{
			name: "clamp_budget_below_bsz",
			prob: [][]float32{
				{0.9, 0.9},
				{0.8, 0.8},
			},
			remap:          nil,
			targetTotalBsz: 1, // clamped to extraBudget=0
			maxStep:        2,
			wantSeqlensQ:   []int32{1, 1},
			wantCuSeqlensQ: []int32{0, 1, 2},
		},
		{
			name: "clamp_budget_above_max",
			prob: [][]float32{
				{0.9, 0.9},
				{0.8, 0.8},
			},
			remap:          nil,
			targetTotalBsz: 100, // clamped to extraBudget = 2*2 = 4
			maxStep:        2,
			wantSeqlensQ:   []int32{3, 3},
			wantCuSeqlensQ: []int32{0, 3, 6},
		},
		{
			name:           "invalid_empty_batch",
			prob:           nil,
			targetTotalBsz: 4,
			maxStep:        2,
			wantErr:        true,
		},
		{
			name: "invalid_remap_out_of_bounds",
			prob: [][]float32{
				{0.5, 0.5},
			},
			remap:          []int32{2},
			targetTotalBsz: 2,
			maxStep:        2,
			wantErr:        true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotQ, gotCu, err := ComputeSeqlensQByConfidenceCPU(tc.prob, tc.remap, tc.targetTotalBsz, tc.maxStep)
			if tc.wantErr {
				if !errors.Is(err, ErrInvalidVarlenSpec) {
					t.Fatalf("expected ErrInvalidVarlenSpec, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ComputeSeqlensQByConfidenceCPU: %v", err)
			}
			if !reflect.DeepEqual(gotQ, tc.wantSeqlensQ) {
				t.Fatalf("seqlensQ = %v, want %v", gotQ, tc.wantSeqlensQ)
			}
			if !reflect.DeepEqual(gotCu, tc.wantCuSeqlensQ) {
				t.Fatalf("cuSeqlensQ = %v, want %v", gotCu, tc.wantCuSeqlensQ)
			}
		})
	}
}

func TestVarlenSeqlensKernelSource(t *testing.T) {
	if _, err := VarlenSeqlensKernelSource(0, 4); !errors.Is(err, ErrInvalidVarlenSpec) {
		t.Fatalf("expected ErrInvalidVarlenSpec for maxBsz=0, got %v", err)
	}
	src, err := VarlenSeqlensKernelSource(64, 5)
	if err != nil {
		t.Fatalf("VarlenSeqlensKernelSource: %v", err)
	}
	for _, needle := range []string{
		`extern "C" __global__ void compute_seqlens_q_by_conf_kernel`,
		`s_cum`,
		`other_val > my_val || (other_val == my_val && other_idx < my_idx)`,
		`cu_seqlens_q[bsz] = prefix`,
	} {
		if !strings.Contains(src, needle) {
			t.Fatalf("kernel source missing %q", needle)
		}
	}
}
