// SPDX-License-Identifier: Apache-2.0

package cudabatch

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestBuildZOSRemapPlan(t *testing.T) {
	tests := []struct {
		name           string
		prevCuSeqlensQ []int32
		acceptedCounts []int32
		nextSeqlensQ   []int32
		basePositions  []int32
		wantPlan       ZOSRemapPlan
		wantErr        bool
	}{
		{
			name:           "three_sequences_varied_acceptance",
			prevCuSeqlensQ: []int32{0, 4, 7, 11},
			acceptedCounts: []int32{3, 1, 4},
			nextSeqlensQ:   []int32{2, 3, 1},
			basePositions:  []int32{100, 50, 200},
			wantPlan: ZOSRemapPlan{
				Remap:            []int32{2, 4, 10},
				SeqlensQ:         []int32{2, 3, 1},
				CuSeqlensQ:       []int32{0, 2, 5, 6},
				Positions:        []int32{103, 104, 51, 52, 53, 204},
				OutToIn:          []int32{0, 0, 1, 1, 1, 2},
				TotalDraftTokens: 6,
			},
		},
		{
			name:           "invalid_empty_batch",
			prevCuSeqlensQ: nil,
			acceptedCounts: nil,
			nextSeqlensQ:   nil,
			basePositions:  nil,
			wantErr:        true,
		},
		{
			name:           "invalid_accepted_exceeds_prev_seqlen",
			prevCuSeqlensQ: []int32{0, 2},
			acceptedCounts: []int32{3},
			nextSeqlensQ:   []int32{2},
			basePositions:  []int32{10},
			wantErr:        true,
		},
		{
			name:           "invalid_zero_accepted",
			prevCuSeqlensQ: []int32{0, 3},
			acceptedCounts: []int32{0},
			nextSeqlensQ:   []int32{1},
			basePositions:  []int32{10},
			wantErr:        true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := BuildZOSRemapPlan(tc.prevCuSeqlensQ, tc.acceptedCounts, tc.nextSeqlensQ, tc.basePositions)
			if tc.wantErr {
				if !errors.Is(err, ErrInvalidZOSPlan) {
					t.Fatalf("expected ErrInvalidZOSPlan, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildZOSRemapPlan: %v", err)
			}
			if !reflect.DeepEqual(got, tc.wantPlan) {
				t.Fatalf("got plan %+v, want %+v", got, tc.wantPlan)
			}
		})
	}
}

func TestApplyZOSRemapCPU(t *testing.T) {
	prevCu := []int32{0, 4, 7}
	accepted := []int32{3, 2}
	nextQ := []int32{3, 2}
	basePos := []int32{10, 20}

	plan, err := BuildZOSRemapPlan(prevCu, accepted, nextQ, basePos)
	if err != nil {
		t.Fatalf("BuildZOSRemapPlan: %v", err)
	}
	// Remap[0] = 0 + 3 - 1 = 2 -> prevTokens[2] = 103
	// Remap[1] = 4 + 2 - 1 = 5 -> prevTokens[5] = 202
	prevTokens := []int32{
		101, 102, 103, 104, // seq 0
		201, 202, 203, // seq 1
	}
	// Stride 2 bonus draft tokens per sequence:
	// seq 0 extra 2 tokens: 1001, 1002
	// seq 1 extra 1 token:  2001, (2002 unused)
	bonusTokens := []int32{
		1001, 1002,
		2001, 2002,
	}
	prevSeqlensK := []int32{14, 23}

	nextIDs, nextK, err := ApplyZOSRemapCPU(prevTokens, bonusTokens, prevSeqlensK, accepted, plan)
	if err != nil {
		t.Fatalf("ApplyZOSRemapCPU: %v", err)
	}

	wantIDs := []int32{103, 1001, 1002, 202, 2001}
	wantK := []int32{17, 25}
	if !reflect.DeepEqual(nextIDs, wantIDs) {
		t.Fatalf("nextInputIDs = %v, want %v", nextIDs, wantIDs)
	}
	if !reflect.DeepEqual(nextK, wantK) {
		t.Fatalf("nextSeqlensK = %v, want %v", nextK, wantK)
	}

	// Out-of-bounds Remap check.
	badPlan := plan
	badPlan.Remap = []int32{99, 5}
	if _, _, err := ApplyZOSRemapCPU(prevTokens, bonusTokens, prevSeqlensK, accepted, badPlan); !errors.Is(err, ErrInvalidZOSPlan) {
		t.Fatalf("expected ErrInvalidZOSPlan for out-of-bounds Remap, got %v", err)
	}
}

func TestZOSRemapKernelSource(t *testing.T) {
	src, err := ZOSRemapKernelSource()
	if err != nil {
		t.Fatalf("ZOSRemapKernelSource: %v", err)
	}
	for _, needle := range []string{
		`extern "C" __global__ void zos_remap_and_pack_kernel`,
		`prev_cu_seqlens_q[b] + acc - 1`,
		`next_seqlens_k[b] = prev_seqlens_k[b] + acc`,
		`prev_tokens[remap_idx]`,
	} {
		if !strings.Contains(src, needle) {
			t.Fatalf("kernel source missing %q", needle)
		}
	}
}
