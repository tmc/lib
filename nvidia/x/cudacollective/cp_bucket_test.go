package cudacollective

import (
	"reflect"
	"testing"
)

func TestComputeCPBucketBytes(t *testing.T) {
	tests := []struct {
		name         string
		spec         CPBucketSpec
		wantPerRank  int64
		wantGathered int64
		wantErr      bool
	}{
		{
			name: "padded_per_rank_tokens",
			spec: CPBucketSpec{
				MaxBatchTokens: 1000,
				CPSize:         4,
				HiddenDim:      128,
				ElemBytes:      2,
				PadMultiple:    64,
			},
			// ceil(1000/4) = 250 -> padded to 64 = 256 tokens/rank
			// perRankBytes = 256 * 128 * 2 = 65536
			// gatheredBytes = 65536 * 4 = 262144
			wantPerRank:  65536,
			wantGathered: 262144,
		},
		{
			name: "exact_multiple_no_extra_pad",
			spec: CPBucketSpec{
				MaxBatchTokens: 512,
				CPSize:         4,
				HiddenDim:      64,
				ElemBytes:      2,
				PadMultiple:    32,
			},
			wantPerRank:  128 * 64 * 2,
			wantGathered: 512 * 64 * 2,
		},
		{
			name:    "invalid_max_tokens",
			spec:    CPBucketSpec{MaxBatchTokens: 0, CPSize: 2, HiddenDim: 64, ElemBytes: 2},
			wantErr: true,
		},
		{
			name:    "invalid_cp_size",
			spec:    CPBucketSpec{MaxBatchTokens: 64, CPSize: 0, HiddenDim: 64, ElemBytes: 2},
			wantErr: true,
		},
		{
			name:    "negative_pad_multiple",
			spec:    CPBucketSpec{MaxBatchTokens: 64, CPSize: 2, HiddenDim: 64, ElemBytes: 2, PadMultiple: -4},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			perRank, gathered, err := ComputeCPBucketBytes(tc.spec)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got (%d, %d)", perRank, gathered)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if perRank != tc.wantPerRank || gathered != tc.wantGathered {
				t.Fatalf("ComputeCPBucketBytes(%+v) = (%d, %d), want (%d, %d)",
					tc.spec, perRank, gathered, tc.wantPerRank, tc.wantGathered)
			}
		})
	}
}

func TestPrepareCPMirrorIndices(t *testing.T) {
	totalSeqLen := 16
	cpSize := 4
	// 2*cpSize = 8 chunks of 2 tokens each:
	// chunk 0: [0, 1], chunk 1: [2, 3], chunk 2: [4, 5], chunk 3: [6, 7],
	// chunk 4: [8, 9], chunk 5: [10, 11], chunk 6: [12, 13], chunk 7: [14, 15].
	wantLocal := [][]int32{
		{0, 1},
		{2, 3},
		{4, 5},
		{6, 7},
	}
	wantMirror := [][]int32{
		{14, 15},
		{12, 13},
		{10, 11},
		{8, 9},
	}

	var expectedCausalWork int
	seen := make([]bool, totalSeqLen)

	for r := 0; r < cpSize; r++ {
		plan, err := PrepareCPMirrorIndices(totalSeqLen, cpSize, r)
		if err != nil {
			t.Fatalf("rank %d unexpected error: %v", r, err)
		}
		if plan.ChunkSize != 2 {
			t.Fatalf("rank %d ChunkSize = %d, want 2", r, plan.ChunkSize)
		}
		if !reflect.DeepEqual(plan.LocalIndices, wantLocal[r]) {
			t.Fatalf("rank %d LocalIndices = %v, want %v", r, plan.LocalIndices, wantLocal[r])
		}
		if !reflect.DeepEqual(plan.MirrorIndices, wantMirror[r]) {
			t.Fatalf("rank %d MirrorIndices = %v, want %v", r, plan.MirrorIndices, wantMirror[r])
		}

		// Causal attention work for token i is proportional to (i + 1).
		// With symmetric mirror pairing, every rank has identical total causal work!
		work := 0
		for _, idx := range plan.LocalIndices {
			seen[idx] = true
			work += int(idx) + 1
		}
		for _, idx := range plan.MirrorIndices {
			seen[idx] = true
			work += int(idx) + 1
		}
		if r == 0 {
			expectedCausalWork = work
		} else if work != expectedCausalWork {
			t.Fatalf("rank %d causal work = %d, want balanced work %d", r, work, expectedCausalWork)
		}
	}

	for i, ok := range seen {
		if !ok {
			t.Fatalf("token index %d was not assigned to any CP rank", i)
		}
	}

	// Error cases
	if _, err := PrepareCPMirrorIndices(15, 4, 0); err == nil {
		t.Fatalf("expected error when totalSeqLen not divisible by 2*cpSize")
	}
	if _, err := PrepareCPMirrorIndices(16, 4, 4); err == nil {
		t.Fatalf("expected error when rank >= cpSize")
	}
}
