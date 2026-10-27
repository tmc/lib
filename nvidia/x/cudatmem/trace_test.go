package cudatmem

import (
	"errors"
	"testing"
)

func validCTA1BlockScaledTrace() []TraceEvent {
	events := []TraceEvent{
		{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		{Kind: EventCp, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		{Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128, HasScaleInTMEM: true, ScaleOffset: 128},
		{Kind: EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		{Kind: EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
	}
	for w := range 4 {
		events = append(events, TraceEvent{
			Kind: EventLd, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1,
			Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
		})
	}
	events = append(events,
		TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
	)
	return events
}

func validCTA2Trace() []TraceEvent {
	events := []TraceEvent{
		{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: EventAlloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: EventClusterSync, CTAGroup: 2},
		{Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: EventWait, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
	}
	for cta := range 2 {
		for w := range 4 {
			events = append(events, TraceEvent{
				Kind: EventLd, CTA: cta, Warp: w, ActiveThreads: 32, CTAGroup: 2,
				Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
			})
		}
	}
	events = append(events,
		TraceEvent{Kind: EventClusterSync, CTAGroup: 2},
		TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		TraceEvent{Kind: EventDealloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
	)
	return events
}

func validTMEMOperandATrace() []TraceEvent {
	events := []TraceEvent{
		{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
		{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
		// S = Q * K^T into offset 0
		{Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
		{Kind: EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
		{Kind: EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
	}
	// Rescale S -> P in TMEM using 4-warp 32-row ld/st
	for w := range 4 {
		events = append(events,
			TraceEvent{Kind: EventLd, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, RowStart: uint32(32 * w), Rows: 32},
			TraceEvent{Kind: EventSt, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, RowStart: uint32(32 * w), Rows: 32},
		)
	}
	// O = P * V reading operand A directly from TMEM offset 0 into offset 64
	events = append(events,
		TraceEvent{Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128, SourceAFromTMEM: true, SourceAOffset: 0},
		TraceEvent{Kind: EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
		TraceEvent{Kind: EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
	)
	for w := range 4 {
		events = append(events, TraceEvent{
			Kind: EventLd, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1,
			Offset: 64, Columns: 64, RowStart: uint32(32 * w), Rows: 32,
		})
	}
	events = append(events,
		TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
		TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
	)
	return events
}

func TestVerifyTraceValid(t *testing.T) {
	tests := []struct {
		name     string
		events   []TraceEvent
		wantCTAs int
		wantGrp  int
		wantCols uint32
		wantMMAs int
	}{
		{
			name:     "empty",
			events:   nil,
			wantCTAs: 0,
			wantGrp:  1,
			wantCols: 0,
			wantMMAs: 0,
		},
		{
			name:     "cta1_block_scaled_gemm",
			events:   validCTA1BlockScaledTrace(),
			wantCTAs: 1,
			wantGrp:  1,
			wantCols: 160,
			wantMMAs: 1,
		},
		{
			name:     "cta2_cooperative_gemm",
			events:   validCTA2Trace(),
			wantCTAs: 2,
			wantGrp:  2,
			wantCols: 128,
			wantMMAs: 1,
		},
		{
			name:     "tmem_sourced_operand_a_attention",
			events:   validTMEMOperandATrace(),
			wantCTAs: 1,
			wantGrp:  1,
			wantCols: 128,
			wantMMAs: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, target := range []Target{SM100A, SM103A, SM107A, SM100F, SM103F, SM107F} {
				sum, err := VerifyTrace(target, tc.events)
				if err != nil {
					t.Fatalf("VerifyTrace(%s, %s) unexpected error: %v", target, tc.name, err)
				}
				if sum.Target != target || sum.CTAs != tc.wantCTAs || sum.CTAGroup != tc.wantGrp || sum.MaxAllocatedColumns != tc.wantCols || sum.MMAs != tc.wantMMAs {
					t.Fatalf("VerifyTrace(%s, %s) summary = %+v, want Target=%s CTAs=%d CTAGroup=%d MaxCols=%d MMAs=%d",
						target, tc.name, sum, target, tc.wantCTAs, tc.wantGrp, tc.wantCols, tc.wantMMAs)
				}
			}
		})
	}
}

func TestVerifyTraceViolations(t *testing.T) {
	tests := []struct {
		name    string
		target  Target
		mutate  func() []TraceEvent
		wantErr error
	}{
		{
			name:   "unsupported_target_sm121a",
			target: SM121A,
			mutate: validCTA1BlockScaledTrace,
			wantErr: ErrUnsupportedTarget,
		},
		{
			name:   "rule1_mma_without_elect_sync",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA1BlockScaledTrace()
				evs[3].ActiveThreads = 32 // MMA issued by full warp instead of 1 elected thread
				return evs
			},
			wantErr: ErrSingleThreadIssue,
		},
		{
			name:   "rule1_cp_without_elect_sync",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA1BlockScaledTrace()
				evs[2].ActiveThreads = 32 // tcgen05.cp issued by 32 threads
				return evs
			},
			wantErr: ErrSingleThreadIssue,
		},
		{
			name:   "rule1_commit_without_elect_sync",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA1BlockScaledTrace()
				evs[4].ActiveThreads = 32 // tcgen05.commit issued by 32 threads
				return evs
			},
			wantErr: ErrSingleThreadIssue,
		},
		{
			name:   "rule2_ld_wrong_warp_row_slice",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA1BlockScaledTrace()
				// Warp 1 (allowed rows 32..63) tries to read rows 0..31.
				evs[7].RowStart = 0
				return evs
			},
			wantErr: ErrWarpRowPartition,
		},
		{
			name:   "rule2_st_wrong_warp_row_slice",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validTMEMOperandATrace()
				// Warp 2 (allowed rows 64..95) tries to store rows 32..63.
				evs[10].RowStart = 32
				return evs
			},
			wantErr: ErrWarpRowPartition,
		},
		{
			name:   "rule2_ld_non_32_active_threads",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA1BlockScaledTrace()
				evs[6].ActiveThreads = 1
				return evs
			},
			wantErr: ErrWarpRowPartition,
		},
		{
			name:   "rule3_block_scale_mma_without_prior_cp",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA1BlockScaledTrace()
				// Remove EventCp at index 2.
				return append(evs[:2], evs[3:]...)
			},
			wantErr: ErrScaleNotStaged,
		},
		{
			name:   "rule4_tmem_operand_a_without_prior_mma",
			target: SM100A,
			mutate: func() []TraceEvent {
				return []TraceEvent{
					{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
					{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
					{Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128, SourceAFromTMEM: true, SourceAOffset: 0},
				}
			},
			wantErr: ErrOperandANotReady,
		},
		{
			name:   "rule4_tmem_operand_a_without_wait",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validTMEMOperandATrace()
				// Remove EventWait (index 4) before the second MMA.
				return append(evs[:4], evs[5:]...)
			},
			wantErr: ErrOperandANotReady,
		},
		{
			name:   "rule5_cta2_mma_issued_by_odd_cta",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA2Trace()
				evs[3].CTA = 1 // CTA 1 issues cta_group::2 MMA instead of CTA 0
				return evs
			},
			wantErr: ErrCTA2Protocol,
		},
		{
			name:   "rule5_cta2_mismatched_peer_allocation",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA2Trace()
				evs[1].Columns = 64 // CTA 1 allocates 64 columns while CTA 0 allocates 128
				return evs
			},
			wantErr: ErrCTA2Protocol,
		},
		{
			name:   "rule5_cta2_dealloc_without_cluster_sync",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA2Trace()
				// Remove final EventClusterSync before EventDealloc.
				syncIdx := len(evs) - 3
				return append(evs[:syncIdx], evs[syncIdx+1:]...)
			},
			wantErr: ErrCTA2Protocol,
		},
		{
			name:   "rule6_cudatmem_increasing_allocation_columns",
			target: SM100A,
			mutate: func() []TraceEvent {
				return []TraceEvent{
					{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 32, Rows: 128},
					{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 32, Columns: 64, Rows: 128},
				}
			},
			wantErr: ErrInvalidUnit,
		},
		{
			name:   "rule6_cudatmem_dealloc_while_mma_in_flight",
			target: SM100A,
			mutate: func() []TraceEvent {
				return []TraceEvent{
					{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
					{Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
					{Kind: EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
					// Missing EventWait before EventDealloc -> ErrBusy
					{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
				}
			},
			wantErr: ErrBusy,
		},
		{
			name:   "rule6_cudatmem_leaked_allocation_at_end",
			target: SM100A,
			mutate: func() []TraceEvent {
				evs := validCTA1BlockScaledTrace()
				return evs[:len(evs)-1] // omit final EventDealloc
			},
			wantErr: ErrBusy,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := VerifyTrace(tc.target, tc.mutate())
			if err == nil {
				t.Fatalf("VerifyTrace(%s) succeeded, want error %v", tc.name, tc.wantErr)
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("VerifyTrace(%s) error = %v, want errors.Is(_, %v)", tc.name, err, tc.wantErr)
			}
		})
	}
}

func TestVerifyTraceShift(t *testing.T) {
	validShift := func() []TraceEvent {
		events := []TraceEvent{
			{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
			{Kind: EventCp, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
			{Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128, HasScaleInTMEM: true, ScaleOffset: 128},
			{Kind: EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: EventShift, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		}
		for w := range 4 {
			events = append(events, TraceEvent{
				Kind: EventLd, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1,
				Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
			})
		}
		events = append(events,
			TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		)
		return events
	}

	for _, target := range []Target{SM100A, SM103A, SM107A, SM100F, SM103F, SM107F} {
		sum, err := VerifyTrace(target, validShift())
		if err != nil {
			t.Fatalf("VerifyTrace(%s, validShift) unexpected error: %v", target, err)
		}
		if sum.Shifts != 1 || sum.MMAs != 1 || sum.Copies != 1 {
			t.Fatalf("VerifyTrace(%s, validShift) summary = %+v, want Shifts=1 MMAs=1 Copies=1", target, sum)
		}
	}

	badTests := []struct {
		name    string
		events  []TraceEvent
		wantErr error
	}{
		{
			name: "shift_wrong_active_threads",
			events: func() []TraceEvent {
				evs := validShift()
				evs[6].ActiveThreads = 32
				return evs
			}(),
			wantErr: ErrSingleThreadIssue,
		},
		{
			name: "shift_wrong_warp",
			events: func() []TraceEvent {
				evs := validShift()
				evs[6].Warp = 1
				return evs
			}(),
			wantErr: ErrSingleThreadIssue,
		},
		{
			name: "shift_unallocated_region",
			events: []TraceEvent{
				{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
				{Kind: EventShift, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
			},
			wantErr: ErrClosed,
		},
		{
			name: "shift_mma_in_flight_without_wait",
			events: []TraceEvent{
				{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
				{Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
				{Kind: EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
				{Kind: EventShift, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			},
			wantErr: ErrBusy,
		},
		{
			name: "shift_cp_in_flight_without_wait",
			events: []TraceEvent{
				{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 32, Rows: 128},
				{Kind: EventCp, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 32, Rows: 128},
				{Kind: EventShift, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 32, Rows: 128},
			},
			wantErr: ErrBusy,
		},
	}

	for _, tc := range badTests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := VerifyTrace(SM100A, tc.events)
			if err == nil {
				t.Fatalf("VerifyTrace(%s) succeeded, want error %v", tc.name, tc.wantErr)
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("VerifyTrace(%s) error = %v, want errors.Is(_, %v)", tc.name, err, tc.wantErr)
			}
		})
	}
}

func TestVerifyTraceRubinLUTAndSyncRestrict(t *testing.T) {
	validRubinTrace := func(lutOffset uint32, syncRestrict bool) []TraceEvent {
		events := []TraceEvent{
			{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
			{Kind: EventCp, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
			{
				Kind: EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1,
				Offset: 0, Columns: 128, Rows: 128,
				HasScaleInTMEM: true, ScaleOffset: 128,
				DecompressLUTB: true, LUTMetadataOffset: lutOffset,
			},
			{
				Kind: EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1,
				Offset: 0, Columns: 128, Rows: 128,
				SyncRestrictReadA: syncRestrict,
			},
			{Kind: EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		}
		for w := range 4 {
			events = append(events, TraceEvent{
				Kind: EventLd, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1,
				Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
			})
		}
		events = append(events,
			TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			TraceEvent{Kind: EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		)
		return events
	}

	for _, target := range []Target{SM107A, SM107F} {
		sum, err := VerifyTrace(target, validRubinTrace(128, true))
		if err != nil {
			t.Fatalf("VerifyTrace(%s, validRubinTrace) unexpected error: %v", target, err)
		}
		if sum.Target != target || sum.MMAs != 1 || sum.Commits != 1 {
			t.Fatalf("VerifyTrace(%s, validRubinTrace) summary = %+v", target, sum)
		}
	}

	// Unaligned LUTMetadataOffset (odd column offset) must be rejected.
	if _, err := VerifyTrace(SM107A, validRubinTrace(129, false)); !errors.Is(err, ErrInvalidUnit) {
		t.Fatalf("VerifyTrace(SM107A, unaligned LUT offset 129) = %v, want ErrInvalidUnit", err)
	}

	// Unallocated LUTMetadataOffset must be rejected.
	if _, err := VerifyTrace(SM107A, validRubinTrace(256, false)); !errors.Is(err, ErrClosed) {
		t.Fatalf("VerifyTrace(SM107A, unallocated LUT offset 256) = %v, want ErrClosed", err)
	}

	// SyncRestrictReadA must be rejected on sm_100a and sm_103a.
	for _, target := range []Target{SM100A, SM103A} {
		if _, err := VerifyTrace(target, validRubinTrace(128, true)); !errors.Is(err, ErrUnsupportedTarget) {
			t.Fatalf("VerifyTrace(%s, SyncRestrictReadA=true) = %v, want ErrUnsupportedTarget", target, err)
		}
	}
}
