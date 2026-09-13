package cuptipcsampling_test

import (
	"bytes"
	"context"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func TestClassifyPipelineStall(t *testing.T) {
	tests := []struct {
		input         string
		wantCanonical string
		wantCategory  cuptipcsampling.StallCategory
		wantCompanion bool
	}{
		{
			input:         "long_scoreboard",
			wantCanonical: "long_scoreboard",
			wantCategory:  cuptipcsampling.StallCategoryMemory,
			wantCompanion: false,
		},
		{
			input:         "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
			wantCanonical: "long_scoreboard",
			wantCategory:  cuptipcsampling.StallCategoryMemory,
			wantCompanion: false,
		},
		{
			input:         "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued",
			wantCanonical: "long_scoreboard",
			wantCategory:  cuptipcsampling.StallCategoryMemory,
			wantCompanion: true,
		},
		{
			input:         "short_scoreboard",
			wantCanonical: "short_scoreboard",
			wantCategory:  cuptipcsampling.StallCategoryExecution,
			wantCompanion: false,
		},
		{
			input:         "stall_mem",
			wantCanonical: "stall_mem",
			wantCategory:  cuptipcsampling.StallCategoryMemory,
			wantCompanion: false,
		},
		{
			input:         "stall_sync",
			wantCanonical: "stall_sync",
			wantCategory:  cuptipcsampling.StallCategorySynchronization,
			wantCompanion: false,
		},
		{
			input:         "stall_barrier",
			wantCanonical: "stall_barrier",
			wantCategory:  cuptipcsampling.StallCategorySynchronization,
			wantCompanion: false,
		},
		{
			input:         "membar",
			wantCanonical: "stall_barrier",
			wantCategory:  cuptipcsampling.StallCategorySynchronization,
			wantCompanion: false,
		},
		{
			input:         "selected",
			wantCanonical: "selected",
			wantCategory:  cuptipcsampling.StallCategoryIssued,
			wantCompanion: false,
		},
		{
			input:         "smsp__pcsamp_warps_issue_stalled_selected",
			wantCanonical: "selected",
			wantCategory:  cuptipcsampling.StallCategoryIssued,
			wantCompanion: false,
		},
		{
			input:         "not_selected",
			wantCanonical: "not_selected",
			wantCategory:  cuptipcsampling.StallCategoryScheduler,
			wantCompanion: false,
		},
		{
			input:         "math_pipe_throttle",
			wantCanonical: "stall_math",
			wantCategory:  cuptipcsampling.StallCategoryExecution,
			wantCompanion: false,
		},
		{
			input:         "no_instruction",
			wantCanonical: "stall_no_instruction",
			wantCategory:  cuptipcsampling.StallCategoryInstructionFetch,
			wantCompanion: false,
		},
		{
			input:         "branch_resolving",
			wantCanonical: "stall_branch",
			wantCategory:  cuptipcsampling.StallCategoryBranch,
			wantCompanion: false,
		},
		{
			input:         "drain",
			wantCanonical: "stall_drain",
			wantCategory:  cuptipcsampling.StallCategoryDrain,
			wantCompanion: false,
		},
		{
			input:         "dispatch_stall",
			wantCanonical: "stall_dispatch",
			wantCategory:  cuptipcsampling.StallCategoryDispatch,
			wantCompanion: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			canonical, cat, desc, isComp := cuptipcsampling.ClassifyPipelineStall(tt.input)
			if canonical != tt.wantCanonical {
				t.Errorf("canonical = %q, want %q", canonical, tt.wantCanonical)
			}
			if cat != tt.wantCategory {
				t.Errorf("category = %q, want %q", cat, tt.wantCategory)
			}
			if isComp != tt.wantCompanion {
				t.Errorf("isCompanion = %v, want %v", isComp, tt.wantCompanion)
			}
			if desc == "" {
				t.Errorf("description is empty")
			}
		})
	}
}

func TestDecodeSASSMetric(t *testing.T) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 28,
		Reasons: []cuptipcsampling.StallReason{
			{Index: 0, Name: "smsp__pcsamp_warps_issue_stalled_selected"},
			{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard"},
			{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued"},
			{Index: 3, Name: "smsp__pcsamp_warps_issue_stalled_barrier"},
			{Index: 4, Name: "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"},
		},
	}

	pc := cuptipcsampling.PC{
		Offset:     0x120,
		Name:       "_Z12gemm_kernelPKfS0_Pf",
		CRC:        0xfeedbeef,
		Reasons: []cuptipcsampling.StallSamples{
			{Index: 0, Samples: 100}, // Selected (issued)
			{Index: 1, Samples: 300}, // long_scoreboard
			{Index: 2, Samples: 300}, // companion (not added to total)
			{Index: 3, Samples: 100}, // barrier
		},
	}

	metric := inv.DecodeSASSMetric(pc)

	if metric.PC != 0x120 {
		t.Errorf("PC = 0x%x, want 0x120", metric.PC)
	}
	if metric.KernelName != "_Z12gemm_kernelPKfS0_Pf" {
		t.Errorf("KernelName = %q", metric.KernelName)
	}
	if metric.TotalSamples != 500 { // 100 selected + 300 long_sb + 100 barrier
		t.Errorf("TotalSamples = %d, want 500", metric.TotalSamples)
	}
	if metric.Selected != 100 {
		t.Errorf("Selected = %d, want 100", metric.Selected)
	}
	if math.Abs(metric.SelectedPct-20.0) > 0.01 {
		t.Errorf("SelectedPct = %.2f, want 20.0", metric.SelectedPct)
	}
	if metric.StallSamples != 400 {
		t.Errorf("StallSamples = %d, want 400", metric.StallSamples)
	}

	if metric.DominantReason() != "long_scoreboard" {
		t.Errorf("DominantReason = %q, want long_scoreboard", metric.DominantReason())
	}
	if metric.Dominant == nil || metric.Dominant.Samples != 300 {
		t.Fatalf("Dominant stall incorrect: %+v", metric.Dominant)
	}
	if math.Abs(metric.Dominant.Percent-75.0) > 0.01 { // 300 / 400 * 100 = 75%
		t.Errorf("Dominant.Percent = %.2f, want 75.0", metric.Dominant.Percent)
	}

	if cost := metric.StallCost("long_scoreboard"); math.Abs(cost-75.0) > 0.01 {
		t.Errorf("StallCost(long_scoreboard) = %.2f, want 75.0", cost)
	}
	if cost := metric.StallCost("stall_barrier"); math.Abs(cost-25.0) > 0.01 {
		t.Errorf("StallCost(stall_barrier) = %.2f, want 25.0", cost)
	}
	if cost := metric.CategoryCost(cuptipcsampling.StallCategoryMemory); math.Abs(cost-75.0) > 0.01 {
		t.Errorf("CategoryCost(Memory) = %.2f, want 75.0", cost)
	}
	if cost := metric.CategoryCost(cuptipcsampling.StallCategorySynchronization); math.Abs(cost-25.0) > 0.01 {
		t.Errorf("CategoryCost(Synchronization) = %.2f, want 25.0", cost)
	}

	// Conversion to cudaprof.SASSMetric
	cp := metric.ToCudaprof()
	if cp.InstructionPC != 0x120 {
		t.Errorf("cp.InstructionPC = 0x%x, want 0x120", cp.InstructionPC)
	}
	if cp.StallReason != "long_scoreboard" || cp.StallWarps != 300 {
		t.Errorf("cp.StallReason/Warps mismatch: %s, %d", cp.StallReason, cp.StallWarps)
	}
	if cp.ExecutedWarps != 100 {
		t.Errorf("cp.ExecutedWarps = %d, want 100", cp.ExecutedWarps)
	}

	// Flight record labels
	labels := metric.FlightRecordLabels()
	if labels["sass.dominant_stall"] != "long_scoreboard" {
		t.Errorf("labels[sass.dominant_stall] = %q", labels["sass.dominant_stall"])
	}
	if labels["sass.total_samples"] != "500" {
		t.Errorf("labels[sass.total_samples] = %q", labels["sass.total_samples"])
	}
}

func TestBatch_SASSMetrics(t *testing.T) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 28,
		Reasons: []cuptipcsampling.StallReason{
			{Index: 0, Name: "selected"},
			{Index: 1, Name: "stall_mem"},
		},
	}

	bInvalid := cuptipcsampling.Batch{Valid: false}
	if metrics := bInvalid.SASSMetrics(inv); metrics != nil {
		t.Errorf("bInvalid.SASSMetrics() returned non-nil: %v", metrics)
	}

	bValid := cuptipcsampling.Batch{
		Valid: true,
		PCs: []cuptipcsampling.PC{
			{
				Offset: 0x10,
				Name:   "kernel1",
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 0, Samples: 50},
					{Index: 1, Samples: 150},
				},
			},
			{
				Offset: 0x20,
				Name:   "kernel1",
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 0, Samples: 200},
				},
			},
		},
	}

	metrics := bValid.SASSMetrics(inv)
	if len(metrics) != 2 {
		t.Fatalf("len(metrics) = %d, want 2", len(metrics))
	}
	if metrics[0].PC != 0x10 || metrics[0].DominantReason() != "stall_mem" {
		t.Errorf("metric[0] mismatch: %+v", metrics[0])
	}
	if metrics[1].PC != 0x20 || metrics[1].DominantReason() != "none" {
		t.Errorf("metric[1] mismatch: %+v", metrics[1])
	}
}

func TestProfileBridge_RecordSASSMetric(t *testing.T) {
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	bridge := &cuptipcsampling.ProfileBridge{Session: session}

	sm := cuptipcsampling.SASSMetric{
		PC:           0x400,
		KernelName:   "test_kernel",
		TotalSamples: 100,
		Selected:     20,
		Dominant: &cuptipcsampling.PipelineStall{
			Name:    "long_scoreboard",
			Samples: 80,
			Percent: 100.0,
		},
	}

	if err := bridge.RecordSASSMetric(context.Background(), sm, "kernels.sass", 10); err != nil {
		t.Fatalf("RecordSASSMetric failed: %v", err)
	}
}
