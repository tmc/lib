package cudaprof_test

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestProfileWorkloadMismatch(t *testing.T) {
	snap, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{
		Version:            cudaprof.ProfileSnapshotVersion,
		Timestamp:          time.Now(),
		CodeDigest:         "1111111111111111111111111111111111111111111111111111111111111111",
		Region:             "fwd_kernel",
		Workload:           "llama-7b-prefill",
		TokensHash:         "tokens-prompt-001",
		Device:             "sm_121a",
		CollectorMode:      "pc_sampling",
		Coverage:           0.85,
		ClockUncertaintyNs: 5000,
		MetricUnits:        map[string]string{"smsp__pcsamp_warps_issue_stalled_long_scoreboard": "samples", "smsp__pcsamp_warps_issue_stalled_short_scoreboard": "samples"},
		Metrics: map[string]int64{
			"smsp__pcsamp_warps_issue_stalled_long_scoreboard":  1500,
			"smsp__pcsamp_warps_issue_stalled_short_scoreboard": 200,
		},
	})
	if err != nil {
		t.Fatalf("NewProfileSnapshot failed: %v", err)
	}

	// 1. Workload mismatch
	_, err = snap.Guidance(profileTarget(snap, "mistral-7b-prefill", "tokens-prompt-001", "sm_121a"))
	if !errors.Is(err, cudaprof.ErrWorkloadMismatch) {
		t.Errorf("expected ErrWorkloadMismatch, got: %v", err)
	}

	// 2. Tokens hash mismatch
	_, err = snap.Guidance(profileTarget(snap, "llama-7b-prefill", "different-tokens", "sm_121a"))
	if !errors.Is(err, cudaprof.ErrWorkloadMismatch) {
		t.Errorf("expected ErrWorkloadMismatch on tokens hash, got: %v", err)
	}

	// 3. Device mismatch
	_, err = snap.Guidance(profileTarget(snap, "llama-7b-prefill", "tokens-prompt-001", "sm_100a"))
	if !errors.Is(err, cudaprof.ErrDeviceMismatch) {
		t.Errorf("expected ErrDeviceMismatch, got: %v", err)
	}

	// 4. Correct match succeeds and identifies primary stall in original units
	guidance, err := snap.Guidance(profileTarget(snap, "llama-7b-prefill", "tokens-prompt-001", "sm_121a"))
	if err != nil {
		t.Fatalf("Guidance failed on matched workload: %v", err)
	}
	if !guidance.Available {
		t.Errorf("guidance.Available = false, want true")
	}
	if guidance.CodeDigest != snap.CodeDigest || guidance.Region != snap.Region || guidance.CollectorMode != snap.CollectorMode {
		t.Errorf("lost attribution: %+v", guidance)
	}
	if guidance.StallCounts["smsp__pcsamp_warps_issue_stalled_long_scoreboard"] != 1500 {
		t.Errorf("expected exact count 1500 in original units, got: %v", guidance.StallCounts)
	}
}

func TestProfileGuidanceUnavailable(t *testing.T) {
	// Snapshot with zero coverage and empty metrics represents unmeasured/unavailable telemetry
	snap, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{
		Version:            cudaprof.ProfileSnapshotVersion,
		Timestamp:          time.Now(),
		CodeDigest:         "2222222222222222222222222222222222222222222222222222222222222222",
		Region:             "fwd_kernel",
		Workload:           "workload_sparse",
		Device:             "sm_121a",
		CollectorMode:      "pc_sampling",
		Coverage:           0.0, // Zero coverage
		ClockUncertaintyNs: -1,
		Metrics:            nil,
	})
	if err != nil {
		t.Fatalf("NewProfileSnapshot failed: %v", err)
	}

	guidance, err := snap.Guidance(profileTarget(snap, "workload_sparse", "", "sm_121a"))
	if !errors.Is(err, cudaprof.ErrGuidanceUnavailable) {
		t.Errorf("expected ErrGuidanceUnavailable, got: %v", err)
	}
	if guidance == nil || guidance.Available {
		t.Errorf("expected non-nil guidance with Available=false, got: %+v", guidance)
	}
	if guidance.UnavailableReason == "" {
		t.Errorf("expected non-empty UnavailableReason")
	}
}

func TestProfileSnapshotVersion(t *testing.T) {
	// Unsupported schema version rejected
	_, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{
		Version:    99, // Unsupported version
		CodeDigest: "3333333333333333333333333333333333333333333333333333333333333333",
		Region:     "k",
		Workload:   "w",
		Device:     "sm_121a",
	})
	if !errors.Is(err, cudaprof.ErrInvalidSnapshotVersion) {
		t.Errorf("expected ErrInvalidSnapshotVersion for version 99, got: %v", err)
	}

	// Valid schema version 1 accepted
	snap, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{
		Version:       1,
		CodeDigest:    "3333333333333333333333333333333333333333333333333333333333333333",
		Region:        "k",
		Workload:      "w",
		Device:        "sm_121a",
		CollectorMode: "pc_sampling",
		Coverage:      1.0,
		Metrics:       map[string]int64{"samples": 100},
	})
	if err != nil {
		t.Fatalf("NewProfileSnapshot v1 failed: %v", err)
	}
	if snap.Version != 1 {
		t.Errorf("Version = %d, want 1", snap.Version)
	}
}

func TestRecipeReportSchemaV1(t *testing.T) {
	rep := &cudaprof.RecipeReport{
		SchemaVersion: 1,
		Artifact: cudaprof.RecipeArtifact{
			SHA256:    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
			SizeBytes: 12288,
			ImageName: "kernel.cubin",
		},
		Selection: cudaprof.RecipeSelection{
			Symbol:   "_Z10copyKernelPfPKfi",
			Function: "copyKernel",
		},
		Target: cudaprof.RecipeTarget{
			RawArchitecture: "sm_121a",
			ComputeCap:      "12.1",
		},
		Request: cudaprof.RecipeRequest{
			Required: []string{"instructions", "resources"},
			Optional: []string{"liveness", "tileir"},
		},
		Result: "complete",
		Features: []cudaprof.RecipeFeature{
			// 1. Distinguish peak live GPR (8) from allocated registers (12)
			{
				Name:     "allocated_registers",
				State:    cudaprof.StateObserved,
				Value:    cudaprof.Float64Ptr(12),
				Unit:     "registers",
				Evidence: []string{"elf:.nv.info:offset_0x120"},
			},
			{
				Name:     "peak_live_gpr",
				State:    cudaprof.StateObserved,
				Value:    cudaprof.Float64Ptr(8),
				Unit:     "registers",
				Evidence: []string{"nvdisasm:life:pc_0x40"},
			},
			// 2. Distinguish local access from confirmed spills
			{
				Name:     "local_memory_access",
				State:    cudaprof.StateObserved,
				Value:    cudaprof.Float64Ptr(0),
				Unit:     "bytes",
				Evidence: []string{"sass:scan:all_pcs"},
			},
			{
				Name:     "confirmed_spills",
				State:    cudaprof.StateObserved,
				Value:    cudaprof.Float64Ptr(0),
				Unit:     "instructions",
				Evidence: []string{"sass:scan:stl_ldl"},
			},
			// 3. Unknown values must be explicit nulls with diagnostic reasons, not zeros
			{
				Name:     "tile_ir_mma_shape",
				State:    cudaprof.StateUnknown,
				Value:    nil, // JSON null
				Unit:     "shape",
				Evidence: []string{"elf:sections:.tileir"},
			},
		},
		Instructions: []cudaprof.RecipeInstruction{
			// 4. Distinguish vector instruction width (128-bit) from scalar (32-bit)
			{
				PC:        0x00,
				Opcode:    "LDG.E.128",
				Operands:  "R4, [R2.64]",
				WidthBits: 128,
				IsVector:  true,
			},
			{
				PC:        0x10,
				Opcode:    "LDG.E",
				Operands:  "R1, [R2.64]",
				WidthBits: 32,
				IsVector:  false,
			},
		},
		Diagnostics: []cudaprof.RecipeDiagnostic{
			{
				Code:     "DIAG_NO_TILEIR_SECTION",
				Feature:  "tile_ir_mma_shape",
				Message:  "CUBIN contains no recognized embedded Tile IR section; shape remains unknown",
				Evidence: []string{"elf:sections"},
			},
		},
		Receipts: []cudaprof.RecipeReceipt{
			{
				Tool:       "nvdisasm",
				Version:    "13.0.85",
				ExitStatus: 0,
				DurationMs: 15,
			},
		},
	}

	// Validate report structure
	if err := rep.Validate(); err != nil {
		t.Fatalf("RecipeReport.Validate failed: %v", err)
	}

	// Serialize to JSON and verify explicit null serialization
	var buf bytes.Buffer
	if err := rep.WriteJSON(&buf); err != nil {
		t.Fatalf("WriteJSON failed: %v", err)
	}

	jsonStr := buf.String()
	// Check that tile_ir_mma_shape has "value": null
	if !strings.Contains(jsonStr, `"name": "tile_ir_mma_shape"`) {
		t.Errorf("missing tile_ir_mma_shape in JSON output")
	}
	if !strings.Contains(jsonStr, `"value": null`) {
		t.Errorf("expected explicit null value in JSON output for unknown feature")
	}

	// Verify round-trip read back
	readBack, err := cudaprof.ReadRecipeReport(&buf)
	if err != nil {
		t.Fatalf("ReadRecipeReport failed: %v", err)
	}
	if err := readBack.Validate(); err != nil {
		t.Fatalf("readBack.Validate failed: %v", err)
	}
	if readBack.Selection.Symbol != rep.Selection.Symbol {
		t.Errorf("Symbol = %q, want %q", readBack.Selection.Symbol, rep.Selection.Symbol)
	}

	// Negative control: unknown feature with non-null value must fail validation
	invalidRep := *rep
	invalidRep.Features = []cudaprof.RecipeFeature{
		{
			Name:  "tile_ir_mma_shape",
			State: cudaprof.StateUnknown,
			Value: cudaprof.Float64Ptr(0), // Falsely supplying zero instead of null
		},
	}
	if err := invalidRep.Validate(); err == nil {
		t.Errorf("expected error validating unknown feature with non-null zero value")
	}
}

func TestFoldedStackExport(t *testing.T) {
	executions := []cudaprof.KernelExecution{
		{
			KernelName: "short_kernel",
			Callstack:  []string{"main.main", "main.runBenchmark"},
			StartNs:    100_000,
			EndNs:      100_500, // 500 ns device duration
		},
		// 2. Overlapping intervals for another kernel: [200_000, 201_000) and [200_500, 201_500)
		// Union duration is 1500 ns (not naive sum 2000 ns)
		{
			KernelName: "stream_kernel",
			Callstack:  []string{"main.main", "main.runStream"},
			StartNs:    200_000,
			EndNs:      201_000,
		},
		{
			KernelName: "stream_kernel",
			Callstack:  []string{"main.main", "main.runStream"},
			StartNs:    200_500,
			EndNs:      201_500,
		},
		// 3. Kernel with no host stack (nil)
		{
			KernelName: "driver_kernel",
			Callstack:  nil,
			StartNs:    300_000,
			EndNs:      300_800,
		},
	}

	var buf bytes.Buffer
	if err := cudaprof.ExportFoldedStacks(&buf, executions); err != nil {
		t.Fatalf("ExportFoldedStacks failed: %v", err)
	}

	out := buf.String()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 folded stack lines, got %d: %q", len(lines), out)
	}

	// Verify line 1: driver_kernel (no host stack) -> [GPU]driver_kernel 800
	if lines[0] != "[GPU]driver_kernel 800" {
		t.Errorf("line 0 = %q, want %q", lines[0], "[GPU]driver_kernel 800")
	}

	// Verify line 2: short_kernel has weight 500 (device time), NOT 10ms host sleep
	if lines[1] != "main.main;main.runBenchmark;[GPU]short_kernel 500" {
		t.Errorf("line 1 = %q, want %q", lines[1], "main.main;main.runBenchmark;[GPU]short_kernel 500")
	}

	// Verify line 3: stream_kernel interval union = 1500 (not 2000)
	if lines[2] != "main.main;main.runStream;[GPU]stream_kernel 1500" {
		t.Errorf("line 2 = %q, want %q", lines[2], "main.main;main.runStream;[GPU]stream_kernel 1500")
	}
}
