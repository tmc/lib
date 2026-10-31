package main

import (
	"fmt"
	"strings"
	"testing"
)

func TestLiveGPUBenchmarkCUDAGeneration(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	iters := srv.buildSimpleKernelStreamIterations(1)
	if len(iters) != 24 {
		t.Fatalf("len(iters) = %d, want 24", len(iters))
	}

	cudaSrc := BuildLiveGPUBenchmarkCUDA(iters, 25, 200)
	for i := range 24 {
		fnName := fmt.Sprintf("cand_%02d_fused_rmsnorm_swiglu_gemv", i)
		if !strings.Contains(cudaSrc, fnName) {
			t.Fatalf("generated CUDA benchmark missing kernel %q", fnName)
		}
	}

	for _, want := range []string{
		"#include <cuda_runtime.h>",
		"#include <stdio.h>",
		"#include <stdlib.h>",
		"#include <math.h>",
		"#include <stdint.h>",
		"cudaGetDeviceProperties(&prop, 0)",
		"cudaEventElapsedTime(&ms, start, stop)",
		"const int warmup_runs = 25;",
		"const int timed_runs = 200;",
		"EG_GPU_JSON:",
		`"iteration":14,"phase":"elide_stage_sync_barrier","executed":false,"skip_reason":"static_barrier_hazard"`,
		`"iteration":15,"phase":"named_barrier_nonuniform","executed":false,"skip_reason":"static_barrier_hazard"`,
		`"iteration":20,"phase":"cp_async_missing_wait_group","executed":false,"skip_reason":"static_barrier_hazard"`,
		"h_ref_07",
	} {
		if !strings.Contains(cudaSrc, want) {
			t.Fatalf("generated CUDA benchmark missing substring %q", want)
		}
	}

	// Ensure unsafe candidates #14, #15, #20 are never launched via <<<1, 256>>>.
	for _, unsafeIdx := range []int{14, 15, 20} {
		launchCall := fmt.Sprintf("cand_%02d_fused_rmsnorm_swiglu_gemv<<<1, 256>>>", unsafeIdx)
		if strings.Contains(cudaSrc, launchCall) {
			t.Fatalf("unsafe candidate #%02d should be skipped, but found launch %q", unsafeIdx, launchCall)
		}
	}
}

func TestLiveGPUApplyBenchmarkOutput(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	iters := srv.buildSimpleKernelStreamIterations(1)

	var stdout strings.Builder
	baseLat := 42.00
	for _, it := range iters {
		idx := it.Iteration
		if idx == 14 || idx == 15 || idx == 20 {
			fmt.Fprintf(&stdout, "EG_GPU_JSON:{\"iteration\":%d,\"phase\":%q,\"executed\":false,\"skip_reason\":\"static_barrier_hazard\",\"latency_us\":0,\"speedup\":0,\"max_abs_err\":0,\"numeric_ok\":false,\"device_name\":\"NVIDIA GB10\"}\n",
				idx, it.Phase)
			continue
		}
		if idx == 8 {
			fmt.Fprintf(&stdout, "EG_GPU_JSON:{\"iteration\":8,\"phase\":%q,\"executed\":true,\"latency_us\":16.50,\"speedup\":2.5455,\"max_abs_err\":0.054200,\"numeric_ok\":false,\"device_name\":\"NVIDIA GB10\"}\n",
				it.Phase)
			continue
		}
		// Scale synthetic hardware latency from baseLat using the candidate's target speedup.
		sp := it.HoldoutSpeedup
		if sp <= 0 {
			sp = 1.0
		}
		lat := baseLat / sp
		fmt.Fprintf(&stdout, "EG_GPU_JSON:{\"iteration\":%d,\"phase\":%q,\"executed\":true,\"latency_us\":%.4f,\"speedup\":%.4f,\"max_abs_err\":0.000012,\"numeric_ok\":true,\"device_name\":\"NVIDIA GB10\"}\n",
			idx, it.Phase, lat, sp)
	}

	updated, meas, err := ApplyLiveGPUBenchmarkOutput(iters, stdout.String(), "live ssh:dgx gpu-lock")
	if err != nil {
		t.Fatalf("ApplyLiveGPUBenchmarkOutput: %v", err)
	}
	if len(meas) != 24 || len(updated) != 24 {
		t.Fatalf("len(meas)=%d len(updated)=%d, want 24", len(meas), len(updated))
	}

	// Verify #00 baseline latency was updated from 48.20 to 42.00.
	if updated[0].LatencyUs != 42.00 || updated[0].HoldoutSpeedup != 1.00 {
		t.Fatalf("updated[0] latency=%.2f speedup=%.2f, want 42.00 / 1.00", updated[0].LatencyUs, updated[0].HoldoutSpeedup)
	}

	// Verify #08 numerical rejection was populated from measured max_abs_err.
	if updated[8].Promoted {
		t.Fatalf("candidate #08 should be rejected (Promoted=false)")
	}
	if !strings.Contains(updated[8].RejectReason, "0.0542") {
		t.Fatalf("candidate #08 RejectReason = %q, want measured max_abs_err 0.0542", updated[8].RejectReason)
	}

	// Verify #14, #15, #20 were skipped and not promoted.
	for _, skipIdx := range []int{14, 15, 20} {
		if meas[skipIdx].Executed {
			t.Fatalf("meas[%d].Executed = true, want false", skipIdx)
		}
		if updated[skipIdx].Promoted {
			t.Fatalf("updated[%d].Promoted = true, want false", skipIdx)
		}
	}

	// Verify final champion #23 envelope and stage cost updates.
	last := updated[23]
	if !last.Promoted || last.BestIteration != 23 || last.BestSpeedup < 6.0 || last.LatencyUs >= updated[0].LatencyUs {
		t.Fatalf("unexpected final promoted iteration #23: %+v", last)
	}
	if len(last.Stages) < 2 || last.Stages[0].EGraphCostNs <= 0 || last.Stages[1].EGraphCostNs <= 0 {
		t.Fatalf("unexpected stage costs on #23: %+v", last.Stages)
	}
	if !strings.Contains(last.InlineAnnotatedText, "/* [STAGE s0") {
		t.Fatalf("expected refreshed InlineAnnotatedText on #23")
	}
}
