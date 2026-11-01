package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestExportBundleRenderStandaloneCUReproducer(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}
	if len(iters) != 24 {
		t.Fatalf("len(iters) = %d, want 24", len(iters))
	}

	champ := iters[23]
	cu := RenderStandaloneCUReproducer(champ, "sm_121a", "live ssh:dgx nvcc (V13.2.51, arch=sm_121a)")

	wantSubstrings := []string{
		"nvcc -O3 -arch=sm_121a -lineinfo -Xptxas -v reproducer_cand_23.cu -o reproducer_cand_23",
		"Formal Barrier Verification Certificate (Passed=true):",
		"RAWBarrierOK:      true",
		"UniformBarrierOK:  true",
		"AsyncWaitGroupOK:  true",
		"CrossWarpReduceOK: true",
		"MaxRegs:           72",
		"TotalSpillBytes:   0",
		"HoldoutSpeedup:    6.05x",
		"#include <cuda_runtime.h>",
		"#include <stdio.h>",
		"#include <stdlib.h>",
		"#include <math.h>",
		"#include <stdint.h>",
		"stage0_rmsnorm(",
		"stage1_swiglu_gemv(",
		"fused_rmsnorm_swiglu_gemv(",
		"cudaStreamBeginCapture(",
		"cudaStreamEndCapture(",
		"cudaGraphInstantiate(",
		"cudaGraphLaunch(",
		"cudaEventElapsedTime(",
	}
	for _, want := range wantSubstrings {
		if !strings.Contains(cu, want) {
			t.Errorf("RenderStandaloneCUReproducer missing substring %q", want)
		}
	}
}

func TestExportBundleBuildExportedEGBundle(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}

	// Test promoted champion #23.
	bundle23, err := BuildExportedEGBundle(iters[23], "sm_121a", "test-provenance")
	if err != nil {
		t.Fatalf("BuildExportedEGBundle(#23): %v", err)
	}
	if len(bundle23.SHA256Hex) != 64 {
		t.Fatalf("SHA256Hex len = %d, want 64 (%q)", len(bundle23.SHA256Hex), bundle23.SHA256Hex)
	}
	if !bundle23.Verification.Passed() {
		t.Fatalf("Verification.Passed() = false for #23, want true: %+v", bundle23.Verification)
	}
	if !bundle23.Promoted || bundle23.Iteration != 23 {
		t.Fatalf("unexpected bundle metadata: promoted=%v iter=%d", bundle23.Promoted, bundle23.Iteration)
	}
	if bundle23.CUDASource == "" || bundle23.StandaloneReproducer == "" {
		t.Fatalf("expected non-empty CUDASource and StandaloneReproducer")
	}
	if len(bundle23.Stages) != 2 {
		t.Fatalf("len(Stages) = %d, want 2", len(bundle23.Stages))
	}

	// Test rejected mutant #14 (missing cross-stage __syncthreads RAW hazard).
	bundle14, err := BuildExportedEGBundle(iters[14], "sm_121a", "test-provenance")
	if err != nil {
		t.Fatalf("BuildExportedEGBundle(#14): %v", err)
	}
	if bundle14.Verification.Passed() || bundle14.Verification.RAWBarrierOK {
		t.Fatalf("expected #14 Verification.Passed()=false and RAWBarrierOK=false, got %+v", bundle14.Verification)
	}

	// Test error on empty CUDALines.
	if _, err := BuildExportedEGBundle(KernelStreamIteration{}, "sm_121a", ""); err == nil {
		t.Fatalf("expected error for empty CUDALines, got nil")
	}
}

func TestExportBundleHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	// 1. Test format=cu (default latest promoted champion #23).
	reqCU := httptest.NewRequest(http.MethodGet, "/v1/kernels/export?format=cu", nil)
	recCU := httptest.NewRecorder()
	srv.handleKernelsExport(recCU, reqCU)

	if recCU.Code != http.StatusOK {
		t.Fatalf("GET /v1/kernels/export?format=cu status = %d, want 200", recCU.Code)
	}
	if ct := recCU.Header().Get("Content-Type"); !strings.Contains(ct, "text/x-cuda") {
		t.Fatalf("Content-Type = %q, want text/x-cuda", ct)
	}
	if cd := recCU.Header().Get("Content-Disposition"); !strings.Contains(cd, "reproducer_cand_23.cu") {
		t.Fatalf("Content-Disposition = %q, want reproducer_cand_23.cu", cd)
	}
	cuBody := recCU.Body.String()
	if !strings.Contains(cuBody, "cudaGraphInstantiate") || !strings.Contains(cuBody, "fused_rmsnorm_swiglu_gemv") {
		t.Fatalf("unexpected .cu body: missing cudaGraphInstantiate or fused_rmsnorm_swiglu_gemv")
	}

	// 2. Test format=egbundle (default format and explicit iter=23).
	reqBundle := httptest.NewRequest(http.MethodGet, "/v1/kernels/export?format=egbundle&iter=23", nil)
	recBundle := httptest.NewRecorder()
	srv.handleKernelsExport(recBundle, reqBundle)

	if recBundle.Code != http.StatusOK {
		t.Fatalf("GET /v1/kernels/export?format=egbundle&iter=23 status = %d, want 200", recBundle.Code)
	}
	if ct := recBundle.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}
	var got ExportedBundleArtifact
	if err := json.NewDecoder(recBundle.Body).Decode(&got); err != nil {
		t.Fatalf("decode ExportedBundleArtifact: %v", err)
	}
	if got.Iteration != 23 || !got.Promoted || !got.Verification.Passed() || len(got.SHA256Hex) != 64 {
		t.Fatalf("unexpected decoded ExportedBundleArtifact: iter=%d promoted=%v ver=%+v sha=%q",
			got.Iteration, got.Promoted, got.Verification, got.SHA256Hex)
	}

	// 3. Test invalid iter returns 400 Bad Request.
	reqBad := httptest.NewRequest(http.MethodGet, "/v1/kernels/export?iter=99", nil)
	recBad := httptest.NewRecorder()
	srv.handleKernelsExport(recBad, reqBad)
	if recBad.Code != http.StatusBadRequest {
		t.Fatalf("GET /v1/kernels/export?iter=99 status = %d, want 400", recBad.Code)
	}
}
