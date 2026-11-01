package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMultiArchComparisonBuild(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    2,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}
	if len(iters) == 0 {
		t.Fatal("expected non-empty simple iterations")
	}

	promoted := iters[len(iters)-1]
	rep, err := BuildMultiArchComparison(promoted)
	if err != nil {
		t.Fatalf("BuildMultiArchComparison: %v", err)
	}
	if rep.SourceIteration != promoted.Iteration {
		t.Errorf("SourceIteration = %d, want %d", rep.SourceIteration, promoted.Iteration)
	}
	if rep.SourcePhase != promoted.Phase {
		t.Errorf("SourcePhase = %q, want %q", rep.SourcePhase, promoted.Phase)
	}
	if rep.KernelName != promoted.KernelName {
		t.Errorf("KernelName = %q, want %q", rep.KernelName, promoted.KernelName)
	}

	wantArchs := []string{"sm_90a", "sm_100a", "sm_121a", "sm_130a"}
	if len(rep.Targets) != len(wantArchs) {
		t.Fatalf("len(Targets) = %d, want %d", len(rep.Targets), len(wantArchs))
	}

	seenTC := make(map[string]string)
	seenMem := make(map[string]string)

	for i, wantArch := range wantArchs {
		tgt := rep.Targets[i]
		if tgt.Arch != wantArch {
			t.Errorf("Targets[%d].Arch = %q, want %q", i, tgt.Arch, wantArch)
		}
		if len(tgt.CUDALines) != 30 {
			t.Errorf("Targets[%d] (%s) len(CUDALines) = %d, want 30", i, tgt.Arch, len(tgt.CUDALines))
		}
		if strings.TrimSpace(tgt.PTXSnippet) == "" {
			t.Errorf("Targets[%d] (%s) has empty PTXSnippet", i, tgt.Arch)
		}
		if len(tgt.SASSInsts) == 0 {
			t.Errorf("Targets[%d] (%s) has empty SASSInsts", i, tgt.Arch)
		}
		if strings.TrimSpace(tgt.TensorCoreInstruction) == "" {
			t.Errorf("Targets[%d] (%s) has empty TensorCoreInstruction", i, tgt.Arch)
		}
		if strings.TrimSpace(tgt.MemoryHierarchyPolicy) == "" {
			t.Errorf("Targets[%d] (%s) has empty MemoryHierarchyPolicy", i, tgt.Arch)
		}
		if prev, dup := seenTC[tgt.TensorCoreInstruction]; dup {
			t.Errorf("duplicate TensorCoreInstruction %q between %s and %s", tgt.TensorCoreInstruction, prev, tgt.Arch)
		}
		seenTC[tgt.TensorCoreInstruction] = tgt.Arch

		if prev, dup := seenMem[tgt.MemoryHierarchyPolicy]; dup {
			t.Errorf("duplicate MemoryHierarchyPolicy %q between %s and %s", tgt.MemoryHierarchyPolicy, prev, tgt.Arch)
		}
		seenMem[tgt.MemoryHierarchyPolicy] = tgt.Arch
	}

	// Verify sm_121a preserves the input iteration's 30 CUDA lines.
	sm121 := rep.Targets[2]
	for idx := range sm121.CUDALines {
		if sm121.CUDALines[idx] != promoted.CUDALines[idx] {
			t.Errorf("sm_121a CUDALines[%d] = %q, want %q", idx, sm121.CUDALines[idx], promoted.CUDALines[idx])
		}
	}
}

func TestMultiArchHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    2,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/kernels/multiarch?iter=23", nil)
	rec := httptest.NewRecorder()
	srv.handleKernelsMultiArch(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("handleKernelsMultiArch status = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}

	var rep MultiArchComparisonReport
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if rep.SourceIteration != 23 {
		t.Errorf("SourceIteration = %d, want 23", rep.SourceIteration)
	}
	if len(rep.Targets) != 4 {
		t.Fatalf("len(Targets) = %d, want 4", len(rep.Targets))
	}

	// Verify POST is rejected with 405 Method Not Allowed.
	postReq := httptest.NewRequest(http.MethodPost, "/v1/kernels/multiarch", nil)
	postRec := httptest.NewRecorder()
	srv.handleKernelsMultiArch(postRec, postReq)
	if postRec.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST handleKernelsMultiArch status = %d, want %d", postRec.Code, http.StatusMethodNotAllowed)
	}
}
