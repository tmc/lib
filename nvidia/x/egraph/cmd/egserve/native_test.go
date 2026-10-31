package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

type egserveNativeGateClosedError struct{}

func (egserveNativeGateClosedError) Error() string {
	return "native qualification pending: CUDATEST_NATIVE=1 required"
}

func (egserveNativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func checkEGServeNativeGate(getenv func(string) string) error {
	if getenv("CUDATEST_NATIVE") != "1" {
		return egserveNativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("egserve: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

// TestSimpleKernelTrajectoryNativeClosedGate verifies the closed-gate control
// returns cudaerrors.ErrHardwarePending when CUDATEST_NATIVE is unset.
func TestSimpleKernelTrajectoryNativeClosedGate(t *testing.T) {
	if os.Getenv("CUDATEST_NATIVE") == "1" {
		t.Skip("this control covers the closed case (CUDATEST_NATIVE unset)")
	}
	err := checkEGServeNativeGate(func(string) string { return "" })
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("closed gate err = %v, want ErrHardwarePending", err)
	}
}

// TestSimpleKernelTrajectoryNativeSM121A compiles and disassembles all 24
// evolutionary candidates in buildSimpleKernelStreamIterations using real nvcc,
// ptxas -v, and nvdisasm -c when CUDATEST_NATIVE=1 is set under gpu-lock.
func TestSimpleKernelTrajectoryNativeSM121A(t *testing.T) {
	if os.Getenv("CUDATEST_NATIVE") != "1" {
		t.Skip("native qualification pending: CUDATEST_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvccPath, ver := cudatest.NVCC(t)
	arch := strings.TrimSpace(os.Getenv("CUDATEST_SM_ARCH"))
	if arch == "" {
		arch = "sm_121a"
	}
	cudatest.SkipUnlessTarget(t, nvccPath, arch)

	srv, err := NewEGServer(ServerConfig{
		Arch:      arch,
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	baseIters := srv.buildSimpleKernelStreamIterations(1)
	nvdisasmPath := filepath.Join(filepath.Dir(nvccPath), "nvdisasm")
	liveIters, prov, err := CompileSimpleKernelTrajectoryLive(ctx, baseIters, LiveNVCCConfig{
		Arch:     arch,
		NVCC:     nvccPath,
		NVDISASM: nvdisasmPath,
	})
	if err != nil {
		t.Fatalf("CompileSimpleKernelTrajectoryLive (%s, CUDA %s): %v", arch, ver, err)
	}
	if len(liveIters) != 24 {
		t.Fatalf("len(liveIters) = %d, want 24", len(liveIters))
	}

	// Mutant control: candidate #14 (missing cross-stage __syncthreads()) and
	// candidate #08 (missing cross-warp reduction) must be flagged as rejected.
	if liveIters[8].Promoted {
		t.Fatalf("mutant candidate #08 (missing cross-warp reduction) was promoted, want rejected")
	}
	if liveIters[14].Promoted || liveIters[14].Stages[1].RAWVerified {
		t.Fatalf("mutant candidate #14 (missing __syncthreads RAW hazard) RAWVerified=%v, want false", liveIters[14].Stages[1].RAWVerified)
	}

	final := liveIters[len(liveIters)-1]
	if !final.Promoted || final.MaxRegs <= 0 || len(final.Stages[0].SASSInsts) == 0 {
		t.Fatalf("unexpected final live-compiled candidate: regs=%d sass=%d", final.MaxRegs, len(final.Stages[0].SASSInsts))
	}
	t.Logf("provenance=%s final_regs=%d s0_insts=%d s1_insts=%d",
		prov, final.MaxRegs, final.Stages[0].TotalInsts, final.Stages[1].TotalInsts)
}
