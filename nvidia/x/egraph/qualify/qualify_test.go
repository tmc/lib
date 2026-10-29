package qualify

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
)

func TestQualifySM100Kernels(t *testing.T) {
	ctx := context.Background()

	receipt, err := QualifySM100Kernels(ctx)
	if err != nil {
		t.Fatalf("QualifySM100Kernels failed: %v", err)
	}
	if !receipt.Passed() {
		t.Fatalf("QualifySM100Kernels: receipt.Passed() = false: %+v", receipt)
	}
	if !receipt.CUDAValid {
		t.Errorf("QualifySM100Kernels: CUDAValid = false")
	}
	if !receipt.TMEMTraceVerified {
		t.Errorf("QualifySM100Kernels: TMEMTraceVerified = false")
	}
	if !receipt.SASSEquivVerified {
		t.Errorf("QualifySM100Kernels: SASSEquivVerified = false")
	}
	if receipt.TMEMPeakColumns != 160 {
		t.Errorf("QualifySM100Kernels: TMEMPeakColumns = %d, want 160", receipt.TMEMPeakColumns)
	}

	suiteReceipt, err := QualifySM100Suite(ctx)
	if err != nil {
		t.Fatalf("QualifySM100Suite failed: %v", err)
	}
	if !suiteReceipt.Passed() || suiteReceipt.TMEMPeakColumns != receipt.TMEMPeakColumns {
		t.Errorf("QualifySM100Suite receipt mismatch: %+v vs %+v", suiteReceipt, receipt)
	}
}

func TestQualifyStepSpecAndPlanSpec(t *testing.T) {
	ctx := context.Background()
	rk := tilebridge.NewRMSNormKernel("qualify_rmsnorm_f32", 64, 1e-5)
	tbStep := tilebridge.Step{Row: &rk}

	const rmsSrc = `
#include <cmath>
#include <cstdint>
extern "C" __global__ void qualify_rmsnorm_f32(int M, const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out) {
    int m = blockIdx.x;
    if (m >= M) return;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < 64; c += blockDim.x) {
        float v = x[m * 64 + c];
        ss += v * v;
    }
    __shared__ float smem[32];
    smem[threadIdx.x] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < (int)blockDim.x; ++i) total += smem[i];
        smem[0] = rsqrtf(total / 64.0f + 1e-5f);
    }
    __syncthreads();
    float inv = smem[0];
    for (int c = threadIdx.x; c < 64; c += blockDim.x) {
        out[m * 64 + c] = x[m * 64 + c] * inv * weight[c];
    }
}
`
	spec := KernelStepSpec{
		Name:           "qualify_rmsnorm_f32",
		Kind:           "row kernel",
		CUDASource:     rmsSrc,
		LauncherSource: "inline void qualify_rmsnorm_f32_launch() {}",
		Launch: cudahostemu.Launch{
			Kernel: "qualify_rmsnorm_f32",
			Grid:   cudahostemu.Dim3{X: 4, Y: 1, Z: 1},
			Block:  cudahostemu.Dim3{X: 32, Y: 1, Z: 1},
		},
		LeadingIntArgs:   []int{4},
		ParamDecls:       []string{"const float* __restrict__ x", "const float* __restrict__ weight", "float* __restrict__ out"},
		DefaultBufBytes:  65536,
		CheckNVFP4SASS:   true,
		TileIRBytecode:   &tbStep,
		TileIRDualTarget: &tbStep,
		DualM:            4,
		DualN:            64,
		DualK:            64,
		Eps:              1e-5,
	}

	receipt, err := QualifyStepSpec(ctx, spec)
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; skipping qualification check")
	}
	if err != nil {
		t.Fatalf("QualifyStepSpec failed: %v", err)
	}
	if !receipt.Passed() {
		t.Fatalf("receipt.Passed() = false: %+v", receipt)
	}
	if !receipt.SASSEquivVerified || !receipt.TileIRAgreement {
		t.Fatalf("unexpected flags: SASSEquivVerified=%v TileIRAgreement=%v", receipt.SASSEquivVerified, receipt.TileIRAgreement)
	}

	planReceipts, err := QualifyPlanSpec(ctx, PlanSpec{
		Name:               "rmsnorm_plan",
		SourceFn:           func() (string, error) { return rmsSrc, nil },
		Steps:              []KernelStepSpec{spec},
		PDLCUDAFn:          func() (string, error) { return "// pdl ok", nil },
		MegakernelSourceFn: func() (string, error) { return "// mega ok", nil },
	})
	if err != nil {
		t.Fatalf("QualifyPlanSpec failed: %v", err)
	}
	if len(planReceipts) != 1 || !planReceipts[0].Passed() {
		t.Fatalf("unexpected plan receipts: %+v", planReceipts)
	}
}

func TestParseParamDecl(t *testing.T) {
	name, ctype := ParseParamDecl("const uint16_t* __restrict__ weight")
	if name != "weight" || ctype != "uint16_t" {
		t.Fatalf("got (%q, %q), want (\"weight\", \"uint16_t\")", name, ctype)
	}
}
