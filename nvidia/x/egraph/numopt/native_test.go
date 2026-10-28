package numopt

import (
	"context"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// TestEmitKLoopNative compiles the mixed-format K loop of the outlier layer
// to a cubin with nvcc. It runs only with CUDA_NUMOPT_NATIVE=1 under
// gpu-lock, and it compiles without launching: the operand descriptors a
// launch needs come from the caller's TMA and TMEM setup, not from here.
//
//	CUDA_NUMOPT_NATIVE=1 gpu-lock go test -run EmitKLoopNative ./x/egraph/numopt
func TestEmitKLoopNative(t *testing.T) {
	if os.Getenv("CUDA_NUMOPT_NATIVE") != "1" {
		t.Skip("native compile pending: CUDA_NUMOPT_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, _ := cudatest.NVCC(t)
	tiles := outlierLayer()
	for _, arch := range []string{"sm_100a", "sm_107a"} {
		t.Run(arch, func(t *testing.T) {
			cudatest.SkipUnlessTarget(t, nvcc, arch)
			_, frontier, err := SolveParetoFrontier(tiles, arch, math.Inf(1))
			if err != nil {
				t.Fatal(err)
			}
			p := frontier[len(frontier)/2]
			groups, err := PlanKLoop(tiles, p.Assignments, arch)
			if err != nil {
				t.Fatal(err)
			}
			loop, err := EmitKLoop("eg_numopt_kloop", arch, groups, len(tiles))
			if err != nil {
				t.Fatal(err)
			}
			src := "#include <cstdint>\n" + loop + `
extern "C" __global__ void eg_numopt_kloop_kernel(
    uint32_t d, const uint64_t* a, const uint64_t* b, const uint32_t* id,
    const uint32_t* sfa, const uint32_t* sfb, const uint32_t* meta, const uint32_t* lut) {
    if (threadIdx.x == 0) eg_numopt_kloop(d, a, b, id, sfa, sfb, meta, lut);
}
`
			dir := t.TempDir()
			cu := filepath.Join(dir, "kloop.cu")
			if err := os.WriteFile(cu, []byte(src), 0o600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
			defer cancel()
			out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-cubin", "-arch="+arch, "-Xptxas", "-v",
				"-o", filepath.Join(dir, "kloop.cubin"), cu).CombinedOutput()
			if err != nil {
				t.Fatalf("nvcc %s: %v\n%s\n%s", arch, err, out, src)
			}
			t.Logf("%s: %v\n%s", arch, p, out)
		})
	}
}
