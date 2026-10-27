package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func TestTCGen05TMAMulticastGatherGEMM(t *testing.T) {
	k := TCGen05TMAMulticastGatherGEMM{
		Name:   "test_tcgen05_tma_multicast_gather_gemm",
		M:      256,
		N:      128,
		K:      32,
		Target: "sm_100a",
	}

	for _, target := range []cudatmem.Target{cudatmem.SM100A, cudatmem.SM103A} {
		sum, err := cudatmem.VerifyTrace(target, k.TraceEvents())
		if err != nil {
			t.Fatalf("VerifyTrace(%s) failed: %v", target, err)
		}
		if sum.CTAs != 2 || sum.CTAGroup != 2 || sum.MaxAllocatedColumns != 128 || sum.MMAs != 1 || sum.Loads != 8 || sum.ClusterSyncs != 2 {
			t.Fatalf("VerifyTrace(%s) unexpected summary: %+v", target, sum)
		}
	}

	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA() failed: %v", err)
	}
	fullUnit, err := k.CUDAFile()
	if err != nil {
		t.Fatalf("CUDAFile() failed: %v", err)
	}
	if !strings.Contains(fullUnit, "EG_PRELUDE") {
		t.Fatalf("CUDAFile() missing EG_PRELUDE")
	}

	for _, want := range []string{
		"__cluster_dims__(2, 1, 1)",
		"gather4",
		"scatter4",
		"multicast::cluster",
		"createpolicy.fractional.L2::evict_last",
		"createpolicy.fractional.L2::evict_first",
		"tcgen05.mma.cta_group::2.kind::tf32",
		"tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64",
		"tcgen05.ld.sync.aligned.32x32b.x1.b32",
		"tcgen05.dealloc.cta_group::2.sync.aligned.b32",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted CUDA missing %q", want)
		}
	}

	// Build a global token pool with 512 rows where all 512 rows are initially
	// poisoned with NaN, and only the 256 non-contiguous permuted gatherRows
	// receive finite values. If the kernel reads any ungathered row, NaN will
	// propagate into the output.
	const totalGlobalRows = 512
	rng := rand.New(rand.NewPCG(2026, 1103))
	perm := rng.Perm(totalGlobalRows)
	gatherRows := make([]int32, k.M)
	gatheredSet := make(map[int]bool, k.M)
	for r := 0; r < k.M; r++ {
		rowIdx := perm[r]
		gatherRows[r] = int32(rowIdx)
		gatheredSet[rowIdx] = true
	}

	aGlobal := make([]float32, totalGlobalRows*k.K)
	nan32 := float32(math.NaN())
	for r := 0; r < totalGlobalRows; r++ {
		for c := 0; c < k.K; c++ {
			if gatheredSet[r] {
				aGlobal[r*k.K+c] = float32(rng.IntN(17)-8) * 0.125
			} else {
				aGlobal[r*k.K+c] = nan32
			}
		}
	}

	b := make([]float32, k.K*k.N)
	for i := range b {
		b[i] = float32(rng.IntN(17)-8) * 0.125
	}

	want, err := k.Reference(aGlobal, totalGlobalRows, gatherRows, b)
	if err != nil {
		t.Fatalf("Reference() failed: %v", err)
	}

	for _, scatterOut := range []bool{false, true} {
		modeName := "direct_epilogue"
		if scatterOut {
			modeName = "scatter4_epilogue"
		}
		t.Run(modeName, func(t *testing.T) {
			kern := k
			kern.ScatterOutput = scatterOut
			if scatterOut {
				kern.Name = "test_tcgen05_tma_multicast_gather_scatter_gemm"
				kern.Target = "sm_103a"
			}
			kSrc, err := kern.CUDA()
			if err != nil {
				t.Fatalf("CUDA() failed: %v", err)
			}

			grid, block := kern.Grid()
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a_global, gather_rows, b, out)",
				grid[0], grid[1], grid[2], block[0], block[1], block[2], kern.Name)
			res := runEmu(t, kSrc, launch,
				emuBuf{"a_global", "float", f32Bytes(aGlobal)},
				emuBuf{"gather_rows", "int32_t", i32Bytes(gatherRows)},
				emuBuf{"b", "float", f32Bytes(b)},
				emuBuf{"out", "float", poison(4 * kern.M * kern.N)},
			)
			got := bytesF32(res["out"])

			var maxDiff float64
			for i, g := range got {
				if math.IsNaN(float64(g)) {
					t.Fatalf("out[%d] is NaN (poisoned ungathered row leaked or uninitialized output)", i)
				}
				diff := math.Abs(float64(g) - float64(want[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff >= 1e-5 {
					t.Fatalf("out[%d] = %g, want %g (diff %g >= 1e-5)", i, g, want[i], diff)
				}
			}
			t.Logf("%s: 2-CTA gather4 + multicast::cluster GEMM max abs diff = %g", modeName, maxDiff)
		})
	}
}

func ExampleTCGen05TMAMulticastGatherGEMM() {
	k := TCGen05TMAMulticastGatherGEMM{
		Name:   "routed_expert_gemm",
		M:      256,
		N:      128,
		K:      32,
		Target: "sm_100a",
	}
	grid, block := k.Grid()
	sum, _ := cudatmem.VerifyTrace(cudatmem.SM100A, k.TraceEvents())
	src, _ := k.CUDA()

	fmt.Printf("grid=%v block=%v cta_group=%d mmas=%d loads=%d\n",
		grid, block, sum.CTAGroup, sum.MMAs, sum.Loads)
	fmt.Printf("has_gather4=%v has_multicast=%v has_evict_last=%v\n",
		strings.Contains(src, ".gather4."),
		strings.Contains(src, ".multicast::cluster."),
		strings.Contains(src, "createpolicy.fractional.L2::evict_last"))
	// Output:
	// grid=[2 1 1] block=[128 1 1] cta_group=2 mmas=1 loads=8
	// has_gather4=true has_multicast=true has_evict_last=true
}
