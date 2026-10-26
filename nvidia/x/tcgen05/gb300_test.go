package tcgen05

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func TestDefaultGB300Spec(t *testing.T) {
	spec := DefaultGB300Spec()
	const gib = 1024 * 1024 * 1024
	if spec.Target != "sm_103a" {
		t.Fatalf("Target = %q, want sm_103a", spec.Target)
	}
	if spec.FamilyTarget != "sm_100f" {
		t.Fatalf("FamilyTarget = %q, want sm_100f", spec.FamilyTarget)
	}
	if spec.HBMBytes != 288*gib {
		t.Fatalf("HBMBytes = %d, want %d (288 GiB)", spec.HBMBytes, 288*gib)
	}
	if spec.RackHBMBytes != 72*288*gib {
		t.Fatalf("RackHBMBytes = %d, want %d (72 * 288 GiB)", spec.RackHBMBytes, 72*288*gib)
	}
	if spec.HBMBandwidthGBps != 8000 {
		t.Fatalf("HBMBandwidthGBps = %v, want 8000", spec.HBMBandwidthGBps)
	}
	if spec.DenseFP4TFLOPS != 15000 {
		t.Fatalf("DenseFP4TFLOPS = %v, want 15000", spec.DenseFP4TFLOPS)
	}
	if spec.DenseFP8TFLOPS != 5000 {
		t.Fatalf("DenseFP8TFLOPS = %v, want 5000", spec.DenseFP8TFLOPS)
	}
	if spec.AttentionSFUMultiplier != 2.0 {
		t.Fatalf("AttentionSFUMultiplier = %v, want 2.0", spec.AttentionSFUMultiplier)
	}
	if spec.ScaleOutBandwidthGBps != 100 {
		t.Fatalf("ScaleOutBandwidthGBps = %v, want 100", spec.ScaleOutBandwidthGBps)
	}
	if spec.GraceLPDDR5XBytes != 512*gib {
		t.Fatalf("GraceLPDDR5XBytes = %d, want %d (512 GiB)", spec.GraceLPDDR5XBytes, 512*gib)
	}
}

func TestTCGen05ScaleVec4XGEMMHostEmuAndSM103ATrace(t *testing.T) {
	const (
		M       = 128
		N       = 128
		K       = 128
		kBytes  = K / 2
		kBlocks = K / 16
	)
	k := TCGen05ScaleVec4XGEMM{
		Name: "test_gb300_scale_vec_4x_gemm",
		M:    M,
		N:    N,
		K:    K,
	}

	sum103, err := cudatmem.VerifyTrace(cudatmem.SM103A, k.TraceEvents())
	if err != nil {
		t.Fatalf("VerifyTrace(SM103A, scale_vec::4X) failed: %v", err)
	}
	if sum103.Target != cudatmem.SM103A || sum103.MaxAllocatedColumns != 160 || sum103.Copies != 1 || sum103.MMAs != 1 {
		t.Fatalf("unexpected SM103A trace summary: %+v", sum103)
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM100F, k.TraceEvents()); err != nil {
		t.Fatalf("VerifyTrace(SM100F, scale_vec::4X) failed: %v", err)
	}

	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA: %v", err)
	}
	for _, want := range []string{
		"tcgen05.cp.cta_group::1.32x128b",
		"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
		"eg_tcgen05_mma_block_scale_nvfp4_4x",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("scale_vec::4X source missing %q", want)
		}
	}

	scaleChoices := []float32{0.5, 1.0, 1.5, 2.0}
	aPacked := make([]uint8, M*kBytes)
	bPacked := make([]uint8, N*kBytes)
	aScales := make([]uint8, M*kBlocks)
	bScales := make([]uint8, N*kBlocks)

	for r := range M {
		for kb := range kBlocks {
			aScales[r*kBlocks+kb] = E4M3Bits(scaleChoices[(r*5+kb)%len(scaleChoices)])
			for j := range 8 {
				lo := uint8((r*7 + kb*3 + j) & 0x7)
				hi := uint8(((r*11 + kb*5 + j) & 0x7) | (((r + kb + j) & 1) << 3))
				aPacked[r*kBytes+kb*8+j] = lo | (hi << 4)
			}
		}
	}
	for n := range N {
		for kb := range kBlocks {
			bScales[n*kBlocks+kb] = E4M3Bits(scaleChoices[(n*3+kb+1)%len(scaleChoices)])
			for j := range 8 {
				lo := uint8((n*5 + kb*7 + j + 1) & 0x7)
				hi := uint8(((n*9 + kb*3 + j) & 0x7) | (((n + j) & 1) << 3))
				bPacked[n*kBytes+kb*8+j] = lo | (hi << 4)
			}
		}
	}

	want := make([]float64, M*N)
	for r := range M {
		for n := range N {
			var acc float64
			for kb := range kBlocks {
				sa := float64(FromE4M3Bits(aScales[r*kBlocks+kb]))
				sb := float64(FromE4M3Bits(bScales[n*kBlocks+kb]))
				var blockDot float64
				for j := range 8 {
					ba := aPacked[r*kBytes+kb*8+j]
					bb := bPacked[n*kBytes+kb*8+j]
					a0 := float64(FromE2M1(ba & 0xf))
					a1 := float64(FromE2M1((ba >> 4) & 0xf))
					b0 := float64(FromE2M1(bb & 0xf))
					b1 := float64(FromE2M1((bb >> 4) & 0xf))
					blockDot += a0*b0 + a1*b1
				}
				acc += blockDot * sa * sb
			}
			want[r*N+n] = acc
		}
	}

	grid, block := k.Grid()
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a_packed, a_scales, b_packed, b_scales, out)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"a_packed", "uint8_t", aPacked},
		emuBuf{"a_scales", "uint8_t", aScales},
		emuBuf{"b_packed", "uint8_t", bPacked},
		emuBuf{"b_scales", "uint8_t", bScales},
		emuBuf{"out", "float", poison(4 * M * N)},
	)
	out := bytesF32(res["out"])

	var maxDiff float64
	for i, got := range out {
		d := math.Abs(float64(got) - want[i])
		if d > maxDiff {
			maxDiff = d
		}
		if math.IsNaN(float64(got)) || d > 1e-3 {
			t.Fatalf("out[%d] = %g, want %g (diff %g)", i, got, want[i], d)
		}
	}
	t.Logf("sm_103a scale_vec::4X GEMM M=%d N=%d K=%d max abs error vs float64 = %g", M, N, K, maxDiff)
}

func TestPlanGB300ResidencyVsGB200(t *testing.T) {
	const (
		gib             = 1024 * 1024 * 1024
		gb200HBMPerGPU  = 192 * gib
		modelParams1p4T = 1_400_000_000_000 // 1.4T parameter MoE model in NVFP4
		weightBits      = 4
		kvBytesPerToken = 128 * 1024 // 128 KiB/token KV cache
		batchSize       = 8
		seqLen          = 32768
		numGPUs         = 4
	)

	gb200Plan := PlanHBMResidencyWithCapacity(
		"sm_100a", gb200HBMPerGPU,
		modelParams1p4T, weightBits, kvBytesPerToken, batchSize, seqLen, numGPUs,
	)
	gb300Plan := PlanGB300Residency(
		modelParams1p4T, weightBits, kvBytesPerToken, batchSize, seqLen, numGPUs,
	)

	if gb200Plan.FitsInHBM {
		t.Fatalf("expected 1.4T NVFP4 model on 4x 192 GiB GB200 to spill to C2C, got FitsInHBM=true (req=%d avail=%d)",
			gb200Plan.TotalRequiredBytes, gb200Plan.AvailableHBMBytes)
	}
	if gb200Plan.C2CColdFraction <= 0 {
		t.Fatalf("expected GB200 C2CColdFraction > 0, got %v", gb200Plan.C2CColdFraction)
	}

	if !gb300Plan.FitsInHBM {
		t.Fatalf("expected 1.4T NVFP4 model on 4x 288 GiB GB300 to fit in HBM3e, got FitsInHBM=false (req=%d avail=%d)",
			gb300Plan.TotalRequiredBytes, gb300Plan.AvailableHBMBytes)
	}
	if gb300Plan.C2CColdFraction != 0.0 {
		t.Fatalf("expected GB300 C2CColdFraction == 0.0, got %v", gb300Plan.C2CColdFraction)
	}
	if gb300Plan.SpilledC2CBytes != 0 {
		t.Fatalf("expected GB300 SpilledC2CBytes == 0, got %d", gb300Plan.SpilledC2CBytes)
	}
	if gb300Plan.MaxHBMContextLen <= gb200Plan.MaxHBMContextLen {
		t.Fatalf("expected GB300 MaxHBMContextLen (%d) > GB200 MaxHBMContextLen (%d)",
			gb300Plan.MaxHBMContextLen, gb200Plan.MaxHBMContextLen)
	}

	// Verify 72-GPU rack residency for a 2T parameter model at 128K context.
	rackPlan := PlanGB300Residency(2_000_000_000_000, 4, kvBytesPerToken, 64, 131072, 72)
	if !rackPlan.FitsInHBM || rackPlan.C2CColdFraction != 0.0 {
		t.Fatalf("expected 2T model on 72-GPU GB300 rack to fit in 20.7 TiB HBM3e, got %+v", rackPlan)
	}
}
