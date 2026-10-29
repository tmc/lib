package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func TestRubinSpecs(t *testing.T) {
	r100 := DefaultRubinSpec()
	if r100.Name != "Vera Rubin R100" {
		t.Fatalf("r100.Name = %q, want %q", r100.Name, "Vera Rubin R100")
	}
	if r100.Target != "sm_107a" || r100.FamilyTarget != "sm_107f" {
		t.Fatalf("r100 targets = (%q, %q), want (sm_107a, sm_107f)", r100.Target, r100.FamilyTarget)
	}
	if r100.HBM4Bytes != 288<<30 {
		t.Fatalf("r100.HBM4Bytes = %d, want %d", r100.HBM4Bytes, int64(288<<30))
	}
	if r100.HBM4BandwidthGBps != 13000 {
		t.Fatalf("r100.HBM4BandwidthGBps = %v, want 13000", r100.HBM4BandwidthGBps)
	}
	if r100.NVLink6BandwidthGBps != 3600 {
		t.Fatalf("r100.NVLink6BandwidthGBps = %v, want 3600", r100.NVLink6BandwidthGBps)
	}
	if r100.MaxClusterCTAs != 32 {
		t.Fatalf("r100.MaxClusterCTAs = %d, want 32", r100.MaxClusterCTAs)
	}
	if !r100.SupportsLUTDecompressB || !r100.SupportsCollectorB || !r100.SupportsSyncRestrictReadA || !r100.SupportsTI16 {
		t.Fatalf("r100 capability flags not all true: %+v", r100)
	}

	r200 := DefaultRubinUltraSpec()
	if r200.Name != "Rubin Ultra R200" {
		t.Fatalf("r200.Name = %q, want %q", r200.Name, "Rubin Ultra R200")
	}
	if r200.Target != "sm_107a" || r200.FamilyTarget != "sm_107f" {
		t.Fatalf("r200 targets = (%q, %q), want (sm_107a, sm_107f)", r200.Target, r200.FamilyTarget)
	}
	if r200.HBM4Bytes != 576<<30 {
		t.Fatalf("r200.HBM4Bytes = %d, want %d", r200.HBM4Bytes, int64(576<<30))
	}
	if r200.HBM4BandwidthGBps != 22000 {
		t.Fatalf("r200.HBM4BandwidthGBps = %v, want 22000", r200.HBM4BandwidthGBps)
	}
	if r200.NVLink6BandwidthGBps != 3600 || r200.MaxClusterCTAs != 32 {
		t.Fatalf("r200 interconnect/cluster spec mismatch: %+v", r200)
	}
	if !r200.SupportsLUTDecompressB || !r200.SupportsCollectorB || !r200.SupportsSyncRestrictReadA || !r200.SupportsTI16 {
		t.Fatalf("r200 capability flags not all true: %+v", r200)
	}
}

func TestQuantize3BitLUTBAndLUTDecompressGEMM(t *testing.T) {
	const (
		M = 128
		N = 128
		K = 64
	)

	// Verify input validation on Quantize3BitLUTB.
	if _, _, _, _, err := Quantize3BitLUTB(make([]float32, 16*8), 16, 8); err == nil {
		t.Fatalf("expected error for non-multiple-of-32 K")
	}
	if _, _, _, _, err := Quantize3BitLUTB(make([]float32, 32*6), 32, 6); err == nil {
		t.Fatalf("expected error for non-multiple-of-8 N")
	}

	rng := rand.New(rand.NewPCG(701, 802))
	aF32 := make([]float32, M*K)
	bF32 := make([]float32, K*N)
	palette := [8]float32{-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75}
	for i := range aF32 {
		aF32[i] = float32(rng.IntN(7)-3) * 0.5
	}
	for kk := range K {
		for n := range N {
			bF32[kk*N+n] = palette[(kk*3+n*5+rng.IntN(8))&7]
		}
	}

	aE4M3, aScales := QuantizeMXFP8E4M3(aF32)
	bIndices, bLUTE4M3, bScales, deqB, err := Quantize3BitLUTB(bF32, K, N)
	if err != nil {
		t.Fatalf("Quantize3BitLUTB failed: %v", err)
	}
	if len(bIndices) != K*N || len(bLUTE4M3) != N || len(bScales) != N*(K/32) || len(deqB) != K*N {
		t.Fatalf("unexpected Quantize3BitLUTB output lengths: idx=%d lut=%d scales=%d deq=%d",
			len(bIndices), len(bLUTE4M3), len(bScales), len(deqB))
	}
	for i, idx := range bIndices {
		if idx > 7 {
			t.Fatalf("bIndices[%d] = %d outside [0..7]", i, idx)
		}
		kk := i / N
		n := i % N
		kb := kk / 32
		wantDeq := FromE4M3Bits(bLUTE4M3[(n/8)*8+int(idx)]) * FromUE8M0Bits(bScales[n*(K/32)+kb])
		if deqB[i] != wantDeq {
			t.Fatalf("deqB[%d] = %g, want %g", i, deqB[i], wantDeq)
		}
		if deqB[i] != bF32[i] {
			t.Fatalf("8-level palette weight deqB[%d] = %g, want exact %g", i, deqB[i], bF32[i])
		}
	}

	k := TCGen05LUTDecompressGEMM{
		Name:       "test_rubin_lut_decompress_gemm",
		M:          M,
		N:          N,
		K:          K,
		Target:     "sm_107f",
		CollectorA: CollectorUse,
		CollectorB: CollectorFill,
	}

	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA() failed: %v", err)
	}
	for _, wantSub := range []string{
		"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32",
		".collector::a::use",
		".collector::b::fill",
		"tcgen05.cp.cta_group::1.32x128b.warpx4",
	} {
		if !strings.Contains(src, wantSub) {
			t.Fatalf("CUDA() missing %q", wantSub)
		}
	}
	if fullTU, err := k.CUDAFile(); err != nil || !strings.Contains(fullTU, "EG_PRELUDE") {
		t.Fatalf("CUDAFile() failed or missing Prelude: %v", err)
	}

	want, err := k.Reference(aE4M3, aScales, bIndices, bLUTE4M3, bScales)
	if err != nil {
		t.Fatalf("Reference() failed: %v", err)
	}

	grid, block := k.Grid()
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a_e4m3, a_scales, b_indices, b_lut_e4m3, b_scales, out)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"a_e4m3", "uint8_t", aE4M3},
		emuBuf{"a_scales", "uint8_t", aScales},
		emuBuf{"b_indices", "uint8_t", bIndices},
		emuBuf{"b_lut_e4m3", "uint8_t", bLUTE4M3},
		emuBuf{"b_scales", "uint8_t", bScales},
		emuBuf{"out", "float", poison(4 * M * N)},
	)
	got := bytesF32(res["out"])

	var maxDiff float64
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if math.IsNaN(float64(got[i])) || diff > 1e-5 {
			t.Fatalf("out[%d] = %g, want %g (diff %g)", i, got[i], want[i], diff)
		}
	}
	if maxDiff != 0 {
		t.Fatalf("expected bit-exact agreement with Reference, got maxDiff=%g", maxDiff)
	}

	// Negative mutant test: corrupting LUT entries in bLUTE4M3 must alter the
	// decompressed GEMM output and be detected against the uncorrupted reference.
	corruptedLUT := slices.Clone(bLUTE4M3)
	for i := range 8 {
		corruptedLUT[i] = E4M3Bits(FromE4M3Bits(corruptedLUT[i]) + 16.0)
	}
	resMutant := runEmu(t, src, launch,
		emuBuf{"a_e4m3", "uint8_t", aE4M3},
		emuBuf{"a_scales", "uint8_t", aScales},
		emuBuf{"b_indices", "uint8_t", bIndices},
		emuBuf{"b_lut_e4m3", "uint8_t", corruptedLUT},
		emuBuf{"b_scales", "uint8_t", bScales},
		emuBuf{"out", "float", poison(4 * M * N)},
	)
	gotMutant := bytesF32(resMutant["out"])
	var mutantDiffs int
	for i := range want {
		if math.Abs(float64(gotMutant[i]-want[i])) > 1e-2 {
			mutantDiffs++
		}
	}
	if mutantDiffs == 0 {
		t.Fatalf("corrupting bLUTE4M3 did not change emulated GEMM output")
	}
}

func TestTCGen05EarlyReleasePipelineGEMM(t *testing.T) {
	k := TCGen05EarlyReleasePipelineGEMM{
		Name:        "test_rubin_early_release_pipeline_gemm",
		M:           128,
		N:           128,
		K:           128,
		Stages:      2,
		ClusterCTAs: 16,
		Target:      "sm_107f",
	}

	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA() failed: %v", err)
	}
	for _, wantSub := range []string{
		"tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64",
		"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64",
		"0x0000ffffu",
	} {
		if !strings.Contains(src, wantSub) {
			t.Fatalf("CUDA() missing %q", wantSub)
		}
	}
	if fullTU, err := k.CUDAFile(); err != nil || !strings.Contains(fullTU, "EG_PRELUDE") {
		t.Fatalf("CUDAFile() failed or missing Prelude: %v", err)
	}

	rng := rand.New(rand.NewPCG(901, 902))
	a := make([]float32, k.M*k.K)
	b := make([]float32, k.K*k.N)
	for i := range a {
		a[i] = float32(rng.IntN(9)-4) * 0.25
	}
	for i := range b {
		b[i] = float32(rng.IntN(9)-4) * 0.25
	}

	want, err := k.Reference(a, b)
	if err != nil {
		t.Fatalf("Reference() failed: %v", err)
	}

	grid, block := k.Grid()
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a, b, out)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"a", "float", f32Bytes(a)},
		emuBuf{"b", "float", f32Bytes(b)},
		emuBuf{"out", "float", poison(4 * k.M * k.N)},
	)
	got := bytesF32(res["out"])

	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if math.IsNaN(float64(got[i])) || diff > 1e-5 {
			t.Fatalf("out[%d] = %g, want %g (diff %g)", i, got[i], want[i], diff)
		}
	}
}

func ExampleTCGen05LUTDecompressGEMM() {
	spec := DefaultRubinSpec()
	k := TCGen05LUTDecompressGEMM{
		Name:       "rubin_lut_b_gemm",
		M:          128,
		N:          128,
		K:          64,
		Target:     spec.FamilyTarget,
		CollectorB: CollectorFill,
	}
	src, err := k.CUDA()
	fmt.Println(
		err == nil,
		spec.Target,
		spec.FamilyTarget,
		spec.MaxClusterCTAs,
		strings.Contains(src, "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32.collector::b::fill"),
	)
	// Output:
	// true sm_107a sm_107f 32 true
}
