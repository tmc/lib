package tcgen05

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func u32Bytes(v []uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, u := range v {
		binary.LittleEndian.PutUint32(b[4*i:], u)
	}
	return b
}

func TestPruneAndUnpack2To4(t *testing.T) {
	const (
		M = 32
		K = 64
	)
	rng := rand.New(rand.NewPCG(2026, 1004))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.IntN(17)-8) * 0.25
	}
	// Inject explicit magnitude ties in row 0 to test deterministic tie-breaking (0 < 1 < 2 < 3).
	a[0] = -2.0
	a[1] = 2.0
	a[2] = -2.0
	a[3] = 2.0

	compressed, meta, prunedDense, err := Prune2To4F32(a, M, K)
	if err != nil {
		t.Fatalf("Prune2To4F32 failed: %v", err)
	}
	if len(compressed) != M*(K/2) {
		t.Fatalf("len(compressed) = %d, want %d", len(compressed), M*(K/2))
	}
	if len(meta) != M*(K/32) {
		t.Fatalf("len(meta) = %d, want %d", len(meta), M*(K/32))
	}
	if len(prunedDense) != M*K {
		t.Fatalf("len(prunedDense) = %d, want %d", len(prunedDense), M*K)
	}

	// Tie-break check on group 0 of row 0: indices 0 and 1 must be chosen.
	if nib0 := meta[0] & 0xf; nib0 != (0 | (1 << 2)) {
		t.Fatalf("tie-break nibble = 0x%x, want 0x4 (idx0=0, idx1=1)", nib0)
	}
	if compressed[0] != -2.0 || compressed[1] != 2.0 {
		t.Fatalf("compressed[0:2] = (%v, %v), want (-2.0, 2.0)", compressed[0], compressed[1])
	}

	// Verify every 4-element group in prunedDense has at most 2 non-zero values.
	for r := range M {
		for g := range K / 4 {
			nonZero := 0
			for j := range 4 {
				if prunedDense[r*K+4*g+j] != 0 {
					nonZero++
				}
			}
			if nonZero > 2 {
				t.Fatalf("row %d group %d has %d non-zero values, want <= 2", r, g, nonZero)
			}
		}
	}

	// Verify Unpack2To4F32 round-trip matches prunedDense bit-exactly.
	unpacked, err := Unpack2To4F32(compressed, meta, M, K)
	if err != nil {
		t.Fatalf("Unpack2To4F32 failed: %v", err)
	}
	for i := range prunedDense {
		if math.Float32bits(unpacked[i]) != math.Float32bits(prunedDense[i]) {
			t.Fatalf("unpacked[%d] = %v (0x%08x), want %v (0x%08x)",
				i, unpacked[i], math.Float32bits(unpacked[i]),
				prunedDense[i], math.Float32bits(prunedDense[i]))
		}
	}

	// Verify Prune2To4NVFP4 on a [M, K/2] packed NVFP4 tensor.
	aPacked := make([]uint8, M*(K/2))
	aScales := make([]uint8, M*(K/16))
	for i := range aPacked {
		c0 := uint8(rng.IntN(16))
		c1 := uint8(rng.IntN(16))
		aPacked[i] = (c0 & 0xf) | ((c1 & 0xf) << 4)
	}
	for i := range aScales {
		aScales[i] = E4M3Bits(0.5)
	}
	compPacked, compScales, nvfp4Meta, err := Prune2To4NVFP4(aPacked, aScales, M, K)
	if err != nil {
		t.Fatalf("Prune2To4NVFP4 failed: %v", err)
	}
	if len(compPacked) != M*(K/4) || len(compScales) != M*(K/16) || len(nvfp4Meta) != M*(K/32) {
		t.Fatalf("unexpected Prune2To4NVFP4 lengths: packed=%d scales=%d meta=%d",
			len(compPacked), len(compScales), len(nvfp4Meta))
	}
	for r := range M {
		for g := range K / 4 {
			nib := (nvfp4Meta[r*(K/32)+(g>>3)] >> (4 * (g & 7))) & 0xf
			idx0 := int(nib & 0x3)
			idx1 := int((nib >> 2) & 0x3)
			if idx0 >= idx1 {
				t.Fatalf("NVFP4 row %d group %d invalid metadata indices (%d, %d)", r, g, idx0, idx1)
			}
		}
	}
}

func TestTCGen05SparseGEMM(t *testing.T) {
	k := TCGen05SparseGEMM{
		Name:     "test_tcgen05_sparse_gemm",
		M:        128,
		N:        128,
		K:        64,
		Target:   "sm_100a",
		CTAGroup: 1,
	}

	d, err := k.InstDescriptor()
	if err != nil {
		t.Fatalf("InstDescriptor() failed: %v", err)
	}
	if !UnpackInstDesc(d).Sparsity {
		t.Fatalf("UnpackInstDesc(0x%08x).Sparsity = false, want true", d)
	}
	if got := DecodeInstDesc(d); got.Sparsity == 0 || got.M != 128 || got.N != 128 {
		t.Fatalf("DecodeInstDesc(0x%08x) = %+v, want Sparsity != 0, M=128, N=128", d, got)
	}

	for _, target := range []cudatmem.Target{cudatmem.SM100A, cudatmem.SM103A} {
		sum, err := cudatmem.VerifyTrace(target, k.TraceEvents())
		if err != nil {
			t.Fatalf("VerifyTrace(%s) failed: %v", target, err)
		}
		if sum.CTAs != 1 || sum.CTAGroup != 1 || sum.MaxAllocatedColumns != 160 || sum.Copies != 1 || sum.MMAs != 1 || sum.Loads != 4 {
			t.Fatalf("VerifyTrace(%s) unexpected summary: %+v", target, sum)
		}
	}

	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA() failed: %v", err)
	}
	for _, want := range []string{
		"tcgen05.cp.cta_group::1.32x128b.warpx4",
		"tcgen05.mma.sp.cta_group::1.kind::tf32",
		"tcgen05.ld.sync.aligned.32x32b.x1.b32",
		"elect.sync",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("CUDA() missing %q", want)
		}
	}

	// Verify BlockScaledNVFP4 codegen on sm_100a and sm_103a.
	kFP4SM103 := TCGen05SparseGEMM{
		Name:             "test_tcgen05_sparse_nvfp4_sm103a",
		M:                128,
		N:                128,
		K:                128,
		Target:           "sm_103a",
		CTAGroup:         1,
		BlockScaledNVFP4: true,
	}
	src103, err := kFP4SM103.CUDA()
	if err != nil {
		t.Fatalf("kFP4SM103.CUDA() failed: %v", err)
	}
	if !strings.Contains(src103, "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X") {
		t.Fatalf("sm_103a BlockScaledNVFP4 CUDA() missing scale_vec::4X sparse mma")
	}

	// Execute via cudahostemu.Run (runEmu) and compare against both Reference and prunedDense * B.
	rng := rand.New(rand.NewPCG(77, 88))
	aDense := make([]float32, k.M*k.K)
	bDense := make([]float32, k.K*k.N)
	for i := range aDense {
		aDense[i] = float32(rng.IntN(11)-5) * 0.25
	}
	for i := range bDense {
		bDense[i] = float32(rng.IntN(9)-4) * 0.125
	}

	aComp, aMeta, prunedDense, err := Prune2To4F32(aDense, k.M, k.K)
	if err != nil {
		t.Fatalf("Prune2To4F32 failed: %v", err)
	}
	wantRef, err := k.Reference(aComp, aMeta, bDense)
	if err != nil {
		t.Fatalf("Reference failed: %v", err)
	}

	grid, block := k.Grid()
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a_compressed, a_meta, b, out)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"a_compressed", "float", f32Bytes(aComp)},
		emuBuf{"a_meta", "uint32_t", u32Bytes(aMeta)},
		emuBuf{"b", "float", f32Bytes(bDense)},
		emuBuf{"out", "float", poison(4 * k.M * k.N)},
	)
	got := bytesF32(res["out"])

	var maxDiffRef, maxDiffDense float64
	for r := range k.M {
		for c := range k.N {
			idx := r*k.N + c
			var denseDot float64
			for kk := range k.K {
				denseDot += float64(prunedDense[r*k.K+kk]) * float64(bDense[kk*k.N+c])
			}
			diffRef := math.Abs(float64(got[idx]) - float64(wantRef[idx]))
			diffDense := math.Abs(float64(got[idx]) - denseDot)
			if diffRef > maxDiffRef {
				maxDiffRef = diffRef
			}
			if diffDense > maxDiffDense {
				maxDiffDense = diffDense
			}
			if math.IsNaN(float64(got[idx])) || diffRef >= 1e-5 || diffDense >= 1e-5 {
				t.Fatalf("out[%d, %d] = %g, wantRef=%g (diff=%g), denseDot=%g (diff=%g)",
					r, c, got[idx], wantRef[idx], diffRef, denseDot, diffDense)
			}
		}
	}
}

func ExampleTCGen05SparseGEMM() {
	k := TCGen05SparseGEMM{
		Name:             "blackwell_sparse_gemm",
		M:                128,
		N:                128,
		K:                64,
		Target:           "sm_103a",
		CTAGroup:         1,
		BlockScaledNVFP4: true,
	}
	idesc, _ := k.InstDescriptor()
	unpacked := UnpackInstDesc(idesc)
	sum, err := cudatmem.VerifyTrace(cudatmem.SM103A, k.TraceEvents())
	src, _ := k.CUDA()
	fmt.Println(
		err == nil,
		unpacked.Sparsity,
		unpacked.ScaleVec,
		sum.MMAs,
		sum.MaxAllocatedColumns,
		strings.Contains(src, "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"),
	)
	// Output:
	// true true 4X 1 160 true
}
