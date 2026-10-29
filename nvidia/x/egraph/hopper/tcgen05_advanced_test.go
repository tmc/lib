package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func TestTCGen05Cluster2GEMM(t *testing.T) {
	k := TCGen05Cluster2GEMM{
		Name: "test_tcgen05_cluster2_gemm",
		M:    256,
		N:    128,
		K:    32,
	}

	// 1. Verify formal sm_100a tcgen05 protocol trace.
	sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, k.TraceEvents())
	if err != nil {
		t.Fatalf("VerifyTrace failed: %v", err)
	}
	if sum.CTAs != 2 || sum.CTAGroup != 2 || sum.MaxAllocatedColumns != 128 || sum.MMAs != 1 || sum.Loads != 8 || sum.ClusterSyncs != 2 {
		t.Fatalf("unexpected trace summary: %+v", sum)
	}

	// 2. Emit CUDA and check sm_100a PTX instruction forms.
	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA() failed: %v", err)
	}
	for _, want := range []string{
		"__cluster_dims__(2, 1, 1)",
		"tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32",
		"tcgen05.mma.cta_group::2.kind::tf32",
		"tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64",
		"tcgen05.ld.sync.aligned.32x32b.x1.b32",
		"tcgen05.dealloc.cta_group::2.sync.aligned.b32",
		"elect.sync",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted CUDA missing %q", want)
		}
	}

	// 3. Execute on host CPU via cudahostemu.Run and compare against float64 reference.
	rng := rand.New(rand.NewPCG(42, 100))
	a := make([]float32, k.M*k.K)
	b := make([]float32, k.K*k.N)
	for i := range a {
		a[i] = float32(rng.IntN(9)-4) * 0.25
	}
	for i := range b {
		b[i] = float32(rng.IntN(9)-4) * 0.125
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

	for r := range k.M {
		for c := range k.N {
			var want float64
			for kk := range k.K {
				want += float64(a[r*k.K+kk]) * float64(b[kk*k.N+c])
			}
			diff := math.Abs(float64(got[r*k.N+c]) - want)
			if math.IsNaN(float64(got[r*k.N+c])) || diff > 1e-4 {
				t.Fatalf("out[%d, %d] = %g, want %g (diff %g)", r, c, got[r*k.N+c], want, diff)
			}
		}
	}
}

func TestTCGen05BlockScaledGEMM(t *testing.T) {
	k := TCGen05BlockScaledGEMM{
		Name: "test_tcgen05_block_scaled_gemm",
		M:    128,
		N:    128,
		K:    64,
	}

	// 1. Verify formal sm_100a tcgen05 protocol trace (including tcgen05.cp scale staging).
	sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, k.TraceEvents())
	if err != nil {
		t.Fatalf("VerifyTrace failed: %v", err)
	}
	if sum.CTAs != 1 || sum.CTAGroup != 1 || sum.MaxAllocatedColumns != 160 || sum.Copies != 1 || sum.MMAs != 1 || sum.Loads != 4 {
		t.Fatalf("unexpected trace summary: %+v", sum)
	}

	// 2. Emit CUDA and check sm_100a PTX instruction forms.
	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA() failed: %v", err)
	}
	for _, want := range []string{
		"tcgen05.cp.cta_group::1.32x128b",
		"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale",
		"tcgen05.ld.sync.aligned.32x32b.x1.b32",
		"elect.sync",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted CUDA missing %q", want)
		}
	}

	// 3. Execute on host CPU via cudahostemu.Run and compare against float64 reference.
	rng := rand.New(rand.NewPCG(7, 77))
	kBytes := k.K / 2
	kBlocks := k.K / 16
	aPacked := make([]uint8, k.M*kBytes)
	bPacked := make([]uint8, k.N*kBytes)
	aScales := make([]uint8, k.M*kBlocks)
	bScales := make([]uint8, k.N*kBlocks)

	for i := range aPacked {
		c0 := uint8(rng.IntN(16))
		c1 := uint8(rng.IntN(16))
		aPacked[i] = (c0 & 0xf) | ((c1 & 0xf) << 4)
	}
	for i := range bPacked {
		c0 := uint8(rng.IntN(16))
		c1 := uint8(rng.IntN(16))
		bPacked[i] = (c0 & 0xf) | ((c1 & 0xf) << 4)
	}
	scaleChoices := []float32{0.25, 0.5, 1.0, 1.5, 2.0}
	for i := range aScales {
		aScales[i] = E4M3Bits(scaleChoices[rng.IntN(len(scaleChoices))])
	}
	for i := range bScales {
		bScales[i] = E4M3Bits(scaleChoices[rng.IntN(len(scaleChoices))])
	}

	grid, block := k.Grid()
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a_packed, a_scales, b_packed, b_scales, out)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"a_packed", "uint8_t", aPacked},
		emuBuf{"a_scales", "uint8_t", aScales},
		emuBuf{"b_packed", "uint8_t", bPacked},
		emuBuf{"b_scales", "uint8_t", bScales},
		emuBuf{"out", "float", poison(4 * k.M * k.N)},
	)
	got := bytesF32(res["out"])

	for r := range k.M {
		for c := range k.N {
			var want float64
			for kb := range kBlocks {
				sa := float64(FromE4M3Bits(aScales[r*kBlocks+kb]))
				sb := float64(FromE4M3Bits(bScales[c*kBlocks+kb]))
				var blockDot float64
				for j := range 8 {
					ba := aPacked[r*kBytes+kb*8+j]
					bb := bPacked[c*kBytes+kb*8+j]
					a0 := float64(FromE2M1(ba & 0xf))
					a1 := float64(FromE2M1((ba >> 4) & 0xf))
					b0 := float64(FromE2M1(bb & 0xf))
					b1 := float64(FromE2M1((bb >> 4) & 0xf))
					blockDot += a0*b0 + a1*b1
				}
				want += blockDot * sa * sb
			}
			diff := math.Abs(float64(got[r*k.N+c]) - want)
			if math.IsNaN(float64(got[r*k.N+c])) || diff > 1e-3 {
				t.Fatalf("out[%d, %d] = %g, want %g (diff %g)", r, c, got[r*k.N+c], want, diff)
			}
		}
	}
}

func TestTCGen05TMEMFlashAttention(t *testing.T) {
	k := TCGen05TMEMFlashAttention{
		Name: "test_tcgen05_tmem_flash_attention",
		M:    128,
		KV:   64,
		D:    64,
	}

	// 1. Verify formal sm_100a tcgen05 protocol trace (including TMEM-sourced operand A).
	sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, k.TraceEvents())
	if err != nil {
		t.Fatalf("VerifyTrace failed: %v", err)
	}
	if sum.CTAs != 1 || sum.CTAGroup != 1 || sum.MaxAllocatedColumns != 128 || sum.MMAs != 2 || sum.Loads != 8 || sum.Stores != 4 {
		t.Fatalf("unexpected trace summary: %+v", sum)
	}

	// 2. Emit CUDA and check sm_100a PTX instruction forms.
	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA() failed: %v", err)
	}
	for _, want := range []string{
		"tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p;",
		"tcgen05.st.sync.aligned.32x32b.x1.b32",
		"tcgen05.ld.sync.aligned.32x32b.x1.b32",
		"elect.sync",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted CUDA missing %q", want)
		}
	}

	// 3. Execute on host CPU via cudahostemu.Run and compare against float64 reference.
	rng := rand.New(rand.NewPCG(99, 123))
	q := make([]float32, k.M*k.D)
	keys := make([]float32, k.KV*k.D)
	vals := make([]float32, k.KV*k.D)
	for i := range q {
		q[i] = float32(rng.NormFloat64() * 0.25)
	}
	for i := range keys {
		keys[i] = float32(rng.NormFloat64() * 0.25)
	}
	for i := range vals {
		vals[i] = float32(rng.NormFloat64() * 0.25)
	}

	grid, block := k.Grid()
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, q, k, v, out)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"q", "float", f32Bytes(q)},
		emuBuf{"k", "float", f32Bytes(keys)},
		emuBuf{"v", "float", f32Bytes(vals)},
		emuBuf{"out", "float", poison(4 * k.M * k.D)},
	)
	got := bytesF32(res["out"])

	scale := 1.0 / math.Sqrt(float64(k.D))
	for r := range k.M {
		logits := make([]float64, k.KV)
		maxLogit := math.Inf(-1)
		for j := range k.KV {
			var dot float64
			for d := range k.D {
				dot += float64(q[r*k.D+d]) * float64(keys[j*k.D+d])
			}
			logits[j] = dot * scale
			if logits[j] > maxLogit {
				maxLogit = logits[j]
			}
		}
		var sumExp float64
		probs := make([]float64, k.KV)
		for j := range k.KV {
			probs[j] = math.Exp(logits[j] - maxLogit)
			sumExp += probs[j]
		}
		for j := range k.KV {
			probs[j] /= sumExp
		}
		for d := range k.D {
			var want float64
			for j := range k.KV {
				want += probs[j] * float64(vals[j*k.D+d])
			}
			diff := math.Abs(float64(got[r*k.D+d]) - want)
			if math.IsNaN(float64(got[r*k.D+d])) || diff > 1e-4 {
				t.Fatalf("out[%d, %d] = %g, want %g (diff %g)", r, d, got[r*k.D+d], want, diff)
			}
		}
	}
}
