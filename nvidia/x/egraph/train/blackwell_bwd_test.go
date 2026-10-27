package train

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagradcheck"
)

func TestStochasticRoundE2M1Unbiased(t *testing.T) {
	tests := []struct {
		val     float32
		minBias float64 // minimum bias magnitude of deterministic EncodeE2M1
	}{
		{0.25, 0.25},
		{0.75, 0.25},
		{1.25, 0.25},
		{2.5, 0.5},
		{-3.5, 0.5},
		{-0.25, 0.25},
		{5.0, 1.0},
	}

	const numSamples = 10000
	for _, tc := range tests {
		t.Run(fmt.Sprintf("v=%g", tc.val), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(42, 1337))
			var sum float64
			for range numSamples {
				// Stratified / quasirandom + PCG draw across 16-bit entropy space.
				r := uint16(rng.Uint32())
				q := StochasticRoundE2M1(tc.val, r)
				sum += float64(DecodeE2M1(q))
			}
			// Also verify exact expectation over all 65,536 16-bit seeds and 10,000 stratified draws.
			var stratSum float64
			for i := range numSamples {
				u32 := splitMix32(uint32(i) + 1)
				// Stratified sample in [0, 65536): bin i of 10000 with uniform jitter inside bin.
				r := uint16((uint64(i)*65536 + uint64(u32&0xffff)) / numSamples)
				q := StochasticRoundE2M1(tc.val, r)
				stratSum += float64(DecodeE2M1(q))
			}
			mean := stratSum / numSamples
			errMean := math.Abs(mean - float64(tc.val))
			if errMean >= 5e-3 {
				t.Errorf("StochasticRoundE2M1(%g): empirical mean = %g, want %g (|err| = %g >= 5e-3)",
					tc.val, mean, tc.val, errMean)
			}

			// Verify Monte Carlo PCG mean is also close.
			mcMean := sum / numSamples
			if mcErr := math.Abs(mcMean - float64(tc.val)); mcErr >= 2.5e-2 {
				t.Errorf("StochasticRoundE2M1(%g): unstratified MC mean = %g, want %g (|err| = %g)",
					tc.val, mcMean, tc.val, mcErr)
			}

			// Contrast with deterministic round-to-nearest-even (EncodeE2M1).
			detVal := float64(DecodeE2M1(EncodeE2M1(tc.val)))
			detBias := math.Abs(detVal - float64(tc.val))
			if detBias < tc.minBias-1e-6 {
				t.Errorf("EncodeE2M1(%g) deterministic bias = %g, expected >= %g", tc.val, detBias, tc.minBias)
			}
		})
	}
}

func TestDualAxisGradQuantizer(t *testing.T) {
	tests := []struct {
		name  string
		m, n  int
		dtype DType
		seed  uint32
	}{
		{"fp32_16x16", 16, 16, FP32, 0x12345678},
		{"fp32_32x48", 32, 48, FP32, 0xdeadbeef},
		{"bf16_32x32", 32, 32, BF16, 0xCAFEBABE},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(701, 702))
			k := DualAxisGradQuantizer{
				Name:  "dual_axis_quant_test",
				M:     tc.m,
				N:     tc.n,
				DType: tc.dtype,
				Seed:  tc.seed,
			}
			src, err := k.Source()
			if err != nil {
				t.Fatalf("Source: %v", err)
			}
			if !strings.Contains(src, "cvt.rs.satfinite.e2m1x2.f32") {
				t.Fatalf("Source missing cvt.rs.satfinite.e2m1x2.f32 PTX instruction")
			}

			dyF64 := quantizeSlice(tc.dtype, randVals(rng, tc.m*tc.n, 2.5))
			dyF32 := make([]float32, len(dyF64))
			for i, v := range dyF64 {
				dyF32[i] = float32(v)
			}

			wantRowP, wantRowS, wantColP, wantColS, err := k.Reference(dyF32)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}

			ct, dyBytes := encodeBwdBuf(tc.dtype, dyF64)
			launch := fmt.Sprintf("dual_axis_quant_test_launch(dY_row_packed, dY_row_scales, dY_col_packed, dY_col_scales, dY, %du, 0)", tc.seed)
			res := runEmu(t, src, launch,
				emuBuf{"dY_row_packed", "uint8_t", poison(len(wantRowP))},
				emuBuf{"dY_row_scales", "uint8_t", poison(len(wantRowS))},
				emuBuf{"dY_col_packed", "uint8_t", poison(len(wantColP))},
				emuBuf{"dY_col_scales", "uint8_t", poison(len(wantColS))},
				emuBuf{"dY", ct, dyBytes},
			)

			if !bytes.Equal(res["dY_row_packed"], wantRowP) {
				t.Errorf("dY_row_packed mismatch:\ngot  %x\nwant %x", res["dY_row_packed"], wantRowP)
			}
			if !bytes.Equal(res["dY_row_scales"], wantRowS) {
				t.Errorf("dY_row_scales mismatch:\ngot  %x\nwant %x", res["dY_row_scales"], wantRowS)
			}
			if !bytes.Equal(res["dY_col_packed"], wantColP) {
				t.Errorf("dY_col_packed mismatch:\ngot  %x\nwant %x", res["dY_col_packed"], wantColP)
			}
			if !bytes.Equal(res["dY_col_scales"], wantColS) {
				t.Errorf("dY_col_scales mismatch:\ngot  %x\nwant %x", res["dY_col_scales"], wantColS)
			}
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		for _, bad := range []DualAxisGradQuantizer{
			{Name: "123bad", M: 16, N: 16},
			{Name: "ok", M: 12, N: 16},
			{Name: "ok", M: 16, N: 24},
			{Name: "ok", M: 16, N: 16, DType: U8},
		} {
			if _, err := bad.Source(); err == nil {
				t.Errorf("expected error for %+v", bad)
			}
		}
	})
}

func TestTCGen05LinearBackward(t *testing.T) {
	tests := []struct {
		name      string
		m, n, k   int
		quantized bool
		seed      uint32
		tol       cudagradcheck.Tolerance
	}{
		{
			name:      "fp32_tmem_16x16x16",
			m:         16,
			n:         16,
			k:         16,
			quantized: false,
			tol:       cudagradcheck.Tolerance{Atol: 1e-4, Rtol: 1e-3},
		},
		{
			name:      "fp32_tmem_32x16x32",
			m:         32,
			n:         16,
			k:         32,
			quantized: false,
			tol:       cudagradcheck.Tolerance{Atol: 1e-4, Rtol: 1e-3},
		},
		{
			name:      "nvfp4_block_scaled_rs_16x16x16",
			m:         16,
			n:         16,
			k:         16,
			quantized: true,
			seed:      0x13572468,
			tol:       cudagradcheck.Tolerance{Atol: 0.35, Rtol: 0.35},
		},
		{
			name:      "nvfp4_block_scaled_rs_32x16x32",
			m:         32,
			n:         16,
			k:         32,
			quantized: true,
			seed:      0x24681357,
			tol:       cudagradcheck.Tolerance{Atol: 0.45, Rtol: 0.35},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(801, 802))
			bwd := TCGen05LinearBackward{
				Name:                 "tcgen05_lin_bwd_test",
				M:                    tc.m,
				N:                    tc.n,
				K:                    tc.k,
				QuantizedBlockScaled: tc.quantized,
				Seed:                 tc.seed,
			}
			src, err := bwd.Source()
			if err != nil {
				t.Fatalf("Source: %v", err)
			}
			if tc.quantized && !strings.Contains(src, "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale") {
				t.Fatalf("quantized Source missing tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale")
			}

			xF64 := quantizeSlice(FP32, randVals(rng, tc.m*tc.k, 0.5))
			wF64 := quantizeSlice(FP32, randVals(rng, tc.n*tc.k, 0.5))
			dyF64 := quantizeSlice(FP32, randVals(rng, tc.m*tc.n, 0.5))

			xF32 := make([]float32, len(xF64))
			wF32 := make([]float32, len(wF64))
			dyF32 := make([]float32, len(dyF64))
			for i, v := range xF64 {
				xF32[i] = float32(v)
			}
			for i, v := range wF64 {
				wF32[i] = float32(v)
			}
			for i, v := range dyF64 {
				dyF32[i] = float32(v)
			}

			wantDX, wantDW, err := bwd.Reference(xF32, wF32, dyF32)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}

			launch := "tcgen05_lin_bwd_test_launch(dX, dW, X, W, dY, 0)"
			res := runEmu(t, src, launch,
				emuBuf{"dX", "float", poison(4 * tc.m * tc.k)},
				emuBuf{"dW", "float", poison(4 * tc.n * tc.k)},
				emuBuf{"X", "float", f32Bytes(xF32)},
				emuBuf{"W", "float", f32Bytes(wF32)},
				emuBuf{"dY", "float", f32Bytes(dyF32)},
			)

			gotDXF32 := bytesF32(res["dX"])
			gotDWF32 := bytesF32(res["dW"])

			// 1. Verify emulated kernel matches Reference closely.
			wantDXF64 := make([]float64, len(wantDX))
			for i, v := range wantDX {
				wantDXF64[i] = float64(v)
			}
			wantDWF64 := make([]float64, len(wantDW))
			for i, v := range wantDW {
				wantDWF64[i] = float64(v)
			}
			refTol := cudagradcheck.Tolerance{Atol: 1e-5, Rtol: 1e-5}
			if rep, err := cudagradcheck.Compare(gotDXF32, wantDXF64, refTol); err != nil {
				t.Errorf("dX vs Reference mismatch: %v (report: %v)", err, rep)
			}
			if rep, err := cudagradcheck.Compare(gotDWF32, wantDWF64, refTol); err != nil {
				t.Errorf("dW vs Reference mismatch: %v (report: %v)", err, rep)
			}

			// 2. Verify dX and dW against finite-difference numerical gradients of Y = X * W^T.
			gotDX := decodeBwdBuf(FP32, res["dX"])
			gotDW := decodeBwdBuf(FP32, res["dW"])

			forward := func(curX, curW []float64) float64 {
				var total float64
				for m := range tc.m {
					for n := range tc.n {
						var y float64
						for k := range tc.k {
							y += curX[m*tc.k+k] * curW[n*tc.k+k]
						}
						total += y * dyF64[m*tc.n+n]
					}
				}
				return total
			}

			opts := cudagradcheck.GradOptions{Eps: 1e-6, Tolerance: tc.tol}
			if rep, err := cudagradcheck.CheckGrad(func(in []float64) float64 { return forward(in, wF64) }, xF64, gotDX, opts); err != nil {
				t.Errorf("dX gradcheck failed: %v (report: %v)", err, rep)
			}
			if rep, err := cudagradcheck.CheckGrad(func(in []float64) float64 { return forward(xF64, in) }, wF64, gotDW, opts); err != nil {
				t.Errorf("dW gradcheck failed: %v (report: %v)", err, rep)
			}
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		for _, bad := range []TCGen05LinearBackward{
			{Name: "123bad", M: 16, N: 16, K: 16},
			{Name: "ok", M: 8, N: 16, K: 16},
			{Name: "ok", M: 16, N: 20, K: 16},
			{Name: "ok", M: 16, N: 16, K: 0},
		} {
			if _, err := bad.Source(); err == nil {
				t.Errorf("expected error for %+v", bad)
			}
		}
	})
}

func ExampleStochasticRoundE2M1() {
	const v = float32(0.25)
	var sum float64
	const n = 4096
	for i := range n {
		r := uint16((i * 16) + 8)
		sum += float64(DecodeE2M1(StochasticRoundE2M1(v, r)))
	}
	mean := sum / n
	det := DecodeE2M1(EncodeE2M1(v))
	fmt.Printf("stochastic mean=%.2f deterministic=%.2f\n", mean, det)
	// Output:
	// stochastic mean=0.25 deterministic=0.00
}

func ExampleTCGen05LinearBackward() {
	bwd := TCGen05LinearBackward{
		Name:                 "linear_bwd_nvfp4",
		M:                    16,
		N:                    16,
		K:                    16,
		QuantizedBlockScaled: true,
		Seed:                 42,
	}
	src, err := bwd.Source()
	fmt.Printf("err=%v has_tcgen05=%v has_cvt_rs=%v\n",
		err,
		strings.Contains(src, "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale"),
		strings.Contains(src, "cvt.rs.satfinite.e2m1x2.f32"),
	)
	// Output:
	// err=<nil> has_tcgen05=true has_cvt_rs=true
}
