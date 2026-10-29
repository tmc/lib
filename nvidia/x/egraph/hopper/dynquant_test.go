package hopper

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"
)

func cosineSimF32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		ai := float64(a[i])
		bi := float64(b[i])
		dot += ai * bi
		na += ai * ai
		nb += bi * bi
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func exactRMSNormF32(x, residual, weight []float32, M, K int, eps float32, hasResidual bool) ([]float32, []float32) {
	hOut := make([]float32, M*K)
	normed := make([]float32, M*K)
	for row := 0; row < M; row++ {
		base := row * K
		var sumSq float64
		for col := 0; col < K; col++ {
			v := x[base+col]
			if hasResidual {
				v += residual[base+col]
			}
			hOut[base+col] = v
			sumSq += float64(v) * float64(v)
		}
		invRMS := float32(1.0 / math.Sqrt(sumSq/float64(K)+float64(eps)))
		for col := 0; col < K; col++ {
			normed[base+col] = hOut[base+col] * invRMS * weight[col]
		}
	}
	return hOut, normed
}

func exactSwiGLUF32(gate, up []float32, M, K int) []float32 {
	out := make([]float32, M*K)
	for i := 0; i < M*K; i++ {
		g := float64(gate[i])
		u := float64(up[i])
		silu := g / (1.0 + math.Exp(-g))
		out[i] = float32(silu * u)
	}
	return out
}

func toBF16Buf(vals []float32) []byte {
	u16 := make([]uint16, len(vals))
	for i, v := range vals {
		u16[i] = BF16Bits(v)
	}
	return u16Bytes(u16)
}

func fromBF16Slice(vals []float32) []float32 {
	out := make([]float32, len(vals))
	for i, v := range vals {
		out[i] = FromBF16Bits(BF16Bits(v))
	}
	return out
}

func TestSwizzleTCGen05ScalesRoundTrip(t *testing.T) {
	const (
		M          = 130
		numKBlocks = 8
	)
	rowMajor := make([]uint8, M*numKBlocks)
	for i := range rowMajor {
		rowMajor[i] = uint8((i*37 + 11) & 0xFF)
	}
	swizzled, err := SwizzleTCGen05Scales(rowMajor, M, numKBlocks)
	if err != nil {
		t.Fatalf("SwizzleTCGen05Scales: %v", err)
	}
	unswizzled, err := UnswizzleTCGen05Scales(swizzled, M, numKBlocks)
	if err != nil {
		t.Fatalf("UnswizzleTCGen05Scales: %v", err)
	}
	if !bytes.Equal(rowMajor, unswizzled) {
		t.Fatalf("UnswizzleTCGen05Scales did not invert SwizzleTCGen05Scales")
	}
}

func TestRMSNormDynamicNVFP4Quant(t *testing.T) {
	const (
		M   = 32
		K   = 64
		eps = 1e-5
	)
	xF32 := make([]float32, M*K)
	resF32 := make([]float32, M*K)
	wF32 := make([]float32, K)
	for i := range xF32 {
		xF32[i] = float32(math.Sin(float64(i)*0.17+0.3)) * 2.5
		resF32[i] = float32(math.Cos(float64(i)*0.11-0.2)) * 1.4
	}
	for i := range wF32 {
		wF32[i] = 0.5 + 0.75*float32(math.Cos(float64(i)*0.09))
	}

	cases := []struct {
		name     string
		residual bool
		inDType  DType
		layout   ScaleLayout
	}{
		{"nores_f32_rowmajor", false, F32, ScaleLayoutRowMajor},
		{"nores_bf16_rowmajor", false, BF16, ScaleLayoutRowMajor},
		{"res_f32_rowmajor", true, F32, ScaleLayoutRowMajor},
		{"res_bf16_rowmajor", true, BF16, ScaleLayoutRowMajor},
		{"nores_f32_swizzled", false, F32, ScaleLayoutTCGen05Swizzled},
		{"res_bf16_swizzled", true, BF16, ScaleLayoutTCGen05Swizzled},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			k := RMSNormDynamicNVFP4Quant{
				Name:        "rmsnorm_dynquant_" + tc.name,
				Cols:        K,
				Eps:         eps,
				In:          tc.inDType,
				Residual:    tc.residual,
				ScaleLayout: tc.layout,
			}
			cudaSrc, err := k.CUDA()
			if err != nil {
				t.Fatalf("CUDA: %v", err)
			}
			launchSrc, err := k.Launcher()
			if err != nil {
				t.Fatalf("Launcher: %v", err)
			}
			src := cudaSrc + "\n" + launchSrc

			var xRef, resRef, wRef []float32
			if tc.inDType == BF16 {
				xRef = fromBF16Slice(xF32)
				if tc.residual {
					resRef = fromBF16Slice(resF32)
				}
				wRef = fromBF16Slice(wF32)
			} else {
				xRef = xF32
				if tc.residual {
					resRef = resF32
				}
				wRef = wF32
			}

			wantH, wantFP4, wantScales, dequant, err := k.Reference(xF32, resF32, wF32, M)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}

			inCType := tc.inDType.cType()
			var xBytes, resBytes, wBytes, hOutBytes []byte
			if tc.inDType == BF16 {
				xBytes = toBF16Buf(xF32)
				resBytes = toBF16Buf(resF32)
				wBytes = toBF16Buf(wF32)
				hOutBytes = poison(M * K * 2)
			} else {
				xBytes = f32Bytes(xF32)
				resBytes = f32Bytes(resF32)
				wBytes = f32Bytes(wF32)
				hOutBytes = poison(M * K * 4)
			}

			var launch string
			var bufs []emuBuf
			if tc.residual {
				launch = fmt.Sprintf("%s_launch(%d, x, residual, weight, h_out, out_fp4, out_scales, nullptr)", k.Name, M)
				bufs = []emuBuf{
					{"x", inCType, xBytes},
					{"residual", inCType, resBytes},
					{"weight", inCType, wBytes},
					{"h_out", inCType, hOutBytes},
					{"out_fp4", "uint8_t", poison(len(wantFP4))},
					{"out_scales", "uint8_t", poison(len(wantScales))},
				}
			} else {
				launch = fmt.Sprintf("%s_launch(%d, x, weight, out_fp4, out_scales, nullptr)", k.Name, M)
				bufs = []emuBuf{
					{"x", inCType, xBytes},
					{"weight", inCType, wBytes},
					{"out_fp4", "uint8_t", poison(len(wantFP4))},
					{"out_scales", "uint8_t", poison(len(wantScales))},
				}
			}

			res := runEmu(t, src, launch, bufs...)
			gotFP4 := res["out_fp4"]
			gotScales := res["out_scales"]

			if !bytes.Equal(gotFP4, wantFP4) {
				diff := 0
				for i := range gotFP4 {
					if gotFP4[i] != wantFP4[i] {
						diff++
					}
				}
				t.Fatalf("packedFP4 mismatch: %d/%d bytes differ", diff, len(wantFP4))
			}

			if !bytes.Equal(gotScales, wantScales) {
				diff := 0
				for i := range gotScales {
					if gotScales[i] != wantScales[i] {
						diff++
					}
				}
				t.Fatalf("scales mismatch: %d/%d bytes differ", diff, len(wantScales))
			}

			if tc.residual {
				var gotH []float32
				if tc.inDType == BF16 {
					u16 := bytesU16(res["h_out"])
					gotH = make([]float32, len(u16))
					for i, b := range u16 {
						gotH[i] = FromBF16Bits(b)
					}
				} else {
					gotH = bytesF32(res["h_out"])
				}
				for i := range wantH {
					if math.Float32bits(gotH[i]) != math.Float32bits(wantH[i]) {
						t.Fatalf("hOut[%d] = %v (0x%08x), want %v (0x%08x)",
							i, gotH[i], math.Float32bits(gotH[i]), wantH[i], math.Float32bits(wantH[i]))
					}
				}
			}

			if tc.layout == ScaleLayoutTCGen05Swizzled {
				kRow := k
				kRow.ScaleLayout = ScaleLayoutRowMajor
				_, _, wantRowScales, _, err := kRow.Reference(xF32, resF32, wF32, M)
				if err != nil {
					t.Fatalf("rowmajor Reference: %v", err)
				}
				unswizzled, err := UnswizzleTCGen05Scales(gotScales, M, K/16)
				if err != nil {
					t.Fatalf("UnswizzleTCGen05Scales: %v", err)
				}
				if !bytes.Equal(unswizzled, wantRowScales) {
					t.Fatalf("UnswizzleTCGen05Scales mismatch against row-major scales")
				}
			}

			_, exactNorm := exactRMSNormF32(xRef, resRef, wRef, M, K, eps, tc.residual)
			cos := cosineSimF32(dequant, exactNorm)
			if cos <= 0.99 {
				t.Fatalf("cosine similarity(dequant, exactNorm) = %.6f, want > 0.99", cos)
			}
		})
	}
}

func TestSwiGLUDynamicNVFP4Quant(t *testing.T) {
	const (
		M = 32
		K = 64
	)
	gateF32 := make([]float32, M*K)
	upF32 := make([]float32, M*K)
	for i := range gateF32 {
		gateF32[i] = float32(math.Sin(float64(i)*0.23+0.1)) * 2.2
		upF32[i] = float32(math.Cos(float64(i)*0.19-0.4)) * 1.8
	}

	cases := []struct {
		name    string
		inDType DType
		layout  ScaleLayout
	}{
		{"f32_rowmajor", F32, ScaleLayoutRowMajor},
		{"bf16_rowmajor", BF16, ScaleLayoutRowMajor},
		{"f32_swizzled", F32, ScaleLayoutTCGen05Swizzled},
		{"bf16_swizzled", BF16, ScaleLayoutTCGen05Swizzled},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			k := SwiGLUDynamicNVFP4Quant{
				Name:        "swiglu_dynquant_" + tc.name,
				Cols:        K,
				In:          tc.inDType,
				ScaleLayout: tc.layout,
			}
			cudaSrc, err := k.CUDA()
			if err != nil {
				t.Fatalf("CUDA: %v", err)
			}
			launchSrc, err := k.Launcher()
			if err != nil {
				t.Fatalf("Launcher: %v", err)
			}
			src := cudaSrc + "\n" + launchSrc

			wantFP4, wantScales, dequant, err := k.Reference(gateF32, upF32, M)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}

			inCType := tc.inDType.cType()
			var gateBytes, upBytes []byte
			var gateRef, upRef []float32
			if tc.inDType == BF16 {
				gateBytes = toBF16Buf(gateF32)
				upBytes = toBF16Buf(upF32)
				gateRef = fromBF16Slice(gateF32)
				upRef = fromBF16Slice(upF32)
			} else {
				gateBytes = f32Bytes(gateF32)
				upBytes = f32Bytes(upF32)
				gateRef = gateF32
				upRef = upF32
			}

			launch := fmt.Sprintf("%s_launch(%d, gate, up, out_fp4, out_scales, nullptr)", k.Name, M)
			res := runEmu(t, src, launch,
				emuBuf{"gate", inCType, gateBytes},
				emuBuf{"up", inCType, upBytes},
				emuBuf{"out_fp4", "uint8_t", poison(len(wantFP4))},
				emuBuf{"out_scales", "uint8_t", poison(len(wantScales))},
			)
			gotFP4 := res["out_fp4"]
			gotScales := res["out_scales"]

			if !bytes.Equal(gotFP4, wantFP4) {
				diff := 0
				for i := range gotFP4 {
					if gotFP4[i] != wantFP4[i] {
						diff++
					}
				}
				t.Fatalf("packedFP4 mismatch: %d/%d bytes differ", diff, len(wantFP4))
			}

			if !bytes.Equal(gotScales, wantScales) {
				diff := 0
				for i := range gotScales {
					if gotScales[i] != wantScales[i] {
						diff++
					}
				}
				t.Fatalf("scales mismatch: %d/%d bytes differ", diff, len(wantScales))
			}

			if tc.layout == ScaleLayoutTCGen05Swizzled {
				kRow := k
				kRow.ScaleLayout = ScaleLayoutRowMajor
				_, wantRowScales, _, err := kRow.Reference(gateF32, upF32, M)
				if err != nil {
					t.Fatalf("rowmajor Reference: %v", err)
				}
				unswizzled, err := UnswizzleTCGen05Scales(gotScales, M, K/16)
				if err != nil {
					t.Fatalf("UnswizzleTCGen05Scales: %v", err)
				}
				if !bytes.Equal(unswizzled, wantRowScales) {
					t.Fatalf("UnswizzleTCGen05Scales mismatch against row-major scales")
				}
			}

			exact := exactSwiGLUF32(gateRef, upRef, M, K)
			cos := cosineSimF32(dequant, exact)
			if cos <= 0.99 {
				t.Fatalf("cosine similarity(dequant, exactSwiGLU) = %.6f, want > 0.99", cos)
			}
		})
	}
}

func TestDynamicQuantEndToEndGEMM(t *testing.T) {
	const (
		M   = 128
		N   = 128
		K   = 64
		eps = 1e-5
	)
	xF32 := make([]float32, M*K)
	wNorm := make([]float32, K)
	wMat := make([]float32, N*K)
	for i := range xF32 {
		xF32[i] = float32(math.Sin(float64(i)*0.13+0.2)) * 2.0
	}
	for i := range wNorm {
		wNorm[i] = 0.8 + 0.4*float32(math.Cos(float64(i)*0.07))
	}
	for i := range wMat {
		wMat[i] = float32(math.Cos(float64(i)*0.17-0.3)) * 1.5
	}

	// Step 1: Run RMSNormDynamicNVFP4Quant kernel via cudahostemu.
	qKernel := RMSNormDynamicNVFP4Quant{
		Name:        "rmsnorm_dynquant_e2e",
		Cols:        K,
		Eps:         eps,
		In:          F32,
		Residual:    false,
		ScaleLayout: ScaleLayoutRowMajor,
	}
	qCUDA, err := qKernel.CUDA()
	if err != nil {
		t.Fatalf("qKernel.CUDA: %v", err)
	}
	qLaunch, err := qKernel.Launcher()
	if err != nil {
		t.Fatalf("qKernel.Launcher: %v", err)
	}
	qRes := runEmu(t, qCUDA+"\n"+qLaunch,
		fmt.Sprintf("%s_launch(%d, x, weight, out_fp4, out_scales, nullptr)", qKernel.Name, M),
		emuBuf{"x", "float", f32Bytes(xF32)},
		emuBuf{"weight", "float", f32Bytes(wNorm)},
		emuBuf{"out_fp4", "uint8_t", poison(M * (K / 2))},
		emuBuf{"out_scales", "uint8_t", poison(M * (K / 16))},
	)
	aFP4 := qRes["out_fp4"]
	aScale := qRes["out_scales"]

	// Step 2: Quantize the B weight matrix [N, K] using NVFP4 block quantization.
	bFP4 := make([]uint8, N*(K/2))
	bScale := make([]uint8, N*(K/16))
	numKBlocks := K / 16
	for n := 0; n < N; n++ {
		for kb := 0; kb < numKBlocks; kb++ {
			base := n*K + kb*16
			var amax float32
			for i := 0; i < 16; i++ {
				av := float32(math.Abs(float64(wMat[base+i])))
				if av > amax {
					amax = av
				}
			}
			sb := EncodeUE4M3(amax / 6.0)
			bScale[n*numKBlocks+kb] = sb
			s := FromE4M3Bits(sb)
			var invS float32
			if s > 0 {
				invS = 1.0 / s
			}
			for pair := 0; pair < 8; pair++ {
				lo := EncodeE2M1(wMat[base+2*pair] * invS)
				hi := EncodeE2M1(wMat[base+2*pair+1] * invS)
				bFP4[n*(K/2)+kb*8+pair] = lo | (hi << 4)
			}
		}
	}

	// Step 3: Run TCGen05BlockScaledGEMM in cudahostemu consuming aFP4 and aScale directly.
	gemmBlockScaled := TCGen05BlockScaledGEMM{
		Name: "block_scaled_gemm_e2e",
		M:    M,
		N:    N,
		K:    K,
	}
	gemmSrc, err := gemmBlockScaled.CUDA()
	if err != nil {
		t.Fatalf("TCGen05BlockScaledGEMM.CUDA: %v", err)
	}
	grid, block := gemmBlockScaled.Grid()
	launchGEMM := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a_packed, a_scales, b_packed, b_scales, out)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], gemmBlockScaled.Name)
	gemmRes := runEmu(t, gemmSrc, launchGEMM,
		emuBuf{"a_packed", "uint8_t", aFP4},
		emuBuf{"a_scales", "uint8_t", aScale},
		emuBuf{"b_packed", "uint8_t", bFP4},
		emuBuf{"b_scales", "uint8_t", bScale},
		emuBuf{"out", "float", poison(M * N * 4)},
	)
	gotD := bytesF32(gemmRes["out"])

	// Step 4: Verify against TCGen05GEMM{Weight: WeightNVFP4}.ReferenceNVFP4.
	gemmRef := TCGen05GEMM{
		Name:   "tcgen05_nvfp4_ref",
		N:      N,
		K:      K,
		Weight: WeightNVFP4,
	}
	wantNVFP4D, err := gemmRef.ReferenceNVFP4(M, aFP4, aScale, bFP4, bScale)
	if err != nil {
		t.Fatalf("ReferenceNVFP4: %v", err)
	}
	for i := range wantNVFP4D {
		if math.Abs(float64(gotD[i]-wantNVFP4D[i])) > 1e-3 {
			t.Fatalf("gotD[%d] = %v, wantNVFP4D = %v", i, gotD[i], wantNVFP4D[i])
		}
	}

	// Step 5: Compare against exact FP32 RMSNorm + FP32 GEMM reference.
	_, exactNorm := exactRMSNormF32(xF32, nil, wNorm, M, K, eps, false)
	exactGEMM := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var acc float64
			for k := 0; k < K; k++ {
				acc += float64(exactNorm[m*K+k]) * float64(wMat[n*K+k])
			}
			exactGEMM[m*N+n] = float32(acc)
		}
	}
	cos := cosineSimF32(gotD, exactGEMM)
	if cos <= 0.99 {
		t.Fatalf("end-to-end cosine similarity(gotD, exactGEMM) = %.6f, want > 0.99", cos)
	}
}

func ExampleRMSNormDynamicNVFP4Quant() {
	k := RMSNormDynamicNVFP4Quant{
		Name:        "rmsnorm_nvfp4_q",
		Cols:        4096,
		Eps:         1e-5,
		In:          BF16,
		Residual:    true,
		ScaleLayout: ScaleLayoutRowMajor,
	}
	src, err := k.CUDA()
	fmt.Println(err == nil, strings.Contains(src, "eg_f32_to_e2m1_code"), k.Block())
	// Output: true true [256 1 1]
}
