package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagradcheck"
)

func fromF16Bits(h uint16) float32 {
	s := uint32(h&0x8000) << 16
	e := uint32(h>>10) & 31
	m := uint32(h & 0x3ff)
	if e == 0 {
		v := float32(m) * (1.0 / 16777216.0)
		if s != 0 {
			return -v
		}
		return v
	}
	if e == 31 {
		return math.Float32frombits(s | 0x7f800000 | (m << 13))
	}
	return math.Float32frombits(s | ((e + 112) << 23) | (m << 13))
}

func roundDType(d DType, v float64) float64 {
	f := float32(v)
	switch d {
	case BF16:
		return float64(FromBF16Bits(BF16Bits(f)))
	case FP16:
		return float64(fromF16Bits(f16Bits(f)))
	default:
		return float64(f)
	}
}

func encodeBwdBuf(d DType, vals []float64) (ctype string, data []byte) {
	switch d {
	case BF16:
		u := make([]uint16, len(vals))
		for i, v := range vals {
			u[i] = BF16Bits(float32(v))
		}
		return "__nv_bfloat16", u16Bytes(u)
	case FP16:
		u := make([]uint16, len(vals))
		for i, v := range vals {
			u[i] = f16Bits(float32(v))
		}
		return "__half", u16Bytes(u)
	default:
		f := make([]float32, len(vals))
		for i, v := range vals {
			f[i] = float32(v)
		}
		return "float", f32Bytes(f)
	}
}

func decodeBwdBuf(d DType, raw []byte) []float64 {
	switch d {
	case BF16:
		u := bytesU16(raw)
		out := make([]float64, len(u))
		for i, b := range u {
			out[i] = float64(FromBF16Bits(b))
		}
		return out
	case FP16:
		u := bytesU16(raw)
		out := make([]float64, len(u))
		for i, b := range u {
			out[i] = float64(fromF16Bits(b))
		}
		return out
	default:
		f := bytesF32(raw)
		out := make([]float64, len(f))
		for i, v := range f {
			out[i] = float64(v)
		}
		return out
	}
}

func gradTol(d DType) cudagradcheck.Tolerance {
	switch d {
	case BF16:
		return cudagradcheck.Tolerance{Atol: 2e-2, Rtol: 2e-2}
	case FP16:
		return cudagradcheck.Tolerance{Atol: 3e-3, Rtol: 3e-3}
	default:
		return cudagradcheck.Tolerance{Atol: 2e-5, Rtol: 1e-3}
	}
}

func quantizeSlice(d DType, v []float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = roundDType(d, x)
	}
	return out
}

func TestRMSNormBackward(t *testing.T) {
	tests := []struct {
		name    string
		rows    int
		cols    int
		threads int
		dtype   DType
	}{
		{"fp32_r3_c32_t32", 3, 32, 32, FP32},
		{"fp32_r2_c96_t64", 2, 96, 64, FP32},
		{"bf16_r3_c64_t32", 3, 64, 32, BF16},
		{"fp16_r3_c64_t32", 3, 64, 32, FP16},
	}

	const eps = 1e-5
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(101, 202))
			k := RMSNormBackward{
				Name:    "rmsnorm_bwd_test",
				Cols:    tc.cols,
				Eps:     eps,
				DType:   tc.dtype,
				Threads: tc.threads,
			}
			src, err := k.Emit()
			if err != nil {
				t.Fatalf("Emit: %v", err)
			}

			x := quantizeSlice(tc.dtype, randVals(rng, tc.rows*tc.cols, 0.8))
			dy := quantizeSlice(tc.dtype, randVals(rng, tc.rows*tc.cols, 0.7))
			w := quantizeSlice(tc.dtype, randVals(rng, tc.cols, 0.9))

			ct, xBytes := encodeBwdBuf(tc.dtype, x)
			_, dyBytes := encodeBwdBuf(tc.dtype, dy)
			_, wBytes := encodeBwdBuf(tc.dtype, w)
			_, dxPoison := encodeBwdBuf(tc.dtype, make([]float64, tc.rows*tc.cols))
			for i := range dxPoison {
				dxPoison[i] = 0xff
			}
			dwZero := f32Bytes(make([]float32, tc.cols))

			launch := fmt.Sprintf("rmsnorm_bwd_test_launch(dX, dW_accum, dY, X, W, %d, 0)", tc.rows)
			res := runEmu(t, src, launch,
				emuBuf{"dX", ct, dxPoison},
				emuBuf{"dW_accum", "float", dwZero},
				emuBuf{"dY", ct, dyBytes},
				emuBuf{"X", ct, xBytes},
				emuBuf{"W", ct, wBytes},
			)

			gotDX := decodeBwdBuf(tc.dtype, res["dX"])
			gotDW := decodeBwdBuf(FP32, res["dW_accum"])

			forward := func(curX, curW []float64) float64 {
				var total float64
				for r := range tc.rows {
					row := curX[r*tc.cols : (r+1)*tc.cols]
					var ss float64
					for _, v := range row {
						ss += v * v
					}
					invRMS := 1.0 / math.Sqrt(ss/float64(tc.cols)+eps)
					for c := range tc.cols {
						y := row[c] * invRMS * curW[c]
						total += y * dy[r*tc.cols+c]
					}
				}
				return total
			}

			opts := cudagradcheck.GradOptions{Eps: 1e-6, Tolerance: gradTol(tc.dtype)}
			if rep, err := cudagradcheck.CheckGrad(func(in []float64) float64 { return forward(in, w) }, x, gotDX, opts); err != nil {
				t.Errorf("dX gradcheck failed: %v (report: %v)", err, rep)
			}
			if rep, err := cudagradcheck.CheckGrad(func(in []float64) float64 { return forward(x, in) }, w, gotDW, opts); err != nil {
				t.Errorf("dW gradcheck failed: %v (report: %v)", err, rep)
			}
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		for _, bad := range []RMSNormBackward{
			{Name: "123bad", Cols: 32},
			{Name: "ok", Cols: 0},
			{Name: "ok", Cols: 32, Eps: -1},
			{Name: "ok", Cols: 32, DType: E4M3},
			{Name: "ok", Cols: 32, Threads: 48},
		} {
			if _, err := bad.Emit(); err == nil {
				t.Errorf("expected error for %+v", bad)
			}
		}
	})
}

func TestSwiGLUBackward(t *testing.T) {
	tests := []struct {
		name    string
		rows    int
		cols    int
		threads int
		dtype   DType
	}{
		{"fp32_r3_c32_t32", 3, 32, 32, FP32},
		{"fp32_r2_c96_t64", 2, 96, 64, FP32},
		{"bf16_r3_c64_t32", 3, 64, 32, BF16},
		{"fp16_r3_c64_t32", 3, 64, 32, FP16},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(303, 404))
			k := SwiGLUBackward{
				Name:    "swiglu_bwd_test",
				Cols:    tc.cols,
				DType:   tc.dtype,
				Threads: tc.threads,
			}
			src, err := k.Emit()
			if err != nil {
				t.Fatalf("Emit: %v", err)
			}

			gateUp := quantizeSlice(tc.dtype, randVals(rng, tc.rows*2*tc.cols, 0.9))
			dy := quantizeSlice(tc.dtype, randVals(rng, tc.rows*tc.cols, 0.8))

			ct, guBytes := encodeBwdBuf(tc.dtype, gateUp)
			_, dyBytes := encodeBwdBuf(tc.dtype, dy)
			_, dguPoison := encodeBwdBuf(tc.dtype, make([]float64, len(gateUp)))
			for i := range dguPoison {
				dguPoison[i] = 0xff
			}

			launch := fmt.Sprintf("swiglu_bwd_test_launch(dGateUp, dY, GateUp, %d, 0)", tc.rows)
			res := runEmu(t, src, launch,
				emuBuf{"dGateUp", ct, dguPoison},
				emuBuf{"dY", ct, dyBytes},
				emuBuf{"GateUp", ct, guBytes},
			)

			gotDGateUp := decodeBwdBuf(tc.dtype, res["dGateUp"])

			forward := func(curGU []float64) float64 {
				var total float64
				for r := range tc.rows {
					base := r * 2 * tc.cols
					for c := range tc.cols {
						g := curGU[base+c]
						u := curGU[base+tc.cols+c]
						silu := g / (1.0 + math.Exp(-g))
						total += (silu * u) * dy[r*tc.cols+c]
					}
				}
				return total
			}

			opts := cudagradcheck.GradOptions{Eps: 1e-6, Tolerance: gradTol(tc.dtype)}
			if rep, err := cudagradcheck.CheckGrad(forward, gateUp, gotDGateUp, opts); err != nil {
				t.Errorf("dGateUp gradcheck failed: %v (report: %v)", err, rep)
			}
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		for _, bad := range []SwiGLUBackward{
			{Name: "bad-name", Cols: 32},
			{Name: "ok", Cols: 0},
			{Name: "ok", Cols: 32, DType: U8},
			{Name: "ok", Cols: 32, Threads: 16},
		} {
			if _, err := bad.Emit(); err == nil {
				t.Errorf("expected error for %+v", bad)
			}
		}
	})
}

func TestCrossEntropySoftcapBackward(t *testing.T) {
	tests := []struct {
		name    string
		rows    int
		vocab   int
		softcap float32
		threads int
		dtype   DType
		targets []int32
	}{
		{"fp32_softcap30", 4, 48, 30.0, 32, FP32, []int32{3, 19, 47, 0}},
		{"fp32_nosoftcap", 3, 64, 0.0, 32, FP32, []int32{11, 63, -1}},
		{"bf16_softcap15", 3, 64, 15.0, 32, BF16, []int32{7, 32, 60}},
		{"fp16_softcap20", 3, 64, 20.0, 32, FP16, []int32{5, 18, 42}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(505, 606))
			k := CrossEntropySoftcapBackward{
				Name:    "ce_softcap_bwd_test",
				Vocab:   tc.vocab,
				Softcap: tc.softcap,
				DType:   tc.dtype,
				Threads: tc.threads,
			}
			src, err := k.Emit()
			if err != nil {
				t.Fatalf("Emit: %v", err)
			}

			logits := quantizeSlice(tc.dtype, randVals(rng, tc.rows*tc.vocab, 2.5))
			dLossF64 := quantizeSlice(FP32, randVals(rng, tc.rows, 0.9))

			ct, logitsBytes := encodeBwdBuf(tc.dtype, logits)
			_, dLossBytes := encodeBwdBuf(FP32, dLossF64)
			_, dLogitsPoison := encodeBwdBuf(tc.dtype, make([]float64, len(logits)))
			for i := range dLogitsPoison {
				dLogitsPoison[i] = 0xff
			}

			launch := fmt.Sprintf("ce_softcap_bwd_test_launch(dLogits, losses, dLoss, logits, targets, %d, 0)", tc.rows)
			res := runEmu(t, src, launch,
				emuBuf{"dLogits", ct, dLogitsPoison},
				emuBuf{"losses", "float", poison(4 * tc.rows)},
				emuBuf{"dLoss", "float", dLossBytes},
				emuBuf{"logits", ct, logitsBytes},
				emuBuf{"targets", "int", i32Bytes(tc.targets)},
			)

			gotLosses := bytesF32(res["losses"])
			gotDLogits := decodeBwdBuf(tc.dtype, res["dLogits"])

			rowLosses := func(curLogits []float64) []float64 {
				out := make([]float64, tc.rows)
				capVal := float64(tc.softcap)
				for r := range tc.rows {
					tgt := int(tc.targets[r])
					if tgt < 0 || tgt >= tc.vocab {
						continue
					}
					row := curLogits[r*tc.vocab : (r+1)*tc.vocab]
					capped := make([]float64, tc.vocab)
					mx := math.Inf(-1)
					for i, z := range row {
						zc := z
						if capVal > 0 {
							zc = capVal * math.Tanh(z/capVal)
						}
						capped[i] = zc
						mx = max(mx, zc)
					}
					var sumExp float64
					for _, zc := range capped {
						sumExp += math.Exp(zc - mx)
					}
					out[r] = -(capped[tgt] - mx - math.Log(sumExp))
				}
				return out
			}

			wantLosses := rowLosses(logits)
			if rep, err := cudagradcheck.Compare(gotLosses, wantLosses, cudagradcheck.Tolerance{Atol: 1e-5, Rtol: 1e-4}); err != nil {
				t.Errorf("losses mismatch: %v (report: %v)", err, rep)
			}

			forward := func(curLogits []float64) float64 {
				return cudagradcheck.Dot(rowLosses(curLogits), dLossF64)
			}
			opts := cudagradcheck.GradOptions{Eps: 1e-6, Tolerance: gradTol(tc.dtype)}
			if rep, err := cudagradcheck.CheckGrad(forward, logits, gotDLogits, opts); err != nil {
				t.Errorf("dLogits gradcheck failed: %v (report: %v)", err, rep)
			}
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		for _, bad := range []CrossEntropySoftcapBackward{
			{Name: "bad name", Vocab: 32},
			{Name: "ok", Vocab: 0},
			{Name: "ok", Vocab: 32, DType: I32},
			{Name: "ok", Vocab: 32, Threads: 2048},
		} {
			if _, err := bad.Emit(); err == nil {
				t.Errorf("expected error for %+v", bad)
			}
		}
	})
}
