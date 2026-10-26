package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagradcheck"
)

func TestFusedAdamW(t *testing.T) {
	tests := []struct {
		name    string
		n       int
		threads int
		dtype   DType
	}{
		{"bf16_n96_t32", 96, 32, BF16},
		{"fp16_n96_t32", 96, 32, FP16},
		{"fp32_n128_t64", 128, 64, FP32},
	}

	const (
		steps       = 6
		lr          = 0.08
		beta1       = 0.9
		beta2       = 0.999
		eps         = 1e-8
		weightDecay = 0.01
		gradScale   = 0.5
	)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(707, 808))
			k := FusedAdamW{
				Name:    "fused_adamw_test",
				DType:   tc.dtype,
				Threads: tc.threads,
			}
			src, err := k.Emit()
			if err != nil {
				t.Fatalf("Emit: %v", err)
			}

			// Quadratic objective: L(w) = 0.5 * sum_i curvature[i] * (w[i] - target[i])^2
			curvature := make([]float64, tc.n)
			target := make([]float64, tc.n)
			masterW := make([]float64, tc.n)
			for i := range tc.n {
				curvature[i] = 0.5 + rng.Float64()*1.5
				target[i] = rng.NormFloat64() * 0.5
				masterW[i] = roundDType(tc.dtype, rng.NormFloat64()*1.2)
			}

			refW := append([]float64(nil), masterW...)
			refM := make([]float64, tc.n)
			refV := make([]float64, tc.n)

			emuW := make([]float32, tc.n)
			for i, w := range masterW {
				emuW[i] = float32(w)
			}
			emuM := make([]float32, tc.n)
			emuV := make([]float32, tc.n)

			quadLoss := func(w []float64) float64 {
				var loss float64
				for i := range tc.n {
					d := w[i] - target[i]
					loss += 0.5 * curvature[i] * d * d
				}
				return loss
			}

			prevLoss := quadLoss(refW)
			for step := 1; step <= steps; step++ {
				bc1 := 1.0 - math.Pow(beta1, float64(step))
				bc2 := 1.0 - math.Pow(beta2, float64(step))

				// Compute unscaled gradient in DType so both GPU kernel and
				// float64 reference see the exact same quantized gradient input.
				rawGrad := make([]float64, tc.n)
				for i := range tc.n {
					g := curvature[i] * (float64(emuW[i]) - target[i])
					rawGrad[i] = roundDType(tc.dtype, g/gradScale)
				}

				for i := range tc.n {
					g := rawGrad[i] * gradScale
					refW[i] *= 1.0 - lr*weightDecay
					refM[i] = beta1*refM[i] + (1.0-beta1)*g
					refV[i] = beta2*refV[i] + (1.0-beta2)*g*g
					refW[i] -= lr * (refM[i] / bc1) / (math.Sqrt(refV[i]/bc2) + eps)
				}

				ct, gradBytes := encodeBwdBuf(tc.dtype, rawGrad)
				_, paramPoison := encodeBwdBuf(tc.dtype, make([]float64, tc.n))
				for i := range paramPoison {
					paramPoison[i] = 0xff
				}

				launch := fmt.Sprintf(
					"fused_adamw_test_launch(param, master_w, exp_avg, exp_avg_sq, grad, %d, %s, %s, %s, %s, %s, %s, %s, %s, 0)",
					tc.n,
					floatLiteral(float32(lr)),
					floatLiteral(float32(beta1)),
					floatLiteral(float32(beta2)),
					floatLiteral(float32(eps)),
					floatLiteral(float32(weightDecay)),
					floatLiteral(float32(bc1)),
					floatLiteral(float32(bc2)),
					floatLiteral(float32(gradScale)),
				)
				res := runEmu(t, src, launch,
					emuBuf{"param", ct, paramPoison},
					emuBuf{"master_w", "float", f32Bytes(emuW)},
					emuBuf{"exp_avg", "float", f32Bytes(emuM)},
					emuBuf{"exp_avg_sq", "float", f32Bytes(emuV)},
					emuBuf{"grad", ct, gradBytes},
				)

				emuW = bytesF32(res["master_w"])
				emuM = bytesF32(res["exp_avg"])
				emuV = bytesF32(res["exp_avg_sq"])
				gotParam := decodeBwdBuf(tc.dtype, res["param"])

				f32Tol := cudagradcheck.Tolerance{Atol: 1e-5, Rtol: 1e-4}
				if rep, err := cudagradcheck.Compare(emuW, refW, f32Tol); err != nil {
					t.Fatalf("step %d master_w mismatch: %v (%v)", step, err, rep)
				}
				if rep, err := cudagradcheck.Compare(emuM, refM, f32Tol); err != nil {
					t.Fatalf("step %d exp_avg mismatch: %v (%v)", step, err, rep)
				}
				if rep, err := cudagradcheck.Compare(emuV, refV, f32Tol); err != nil {
					t.Fatalf("step %d exp_avg_sq mismatch: %v (%v)", step, err, rep)
				}

				wantParam := make([]float64, tc.n)
				gotParamF32 := make([]float32, tc.n)
				for i := range tc.n {
					wantParam[i] = roundDType(tc.dtype, float64(emuW[i]))
					gotParamF32[i] = float32(gotParam[i])
				}
				if rep, err := cudagradcheck.Compare(gotParamF32, wantParam, cudagradcheck.Tolerance{Atol: 1e-6, Rtol: 1e-6}); err != nil {
					t.Fatalf("step %d downcast param mismatch: %v (%v)", step, err, rep)
				}

				curW := make([]float64, tc.n)
				for i, v := range emuW {
					curW[i] = float64(v)
				}
				curLoss := quadLoss(curW)
				if !(curLoss < prevLoss) {
					t.Fatalf("step %d loss did not decrease monotonically: got %g >= prev %g", step, curLoss, prevLoss)
				}
				prevLoss = curLoss
			}
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		for _, bad := range []FusedAdamW{
			{Name: "9bad"},
			{Name: "ok", DType: E2M1},
			{Name: "ok", Threads: 100},
		} {
			if _, err := bad.Emit(); err == nil {
				t.Errorf("expected error for %+v", bad)
			}
		}
	})
}
