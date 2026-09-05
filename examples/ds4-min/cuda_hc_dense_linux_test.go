//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/lm/quant"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Frozen before the first physical run. This extends the existing HC gate to
// dense projections; it does not replace a failure with a looser F16 gate.
const cudaHCDenseATol = 2e-5
const cudaHCDenseRTol = 2e-5

// TestCUDAHCDenseProjection separates RMS, identical-input projection, and HC
// outputs. Pinned ds4.c:24675-24704 computes a full cuBLAS mix once and caches
// its split. ds4_cuda.cu:15887-15927 specifies half operands and float output.
// The production HC kernels instead recompute a float32 reduction. The sparse
// fixture in cuda_hc_default_linux_test.go does not qualify that reduction.
func TestCUDAHCDenseProjection(t *testing.T) {
	// Attempts 12 and 13 established that cuBLAS itself fails this
	// mathematical cancellation contract. Keep that diagnostic available;
	// TestCUDAHCOracleEquivalence tests the production oracle contract.
	if os.Getenv("DS4_CUDA_HC_MATH_DIAGNOSTIC") != "1" {
		t.Skip("set DS4_CUDA_HC_MATH_DIAGNOSTIC=1 to run the mathematical diagnostic")
	}
	for _, name := range []string{"cancellation", "varied"} {
		t.Run(name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			state, block := make([]float32, 16384), make([]float32, 4096)
			for i := range state {
				state[i] = 1
				if name == "varied" {
					state[i] = float32((i*37)%251-125) / 64
				}
			}
			for i := range block {
				block[i] = float32(i%31-15) / 16
			}
			raw := make([]byte, 24*16384*2)
			weights := make([]float32, 24*16384)
			for row := range 24 {
				for col := range 16384 {
					var w float32
					if name == "cancellation" {
						// Every weight is nonzero and representable in binary16.
						// For half-rounded unit input, the exact row dot is +/-8;
						// the current 256-lane float tree loses the small terms.
						w = 1.0 / 1024
						if col%256 < 64 {
							w = 65504
						} else if col%256 < 128 {
							w = -65504
						}
						if row%2 != 0 {
							w = -w
						}
					} else {
						w = float32((col*19+row*43)%127-63) / 4096
						if w == 0 {
							w = 1.0 / 8192
						}
					}
					bits := cudaQ81HalfReference(w)
					weights[row*16384+col] = quant.Float16ToFloat32(bits)
					binary.LittleEndian.PutUint16(raw[2*(row*16384+col):], bits)
				}
			}
			fn := cudaSynthTensor(t, r, "dense hc fn", 1, []uint64{16384, 24}, raw)
			ds, db := cudaSynthUpload(t, r, state), cudaSynthUpload(t, r, block)
			normalized := cudaTestAlloc(t, r, len(state))
			cudaHCDensePoison(t, normalized)
			cudaQualificationMeasure(t, r, func() error { return r.encodeRMS(ds, cudaModelTensor{}, normalized, 1, false) })
			x := cudaTestRead(t, normalized)
			cudaHCDenseCheck(t, "RMS vs independent reference", x, cudaHCDefaultNormalized(state))
			// All projection implementations receive these identical activation
			// bytes, so this comparison excludes RMS drift and half-cell changes.
			half := make([]float32, len(x))
			for i, v := range x {
				half[i] = quant.Float16ToFloat32(cudaQ81HalfReference(v))
			}
			want := make([]float32, 24)
			for row := range want {
				var sum float64
				for col, v := range half {
					sum += float64(weights[row*16384+col]) * float64(v)
				}
				want[row] = float32(sum)
			}
			if name == "cancellation" {
				for i, v := range half {
					if v != 1 {
						t.Fatalf("cancellation half input %d: got %g, want 1", i, v)
					}
				}
				for row, v := range want {
					expected := float32(8)
					if row%2 != 0 {
						expected = -8
					}
					if v != expected {
						t.Fatalf("cancellation row %d: got %g, want %g", row, v, expected)
					}
				}
			}
			blas := cudaF16DefaultCUBLAS(t, r, fn, x)
			cudaHCDenseCheck(t, "actual cuBLAS vs float64 projection", blas, want)
			// This existing primitive exposes the same float32 product/tree as
			// HC after conversion, but is only a projection diagnostic. The
			// actual production HC entry points are checked separately below.
			projection := cudaTestAlloc(t, r, 24)
			cudaHCDensePoison(t, projection)
			dh := cudaSynthUpload(t, r, half)
			cudaQualificationMeasure(t, r, func() error { return r.encodeF16Matvec(fn, dh, projection) })
			got := cudaTestRead(t, projection)
			cudaHCDenseCheck(t, "float32 projection vs cuBLAS", got, blas)
			cudaHCDenseCheck(t, "float32 projection vs float64", got, want)
			wrong := append([]float32(nil), want...)
			wrong[0] += 1
			if cudaQualificationClose(wrong, want, cudaHCDenseATol, cudaHCDenseRTol) == nil {
				t.Fatal("projection comparator accepted injected error")
			}

			scale := []float32{.125, .125, .125}
			base := make([]float32, 24)
			for i := range base {
				base[i] = float32(i%7-3) / 16
			}
			norm := make([]float32, 4096)
			for i := range norm {
				norm[i] = 1 + float32(i%5-2)/32
			}
			st := cudaCompressedTensor(t, r, "scale", scale)
			bt := cudaCompressedTensor(t, r, "base", base)
			nt := cudaCompressedTensor(t, r, "norm", norm)
			pre, post := cudaTestAlloc(t, r, 8192), cudaTestAlloc(t, r, 16384)
			cudaHCDensePoison(t, pre)
			cudaHCDensePoison(t, post)
			cudaQualificationMeasure(t, r, func() error { return r.encodeHCPrePrecision(ds, fn, st, bt, nt, pre, true) })
			cudaQualificationMeasure(t, r, func() error { return r.encodeHCPostPrecision(ds, db, fn, st, bt, post, true) })
			wantPre, wantPost := cudaHCDenseFromMix(state, block, blas, scale, base, norm)
			cudaHCDenseCheck(t, "production HC pre vs cuBLAS mix", cudaTestRead(t, pre), wantPre)
			cudaHCDenseCheck(t, "production HC post vs cuBLAS mix", cudaTestRead(t, post), wantPost)
			headFn := fn
			headFn.dimensions = []uint64{16384, 4}
			headFn.bytes = 4 * 16384 * 2
			headBLAS := cudaF16DefaultCUBLAS(t, r, headFn, x)
			cudaHCDenseCheck(t, "output projection cuBLAS vs float64", headBLAS, want[:4])
			headMix := make([]float32, 24)
			copy(headMix, headBLAS)
			wantHead, _ := cudaHCDenseFromMix(state, block, headMix, scale, base, norm)
			head := cudaOutput{hcFn: headFn, hcScale: st, hcBase: bt, norm: nt,
				weight: cudaSynthTensor(t, r, "unused output projection", 8, []uint64{4096, 8}, make([]byte, 8*128*34))}
			mixed, logits := cudaTestAlloc(t, r, 8192), cudaTestAlloc(t, r, 8)
			cudaHCDensePoison(t, mixed)
			cudaQualificationMeasure(t, r, func() error { return r.encodeOutputTransform(ds, head, mixed, logits, true) })
			cudaHCDenseCheck(t, "production output collapse vs cuBLAS mix", cudaTestRead(t, mixed), wantHead)
			wrongPost := append([]float32(nil), wantPost...)
			wrongPost[31] += 1
			if cudaQualificationClose(wrongPost, wantPost, cudaHCDenseATol, cudaHCDenseRTol) == nil {
				t.Fatal("HC comparator accepted injected error")
			}
		})
	}
}

// Added after attempt12: poisoning and the explicit +/-8 assertion strengthen
// missing-store and fixture-drift controls without changing values or tolerances.
func cudaHCDensePoison(t *testing.T, out cudamem.Slice[float32]) {
	t.Helper()
	poison := make([]float32, out.Len())
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	cudaCompressedUpload(t, out, poison)
}

func cudaHCDenseCheck(t *testing.T, name string, got, want []float32) {
	t.Helper()
	var maximum float64
	for i := range got {
		maximum = max(maximum, math.Abs(float64(got[i])-float64(want[i])))
	}
	t.Logf("%s: max absolute delta %g; atol=%g rtol=%g", name, maximum, cudaHCDenseATol, cudaHCDenseRTol)
	if err := cudaQualificationClose(got, want, cudaHCDenseATol, cudaHCDenseRTol); err != nil {
		t.Errorf("%s: %v", name, err)
	}
}

// cudaHCDenseFromMix uses actual cuBLAS projection values and independent
// float64 HC equations. It does not emulate the device reduction or intrinsics.
func cudaHCDenseFromMix(state, block, mix, scale, base, norm []float32) ([]float32, []float32) {
	var pre, post [4]float64
	var comb [4][4]float64
	for i := range 4 {
		pre[i] = 1/(1+math.Exp(-(float64(mix[i])*float64(scale[0])+float64(base[i])))) + 1e-6
		post[i] = 2 / (1 + math.Exp(-(float64(mix[4+i])*float64(scale[1]) + float64(base[4+i]))))
		maximum := math.Inf(-1)
		for j := range 4 {
			comb[i][j] = float64(mix[8+4*i+j])*float64(scale[2]) + float64(base[8+4*i+j])
			maximum = max(maximum, comb[i][j])
		}
		var sum float64
		for j := range 4 {
			comb[i][j] = math.Exp(comb[i][j] - maximum)
			sum += comb[i][j]
		}
		for j := range 4 {
			comb[i][j] = comb[i][j]/sum + 1e-6
		}
	}
	for iteration := range 20 {
		for col := range 4 {
			var sum float64
			for row := range 4 {
				sum += comb[row][col]
			}
			for row := range 4 {
				comb[row][col] /= sum + 1e-6
			}
		}
		if iteration == 19 {
			break
		}
		for row := range 4 {
			var sum float64
			for col := range 4 {
				sum += comb[row][col]
			}
			for col := range 4 {
				comb[row][col] /= sum + 1e-6
			}
		}
	}
	before, after := make([]float32, 8192), make([]float32, 16384)
	for d, b := range block {
		var sum float64
		for src := range 4 {
			sum += float64(state[src*4096+d]) * pre[src]
		}
		before[d] = float32(sum)
		for dst := range 4 {
			sum := float64(b) * post[dst]
			for src := range 4 {
				sum += float64(state[src*4096+d]) * comb[src][dst]
			}
			after[dst*4096+d] = float32(sum)
		}
	}
	flat := cudaHCDefaultNormalized(before[:4096])
	for i, v := range flat {
		before[4096+i] = v * norm[i]
	}
	return before, after
}
