//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"testing"
)

// Pinned ds4.c:22960-23050 cannot take the Apple-only HC norm/mix fusion on
// CUDA. ds4_cuda.cu:15887-15919 converts normalized float32 activations to
// binary16 before the default cuBLAS F16 projection. These tests retain the
// established HC fixture gate, 2e-5 absolute plus 2e-5 relative.
func TestCUDAHCDefaultPrecision(t *testing.T) {
	r := cudaSynthFoundation(t)
	state, block, weights, scale, base, norm, weightBytes := cudaHCDefaultFixture(t)
	fn := cudaSynthTensor(t, r, "hc default fn", 1, []uint64{16384, 24}, weightBytes)
	ds, db := cudaTestAlloc(t, r, 16384), cudaTestAlloc(t, r, 4096)
	cudaCompressedUpload(t, ds, state)
	cudaCompressedUpload(t, db, block)
	st := cudaCompressedTensor(t, r, "scale", scale)
	bt := cudaCompressedTensor(t, r, "base", base)
	nt := cudaCompressedTensor(t, r, "norm", norm)
	head := cudaOutput{
		hcFn:    cudaSynthTensor(t, r, "head fn", 1, []uint64{16384, 4}, weightBytes[:4*16384*2]),
		hcScale: cudaCompressedTensor(t, r, "head scale", scale[:1]),
		hcBase:  cudaCompressedTensor(t, r, "head base", base[:4]), norm: nt,
		weight: cudaSynthTensor(t, r, "unused head projection", 8, []uint64{4096, 8}, make([]byte, 8*128*34)),
	}
	logits := cudaTestAlloc(t, r, 8)
	for _, name := range []string{"pre", "post", "output"} {
		t.Run(name, func(t *testing.T) {
			n := 8192
			if name == "post" {
				n = 16384
			}
			out := cudaTestAlloc(t, r, n)
			wantPre, wantPost := cudaHCDefaultReference(state, block, weights, scale, base, norm, true)
			want := wantPre
			if name == "post" {
				want = wantPost
			}
			launch := func(half bool) error {
				switch name {
				case "pre":
					return r.encodeHCPrePrecision(ds, fn, st, bt, nt, out, half)
				case "post":
					return r.encodeHCPostPrecision(ds, db, fn, st, bt, out, half)
				default:
					return r.encodeOutputTransform(ds, head, out, logits, half)
				}
			}
			poison := make([]float32, n)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			cudaCompressedUpload(t, out, poison)
			cudaQualificationMeasure(t, r, func() error { return launch(true) })
			got := cudaTestRead(t, out)
			if err := cudaQualificationClose(got, want, cudaHCPostQualificationATol, cudaHCPostQualificationRTol); err != nil {
				t.Fatal(err)
			}
			// The old unrounded production branch must fail the same oracle and gate.
			cudaCompressedUpload(t, out, poison)
			cudaQualificationMeasure(t, r, func() error { return launch(false) })
			if cudaQualificationClose(cudaTestRead(t, out), want, cudaHCPostQualificationATol, cudaHCPostQualificationRTol) == nil {
				t.Fatal("old unrounded input branch passed default CUDA gate")
			}
		})
	}

}

func cudaHCDefaultFixture(t *testing.T) (state, block, weights, scale, base, norm []float32, weightBytes []byte) {
	state, block = make([]float32, 16384), make([]float32, 4096)
	for i := range state {
		state[i] = float32((i*29+i/4096*113)%1999-999)/1000 + float32(i/4096)/8
	}
	for i := range block {
		block[i] = float32((i*17)%127-63) / 64
	}
	normalized := cudaHCDefaultNormalized(state)
	weights = make([]float32, 24*16384)
	weightBytes = make([]byte, 2*len(weights))
	base = make([]float32, 24)
	scale = []float32{1, 1, 1}
	for row := range 24 {
		// Find a normal activation safely inside a half cell with a visible rounding
		// error. A sparse projection isolates conversion from reduction-order noise.
		col := (row*719 + 17) % len(state)
		found := false
		for range len(state) {
			v := normalized[col]
			delta := math.Abs(float64(v - cudaHCDefaultHalf(v)))
			if math.Abs(float64(v)) >= .5 && math.Abs(float64(v)) <= 1.5 && delta >= .00012 && delta <= .0002 {
				found = true
				break
			}
			col = (col + 1) % len(state)
		}
		if !found {
			t.Fatal("no stable half-cell fixture activation")
		}
		weight := float32(8)
		encoding := uint16(0x4800)
		if row%3 == 1 {
			weight = -8
			encoding = 0xc800
		}
		weights[row*len(state)+col] = weight
		binary.LittleEndian.PutUint16(weightBytes[2*(row*len(state)+col):], encoding)
		base[row] = -weight*cudaHCDefaultHalf(normalized[col]) + float32(row%7-3)/8
	}
	norm = make([]float32, 4096)
	for i := range norm {
		norm[i] = 1 + float32(i%5-2)/16
	}
	return
}

func cudaHCDefaultNormalized(x []float32) []float32 {
	squares := 0.0
	for _, v := range x {
		squares += float64(v) * float64(v)
	}
	// The upstream RMS output is float32 before the separate conversion kernel.
	mean := float32(squares / float64(len(x)))
	denom := mean + float32(1e-6)
	inv := float32(1 / math.Sqrt(float64(denom)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv
	}
	return out
}

// Independent binary16 rounding by an arithmetic lattice, not CUDA intrinsics
// or the production bit-manipulation helper. Fixtures are finite and bounded.
func cudaHCDefaultHalf(v float32) float32 {
	_, exp := math.Frexp(math.Abs(float64(v)))
	step := math.Ldexp(1, max(exp-1, -14)-10)
	return float32(math.RoundToEven(float64(v)/step) * step)
}

func cudaHCDefaultReference(state, block, weights, scale, base, norm []float32, half bool) ([]float32, []float32) {
	normalized := cudaHCDefaultNormalized(state)
	if half {
		for i := range normalized {
			normalized[i] = cudaHCDefaultHalf(normalized[i])
		}
	}
	mix := make([]float64, 24)
	for row := range mix {
		for col, v := range normalized {
			mix[row] += float64(weights[row*len(state)+col]) * float64(v)
		}
	}
	pre, post := make([]float64, 4), make([]float64, 4)
	var matrix [4][4]float64
	for h := range 4 {
		pre[h] = 1/(1+math.Exp(-(mix[h]*float64(scale[0])+float64(base[h])))) + 1e-6
		post[h] = 2 / (1 + math.Exp(-(mix[4+h]*float64(scale[1]) + float64(base[4+h]))))
	}
	for row := range 4 {
		maximum := math.Inf(-1)
		for col := range 4 {
			matrix[row][col] = mix[8+4*row+col]*float64(scale[2]) + float64(base[8+4*row+col])
			maximum = math.Max(maximum, matrix[row][col])
		}
		sum := 0.0
		for col := range 4 {
			matrix[row][col] = math.Exp(matrix[row][col] - maximum)
			sum += matrix[row][col]
		}
		for col := range 4 {
			matrix[row][col] = matrix[row][col]/sum + 1e-6
		}
	}
	for iteration := range 20 {
		for col := range 4 {
			sum := 0.0
			for row := range 4 {
				sum += matrix[row][col]
			}
			for row := range 4 {
				matrix[row][col] /= sum + 1e-6
			}
		}
		if iteration == 19 {
			break
		}
		for row := range 4 {
			sum := 0.0
			for _, v := range matrix[row] {
				sum += v
			}
			for col := range 4 {
				matrix[row][col] /= sum + 1e-6
			}
		}
	}
	before, after := make([]float32, 8192), make([]float32, 16384)
	for d, v := range block {
		sum := 0.0
		for h := range 4 {
			sum += float64(state[h*4096+d]) * pre[h]
		}
		before[d] = float32(sum)
		for dst := range 4 {
			sum := float64(v) * post[dst]
			for src := range 4 {
				sum += float64(state[src*4096+d]) * matrix[src][dst]
			}
			after[dst*4096+d] = float32(sum)
		}
	}
	currentNorm := cudaHCDefaultNormalized(before[:4096])
	for i, v := range currentNorm {
		before[4096+i] = v * norm[i]
	}
	return before, after
}

func TestCUDAHCDefaultReferenceSensitivity(t *testing.T) {
	state, block, weights, scale, base, norm, _ := cudaHCDefaultFixture(t)
	before, after := cudaHCDefaultReference(state, block, weights, scale, base, norm, true)
	oldBefore, oldAfter := cudaHCDefaultReference(state, block, weights, scale, base, norm, false)
	for name, pair := range map[string][2][]float32{"pre": {oldBefore, before}, "post": {oldAfter, after}} {
		if cudaQualificationClose(pair[0], pair[1], cudaHCPostQualificationATol, cudaHCPostQualificationRTol) == nil {
			t.Fatalf("%s fixture cannot distinguish activation rounding", name)
		}
	}
}
