//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"testing"
)

// These tolerances are fixed before device execution. The reference accumulates
// in float64; the kernel uses float32 reductions and transcendental functions.
// They apply only to these bounded synthetic inputs, not checkpoint parity.
const cudaHCPostQualificationATol = 2e-5
const cudaHCPostQualificationRTol = 2e-5

func TestCUDAQualificationHCPost(t *testing.T) {
	r := cudaSynthFoundation(t)
	const width = 4096
	weights := make([]float64, 24*4*width)
	weightBytes := make([]byte, 2*len(weights))
	// All four weights are exactly representable in binary16. The pattern
	// varies across both rows and columns, including reduction-lane boundaries.
	for row := range 24 {
		for col := range 4 * width {
			i := row*4*width + col
			code := (row*31 + col*17 + col/251 + row*col%13) % 4
			bits := []uint16{0x1400, 0x1000, 0x9400, 0x9000}[code]
			weights[i] = []float64{1.0 / 1024, 1.0 / 2048, -1.0 / 1024, -1.0 / 2048}[code]
			binary.LittleEndian.PutUint16(weightBytes[2*i:], bits)
		}
	}
	scale := []float32{0.75, 1.25, 0.875}
	base := make([]float32, 24)
	for i := range base {
		base[i] = float32((i*7+i*i*3)%23-11) / 8
	}
	fn := cudaSynthTensor(t, r, "synthetic hc fn", 1, []uint64{4 * width, 24}, weightBytes)
	scaleTensor := cudaSynthTensor(t, r, "synthetic hc scale", 0, []uint64{3}, cudaHCPostQualificationBytes(scale))
	baseTensor := cudaSynthTensor(t, r, "synthetic hc base", 0, []uint64{24}, cudaHCPostQualificationBytes(base))
	for _, name := range []string{"asymmetric", "zero_state", "zero_block"} {
		t.Run(name, func(t *testing.T) {
			state := make([]float32, 4*width)
			block := make([]float32, width)
			for i := range state {
				state[i] = float32((i*29+i/width*19+7)%251-125)/128 + float32(i/width)/4
			}
			for i := range block {
				block[i] = float32((i*13+3)%199-99) / 256
			}
			if name == "zero_state" {
				clear(state)
			}
			if name == "zero_block" {
				clear(block)
			}
			want := cudaHCPostQualificationReference(state, block, weights, scale, base, false)
			dState := cudaTestAlloc(t, r, len(state))
			dBlock := cudaTestAlloc(t, r, len(block))
			dOut := cudaTestAlloc(t, r, len(state))
			for _, err := range []error{dState.CopyFromHost(state), dBlock.CopyFromHost(block)} {
				if err != nil {
					t.Fatal(err)
				}
			}
			poison := make([]float32, len(state))
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			if err := dOut.CopyFromHost(poison); err != nil {
				t.Fatal(err)
			}
			for _, tc := range []struct {
				name            string
				fn, scale, base cudaModelTensor
			}{
				{"projection dimensions", cudaModelTensor{typeID: 1, dimensions: []uint64{4 * width, 23}}, scaleTensor, baseTensor},
				{"projection type", cudaModelTensor{typeID: 0, dimensions: fn.dimensions}, scaleTensor, baseTensor},
				{"scale dimensions", fn, cudaModelTensor{typeID: 0, dimensions: []uint64{2}}, baseTensor},
				{"base dimensions", fn, scaleTensor, cudaModelTensor{typeID: 0, dimensions: []uint64{23}}},
			} {
				if err := r.encodeHCPost(dState, dBlock, tc.fn, tc.scale, tc.base, dOut); err == nil {
					t.Fatalf("accepted wrong %s", tc.name)
				}
			}
			for label, err := range map[string]error{
				"state length":  r.encodeHCPost(dState.Slice(0, dState.Len()-1), dBlock, fn, scaleTensor, baseTensor, dOut),
				"block length":  r.encodeHCPost(dState, dBlock.Slice(0, dBlock.Len()-1), fn, scaleTensor, baseTensor, dOut),
				"output length": r.encodeHCPost(dState, dBlock, fn, scaleTensor, baseTensor, dOut.Slice(0, dOut.Len()-1)),
			} {
				if err == nil {
					t.Fatalf("accepted wrong %s", label)
				}
			}
			cudaQualificationMeasure(t, r, func() error {
				return r.encodeHCPost(dState, dBlock, fn, scaleTensor, baseTensor, dOut)
			})
			got := cudaTestRead(t, dOut)
			if err := cudaQualificationClose(got, want, cudaHCPostQualificationATol, cudaHCPostQualificationRTol); err != nil {
				t.Fatal(err)
			}
			var maxAbs float64
			for i := range got {
				maxAbs = max(maxAbs, math.Abs(float64(got[i])-float64(want[i])))
			}
			t.Logf("physical HC-post: %d outputs, max absolute error %.9g, atol=%g rtol=%g", len(got), maxAbs, cudaHCPostQualificationATol, cudaHCPostQualificationRTol)
			if err := cudaQualificationClose(make([]float32, len(want)), want, cudaHCPostQualificationATol, cudaHCPostQualificationRTol); err == nil {
				t.Fatal("fixture does not reject zero output")
			}
			if name == "asymmetric" {
				for label, wrong := range map[string][]float32{
					"omitted projection":     cudaHCPostQualificationReference(state, block, make([]float64, len(weights)), scale, base, false),
					"transposed combination": cudaHCPostQualificationReference(state, block, weights, scale, base, true),
				} {
					if err := cudaQualificationClose(wrong, want, cudaHCPostQualificationATol, cudaHCPostQualificationRTol); err == nil {
						t.Fatalf("fixture does not reject %s", label)
					}
				}
			}
		})
	}
}

func cudaHCPostQualificationBytes(values []float32) []byte {
	p := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(p[4*i:], math.Float32bits(v))
	}
	return p
}

// cudaHCPostQualificationReference evaluates the CPU hcSplit/hcPost contract
// with float64 scalar sums and a two-dimensional matrix. It deliberately does
// not reproduce the device's reduction tree or call its mirrored test helpers.
func cudaHCPostQualificationReference(state, block []float32, weights []float64, scale, base []float32, transpose bool) []float32 {
	var squares float64
	for _, v := range state {
		squares += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(squares/float64(len(state))+1e-6)
	var mix [24]float64
	for row := range mix {
		for col, v := range state {
			mix[row] += weights[row*len(state)+col] * float64(v) * inv
		}
	}
	var gate [4]float64
	var matrix [4][4]float64
	for row := range matrix {
		gate[row] = 2 / (1 + math.Exp(-(mix[4+row]*float64(scale[1]) + float64(base[4+row]))))
		mx := math.Inf(-1)
		for col := range matrix[row] {
			matrix[row][col] = mix[8+4*row+col]*float64(scale[2]) + float64(base[8+4*row+col])
			mx = max(mx, matrix[row][col])
		}
		var sum float64
		for col := range matrix[row] {
			matrix[row][col] = math.Exp(matrix[row][col] - mx)
			sum += matrix[row][col]
		}
		for col := range matrix[row] {
			matrix[row][col] = matrix[row][col]/sum + 1e-6
		}
	}
	for iteration := range 20 {
		for col := range 4 {
			var sum float64
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
		for row := range matrix {
			var sum float64
			for _, v := range matrix[row] {
				sum += v
			}
			for col := range matrix[row] {
				matrix[row][col] /= sum + 1e-6
			}
		}
	}
	out := make([]float32, len(state))
	for dst := range 4 {
		for d, v := range block {
			sum := float64(v) * gate[dst]
			for src := range 4 {
				coefficient := matrix[src][dst]
				if transpose {
					coefficient = matrix[dst][src]
				}
				sum += coefficient * float64(state[src*len(block)+d])
			}
			out[dst*len(block)+d] = float32(sum)
		}
	}
	return out
}
