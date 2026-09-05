//go:build linux

package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/tmc/lib/lm/quant"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Quantization is byte-exact. Projection tolerances are registered before
// physical execution and match manual-gates.md's synthetic f32 contract.
const cudaQ80ATol = 1e-4
const cudaQ80RTol = 1e-4

func TestCUDAQualificationQ80Quantization(t *testing.T) {
	r := cudaSynthFoundation(t)
	for _, name := range []string{"zero", "positive_max", "negative_max", "opposite_tie", "fractional_scale"} {
		t.Run(name, func(t *testing.T) {
			x := make([]float32, 64)
			if name != "zero" {
				for i := range x {
					x[i] = float32((i*17)%101-50) / 2
				}
				for b := range 2 {
					switch name {
					case "positive_max":
						x[b*32] = 127
					case "negative_max":
						x[b*32] = -127
					case "opposite_tie":
						x[b*32+1] = -127
						x[b*32+16] = 127
					case "fractional_scale":
						x[b*32+7] = 131.375
					}
				}
			}
			want := cudaQ80ActivationReference(x)
			if name == "opposite_tie" && math.Float32frombits(binary.LittleEndian.Uint32(want)) != 1 {
				t.Fatal("reference lost unsigned absolute-maximum scale")
			}
			if name == "positive_max" && (int8(want[4+1]) != -16 || int8(want[4+3]) != 0) {
				t.Fatal("reference lost nearest-even rounding")
			}
			in := cudaSynthUpload(t, r, x)
			out := cudaExpertAllocBytes(t, r, len(want))
			if err := out.CopyFromHost(bytes.Repeat([]byte{0xa5}, len(want))); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeQuantQ80(in, out, 32) })
			cudaQ80CheckBytes(t, out, want)
			for label, err := range map[string]error{
				"partial block": r.encodeQuantQ80(in, out, 31),
				"output extent": r.encodeQuantQ80(in, out.Slice(0, out.Len()-1), 32),
				"input extent":  r.encodeQuantQ80(in.Slice(0, in.Len()-1), out, 32),
			} {
				if err == nil {
					t.Fatalf("accepted bad %s", label)
				}
			}
		})
	}
}

func TestCUDAQualificationQ80Matvec(t *testing.T) {
	for _, cols := range []int{32, 96, 4096, 8192} {
		t.Run(strconv.Itoa(cols), func(t *testing.T) {
			r := cudaSynthFoundation(t)
			const rows = 17
			weights := cudaSynthQ80(rand.New(rand.NewSource(int64(cols))), rows, cols, -8)
			tensor := cudaSynthTensor(t, r, "quantized Q8_0 fixture", 8, []uint64{uint64(cols), rows}, weights)
			x := make([]float32, cols)
			for i := range x {
				x[i] = float32((i*37+i/17)%257-128) / 128
			}
			raw := cudaQ80ActivationReference(x)
			in := cudaSynthUpload(t, r, x)
			q := cudaExpertAllocBytes(t, r, len(raw))
			cudaQualificationMeasure(t, r, func() error { return r.encodeQuantQ80(in, q, cols) })
			cudaQ80CheckBytes(t, q, raw)
			out := cudaTestAlloc(t, r, rows)
			poison := make([]float32, rows)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			if err := out.CopyFromHost(poison); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeQ80MatvecQuantized(tensor, q, out) })
			want := cudaQ80MatvecReference(t, weights, raw, cols, rows)
			got := cudaTestRead(t, out)
			if err := cudaQualificationClose(got, want, cudaQ80ATol, cudaQ80RTol); err != nil {
				t.Fatal(err)
			}
			var maxAbs float64
			for i := range got {
				maxAbs = max(maxAbs, math.Abs(float64(got[i])-float64(want[i])))
			}
			t.Logf("quantized Q8_0 %dx%d: max absolute error %g, atol=%g rtol=%g", rows, cols, maxAbs, cudaQ80ATol, cudaQ80RTol)
			// Prove this input detects replacing the quantized path with a raw-F32 dot.
			unquantized := make([]float32, rows)
			decoded, err := quant.DequantizeQ8_0(weights, rows*cols)
			if err != nil {
				t.Fatal(err)
			}
			for row := range rows {
				var sum float64
				for col, v := range x {
					sum += float64(decoded[row*cols+col]) * float64(v)
				}
				unquantized[row] = float32(sum)
			}
			if err := cudaQualificationClose(unquantized, want, cudaQ80ATol, cudaQ80RTol); err == nil {
				t.Fatal("fixture cannot detect omitted activation quantization")
			}
			shifted := append([]float32(nil), want[1:]...)
			shifted = append(shifted, want[0])
			if err := cudaQualificationClose(shifted, want, cudaQ80ATol, cudaQ80RTol); err == nil {
				t.Fatal("fixture cannot detect wrong output row")
			}
			for _, bad := range []struct {
				name   string
				typeID uint32
				dims   []uint64
				size   uint64
			}{
				{"type", 0, tensor.dimensions, tensor.bytes},
				{"rank", 8, []uint64{uint64(cols)}, tensor.bytes},
				{"partial block", 8, []uint64{uint64(cols - 1), rows}, tensor.bytes},
				{"extent", 8, tensor.dimensions, tensor.bytes - 1},
			} {
				v := tensor
				v.typeID = bad.typeID
				v.dimensions = bad.dims
				v.bytes = bad.size
				if err := r.encodeQ80MatvecQuantized(v, q, out); err == nil {
					t.Fatalf("accepted bad %s", bad.name)
				}
			}
			if err := r.encodeQ80MatvecQuantized(tensor, q.Slice(0, q.Len()-1), out); err == nil {
				t.Fatal("accepted short activations")
			}
			if err := r.encodeQ80MatvecQuantized(tensor, q, out.Slice(0, rows-1)); err == nil {
				t.Fatal("accepted short output")
			}
		})
	}
}

// cudaQ80ActivationReference follows ds4_cuda.cu:5037 with a scalar absolute
// maximum and explicit float32 rounding. Unlike Q8_K, the scale is positive.
func cudaQ80ActivationReference(x []float32) []byte {
	raw := make([]byte, len(x)/32*36)
	for b := range len(x) / 32 {
		row := x[b*32 : (b+1)*32]
		var amax float32
		for _, v := range row {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		d := amax / float32(127)
		var inverse float32
		if d != 0 {
			inverse = 1 / d
		}
		p := raw[b*36 : (b+1)*36]
		binary.LittleEndian.PutUint32(p, math.Float32bits(d))
		for i, v := range row {
			scaled := float32(v * inverse)
			q := min(127, max(-128, int(math.RoundToEven(float64(scaled)))))
			p[4+i] = byte(int8(q))
		}
	}
	return raw
}

// cudaQ80MatvecReference computes exact integer block dots and accumulates their
// scaled values in float64, independently of the CUDA warp reduction.
func cudaQ80MatvecReference(t *testing.T, weights, activation []byte, cols, rows int) []float32 {
	t.Helper()
	out := make([]float32, rows)
	for row := range rows {
		var sum float64
		for b := range cols / 32 {
			w := weights[(row*(cols/32)+b)*34:]
			x := activation[b*36:]
			var dot int64
			for i := range 32 {
				dot += int64(int8(w[2+i])) * int64(int8(x[4+i]))
			}
			wd := quant.Float16ToFloat32(binary.LittleEndian.Uint16(w))
			xd := math.Float32frombits(binary.LittleEndian.Uint32(x))
			sum += float64(wd) * float64(xd) * float64(dot)
		}
		out[row] = float32(sum)
	}
	return out
}

func cudaQ80CheckBytes(t *testing.T, device cudamem.Slice[byte], want []byte) {
	t.Helper()
	got := make([]byte, device.Len())
	if err := device.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		for i := range min(len(got), len(want)) {
			if got[i] != want[i] {
				t.Fatalf("Q8_0 activation byte%d (block%d offset%d): got%#x want%#x", i, i/36, i%36, got[i], want[i])
			}
		}
		t.Fatalf("Q8_0 activation lengths %d and %d", len(got), len(want))
	}
	t.Logf("Q8_0 activation %d bytes match exactly", len(got))
}
