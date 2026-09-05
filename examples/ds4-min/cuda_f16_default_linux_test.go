//go:build linux

package main

import (
	"encoding/binary"
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/lm/quant"
)

// Registered before physical execution. This gate checks half-input arithmetic
// against independent float64 dots, not bitwise cuBLAS reduction equivalence.
const cudaF16DefaultATol = 1e-4
const cudaF16DefaultRTol = 1e-4

func TestCUDAQualificationF16Default(t *testing.T) {
	for _, tc := range []struct {
		name       string
		cols, rows int
	}{
		{"rounding_boundaries", 37, 7}, {"router", 4096, 256}, {"hc_mix", 16384, 24},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			// Values straddle halfway points, including negative and subnormal inputs.
			boundary := []float32{1 + 1.0/2048, 1 + 3.0/2048, -1 - 1.0/2048, -1 - 3.0/2048, math.Nextafter32(1+1.0/2048, 1), math.Nextafter32(1+1.0/2048, 2), 1.0 / 33554432, 3.0 / 33554432}
			input := make([]float32, tc.cols)
			rounded := make([]float32, tc.cols)
			for i := range input {
				input[i] = boundary[i%len(boundary)]
				rounded[i] = quant.Float16ToFloat32(cudaQ81HalfReference(input[i]))
			}
			raw := make([]byte, 2*tc.cols*tc.rows)
			want := make([]float32, tc.rows)
			wrong := make([]float32, tc.rows)
			for row := range tc.rows {
				var sum, unrounded float64
				for col, x := range input {
					// Row0 weights align with rounding error signs, making omission of the
					// binary16 boundary visible even for short, cancellation-heavy inputs.
					bits := uint16(0x2c00 + ((row*37 + col*13) % 1024))
					if x-rounded[col] < 0 {
						bits |= 0x8000
					}
					binary.LittleEndian.PutUint16(raw[(row*tc.cols+col)*2:], bits)
					w := quant.Float16ToFloat32(bits)
					sum += float64(w) * float64(rounded[col])
					unrounded += float64(w) * float64(x)
				}
				want[row] = float32(sum)
				wrong[row] = float32(unrounded)
			}
			tensor := cudaSynthTensor(t, r, "default F16 fixture", 1, []uint64{uint64(tc.cols), uint64(tc.rows)}, raw)
			in := cudaSynthUpload(t, r, input)
			out := cudaTestAlloc(t, r, tc.rows)
			poison := make([]float32, tc.rows)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			if err := out.CopyFromHost(poison); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeF16MatvecDefault(tensor, in, out) })
			got := cudaTestRead(t, out)
			if err := cudaQualificationClose(got, want, cudaF16DefaultATol, cudaF16DefaultRTol); err != nil {
				t.Fatal(err)
			}
			blas := cudaF16DefaultCUBLAS(t, r, tensor, input)
			var blasError, candidateError float64
			for i := range blas {
				blasError = max(blasError, math.Abs(float64(blas[i])-float64(want[i])))
				candidateError = max(candidateError, math.Abs(float64(blas[i])-float64(got[i])))
			}
			t.Logf("actual cuBLAS GemmEx: max reference delta%g candidate delta%g", blasError, candidateError)
			if err := cudaQualificationClose(blas, want, cudaF16DefaultATol, cudaF16DefaultRTol); err != nil {
				t.Errorf("cuBLAS vs mathematical reference: %v", err)
			}
			if err := cudaQualificationClose(got, blas, cudaF16DefaultATol, cudaF16DefaultRTol); err != nil {
				t.Errorf("candidate vs cuBLAS: %v", err)
			}
			if err := cudaQualificationClose(wrong, want, cudaF16DefaultATol, cudaF16DefaultRTol); err == nil {
				t.Fatal("fixture accepts omitted binary16 input rounding")
			}
			var maxAbs float64
			for i := range got {
				maxAbs = max(maxAbs, math.Abs(float64(got[i])-float64(want[i])))
			}
			t.Logf("half-input F16 %dx%d: max absolute error %g, atol=%g rtol=%g; cuBLAS order not certified", tc.rows, tc.cols, maxAbs, cudaF16DefaultATol, cudaF16DefaultRTol)
			if err := r.encodeF16MatvecDefault(tensor, in.Slice(0, in.Len()-1), out); err == nil {
				t.Fatal("accepted short input")
			}
			if err := r.encodeF16MatvecDefault(tensor, in, out.Slice(0, out.Len()-1)); err == nil {
				t.Fatal("accepted short output")
			}
			bad := tensor
			bad.bytes--
			if err := r.encodeF16MatvecDefault(bad, in, out); err == nil {
				t.Fatal("accepted truncated weights")
			}
			bad = tensor
			bad.typeID = 0
			if err := r.encodeF16MatvecDefault(bad, in, out); err == nil {
				t.Fatal("accepted wrong weight type")
			}
		})
	}
}

func TestCUDAQualificationF16PairOrdered(t *testing.T) {
	for _, tc := range []struct {
		name       string
		cols, rows int
	}{
		{"cancellation", 256, 1}, {"compressor", 4096, 128}, {"partial_chunk", 263, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			x := make([]float32, tc.cols)
			a, b := make([]byte, tc.cols*tc.rows*2), make([]byte, tc.cols*tc.rows*2)
			for i := range x {
				x[i] = float32((i*29)%251-125) / 128
			}
			if tc.name == "cancellation" {
				for i := range x {
					x[i] = 1
				}
			}
			wantA, wantB := make([]float32, tc.rows), make([]float32, tc.rows)
			for row := range tc.rows {
				var sumA, sumB float64
				for col := range tc.cols {
					wa := uint16(0x2000 + (row*37+col*17)%1024)
					wb := uint16(0xa000 + (row*19+col*31)%1024)
					if tc.name == "cancellation" {
						switch {
						case col < 64:
							wa = 0x7bff
						case col < 128:
							wa = 0xfbff
						default:
							wa = 0x1400
						}
						wb = wa ^ 0x8000
					}
					binary.LittleEndian.PutUint16(a[(row*tc.cols+col)*2:], wa)
					binary.LittleEndian.PutUint16(b[(row*tc.cols+col)*2:], wb)
					sumA += float64(quant.Float16ToFloat32(wa)) * float64(x[col])
					sumB += float64(quant.Float16ToFloat32(wb)) * float64(x[col])
				}
				wantA[row], wantB[row] = float32(sumA), float32(sumB)
			}
			if tc.name == "cancellation" && (wantA[0] != 0.125 || wantB[0] != -0.125) {
				t.Fatal("invalid cancellation control")
			}
			dims := []uint64{uint64(tc.cols), uint64(tc.rows)}
			ta := cudaSynthTensor(t, r, "ordered F16 A", 1, dims, a)
			tb := cudaSynthTensor(t, r, "ordered F16 B", 1, dims, b)
			in := cudaSynthUpload(t, r, x)
			outA, outB := cudaTestAlloc(t, r, tc.rows), cudaTestAlloc(t, r, tc.rows)
			poison := make([]float32, tc.rows)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			if err := outA.CopyFromHost(poison); err != nil {
				t.Fatal(err)
			}
			if err := outB.CopyFromHost(poison); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeF16PairOrdered(ta, tb, in, outA, outB) })
			if err := cudaQualificationClose(cudaTestRead(t, outA), wantA, cudaF16DefaultATol, cudaF16DefaultRTol); err != nil {
				t.Fatal(err)
			}
			if err := cudaQualificationClose(cudaTestRead(t, outB), wantB, cudaF16DefaultATol, cudaF16DefaultRTol); err != nil {
				t.Fatal(err)
			}
			if tc.name == "cancellation" {
				cudaQualificationMeasure(t, r, func() error { return r.encodeF16Matvec(ta, in, outA) })
				wrong := cudaTestRead(t, outA)
				if err := cudaQualificationClose(wrong, wantA, cudaF16DefaultATol, cudaF16DefaultRTol); err == nil {
					t.Fatal("gate accepted interleaved reduction instead of ordered chunks")
				}
				t.Logf("physical wrong-tree control rejected: got%g want%g", wrong[0], wantA[0])
			}
			if err := r.encodeF16PairOrdered(ta, tb, in.Slice(0, in.Len()-1), outA, outB); err == nil {
				t.Fatal("accepted short input")
			}
			if err := r.encodeF16PairOrdered(ta, tb, in, outA.Slice(0, outA.Len()-1), outB); err == nil {
				t.Fatal("accepted short output")
			}
			bad := tb
			bad.bytes--
			if err := r.encodeF16PairOrdered(ta, bad, in, outA, outB); err == nil {
				t.Fatal("accepted truncated B")
			}
		})
	}
}

// cudaF16DefaultCUBLAS invokes the actual upstream default operand types and
// math mode. It is a second physical comparator, not an emulation of cuBLAS.
func cudaF16DefaultCUBLAS(t *testing.T, r *cudaFoundation, tensor cudaModelTensor, input []float32) []float32 {
	t.Helper()
	bits := make([]uint16, len(input))
	for i, v := range input {
		bits[i] = cudaQ81HalfReference(v)
	}
	in, err := cudamem.Alloc[uint16](r.device, len(bits))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := in.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := in.CopyFromHost(bits); err != nil {
		t.Fatal(err)
	}
	out := cudaTestAlloc(t, r, int(tensor.dimensions[1]))
	var handle cublas.Handle
	if err := r.device.WithContext(func() error {
		var err error
		handle, err = cublas.Create()
		if err != nil {
			return err
		}
		return handle.SetMathMode(cublas.TF32TensorOpMath)
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.device.WithContext(handle.Destroy); err != nil {
			t.Error(err)
		}
	})
	if err := r.device.DefaultStream().WithHandle(func(stream driver.Stream) error { return handle.SetStream(cublas.Stream(stream)) }); err != nil {
		t.Fatal(err)
	}
	cudaQualificationMeasure(t, r, func() error {
		return r.device.WithContext(func() error {
			alpha, beta := float32(1), float32(0)
			cols, rows := int32(tensor.dimensions[0]), int32(tensor.dimensions[1])
			err := handle.GemmEx(cublas.OpT, cublas.OpN, rows, 1, cols,
				uintptr(unsafe.Pointer(&alpha)), uintptr(tensor.device), cublas.R16F, cols,
				uintptr(in.Ptr()), cublas.R16F, cols, uintptr(unsafe.Pointer(&beta)), uintptr(out.Ptr()), cublas.R32F, rows, cublas.Compute32F, cublas.GemmDefault)
			runtime.KeepAlive(&alpha)
			runtime.KeepAlive(&beta)
			return err
		})
	})
	return cudaTestRead(t, out)
}
