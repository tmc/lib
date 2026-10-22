//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"math"
	"runtime"
	"testing"

	"github.com/tmc/lib/lm/quant"
)

// Registered before physical execution. This gate checks half-input arithmetic
// against independent float64 dots, not bitwise cuBLAS reduction equivalence.
const cudaF16DefaultATol = 1e-4
const cudaF16DefaultRTol = 1e-4

// cudaF16DefaultBound returns the error bound for one row of
// ds4_f16_matvec_half_input given the row's sum of |w*x|. It is the float
// accumulation error of the kernel's order: each of 256 threads fuses up to
// k = ceil(cols/256) products into a running sum, then a 256-way tree adds
// log2(256) = 8 levels, so each term passes through at most k+8 roundings,
// plus 2 for slack. A fixed atol cannot hold for long cancelling rows: on
// hc_mix (24x16384) cuBLAS GemmEx measured 2.3e-6 against the float64
// reference and this kernel 1.6e-4, against a fixed limit of 1.19e-4.
func cudaF16DefaultBound(cols int, abs float64) float64 {
	const u = 0x1p-24 // float32 unit roundoff
	k := (cols + 255) / 256
	return float64(k+8+2) * u * abs
}

// cudaF16DefaultClose is cudaQualificationClose with the fixed limit raised
// to bound[i] where that is larger.
func cudaF16DefaultClose(got, want []float32, bound []float64) error {
	if len(got) == 0 || len(got) != len(want) || len(bound) != len(want) {
		return fmt.Errorf("sample lengths: got %d, want %d, bound %d nonempty values", len(got), len(want), len(bound))
	}
	for i, value := range got {
		g, w := float64(value), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			return fmt.Errorf("sample %d: non-finite values %g, %g", i, g, w)
		}
		limit := max(cudaF16DefaultATol+cudaF16DefaultRTol*math.Abs(w), bound[i])
		if delta := math.Abs(g - w); delta > limit {
			return fmt.Errorf("sample %d: got %g, want %g (delta %g > %g)", i, g, w, delta, limit)
		}
	}
	return nil
}

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
			bound := make([]float64, tc.rows)
			for row := range tc.rows {
				var sum, unrounded, abs float64
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
					abs += math.Abs(float64(w) * float64(rounded[col]))
				}
				bound[row] = cudaF16DefaultBound(tc.cols, abs)
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
			// The bound must stay small against the outputs it judges, or it
			// accepts anything.
			var maxBound, maxWant float64
			for i := range want {
				maxBound = max(maxBound, bound[i])
				maxWant = max(maxWant, math.Abs(float64(want[i])))
			}
			t.Logf("accumulation bound: max %g, %.3g of max |want| %g", maxBound, maxBound/maxWant, maxWant)
			if maxBound > maxWant/16 {
				t.Fatalf("accumulation bound %g exceeds 1/16 of max |want| %g", maxBound, maxWant)
			}
			got := cudaTestRead(t, out)
			if err := cudaF16DefaultClose(got, want, bound); err != nil {
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
			if err := cudaF16DefaultClose(got, blas, bound); err != nil {
				t.Errorf("candidate vs cuBLAS: %v", err)
			}
			if err := cudaF16DefaultClose(wrong, want, bound); err == nil {
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
		if err := r.device.WithContext(handle.Close); err != nil {
			t.Error(err)
		}
	})
	// Inside the device context, like newCUDAHC does it. cublasSetStream is a
	// context-bound call: with no context current on the calling thread it
	// returns CUBLAS_STATUS_INTERNAL_ERROR, which is what this comparator did on
	// every run until 2026-09-18 -- so the "cuBLAS failure" here was the fixture
	// never reaching cuBLAS, not cuBLAS failing. Once the context is current,
	// GemmEx succeeds on this device in every math mode, compute type and
	// algorithm probed.
	if err := r.device.WithContext(func() error {
		return r.device.DefaultStream().WithHandle(func(stream driver.Stream) error {
			return handle.SetStream(driver.Stream(stream))
		})
	}); err != nil {
		t.Fatal(err)
	}
	cudaQualificationMeasure(t, r, func() error {
		return r.device.WithContext(func() error {
			alpha, beta := float32(1), float32(0)
			cols, rows := int32(tensor.dimensions[0]), int32(tensor.dimensions[1])
			err := handle.GemmEx(cublas.OpT, cublas.OpN, rows, 1, cols, cublas.Float32Scalar(alpha), driver.DevicePtr(tensor.device), cublas.R16F, cols, driver.DevicePtr(in.Ptr()), cublas.R16F, cols, cublas.Float32Scalar(beta), driver.DevicePtr(out.Ptr()), cublas.R32F, rows, cublas.Compute32F, cublas.GemmDefault)
			runtime.KeepAlive(&alpha)
			runtime.KeepAlive(&beta)
			return err
		})
	})
	return cudaTestRead(t, out)
}
