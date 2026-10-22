//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"runtime"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaF16DefaultSource reproduces the binary16 activation boundary in the
// default single-token cuBLAS dispatch in pinned antirez/ds4 b0a147a,
// ds4_cuda.cu:15887-15927. It accumulates in float, as the reference does; it is not cuBLAS's implementation.
//
//go:embed cuda_f16_default.cu
var cudaF16DefaultSource string

// encodeF16MatvecDefault projects F16 weights after rounding input to F16,
// matching the default upstream cuBLAS operand types. It does not promise
// cuBLAS's accumulation order; the original raw-F32-input primitive is separate.
func (r *cudaFoundation) encodeF16MatvecDefault(t cudaModelTensor, in, out cudamem.Slice[float32]) error {
	if t.typeID != 1 || len(t.dimensions) != 2 || t.device == 0 {
		return fmt.Errorf("cuda: default F16 matvec %q: invalid tensor", t.name)
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if cols == 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 || uint64(in.Len()) != cols || uint64(out.Len()) != rows || rows > t.bytes/(cols*2) {
		return fmt.Errorf("cuda: default F16 matvec %q: invalid shape or extent", t.name)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		r.countWeightBytes("ds4_f16_matvec_half_input", rows*cols*2)
		if err := cudalaunch.Kernel(r.f16DefaultFn).Grid(uint32(rows), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(t.device, in, out, uint32(cols)); err != nil {
			return fmt.Errorf("cuda: launch default F16 matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeF16MatvecDefaultBatch projects F16 weights against batch input vectors
// separated by stride elements, matching ds4_f16_matvec_half_input bit-identically.
func (r *cudaFoundation) encodeF16MatvecDefaultBatch(t cudaModelTensor, in cudamem.Slice[float32], stride int, out cudamem.Slice[float32], batch int) error {
	if t.typeID != 1 || len(t.dimensions) != 2 || t.device == 0 {
		return fmt.Errorf("cuda: default F16 matvec batch %q: invalid tensor", t.name)
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if cols == 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 || batch <= 0 || stride < int(cols) {
		return fmt.Errorf("cuda: default F16 matvec batch %q: invalid shape", t.name)
	}
	if uint64(in.Len()) < uint64(batch-1)*uint64(stride)+cols || uint64(out.Len()) < uint64(batch)*rows || rows > t.bytes/(cols*2) {
		return fmt.Errorf("cuda: default F16 matvec batch %q: invalid extent", t.name)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.f16DefaultBatchFn).Grid(uint32(rows), uint32(batch), 1).Block(256, 1, 1).On(r.stream()).Launch(
			t.device, in, out, uint32(cols), uint32(stride), uint32(rows), uint32(batch)); err != nil {
			return fmt.Errorf("cuda: launch default F16 matvec batch %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeF16PairOrdered follows the raw-F32-input single-token pair dispatch in
// pinned ds4_cuda.cu:16194. Thirty-two contiguous partials are summed serially.
func (r *cudaFoundation) encodeF16PairOrdered(a, b cudaModelTensor, in, outA, outB cudamem.Slice[float32]) error {
	if a.typeID != 1 || b.typeID != 1 || len(a.dimensions) != 2 || len(b.dimensions) != 2 || a.device == 0 || b.device == 0 {
		return fmt.Errorf("cuda: ordered F16 pair: invalid tensor")
	}
	cols, rows := a.dimensions[0], a.dimensions[1]
	if cols == 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 || b.dimensions[0] != cols || b.dimensions[1] != rows || uint64(in.Len()) != cols || uint64(outA.Len()) != rows || outB.Len() != outA.Len() || rows > a.bytes/(cols*2) || rows > b.bytes/(cols*2) {
		return fmt.Errorf("cuda: ordered F16 pair: invalid shape or extent")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		r.countWeightBytes("ds4_f16_pair_ordered_chunks", 2*rows*cols*2)
		if err := cudalaunch.Kernel(r.f16PairOrderedFn).Grid(uint32(rows), 1, 1).Block(32, 1, 1).On(r.stream()).Launch(a.device, b.device, in, outA, outB, uint32(cols)); err != nil {
			return fmt.Errorf("cuda: launch ordered F16 pair: %w", err)
		}
		return nil
	})
}

// narrowActivations converts n = batch*cols floats to half, reading rows that
// are srcStride elements apart and writing them packed.
func (r *cudaRuntime) narrowActivations(in cudamem.Slice[float32], half cudamem.Slice[uint16], cols, srcStride, batch int) error {
	if err := r.loadMathKernels(); err != nil {
		return err
	}
	grid := uint32(min((cols+255)/256, 64))
	if err := cudalaunch.Kernel(r.f32ToF16Fn).Grid(grid, uint32(batch), 1).Block(256, 1, 1).On(r.stream()).Launch(in, half, uint32(cols), uint32(srcStride)); err != nil {
		return fmt.Errorf("narrow activations: %w", err)
	}
	return nil
}

// encodeF16MatvecGEMM is encodeF16MatvecDefaultBatch through cuBLAS. The kernel
// it replaces rounds each activation to half before multiplying, so the operands
// here carry exactly the precision that one used; only the order the products
// are summed in differs.
//
// The kernel gives each (row, token) pair its own block, so every token re-reads
// the whole weight matrix from memory. At 2048 tokens that is the entire cost:
// one GEMM reads the weights once and reuses them across the batch.
//
// half is scratch of at least batch*cols elements; it is overwritten.
func (r *cudaRuntime) encodeF16MatvecGEMM(blas cublas.Handle, t cudaModelTensor, in cudamem.Slice[float32], stride int, half cudamem.Slice[uint16], out cudamem.Slice[float32], batch int) error {
	if blas == 0 {
		return errors.New("cuda: matvec GEMM: no cuBLAS handle")
	}
	if t.typeID != 1 || len(t.dimensions) != 2 || t.device == 0 || batch <= 0 {
		return fmt.Errorf("cuda: matvec GEMM %q: invalid tensor or batch %d", t.name, batch)
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if cols == 0 || cols > math.MaxInt32 || rows == 0 || rows > math.MaxInt32 || stride < int(cols) ||
		uint64(in.Len()) < uint64(batch-1)*uint64(stride)+cols || uint64(half.Len()) < uint64(batch)*cols ||
		uint64(out.Len()) < uint64(batch)*rows || rows > t.bytes/(cols*2) {
		return fmt.Errorf("cuda: matvec GEMM %q: invalid shape or extent", t.name)
	}
	return r.withContext(func() error {
		if err := r.narrowActivations(in, half, int(cols), stride, batch); err != nil {
			return fmt.Errorf("cuda: matvec GEMM %q: %w", t.name, err)
		}
		var pin runtime.Pinner
		defer pin.Unpin()
		alpha, beta := float32(1), float32(0)
		pin.Pin(&alpha)
		pin.Pin(&beta)
		if err := blas.GemmEx(cublas.OpT, cublas.OpN, int32(rows), int32(batch), int32(cols), cublas.Float32Scalar(alpha), driver.DevicePtr(t.device), cublas.R16F, int32(cols), driver.DevicePtr(half.Ptr()), cublas.R16F, int32(cols), cublas.Float32Scalar(beta), driver.DevicePtr(out.Ptr()), cublas.R32F, int32(rows),
			cublas.Compute32F, cublas.GemmDefault); err != nil {
			return fmt.Errorf("cuda: matvec GEMM %q: project %d rows: %w", t.name, rows, err)
		}
		return nil
	})
}

// encodeF16PairGEMM projects the same weight pair as encodeF16PairOrderedBatch
// through cuBLAS instead of the hand-written kernel. Weights are half and
// row-major (rows x cols), so in cuBLAS column-major terms they are a
// (cols x rows) matrix read transposed; the activations are (batch x cols)
// row-major, which is (cols x batch) untransposed; and the product lands
// column-major as (rows x batch), which is exactly the token-major, row-minor
// layout the caller already expects. Hence OpT, OpN, lda = ldb = cols,
// ldc = rows -- the same shape cudaHC.pre uses for its own projection.
//
// The activations are narrowed to half first. The hand-written kernel keeps
// them in float, so this is a real loss of precision on that operand, and it is
// the reason the two paths do not produce the same bits. It is also what the
// reference implementation does: its prefill projections run through library
// GEMMs on half inputs.
//
// half is scratch of at least batch*cols elements; it is overwritten.
func (r *cudaRuntime) encodeF16PairGEMM(blas cublas.Handle, a, b cudaModelTensor, in cudamem.Slice[float32], half cudamem.Slice[uint16], outA, outB cudamem.Slice[float32], batch int) error {
	if blas == 0 {
		return errors.New("cuda: pair GEMM: no cuBLAS handle")
	}
	if a.typeID != 1 || b.typeID != 1 || len(a.dimensions) != 2 || len(b.dimensions) != 2 || a.device == 0 || b.device == 0 || batch <= 0 {
		return fmt.Errorf("cuda: pair GEMM: invalid tensor or batch %d", batch)
	}
	cols, rows := a.dimensions[0], a.dimensions[1]
	if cols == 0 || cols > math.MaxInt32 || rows == 0 || rows > math.MaxInt32 || b.dimensions[0] != cols || b.dimensions[1] != rows ||
		uint64(in.Len()) != uint64(batch)*cols || uint64(half.Len()) < uint64(batch)*cols ||
		uint64(outA.Len()) != uint64(batch)*rows || outB.Len() != outA.Len() ||
		rows > a.bytes/(cols*2) || rows > b.bytes/(cols*2) {
		return fmt.Errorf("cuda: pair GEMM: invalid shape or extent")
	}
	return r.withContext(func() error {
		if err := r.narrowActivations(in, half, int(cols), int(cols), batch); err != nil {
			return fmt.Errorf("cuda: pair GEMM: %w", err)
		}
		// cuBLAS reads the scalars through these pointers while the call runs,
		// and the uintptr conversion hides them from the collector, so pin them.
		var pin runtime.Pinner
		defer pin.Unpin()
		alpha, beta := float32(1), float32(0)
		pin.Pin(&alpha)
		pin.Pin(&beta)
		for _, w := range []struct {
			t   cudaModelTensor
			out cudamem.Slice[float32]
		}{{a, outA}, {b, outB}} {
			if err := blas.GemmEx(cublas.OpT, cublas.OpN, int32(rows), int32(batch), int32(cols), cublas.Float32Scalar(alpha), driver.DevicePtr(w.t.device), cublas.R16F, int32(cols), driver.DevicePtr(half.Ptr()), cublas.R16F, int32(cols), cublas.Float32Scalar(beta), driver.DevicePtr(w.out.Ptr()), cublas.R32F, int32(rows),
				cublas.Compute32F, cublas.GemmDefault); err != nil {
				return fmt.Errorf("cuda: pair GEMM: project %d rows: %w", rows, err)
			}
		}
		return nil
	})
}

// cudaF16PairMaxBatch is the compile-time maximum batch width processed per thread block by
// ds4_f16_pair_ordered_chunks_batch in cuda_f16_default.cu. Larger batches are tiled across gridDim.y.
const cudaF16PairMaxBatch = 16

// encodeF16PairOrderedBatch projects batch token-major activation rows against
// one weight pair with a single launch, reading the weights once for all
// tokens. Each token keeps the chunk partition and summation order of
// encodeF16PairOrdered, so the outputs are bit-identical to one launch per
// token. Input is batch adjacent rows of cols elements; the outputs are batch
// adjacent rows of rows elements.
func (r *cudaFoundation) encodeF16PairOrderedBatch(a, b cudaModelTensor, in, outA, outB cudamem.Slice[float32], batch int) error {
	if a.typeID != 1 || b.typeID != 1 || len(a.dimensions) != 2 || len(b.dimensions) != 2 || a.device == 0 || b.device == 0 || batch <= 0 {
		return fmt.Errorf("cuda: batched ordered F16 pair: invalid tensor or batch %d", batch)
	}
	cols, rows := a.dimensions[0], a.dimensions[1]
	if cols == 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 || b.dimensions[0] != cols || b.dimensions[1] != rows ||
		uint64(in.Len()) != uint64(batch)*cols || uint64(outA.Len()) != uint64(batch)*rows || outB.Len() != outA.Len() ||
		rows > a.bytes/(cols*2) || rows > b.bytes/(cols*2) {
		return fmt.Errorf("cuda: batched ordered F16 pair: invalid shape or extent")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		gridRows := uint32(rows)
		gridBatches := (uint32(batch) + cudaF16PairMaxBatch - 1) / cudaF16PairMaxBatch
		if err := cudalaunch.Kernel(r.f16PairOrderedBatchFn).Grid(gridRows, gridBatches, 1).Block(256, 1, 1).On(r.stream()).Launch(
			a.device, b.device, in, outA, outB, uint32(cols), uint32(batch)); err != nil {
			return fmt.Errorf("cuda: launch batched ordered F16 pair: %w", err)
		}
		return nil
	})
}
