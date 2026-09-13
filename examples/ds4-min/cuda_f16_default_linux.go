//go:build linux

package main

import (
	_ "embed"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaF16DefaultSource reproduces the binary16 activation boundary in the
// default single-token cuBLAS dispatch in pinned antirez/ds4 b0a147a,
// ds4_cuda.cu:15887-15927. Float64 accumulation avoids cancellation loss; it is not cuBLAS's implementation.
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.f16DefaultFn).Grid(uint32(rows)).Block(256).On(r.stream()).Launch(t.device, in, out, uint32(cols)); err != nil {
			return fmt.Errorf("cuda: launch default F16 matvec %q: %w", t.name, err)
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.f16PairOrderedFn).Grid(uint32(rows)).Block(128).On(r.stream()).Launch(a.device, b.device, in, outA, outB, uint32(cols)); err != nil {
			return fmt.Errorf("cuda: launch ordered F16 pair: %w", err)
		}
		return nil
	})
}
