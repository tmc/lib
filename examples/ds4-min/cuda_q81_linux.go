//go:build linux

package main

import (
	_ "embed"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaQ81Source follows antirez/ds4 b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd
// cuda/mmq/quantize.cu:5 and cuda/mmq/ds4_mmq.cu:3950.
// Activation scales and block sums are binary16; the dot does not consume sums.
//go:embed cuda_q81.cu
var cudaQ81Source string

// encodeQuantQ81 appends the pinned CUDA Q8_1 activation quantizer. Each
// 32-element block occupies 36 bytes: two binary16 values and 32 signed bytes.
func (r *cudaFoundation) encodeQuantQ81(in cudamem.Slice[float32], out cudamem.Slice[byte], cols int) error {
	if cols <= 0 || cols%32 != 0 || in.Len() == 0 || in.Len()%cols != 0 || uint64(in.Len()/32) > math.MaxInt32 || uint64(out.Len()) != uint64(in.Len()/32)*36 {
		return fmt.Errorf("cuda: Q8_1 activation quantization: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.quantQ81Fn).Grid(uint32(in.Len()/32)).Block(32).On(r.stream()).Launch(in, out); err != nil {
			return fmt.Errorf("cuda: launch Q8_1 activation quantization: %w", err)
		}
		return nil
	})
}

// encodeQuantQ81Batch quantizes batch rows separated by stride elements into
// adjacent Q8_1 rows.
func (r *cudaFoundation) encodeQuantQ81Batch(in cudamem.Slice[float32], out cudamem.Slice[byte], cols, stride, batch int) error {
	if cols <= 0 || cols%32 != 0 || stride < cols || batch <= 0 ||
		uint64(batch) > math.MaxUint32 || uint64(cols/32) > math.MaxUint32 || uint64(stride) > math.MaxUint32 ||
		uint64(batch)*uint64(cols/32) > math.MaxUint32 ||
		uint64(in.Len()) != uint64(batch-1)*uint64(stride)+uint64(cols) ||
		uint64(out.Len()) != uint64(batch)*uint64(cols/32)*36 {
		return fmt.Errorf("cuda: batched Q8_1 activation quantization: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := uint32(cols / 32)
		if err := cudalaunch.Kernel(r.quantQ81BatchFn).Grid(blocks*uint32(batch)).Block(32).On(r.stream()).Launch(
			in, out, blocks, uint32(stride)); err != nil {
			return fmt.Errorf("cuda: launch batched Q8_1 activation quantization: %w", err)
		}
		return nil
	})
}

// encodeQ80MatvecQ81 projects Q8_0 weights using quantized activations.
func (r *cudaFoundation) encodeQ80MatvecQ81(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.device == 0 {
		return fmt.Errorf("cuda: quantized Q8_0 matvec %q: invalid tensor", t.name)
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 || uint64(in.Len()) != cols/32*36 || uint64(out.Len()) != rows || rows > t.bytes/(cols/32*34) {
		return fmt.Errorf("cuda: quantized Q8_0 matvec %q: invalid shape or extent", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q81MatvecFn).Grid(uint32(rows)).Block(32).On(r.stream()).Launch(t.device, in, out, uint32(cols/32)); err != nil {
			return fmt.Errorf("cuda: launch quantized Q8_0 matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeQ80MatvecQ81Batch projects batch adjacent Q8_1 activation rows with
// one two-dimensional launch. Input and output are token-major.
func (r *cudaFoundation) encodeQ80MatvecQ81Batch(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32], batch int) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.device == 0 || batch <= 0 || uint64(batch) > math.MaxUint32 {
		return fmt.Errorf("cuda: batched quantized Q8_0 matvec %q: invalid tensor or batch", t.name)
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 ||
		uint64(in.Len()) != uint64(batch)*cols/32*36 || uint64(out.Len()) != uint64(batch)*rows || rows > t.bytes/(cols/32*34) {
		return fmt.Errorf("cuda: batched quantized Q8_0 matvec %q: invalid shape or extent", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q81MatvecBatchFn).Grid(uint32(rows), uint32(batch)).Block(32).On(r.stream()).Launch(
			t.device, in, out, uint32(cols/32), uint32(rows), uint32(batch)); err != nil {
			return fmt.Errorf("cuda: launch batched quantized Q8_0 matvec %q: %w", t.name, err)
		}
		return nil
	})
}
