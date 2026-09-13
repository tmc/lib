//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_q80_layout.cu
var cudaQ80Layout string

//go:embed cuda_q80_quant.cu
var cudaQ80QuantBody string

var cudaQ80QuantSource = cudaQ80Layout + cudaQ80QuantBody

//go:embed cuda_q80_matvec.cu
var cudaQ80MatvecBody string

// cudaQ80Source follows antirez/ds4 b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd
// ds4_cuda.cu:5037 (activation quantization) and :5152 (single-token matvec).
// Activations retain float32 scales; GGUF weight scales remain binary16.
var cudaQ80Source = cudaQ80QuantSource + cudaQ80MatvecBody

// encodeQuantQ80 appends the pinned CUDA Q8_0 activation quantizer. Each
// 32-element block occupies 36 bytes: one float32 scale and 32 signed bytes.
// The isolated module uses the pinned NVCC fast-math quantizer semantics; the
// matvec and other model kernels retain their existing compiler options.
func (r *cudaFoundation) encodeQuantQ80(in cudamem.Slice[float32], out cudamem.Slice[byte], cols int) error {
	if cols <= 0 || cols%32 != 0 || in.Len() == 0 || in.Len()%cols != 0 || uint64(in.Len()/32) > math.MaxInt32 || uint64(out.Len()) != uint64(in.Len()/32)*36 {
		return fmt.Errorf("cuda: Q8_0 activation quantization: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadQ80Quantizer(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.quantQ80Fn).Grid(uint32(in.Len()/32)).Block(32).On(r.device.DefaultStream()).Launch(in, out); err != nil {
			return fmt.Errorf("cuda: launch Q8_0 activation quantization: %w", err)
		}
		return nil
	})
}

// encodeQ80MatvecQuantized projects Q8_0 weights using quantized activations.
func (r *cudaFoundation) encodeQ80MatvecQuantized(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
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
		gridRows := (uint32(rows) + 7) / 8
		if err := cudalaunch.Kernel(r.q80QuantizedFn).Grid(gridRows).Block(256).On(r.device.DefaultStream()).Launch(t.device, in, out, uint32(cols/32), uint32(rows)); err != nil {
			return fmt.Errorf("cuda: launch quantized Q8_0 matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeQ80MatvecQuantizedBlockDiag projects block-diagonal Q8_0 weights using quantized activations.
func (r *cudaFoundation) encodeQ80MatvecQuantizedBlockDiag(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32], diag int) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.device == 0 || diag <= 0 {
		return fmt.Errorf("cuda: quantized Q8_0 block-diagonal matvec %q: invalid tensor", t.name)
	}
	cols, totalRows := t.dimensions[0], t.dimensions[1]
	if totalRows%uint64(diag) != 0 {
		return fmt.Errorf("cuda: quantized Q8_0 block-diagonal matvec %q: total rows not divisible by diag", t.name)
	}
	rows := totalRows / uint64(diag)
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 ||
		uint64(in.Len()) != (cols/32*36)*uint64(diag) || uint64(out.Len()) != totalRows || totalRows > t.bytes/(cols/32*34) {
		return fmt.Errorf("cuda: quantized Q8_0 block-diagonal matvec %q: invalid shape or extent", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		gridRows := (uint32(rows) + 7) / 8
		if err := cudalaunch.Kernel(r.q80QuantizedBlockDiagFn).Grid(gridRows, uint32(diag)).Block(256).On(r.device.DefaultStream()).Launch(t.device, in, out, uint32(cols/32), uint32(rows)); err != nil {
			return fmt.Errorf("cuda: launch quantized Q8_0 block-diagonal matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// loadQ80Quantizer is called with the device context current. The module is
// owned immediately after load, but is usable only after function lookup. A
// failed cleanup retains the module for a later lookup or Close attempt.
func (r *cudaFoundation) loadQ80Quantizer() error {
	if r.quantQ80Mod != 0 && r.quantQ80Fn != 0 {
		return nil
	}
	if r.quantQ80Mod == 0 {
		compiled, err := nvrtcx.Compile(cudaQ80QuantSource,
			nvrtcx.WithName("ds4_quant_q8_0.cu"),
			nvrtcx.WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", r.major, r.minor)),
			nvrtcx.WithFlag("--generate-line-info"),
			nvrtcx.WithFlag("--use_fast_math"),
		)
		if err != nil {
			return fmt.Errorf("cuda: compile Q8_0 quantizer: %w", err)
		}
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load Q8_0 quantizer: %w", err)
		}
		r.quantQ80Mod = mod
	}
	fn, err := r.quantQ80Mod.GetFunction("ds4_quant_q8_0")
	if err != nil {
		unloadErr := r.quantQ80Mod.Unload()
		if unloadErr == nil {
			r.quantQ80Mod, r.quantQ80Fn = 0, 0
		}
		return errors.Join(fmt.Errorf("cuda: find Q8_0 quantizer: %w", err), unloadErr)
	}
	r.quantQ80Fn = fn
	return nil
}
