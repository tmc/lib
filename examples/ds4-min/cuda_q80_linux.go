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

// cudaQ80Source follows antirez/ds4 b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd:
// ds4_cuda.cu:5037 quantize_q8_0_f32_kernel (activation quantization) and :5152
// matmul_q8_0_preq_warp8_kernel, specialized here to a single token.
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
	return r.withContext(func() error {
		if err := r.loadQ80Quantizer(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.quantQ80Fn).Grid(uint32(in.Len()/32), 1, 1).Block(32, 1, 1).On(r.stream()).Launch(in, out); err != nil {
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
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		gridRows := (uint32(rows) + 7) / 8
		r.countWeightBytes("ds4_q8_0_matvec_quantized", rows*cols/32*34)
		if err := cudalaunch.Kernel(r.q80QuantizedFn).Grid(gridRows, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(t.device, in, out, uint32(cols/32), uint32(rows)); err != nil {
			return fmt.Errorf("cuda: launch quantized Q8_0 matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeQ80MatvecQuantizedAligned is encodeQ80MatvecQuantized reading the
// weights out of the dense artifact's aligned dq/qs planes in device memory
// instead of the model mapping. See ds4_q8_0_matvec_q8_1_aligned in cuda_q81.cu
// for why the two kernels are bit-identical and why the per-kernel gate, not
// the argument, is what settles it.
func (r *cudaFoundation) encodeQ80MatvecQuantizedAligned(aligned cudamem.Slice[byte], cols, rows uint64, in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 {
		return fmt.Errorf("cuda: aligned quantized Q8_0 matvec: invalid shape %dx%d", cols, rows)
	}
	if uint64(in.Len()) != cols/32*36 || uint64(out.Len()) != rows {
		return fmt.Errorf("cuda: aligned quantized Q8_0 matvec: invalid extent")
	}
	want := denseQ8AlignedBytes(int(rows), int(cols))
	if want == 0 || aligned.Len() != want {
		return fmt.Errorf("cuda: aligned quantized Q8_0 matvec: artifact is %d bytes, want %d", aligned.Len(), want)
	}
	nblk := rows * (cols / 32)
	dqBytes := int((nblk*2 + 63) &^ 63)
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		gridRows := (uint32(rows) + 7) / 8
		r.countWeightBytes("ds4_q8_0_matvec_quantized_aligned", rows*cols/32*34)
		if err := cudalaunch.Kernel(r.q80QuantizedAlignedFn).Grid(gridRows, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			aligned.Slice(0, dqBytes), aligned.Slice(dqBytes, aligned.Len()), in, out,
			uint32(cols/32), uint32(rows)); err != nil {
			return fmt.Errorf("cuda: launch aligned quantized Q8_0 matvec: %w", err)
		}
		return nil
	})
}

// encodeQ80MatvecQuantizedBlockDiagAligned is the same fetch change for the
// block-diagonal projection. buildDenseArtifact repacks each diagonal block
// into its own aligned plane laid end to end, so the kernel takes the plane
// stride rather than a row offset.
func (r *cudaFoundation) encodeQ80MatvecQuantizedBlockDiagAligned(aligned cudamem.Slice[byte], cols, totalRows uint64, in cudamem.Slice[byte], out cudamem.Slice[float32], diag int) error {
	if diag <= 0 || totalRows == 0 || totalRows%uint64(diag) != 0 {
		return fmt.Errorf("cuda: aligned block-diagonal Q8_0 matvec: %d rows over %d diagonals", totalRows, diag)
	}
	rows := totalRows / uint64(diag)
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows > math.MaxInt32 {
		return fmt.Errorf("cuda: aligned block-diagonal Q8_0 matvec: invalid shape %dx%d", cols, rows)
	}
	if uint64(in.Len()) != cols/32*36*uint64(diag) || uint64(out.Len()) != totalRows {
		return fmt.Errorf("cuda: aligned block-diagonal Q8_0 matvec: invalid extent")
	}
	plane := denseQ8AlignedBytes(int(rows), int(cols))
	if plane == 0 || aligned.Len() != plane*diag {
		return fmt.Errorf("cuda: aligned block-diagonal Q8_0 matvec: artifact is %d bytes, want %d", aligned.Len(), plane*diag)
	}
	nblk := rows * (cols / 32)
	dqBytes := (nblk*2 + 63) &^ 63
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		gridRows := (uint32(rows) + 7) / 8
		r.countWeightBytes("ds4_q8_0_matvec_quantized_blockdiag_aligned", totalRows*cols/32*34)
		if err := cudalaunch.Kernel(r.q80QuantizedBlockDiagAlignedFn).Grid(gridRows, uint32(diag), 1).Block(256, 1, 1).On(r.stream()).Launch(
			aligned, in, out, uint32(cols/32), uint32(rows),
			uint64(plane), dqBytes); err != nil {
			return fmt.Errorf("cuda: launch aligned block-diagonal Q8_0 matvec: %w", err)
		}
		return nil
	})
}

// cudaQ80MaxBatch is the compile-time maximum batch width processed per thread block by
// ds4_q8_0_matvec_quantized_batch in cuda_q80_matvec.cu. Larger batches are tiled across gridDim.y.
const cudaQ80MaxBatch = 32

// cudaQ80SmallMaxBatch mirrors the accumulator tile in
// ds4_q8_0_matvec_quantized_small. It is the maximum speculative verifier
// width, not a general prefill limit.
const cudaQ80SmallMaxBatch = 8

// encodeQ80MatvecQuantizedBatch projects batch adjacent Q8_0 activation rows
// against one weight matrix with a single launch. Input and output are
// token-major: in holds batch consecutive quantized rows of cols/32 blocks and
// out holds batch consecutive rows of rows floats.
func (r *cudaFoundation) encodeQ80MatvecQuantizedBatch(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32], batch int) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.device == 0 || batch <= 0 {
		return fmt.Errorf("cuda: batched quantized Q8_0 matvec %q: invalid tensor or batch %d", t.name, batch)
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 ||
		uint64(in.Len()) != uint64(batch)*cols/32*36 || uint64(out.Len()) != uint64(batch)*rows || rows > t.bytes/(cols/32*34) {
		return fmt.Errorf("cuda: batched quantized Q8_0 matvec %q: invalid shape or extent", t.name)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if batch <= cudaQ80SmallMaxBatch {
			gridRows := (uint32(rows) + 7) / 8
			if err := cudalaunch.Kernel(r.q80QuantizedSmallFn).Grid(gridRows, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
				t.device, in, out, uint32(cols/32), uint32(rows), uint32(batch)); err != nil {
				return fmt.Errorf("cuda: launch small batched quantized Q8_0 matvec %q: %w", t.name, err)
			}
			return nil
		}
		gridRows := (uint32(rows) + 7) / 8
		gridBatches := (uint32(batch) + cudaQ80MaxBatch - 1) / cudaQ80MaxBatch
		if err := cudalaunch.Kernel(r.q80QuantizedBatchFn).Grid(gridRows, gridBatches, 1).Block(256, 1, 1).On(r.stream()).Launch(
			t.device, in, out, uint32(cols/32), uint32(rows), uint32(batch)); err != nil {
			return fmt.Errorf("cuda: launch batched quantized Q8_0 matvec %q: %w", t.name, err)
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
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		gridRows := (uint32(rows) + 7) / 8
		// Every diagonal block reads its own rows, so the launch reads the
		// whole plane, not the rows one block covers.
		r.countWeightBytes("ds4_q8_0_matvec_quantized_blockdiag", totalRows*cols/32*34)
		if err := cudalaunch.Kernel(r.q80QuantizedBlockDiagFn).Grid(gridRows, uint32(diag), 1).Block(256, 1, 1).On(r.stream()).Launch(t.device, in, out, uint32(cols/32), uint32(rows)); err != nil {
			return fmt.Errorf("cuda: launch quantized Q8_0 block-diagonal matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// cudaQ80BlockDiagMaxBatch is the compile-time batch tile width of
// ds4_q8_0_matvec_quantized_blockdiag_batch in cuda_q80_matvec.cu.
// Larger batches are tiled across gridDim.z.
const cudaQ80BlockDiagMaxBatch = 8

// encodeQ80MatvecQuantizedBlockDiagBatch projects batch tokens through the same
// block-diagonal Q8_0 weights with a single launch. The weight bytes a warp
// reads depend only on the diagonal and the row, so the batch streams the weight
// tensor once per batch tile instead of once per token.
//
// Input and output are token major: in holds batch consecutive activation
// groups of diag*cols/32 blocks, out holds batch consecutive groups of
// diag*rows floats, and each group has the layout encodeQ80MatvecQuantizedBlockDiag
// uses for one token.
func (r *cudaFoundation) encodeQ80MatvecQuantizedBlockDiagBatch(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32], diag, batch int) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.device == 0 || diag <= 0 || diag > math.MaxUint16 ||
		batch <= 0 || batch > math.MaxUint16*cudaQ80BlockDiagMaxBatch {
		return fmt.Errorf("cuda: batched quantized Q8_0 block-diagonal matvec %q: invalid tensor, diagonal count, or batch %d", t.name, batch)
	}
	cols, totalRows := t.dimensions[0], t.dimensions[1]
	if totalRows%uint64(diag) != 0 {
		return fmt.Errorf("cuda: batched quantized Q8_0 block-diagonal matvec %q: total rows not divisible by diag", t.name)
	}
	rows := totalRows / uint64(diag)
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 ||
		in.Len()%batch != 0 || uint64(in.Len()/batch) != (cols/32*36)*uint64(diag) ||
		out.Len()%batch != 0 || uint64(out.Len()/batch) != totalRows ||
		totalRows > t.bytes/(cols/32*34) {
		return fmt.Errorf("cuda: batched quantized Q8_0 block-diagonal matvec %q: invalid shape or extent", t.name)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		gridRows := (uint32(rows) + 7) / 8
		gridBatches := (uint32(batch) + cudaQ80BlockDiagMaxBatch - 1) / cudaQ80BlockDiagMaxBatch
		if err := cudalaunch.Kernel(r.q80BlockDiagBatchFn).Grid(gridRows, uint32(diag), gridBatches).Block(256, 1, 1).On(r.stream()).Launch(
			t.device, in, out, uint32(cols/32), uint32(rows), uint32(diag), uint32(batch)); err != nil {
			return fmt.Errorf("cuda: launch batched quantized Q8_0 block-diagonal matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// loadQ80Quantizer is called with the device context current. The module is
// owned immediately after load, but is usable only after function lookup. A
// failed cleanup retains the module for a later lookup or Close attempt.
func (r *cudaFoundation) loadQ80Quantizer() error {
	if err := r.checkOpen(); err != nil {
		return err
	}
	if r.quantQ80Mod != 0 && r.quantQ80Fn != 0 {
		return nil
	}
	if r.quantQ80Mod == 0 {
		// --use_fast_math here is load-bearing and asymmetric: this is the
		// only module in ds4-min that carries it, and ds4_quant_q8_0 produces
		// the q8_1 input every Q8_0 matvec reads. The flag is not decoration.
		// Measured through NVRTC 13.0 on this source family: it drives FCHK to
		// zero, rewrites every FFMA to FFMA.FTZ, and cuts MUFU.RCP, so the
		// emitted SASS flushes denormals and takes approximate division.
		// Moving it in either direction -- adding it to the other modules or
		// dropping it from this one -- moves the registered logits digests and
		// invalidates every gate that leans on them, so it is not a tidy-up.
		//
		// It also cannot simply be matched elsewhere. NVRTC 13.0 accepts
		// --ftz=true and friends and applies them to the PTX it returns, then
		// emits a CUBIN byte-identical to the unflagged one; --use_fast_math
		// is the only one of those flags that reaches SASS through NVRTC.
		// The C oracle compiles with nvcc, which honors all of them. The two
		// implementations therefore differ in which configurations they can
		// reach, not merely in which they chose.
		compiled, err := nvrtcx.Compile(cudaQ80QuantSource,
			nvrtcx.WithName(cudaSourceRoot+"ds4_quant_q8_0.cu"),
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
	fn, err := r.quantQ80Mod.Function("ds4_quant_q8_0")
	if err != nil {
		unloadErr := r.quantQ80Mod.Close()
		if unloadErr == nil {
			r.quantQ80Mod, r.quantQ80Fn = 0, 0
		}
		return errors.Join(fmt.Errorf("cuda: find Q8_0 quantizer: %w", err), unloadErr)
	}
	r.quantQ80Fn = fn
	return nil
}
