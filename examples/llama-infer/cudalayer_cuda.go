//go:build !darwin && !tpu

package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalayer"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudalayerEnabled reports whether batched decode should run its per-layer
// projections through x/cudalayer's fused kernels. Off unless
// LLAMA_INFER_CUDALAYER=1.
func cudalayerEnabled() bool { return os.Getenv("LLAMA_INFER_CUDALAYER") == "1" }

// cudalayerThreads is the block width every cudalayer kernel is launched
// with. ProjectionResidualRMSNorm reduces through blockDim floats of shared
// memory, so the width fixes that request.
const cudalayerThreads = 256

// cudalayerRuntime runs one decoder layer of a batched decode step through
// x/cudalayer's four kernels instead of forward.cu's batch kernels and the
// cublasLt gate/up plans: RMSNormQKV replaces BatchRMSNorm and three
// BatchLinearGEMVs; ProjectionResidualRMSNorm replaces the output
// BatchLinearGEMV, BatchAddResidual and the post BatchRMSNorm; GateUpSiLU
// replaces two cublasLt GEMMs and BatchSwiGLU; DownResidual replaces the
// down BatchLinearGEMV and BatchAddResidual. Attention between them is the
// unchanged per-row path.
//
// UNQUALIFIED. Nothing here has run on a device. The kernels compile under
// x/cudalayer's own test on a CUDA host; nothing has checked that a step
// through them matches a step through the dense path, which is the test
// that would qualify this: TestPagedKVMatchesDense's shape, for this path.
//
// Do not assume this is faster. The cudalayer kernels are one block per
// token with a plain inner loop over the input dimension per output element;
// forward.cu's GEMVs reduce across a warp and the prefill/gate-up GEMMs run
// on cublasLt. Fewer launches, more work per launch. Which side wins for a
// given batch and model is the measurement that has not been made, and the
// package record said as much before this was written.
//
// The kernels only read float32 weights and take no bias, so the runtime
// applies only when the model was uploaded as WeightF32 and the layer has no
// QKV bias; the caller falls back to the dense path otherwise, per layer.
//
// x/cudalayer addresses tokens through a row table with generation stamps so
// a serving scheduler can retire rows without a device sync. This example's
// batch rows are already dense and validated on the host, so the table is
// the identity, every generation is 1, and row_status is read back after the
// step's existing synchronize: a nonzero status means a kernel saw an
// invalid row, which under the identity table means a bug here, not a stale
// slot.
type cudalayerRuntime struct {
	mod      *cuda.Module
	normQKV  *cuda.Kernel
	projNorm *cuda.Kernel
	gateUp   *cuda.Kernel
	down     *cuda.Kernel

	tokenRows cudamem.Slice[int32]
	rowGen    cudamem.Slice[uint64]
	tokenGen  cudamem.Slice[uint64]
	rowStatus cudamem.Slice[int32]
	hostRows  []int32

	capacity int
	arena    *cudamem.Arena
}

// newCudalayerRuntime compiles the kernels for dev and allocates the row
// tables for capacity rows.
func newCudalayerRuntime(dev *cuda.Device, tracker *cudamem.Tracker, capacity int) (_ *cudalayerRuntime, resultErr error) {
	if dev == nil || capacity <= 0 {
		return nil, errors.New("cudalayer runtime: invalid configuration")
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("cudalayer runtime: compute capability: %w", err)
	}
	compiled, err := cudalayer.Compile(major, minor)
	if err != nil {
		return nil, fmt.Errorf("cudalayer runtime: %w", err)
	}
	mod, err := loadCUDAProgram(dev, compiled)
	if err != nil {
		return nil, fmt.Errorf("cudalayer runtime: %w", err)
	}
	r := &cudalayerRuntime{mod: mod, capacity: capacity, arena: cudamem.NewTrackedArena(dev, tracker)}
	defer func() {
		if resultErr != nil {
			_ = r.close()
		}
	}()
	for _, k := range []struct {
		name string
		dst  **cuda.Kernel
	}{
		{"RMSNormQKV", &r.normQKV},
		{"ProjectionResidualRMSNorm", &r.projNorm},
		{"GateUpSiLU", &r.gateUp},
		{"DownResidual", &r.down},
	} {
		if *k.dst, err = mod.Function(k.name); err != nil {
			return nil, fmt.Errorf("cudalayer runtime: %w", err)
		}
	}

	r.tokenRows = cudamem.AllocIn[int32](r.arena, "cudalayer-token-rows", capacity)
	r.rowGen = cudamem.AllocIn[uint64](r.arena, "cudalayer-row-generations", capacity)
	r.tokenGen = cudamem.AllocIn[uint64](r.arena, "cudalayer-token-generations", capacity)
	r.rowStatus = cudamem.AllocIn[int32](r.arena, "cudalayer-row-status", capacity)
	if err := r.arena.Err(); err != nil {
		return nil, fmt.Errorf("cudalayer runtime: %w", err)
	}
	rows := make([]int32, capacity)
	gens := make([]uint64, capacity)
	for i := range rows {
		rows[i] = int32(i)
		gens[i] = 1
	}
	r.hostRows = make([]int32, capacity)
	if err := r.tokenRows.CopyFromHost(rows); err != nil {
		return nil, fmt.Errorf("cudalayer runtime: %w", err)
	}
	if err := r.rowGen.CopyFromHost(gens); err != nil {
		return nil, fmt.Errorf("cudalayer runtime: %w", err)
	}
	if err := r.tokenGen.CopyFromHost(gens); err != nil {
		return nil, fmt.Errorf("cudalayer runtime: %w", err)
	}
	if err := r.rowStatus.CopyFromHost(r.hostRows); err != nil {
		return nil, fmt.Errorf("cudalayer runtime: %w", err)
	}
	return r, nil
}

// applies reports whether this layer of a decode batch can run through the
// cudalayer kernels: float32 weights, no QKV bias, and a batch the row tables
// cover.
func (r *cudalayerRuntime) applies(w *GPULayerWeights, batch int, prefill bool) bool {
	if r == nil || w == nil || prefill || batch <= 0 || batch > r.capacity {
		return false
	}
	return w.WQ.DType() == WeightF32 && w.QBias.Len() == 0 && w.KBias.Len() == 0 && w.VBias.Len() == 0
}

// qkv runs the input norm and the Q, K and V projections for batch rows.
func (r *cudalayerRuntime) qkv(launcher func(*cuda.Kernel) *cudalaunch.Launcher, sp *GPUBatchScratchpad, w *GPULayerWeights, cfg *ModelConfig, batch int) error {
	dim := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
	return launcher(r.normQKV).Grid(uint32(batch)).Block(cudalayerThreads).
		Launch(sp.X, w.InputNorm, w.WQ, w.WK, w.WV,
			r.tokenRows, r.rowGen, r.tokenGen, r.rowStatus,
			sp.NormX, sp.Q, sp.K, sp.V,
			int32(dim), int32(qDim), int32(kvDim), int32(batch), int32(r.capacity), cfg.RmsNormEps)
}

// ffn runs everything after attention for batch rows: the output projection
// with its residual and post norm, gate/up with SiLU, and down with its
// residual. sp.X is updated in place; both residual kernels read only the
// element they write.
func (r *cudalayerRuntime) ffn(launcher func(*cuda.Kernel) *cudalaunch.Launcher, sp *GPUBatchScratchpad, w *GPULayerWeights, cfg *ModelConfig, batch int) error {
	dim := cfg.HiddenSize
	inter := cfg.IntermediateSize
	shared, err := cudalayer.ProjectionNormConfig{Dim: dim, Threads: cudalayerThreads, Epsilon: cfg.RmsNormEps}.SharedMemoryBytes()
	if err != nil {
		return fmt.Errorf("cudalayer: %w", err)
	}
	if err := launcher(r.projNorm).Grid(uint32(batch)).Block(cudalayerThreads).SharedMem(uint32(shared)).
		Launch(sp.AttnOut, sp.X, w.WO, w.PostNorm, r.tokenRows, r.rowStatus, sp.X, sp.NormX,
			int32(dim), int32(batch), int32(r.capacity), cfg.RmsNormEps); err != nil {
		return err
	}
	if err := launcher(r.gateUp).Grid(uint32(batch)).Block(cudalayerThreads).
		Launch(sp.NormX, w.WGate, w.WUp, r.tokenRows, r.rowStatus, sp.Gate, sp.Up, sp.FFNAct,
			int32(dim), int32(inter), int32(batch), int32(r.capacity)); err != nil {
		return err
	}
	return launcher(r.down).Grid(uint32(batch)).Block(cudalayerThreads).
		Launch(sp.FFNAct, sp.X, w.WDown, r.tokenRows, r.rowStatus, sp.X,
			int32(dim), int32(inter), int32(batch), int32(r.capacity))
}

// readStatus queues the row-status copy on stream; check reads it after the
// caller has synchronized.
func (r *cudalayerRuntime) readStatus(stream *cuda.Stream, batch int) error {
	return r.rowStatus.Slice(0, batch).CopyToHostAsync(r.hostRows[:batch], stream)
}

// check reports any row a kernel marked invalid during the step, and clears
// the marks so the next step starts clean.
func (r *cudalayerRuntime) check(batch int) error {
	var bad []int
	for i, s := range r.hostRows[:batch] {
		if s != 0 {
			bad = append(bad, i)
		}
	}
	if len(bad) == 0 {
		return nil
	}
	clear(r.hostRows[:batch])
	if err := r.rowStatus.Slice(0, batch).CopyFromHost(r.hostRows[:batch]); err != nil {
		return fmt.Errorf("cudalayer: rows %v marked invalid, and clearing the marks failed: %w", bad, err)
	}
	return fmt.Errorf("cudalayer: kernels marked rows %v invalid; outputs for those rows are NaN", bad)
}

// close releases the module and the row tables. Safe on nil and repeated.
func (r *cudalayerRuntime) close() error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.arena != nil {
		errs = append(errs, r.arena.Free())
		r.arena = nil
	}
	if r.mod != nil {
		errs = append(errs, r.mod.Close())
		r.mod = nil
	}
	return errors.Join(errs...)
}
