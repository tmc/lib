package cudaoptim

import (
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ChunkSize is the number of elements one thread block of a step handles.
// Tensors are split into chunks of this size, and the gradient norm is summed
// one chunk at a time, so it fixes the reduction order.
const ChunkSize = 1 << 16

// A Tensor is one trainable tensor: its parameters, their gradient, and the
// AdamW first and second moment estimates. All four slices must have the
// same length. M and V must start zeroed.
type Tensor struct {
	Param, Grad, M, V cudamem.Slice[float32]

	// NoDecay excludes the tensor from its group's weight decay, as is usual
	// for biases and normalization weights.
	NoDecay bool
}

// A Group is a set of tensors sharing a learning-rate scale and a weight
// decay. The learning rate a group sees is the optimizer's current
// learning rate, [State].LR, times LRScale.
type Group struct {
	Tensors     []Tensor
	LRScale     float32
	WeightDecay float32
}

// Config holds the hyperparameters shared by every group.
type Config struct {
	LR    float32 // initial learning rate; see [AdamW.SetLR]
	Beta1 float32
	Beta2 float32
	Eps   float32

	// MaxNorm clips the gradients, before the update, so that their global
	// L2 norm is at most MaxNorm, as torch.nn.utils.clip_grad_norm_ does.
	// Zero disables clipping.
	MaxNorm float32
}

// State is the optimizer state kept in device memory. Kernels read the
// learning rate from it and write the other fields, so a step captured in a
// CUDA graph uses the current values each time it is replayed.
type State struct {
	// Step counts the updates applied.
	Step int32

	// Nonfinite is nonzero once a gradient norm has been NaN or infinite,
	// which is the case whenever any gradient element is. It stays set: every
	// later [AdamW.Step] leaves the parameters, the moments and Step
	// unchanged until [AdamW.ClearNonfinite].
	Nonfinite int32

	// LR is the learning rate the next update uses.
	LR float32

	// Norm is the global L2 norm of the gradients at the last step, before
	// clipping, and ClipScale the factor the gradients were multiplied by.
	Norm      float32
	ClipScale float32

	bias1        float32
	invSqrtBias2 float32
	_            int32
}

// tensorEntry mirrors struct cudaoptim_tensor.
type tensorEntry struct {
	p, g, m, v  driver.DevicePtr
	n           int64
	lrScale, wd float32
}

// chunkEntry mirrors struct cudaoptim_chunk.
type chunkEntry struct {
	tensor int32
	_      int32
	start  int64
}

// AdamW updates a fixed list of tensors with the AdamW algorithm of
// torch.optim.AdamW, after clipping the gradients by their global norm.
//
// A step is three kernel launches over pointer tables built once by
// [NewAdamW], however many tensors there are. It makes no allocations and
// no host synchronization, and everything it reads that changes from step to
// step lives in device memory, so it can be captured in a CUDA graph and
// replayed. Its results are bitwise reproducible.
type AdamW struct {
	cfg     Config
	mod     *cuda.Module
	norm    *cuda.Kernel
	final   *cuda.Kernel
	update  *cuda.Kernel
	tensors cudamem.Slice[tensorEntry]
	chunks  cudamem.Slice[chunkEntry]
	partial cudamem.Slice[float64]
	state   cudamem.Slice[State]
	nchunks int
}

// NewAdamW compiles the optimizer's kernels for dev and uploads the pointer
// tables for groups. The tensors must stay allocated for the life of the
// optimizer.
func NewAdamW(dev *cuda.Device, groups []Group, cfg Config) (_ *AdamW, err error) {
	entries, chunks, err := layout(groups)
	if err != nil {
		return nil, err
	}
	cm, err := nvrtcx.Compile(kernelSource,
		nvrtcx.WithName("cudaoptim.cu"),
		nvrtcx.WithDeviceArch(dev),
		nvrtcx.WithHeader("cudablock.cuh", cudablock.Header),
		nvrtcx.WithToolkitIncludes())
	if err != nil {
		return nil, fmt.Errorf("cudaoptim: compile kernels: %w", err)
	}
	o := &AdamW{cfg: cfg, nchunks: len(chunks)}
	defer func() {
		if err != nil {
			o.Close()
		}
	}()
	if o.mod, err = cm.LoadModule(dev); err != nil {
		return nil, fmt.Errorf("cudaoptim: load kernels: %w", err)
	}
	for _, k := range []struct {
		dst  **cuda.Kernel
		name string
	}{{&o.norm, "cudaoptim_norm_partials"}, {&o.final, "cudaoptim_finalize"}, {&o.update, "cudaoptim_adamw"}} {
		if *k.dst, err = o.mod.Function(k.name); err != nil {
			return nil, fmt.Errorf("cudaoptim: %w", err)
		}
	}
	a := cudamem.NewArena(dev)
	o.tensors = cudamem.AllocIn[tensorEntry](a, "tensor table", max(len(entries), 1))
	o.chunks = cudamem.AllocIn[chunkEntry](a, "chunk table", max(len(chunks), 1))
	o.partial = cudamem.AllocIn[float64](a, "norm partials", max(len(chunks), 1))
	o.state = cudamem.AllocIn[State](a, "state", 1)
	if err := a.Err(); err != nil {
		a.Free()
		return nil, err
	}
	if len(entries) > 0 {
		if err := o.tensors.CopyFromHost(entries); err != nil {
			return nil, fmt.Errorf("cudaoptim: upload tensor table: %w", err)
		}
		if err := o.chunks.CopyFromHost(chunks); err != nil {
			return nil, fmt.Errorf("cudaoptim: upload chunk table: %w", err)
		}
	}
	init := []State{{LR: cfg.LR, ClipScale: 1, bias1: 1, invSqrtBias2: 1}}
	if err := o.state.CopyFromHost(init); err != nil {
		return nil, fmt.Errorf("cudaoptim: upload state: %w", err)
	}
	return o, nil
}

// layout flattens groups into the tensor table and splits every tensor into
// chunks of at most ChunkSize elements.
func layout(groups []Group) ([]tensorEntry, []chunkEntry, error) {
	var entries []tensorEntry
	for gi, g := range groups {
		for ti, t := range g.Tensors {
			n := t.Param.Len()
			if t.Grad.Len() != n || t.M.Len() != n || t.V.Len() != n {
				return nil, nil, fmt.Errorf("cudaoptim: group %d tensor %d: lengths param %d, grad %d, m %d, v %d differ",
					gi, ti, n, t.Grad.Len(), t.M.Len(), t.V.Len())
			}
			if n == 0 {
				continue
			}
			if t.Param.Ptr() == 0 || t.Grad.Ptr() == 0 || t.M.Ptr() == 0 || t.V.Ptr() == 0 {
				return nil, nil, fmt.Errorf("cudaoptim: group %d tensor %d: freed or unallocated slice", gi, ti)
			}
			wd := g.WeightDecay
			if t.NoDecay {
				wd = 0
			}
			if len(entries) == math.MaxInt32 {
				return nil, nil, errors.New("cudaoptim: too many tensors")
			}
			entries = append(entries, tensorEntry{
				p: t.Param.Ptr(), g: t.Grad.Ptr(), m: t.M.Ptr(), v: t.V.Ptr(),
				n: int64(n), lrScale: g.LRScale, wd: wd,
			})
		}
	}
	chunks, err := splitChunks(entries, ChunkSize)
	return entries, chunks, err
}

// splitChunks returns the chunks of size elements that cover every tensor.
func splitChunks(entries []tensorEntry, size int64) ([]chunkEntry, error) {
	var chunks []chunkEntry
	for i, e := range entries {
		for start := int64(0); start < e.n; start += size {
			chunks = append(chunks, chunkEntry{tensor: int32(i), start: start})
		}
	}
	if len(chunks) > math.MaxInt32 {
		return nil, errors.New("cudaoptim: too many chunks")
	}
	return chunks, nil
}

// State returns the device-resident optimizer state. It holds one element.
func (o *AdamW) State() cudamem.Slice[State] {
	return o.state
}

// ReadState copies the optimizer state to the host. It synchronizes with
// the device, so work queued on other streams must be complete.
func (o *AdamW) ReadState() (State, error) {
	s := make([]State, 1)
	if err := o.state.CopyToHost(s); err != nil {
		return State{}, fmt.Errorf("cudaoptim: read state: %w", err)
	}
	return s[0], nil
}

// SetLR enqueues on stream a write of lr to [State].LR. It is a memset node
// in a captured graph, so it records the value lr has now.
func (o *AdamW) SetLR(stream *cuda.Stream, lr float32) error {
	return o.setWord(stream, unsafe.Offsetof(State{}.LR), math.Float32bits(lr))
}

// ClearNonfinite enqueues on stream a reset of [State].Nonfinite, so that
// steps update the parameters again.
func (o *AdamW) ClearNonfinite(stream *cuda.Stream) error {
	return o.setWord(stream, unsafe.Offsetof(State{}.Nonfinite), 0)
}

func (o *AdamW) setWord(stream *cuda.Stream, offset uintptr, word uint32) error {
	ptr := o.state.Ptr() + driver.DevicePtr(offset)
	return stream.WithHandle(func(h driver.Stream) error {
		return stream.Device().WithContext(func() error {
			if err := ptr.MemsetD32Async(word, 1, h); err != nil {
				return fmt.Errorf("cudaoptim: write state: %w", err)
			}
			return nil
		})
	})
}

// GradNorm enqueues on stream the computation of the gradients' global norm,
// clip scale and non-finite flag into [State], without updating anything.
func (o *AdamW) GradNorm(stream *cuda.Stream) error {
	return o.gradNorm(stream, 0)
}

// Step enqueues one optimizer step on stream: the global gradient norm, the
// clip scale, and the AdamW update of every tensor using the clipped
// gradients. The gradients themselves are not modified. If the norm is not
// finite, the step sets [State].Nonfinite and changes nothing else.
func (o *AdamW) Step(stream *cuda.Stream) error {
	if err := o.gradNorm(stream, 1); err != nil {
		return err
	}
	if o.nchunks == 0 {
		return nil
	}
	tensors, chunks, state := o.tensors.Ptr(), o.chunks.Ptr(), o.state.Ptr()
	chunk := int64(ChunkSize)
	b1, b2, eps := o.cfg.Beta1, o.cfg.Beta2, o.cfg.Eps
	return o.launch(o.update, stream, o.nchunks,
		unsafe.Pointer(&tensors), unsafe.Pointer(&chunks), unsafe.Pointer(&chunk), unsafe.Pointer(&state),
		unsafe.Pointer(&b1), unsafe.Pointer(&b2), unsafe.Pointer(&eps))
}

func (o *AdamW) gradNorm(stream *cuda.Stream, advance int32) error {
	tensors, chunks, partial, state := o.tensors.Ptr(), o.chunks.Ptr(), o.partial.Ptr(), o.state.Ptr()
	if o.nchunks > 0 {
		chunk := int64(ChunkSize)
		if err := o.launch(o.norm, stream, o.nchunks,
			unsafe.Pointer(&tensors), unsafe.Pointer(&chunks), unsafe.Pointer(&chunk), unsafe.Pointer(&partial)); err != nil {
			return err
		}
	}
	n := int32(o.nchunks)
	maxNorm, b1, b2 := o.cfg.MaxNorm, o.cfg.Beta1, o.cfg.Beta2
	return o.launch(o.final, stream, 1,
		unsafe.Pointer(&partial), unsafe.Pointer(&n), unsafe.Pointer(&maxNorm),
		unsafe.Pointer(&b1), unsafe.Pointer(&b2), unsafe.Pointer(&advance), unsafe.Pointer(&state))
}

func (o *AdamW) launch(k *cuda.Kernel, stream *cuda.Stream, blocks int, args ...unsafe.Pointer) error {
	err := stream.WithHandle(func(h driver.Stream) error {
		return k.Launch([3]uint32{uint32(blocks), 1, 1}, [3]uint32{blockThreads, 1, 1}, 0, h, args)
	})
	if err != nil {
		return fmt.Errorf("cudaoptim: %w", err)
	}
	return nil
}

// Close frees the optimizer's tables and unloads its kernels. Work it
// enqueued must be complete.
func (o *AdamW) Close() error {
	var errs []error
	for _, free := range []func() error{o.tensors.Free, o.chunks.Free, o.partial.Free, o.state.Free} {
		if err := free(); err != nil {
			errs = append(errs, err)
		}
	}
	if o.mod != nil {
		errs = append(errs, o.mod.Close())
		o.mod = nil
	}
	return errors.Join(errs...)
}
