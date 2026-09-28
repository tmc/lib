package puffernet

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// KernelSource is the CUDA source of the step kernel.
//
// One block runs one agent through the whole network. The block holds the
// agent's activations in shared memory; each thread computes whole dot
// products, summing in index order as puffercpu.c does. The weights are
// read from global memory, where a small policy stays resident in L2.
const KernelSource = `
__device__ __forceinline__ float puffernet_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// puffernet_linear computes out[o] = dot(in, w[o]) for o < n.
__device__ __forceinline__ void puffernet_linear(float* out, const float* in,
        const float* __restrict__ w, int k, int n) {
    for (int o = threadIdx.x; o < n; o += blockDim.x) {
        const float* row = w + (long long)o * k;
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += in[i] * row[i];
        }
        out[o] = sum;
    }
}

// Shared memory: a (max(in_dim, hidden)), b (hidden), fused (3*hidden).
extern "C" __global__ void puffernet_step(
        const float* __restrict__ w,
        long long enc, long long dec, long long layer0, long long layer_stride,
        int in_dim, int hidden, int layers, int out_dim, int batch,
        const float* __restrict__ obs, const float* __restrict__ terminals,
        float* __restrict__ state, float* __restrict__ out) {
    extern __shared__ float smem[];
    int agent = blockIdx.x;
    float* a = smem;
    float* b = a + max(in_dim, hidden);
    float* fused = b + hidden;

    const float* o = obs + (long long)agent * in_dim;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        a[i] = o[i];
    }
    bool reset = terminals != 0 && terminals[agent] > 0.5f;
    __syncthreads();

    puffernet_linear(b, a, w + enc, in_dim, hidden);
    __syncthreads();

    float* x = b;
    float* y = a;
    for (int l = 0; l < layers; l++) {
        puffernet_linear(fused, x, w + layer0 + l * layer_stride, hidden, 3 * hidden);
        __syncthreads();
        float* s = state + ((long long)l * batch + agent) * hidden;
        for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
            float hid = fused[h];
            float gate = fused[hidden + h];
            float hw = fused[2 * hidden + h];
            float prev = reset ? 0.0f : s[h];
            float tilde = hid >= 0.0f ? hid + 0.5f : puffernet_sigmoid(hid);
            float m = prev + puffernet_sigmoid(gate) * (tilde - prev);
            float hs = puffernet_sigmoid(hw);
            y[h] = hs * m + (1.0f - hs) * x[h];
            s[h] = m;
        }
        __syncthreads();
        float* t = x;
        x = y;
        y = t;
    }

    puffernet_linear(out + (long long)agent * out_dim, x, w + dec, hidden, out_dim);
}
`

// Net runs the network on a CUDA device for a fixed number of agents, and
// holds their recurrent state.
type Net struct {
	cfg     Config
	batch   int
	mod     *cuda.Module
	step    *cuda.Kernel
	arena   *cudamem.Arena
	weights cudamem.Slice[float32]
	state   cudamem.Slice[float32]
	threads uint32
	shared  uint32
}

// New compiles the step kernel for dev, uploads w, and allocates a zero
// recurrent state for batch agents.
func New(dev *cuda.Device, w *Weights, batch int) (_ *Net, err error) {
	cfg := w.Config
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if batch <= 0 || batch > 1<<31-1 {
		return nil, fmt.Errorf("puffernet: invalid batch %d", batch)
	}
	h := cfg.HiddenSize
	threads := min(256, (max(3*h, cfg.OutputSize())+31)&^31)
	shared := (max(cfg.InputSize, h) + 4*h) * 4
	if shared > 48<<10 {
		return nil, fmt.Errorf("puffernet: activations need %d bytes of shared memory, more than 48 KiB", shared)
	}
	n := &Net{cfg: cfg, batch: batch, threads: uint32(threads), shared: uint32(shared)}
	defer func() {
		if err != nil {
			n.Close()
		}
	}()
	cm, err := nvrtcx.Compile(KernelSource,
		nvrtcx.WithName("puffernet.cu"),
		nvrtcx.WithDeviceArch(dev))
	if err != nil {
		return nil, fmt.Errorf("puffernet: compile kernel: %w", err)
	}
	if n.mod, err = cm.LoadModule(dev); err != nil {
		return nil, fmt.Errorf("puffernet: load kernel: %w", err)
	}
	if n.step, err = n.mod.Function("puffernet_step"); err != nil {
		return nil, fmt.Errorf("puffernet: %w", err)
	}
	n.arena = cudamem.NewArena(dev)
	n.weights = cudamem.AllocIn[float32](n.arena, "weights", len(w.flat))
	n.state = cudamem.AllocIn[float32](n.arena, "state", cfg.NumLayers*batch*h)
	if err := n.arena.Err(); err != nil {
		return nil, fmt.Errorf("puffernet: %w", err)
	}
	if err := n.weights.CopyFromHost(w.flat); err != nil {
		return nil, fmt.Errorf("puffernet: upload weights: %w", err)
	}
	if err := n.state.CopyFromHost(make([]float32, n.state.Len())); err != nil {
		return nil, fmt.Errorf("puffernet: zero state: %w", err)
	}
	return n, nil
}

// Close frees the device memory and unloads the kernel. Work the Net
// enqueued must be complete.
func (n *Net) Close() error {
	var err error
	if n.arena != nil {
		err = n.arena.Free()
		n.arena = nil
	}
	if n.mod != nil {
		if cerr := n.mod.Close(); err == nil {
			err = cerr
		}
		n.mod = nil
	}
	return err
}

// State returns the recurrent state, NumLayers×batch×HiddenSize values.
func (n *Net) State() cudamem.Slice[float32] {
	return n.state
}

// Reset enqueues on stream a zeroing of every agent's recurrent state.
func (n *Net) Reset(stream *cuda.Stream) error {
	if err := n.state.MemsetAsync(0, stream); err != nil {
		return fmt.Errorf("puffernet: reset: %w", err)
	}
	return nil
}

// Step enqueues on stream one step of every agent, reading batch×InputSize
// observations from obs and writing batch×OutputSize values to out. If
// terminals is not empty it holds one flag per agent, and the state of each
// agent whose flag is above 0.5 is zeroed first. See [Host.Step].
func (n *Net) Step(stream *cuda.Stream, out, obs, terminals cudamem.Slice[float32]) error {
	cfg := n.cfg
	if obs.Len() != n.batch*cfg.InputSize || out.Len() != n.batch*cfg.OutputSize() ||
		terminals.Len() != 0 && terminals.Len() != n.batch {
		return fmt.Errorf("puffernet: step: have obs %d, out %d, terminals %d for batch %d",
			obs.Len(), out.Len(), terminals.Len(), n.batch)
	}
	w, o, t, s, y := n.weights.Ptr(), obs.Ptr(), terminals.Ptr(), n.state.Ptr(), out.Ptr()
	if terminals.Len() == 0 {
		t = 0
	}
	off := n.offsets()
	in, hid, layers, outDim, batch := int32(cfg.InputSize), int32(cfg.HiddenSize),
		int32(cfg.NumLayers), int32(cfg.OutputSize()), int32(n.batch)
	args := []unsafe.Pointer{
		unsafe.Pointer(&w),
		unsafe.Pointer(&off.enc), unsafe.Pointer(&off.dec), unsafe.Pointer(&off.layer0), unsafe.Pointer(&off.stride),
		unsafe.Pointer(&in), unsafe.Pointer(&hid), unsafe.Pointer(&layers), unsafe.Pointer(&outDim), unsafe.Pointer(&batch),
		unsafe.Pointer(&o), unsafe.Pointer(&t), unsafe.Pointer(&s), unsafe.Pointer(&y),
	}
	err := stream.WithHandle(func(h driver.Stream) error {
		return n.step.Launch([3]uint32{uint32(n.batch), 1, 1}, [3]uint32{n.threads, 1, 1}, n.shared, h, args)
	})
	if err != nil {
		return fmt.Errorf("puffernet: step: %w", err)
	}
	return nil
}

type offsets struct {
	enc, dec, layer0, stride int64
}

// offsets returns where the tensors start in the flat weights.
func (n *Net) offsets() offsets {
	cfg := n.cfg
	h := cfg.HiddenSize
	dec := align8(h * cfg.InputSize)
	layer0 := align8(dec + cfg.OutputSize()*h)
	if cfg.Continuous() {
		layer0 = align8(layer0 + len(cfg.ActionSizes))
	}
	return offsets{0, int64(dec), int64(layer0), int64(align8(3 * h * h))}
}
