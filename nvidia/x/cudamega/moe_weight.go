package cudamega

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudamoe"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// MoEWeightConfig declares one finite FP32 gate/up task over the shared
// E4M3FN-interleave8 slab. The first integration supports three global layers,
// two experts, I=8/16 and H=17/32. It does not change task scheduling.
type MoEWeightConfig struct{ Layers, Experts, Intermediate, Hidden int }

func (c MoEWeightConfig) validate() error {
	if c.Layers != 3 || c.Experts != 2 || c.Intermediate != 8 && c.Intermediate != 16 || c.Hidden != 17 && c.Hidden != 32 {
		return errors.New("cudamega: unsupported shared-weight shape")
	}
	return nil
}

func moeWeightProgram(c MoEWeightConfig, target Target) (*Program, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	if target != "sm_121" {
		return nil, errors.New("cudamega: shared-weight consumer requires sm_121")
	}
	g, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "gate-up", Scope: cudataskgraph.ScopeDevice}})
	if err != nil {
		return nil, err
	}
	return newProgram(g, map[string]Operation{"gate-up": {Kind: MoEGateUp, moe: &c}}, Grid{CTAs: 1, ThreadsPerCTA: 32}, target, true)
}

// moeWeightCUDA reuses the existing selection, task guard, ownership, fences
// and publication renderer. Only the typed ABI and Compute payload differ.
type moeWeightCUDA struct {
	*cudaSource
	config MoEWeightConfig
}

func (c *moeWeightCUDA) Begin(p *Program) error {
	if c.p != p || len(p.steps) != 1 || p.steps[0].Operation.moe == nil || *p.steps[0].Operation.moe != c.config {
		return errors.New("cudamega: different typed program")
	}
	c.defs.WriteString("constexpr unsigned task_count=1, grid_ctas=1;\n")
	return nil
}

func (c *moeWeightCUDA) Compute(i, src int, op Operation) error {
	if i != 0 || src != -1 || op.Kind != MoEGateUp || op.moe == nil || *op.moe != c.config {
		return errors.New("cudamega: unsupported typed task")
	}
	c.body.WriteString(`        for (unsigned neuron=threadIdx.x; neuron<n; neuron+=blockDim.x) {
          unsigned long long base=((unsigned long long)layer*2+expert)*(2ULL*intermediate);
          unsigned long long gate_row=base+(neuron/8)*16+neuron%8, up_row=gate_row+8;
          float gate=0.0f, up=0.0f;
          for (unsigned column=0; column<hidden; ++column) {
            gate=__fadd_rn(gate,__fmul_rn(weight_decode(weights[gate_row*hidden+column]),input[column]));
            up=__fadd_rn(up,__fmul_rn(weight_decode(weights[up_row*hidden+column]),input[column]));
          }
          gate=__fmul_rn(gate,scales[gate_row]);
          up=__fmul_rn(up,scales[up_row]);
          output[neuron]=__fmul_rn(gate/__fadd_rn(1.0f,expf(-gate)),up);
        }
`)
	return nil
}

func (c *moeWeightCUDA) End() error { return nil }

// MoEWeightSource renders a typed payload through the existing finite Runner.
// Its one CTA uses the same task selection and completion publication as scalar
// programs. The launched artifact still requires measured occupancy admission.
func MoEWeightSource(config MoEWeightConfig, target Target) (string, error) {
	p, err := moeWeightProgram(config, target)
	if err != nil {
		return "", err
	}
	c := &moeWeightCUDA{cudaSource: &cudaSource{p: p}, config: config}
	r, err := NewRunner(p, c)
	if err != nil {
		return "", err
	}
	if err := r.Run(); err != nil {
		return "", err
	}
	// Decode is copied exactly from the qualified ordinary consumer. Keeping
	// its source identity separate prevents claiming arbitrary renderer identity.
	decode := `static __device__ float weight_decode(unsigned char raw) {
    unsigned int magnitude=raw&127, e=magnitude>>3, m=magnitude&7;
    if (magnitude==127) return __int_as_float(0x7fc00000);
    float v=e==0 ? __fmul_rn((float)m,0.001953125f) : __int_as_float(((e+120)<<23)|(m<<20));
    return __int_as_float(__float_as_int(v)|((unsigned int)(raw&128)<<24));
}
`
	entry := fmt.Sprintf(`extern "C" __global__ void moe_weight_worker(const unsigned char *weights,const float *scales,const float *input,float *output,unsigned *completed,unsigned long long *start_order,volatile unsigned *abort_code,int layer,int expert,unsigned n,unsigned long long epoch) {
  constexpr unsigned intermediate=%d, hidden=%d;
  if (gridDim.x!=1||gridDim.y!=1||gridDim.z!=1||blockDim.x!=32||blockDim.y!=1||blockDim.z!=1) { if(threadIdx.x==0) atomicCAS((unsigned*)abort_code,0u,%du); return; }
  if (n!=intermediate||!epoch||layer<0||layer>=3||layer==1||expert<0||expert>=2) { if(threadIdx.x==0) atomicCAS((unsigned*)abort_code,0u,%du); return; }
`, config.Intermediate, config.Hidden, AbortBadLaunch, AbortBadExtent)
	return decode + c.defs.String() + entry + c.body.String() + "}\n", nil
}

// MoEWeightSchema is the typed worker ABI. Floats are never uint32 arithmetic.
func MoEWeightSchema() (cudakernel.Schema, error) {
	var p []cudakernel.Param
	for _, n := range []string{"weights", "scales", "input", "output", "completed", "start_order", "abort_code"} {
		p = append(p, cudakernel.Param{Name: n, Type: "ptr", ByteWidth: 8, Alignment: 8, IsPointer: true})
	}
	for _, n := range []string{"layer", "expert", "n"} {
		typ := "s32"
		if n == "n" {
			typ = "u32"
		}
		p = append(p, cudakernel.Param{Name: n, Type: typ, ByteWidth: 4, Alignment: 4})
	}
	p = append(p, cudakernel.Param{Name: "epoch", Type: "u64", ByteWidth: 8, Alignment: 8})
	return cudakernel.NewSchema(p...)
}

func checkedMoEArtifact(config MoEWeightConfig, a *cudakernel.Artifact) (*cudakernel.Artifact, error) {
	s, err := MoEWeightSource(config, "sm_121")
	if err != nil {
		return nil, err
	}
	if err := a.Validate(); err != nil {
		return nil, err
	}
	schema, _ := MoEWeightSchema()
	if a.Kind != cudakernel.KindCUBIN || a.Target != "sm_121" || a.EntryPoint != "moe_weight_worker" || !reflect.DeepEqual(a.Schema, schema) || !reflect.DeepEqual(a.CompilerOpts, []string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}) {
		return nil, errors.New("cudamega: typed weight artifact identity differs")
	}
	count := 0
	want := fmt.Sprintf("%x", sha256.Sum256([]byte(s)))
	for _, d := range a.Dependencies {
		if d.Path == "cudamega_moe_weight.cu" {
			count++
			if !d.Tracked || d.Digest != want || d.ErrorMsg != "" {
				return nil, errors.New("cudamega: typed source binding differs")
			}
		}
	}
	if count != 1 {
		return nil, errors.New("cudamega: typed source binding missing or duplicated")
	}
	b := *a
	b.Raw = append([]byte(nil), a.Raw...)
	b.Dependencies = append([]cudakernel.Dependency(nil), a.Dependencies...)
	b.CompilerOpts = append([]string(nil), a.CompilerOpts...)
	b.Schema.Params = append([]cudakernel.Param(nil), a.Schema.Params...)
	return &b, nil
}

// MoEWeightBuffers are caller-owned typed data and publication buffers. Keep
// these, the device and stream live and exclude concurrent use until Wait.
type MoEWeightBuffers struct {
	Input, Output    cudamem.Slice[float32]
	Completed, Abort cudamem.Slice[uint32]
	StartOrder       cudamem.Slice[uint64]
}

// MoEWeightConsumer owns one typed persistent worker module. The zero value
// refuses Execute. Close/Wait retry actual drain before releasing leases.
// Keep its device and any submitted stream live until Close succeeds. Do not
// close either concurrently with Submit, Execute, Wait or Close.
type MoEWeightConsumer struct {
	mu          sync.Mutex
	device      *cuda.Device
	config      MoEWeightConfig
	module      *cuda.Module
	raw         driver.Module
	kernel      *cuda.Kernel
	pending     moeWeightCompletion
	failure     error
	blocksPerSM int32
}

type moeWeightCompletion interface{ Wait(context.Context) error }

// LoadMoEWeightConsumer loads a trusted source-bound CUBIN and measures its
// actual 32-thread occupancy before permitting the single cooperative CTA.
// Receipt association is not compiler authentication or binary semantic proof.
// Close every nonnil error result. Artifact variable slices are copied.
func LoadMoEWeightConsumer(dev *cuda.Device, config MoEWeightConfig, a *cudakernel.Artifact) (*MoEWeightConsumer, error) {
	owned, err := checkedMoEArtifact(config, a)
	if err != nil {
		return nil, err
	}
	return loadMoEWeightImage(dev, config, owned)
}

func loadMoEWeightImage(dev *cuda.Device, config MoEWeightConfig, owned *cudakernel.Artifact) (*MoEWeightConsumer, error) {
	var err error
	if dev == nil {
		return nil, errors.New("cudamega: nil weight device")
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	if major != 12 || minor != 1 {
		return nil, errors.New("cudamega: typed weight device differs")
	}
	c := &MoEWeightConsumer{device: dev, config: config}
	c.raw, err = owned.Load(dev)
	if err != nil {
		return c, err
	}
	c.module, err = cuda.OwnModule(dev, c.raw)
	if err != nil {
		return c, err
	}
	c.raw = 0
	c.kernel, err = c.module.Function(owned.EntryPoint)
	if err != nil {
		return c, err
	}
	err = dev.WithContext(func() error {
		var e error
		c.blocksPerSM, e = c.kernel.Raw().MaxActiveBlocksPerMultiprocessor(32, 0)
		return e
	})
	if err != nil {
		return c, err
	}
	if err := checkMoEOccupancy(c.blocksPerSM); err != nil {
		return c, err
	}
	return c, nil
}

func checkMoEOccupancy(blocks int32) error {
	if blocks < 1 {
		return errors.New("cudamega: measured worker occupancy is empty")
	}
	return nil
}

// BlocksPerSM returns the actual compiled-worker occupancy measurement, not an
// SM-count guess. Only one CTA is launched by this finite consumer.
func (c *MoEWeightConsumer) BlocksPerSM() int32 {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.blocksPerSM
}

func (c *MoEWeightConsumer) wait(ctx context.Context) error {
	if c.pending == nil {
		return nil
	}
	if err := c.pending.Wait(ctx); err != nil {
		return err
	}
	c.pending = nil
	err := c.failure
	c.failure = nil
	return err
}

func checkMoEBorrow(c MoEWeightConfig, dev *cuda.Device, b cudamoe.MoEWeightBorrowed) error {
	d, s := b.Descriptor, b.Shape
	if d.Device != dev || s.Layers != c.Layers || s.Experts != c.Experts || s.Intermediate != c.Intermediate || s.Hidden != c.Hidden || len(s.ActiveLayers) != 2 || !(s.ActiveLayers[0] == 0 && s.ActiveLayers[1] == 2 || s.ActiveLayers[0] == 2 && s.ActiveLayers[1] == 0) || d.Format != "e4m3fn-gate-up-interleave8-v1" || d.Generation == 0 || d.Layer != 0 && d.Layer != 2 || d.Expert < 0 || d.Expert >= s.Experts || d.Intermediate != s.Intermediate || d.Hidden != s.Hidden || d.WeightOffsetBytes != uint64((d.Layer*s.Experts+d.Expert)*2*d.Intermediate*d.Hidden) || d.ScaleOffsetBytes != uint64((d.Layer*s.Experts+d.Expert)*2*d.Intermediate*4) || b.WeightBytes != uint64(s.Layers*s.Experts*2*s.Intermediate*s.Hidden) || b.ScaleBytes != uint64(s.Layers*s.Experts*2*s.Intermediate*4) {
		return errors.New("cudamega: typed weight owner metadata differs")
	}
	return nil
}

// Execute borrows the actual ready view independently, resets publication
// buffers, submits one cooperative task and waits for real stream completion.
// Canceled/failed drain retains all leases. Epoch must be nonzero. Successful
// publication is required before finite output admission; caller checks numerics.
func (c *MoEWeightConsumer) Execute(ctx context.Context, stream *cuda.Stream, view *cudamoe.MoEWeightView, b MoEWeightBuffers, epoch uint64) error {
	if c == nil {
		return errors.New("cudamega: nil typed weight consumer")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pending != nil {
		return cudamem.ErrInUse
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if err := c.submit(stream, view, b, epoch); err != nil {
		return err
	}
	if err := c.wait(ctx); err != nil {
		return err
	}
	return checkMoEPublication(b)
}

// Submit borrows and queues one actual task without waiting. It refuses reuse
// while work is pending. Even a launch error can leave owned work that requires
// Wait or Close; keep every resource live until a successful actual drain.
func (c *MoEWeightConsumer) Submit(stream *cuda.Stream, view *cudamoe.MoEWeightView, b MoEWeightBuffers, epoch uint64) error {
	if c == nil {
		return errors.New("cudamega: nil typed weight consumer")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pending != nil {
		return cudamem.ErrInUse
	}
	if err := c.submit(stream, view, b, epoch); err != nil {
		return err
	}
	return c.failure
}

func (c *MoEWeightConsumer) submit(stream *cuda.Stream, view *cudamoe.MoEWeightView, b MoEWeightBuffers, epoch uint64) error {
	if c.kernel == nil || c.blocksPerSM < 1 || stream == nil || stream.Device() != c.device || epoch == 0 {
		return errors.New("cudamega: closed consumer, epoch or stream differs")
	}
	owned, release, err := view.Borrow(c.device)
	if err != nil {
		return err
	}
	releases := []func(){release}
	d := owned.Descriptor
	rollback := func() {
		for _, r := range releases {
			r()
		}
	}
	if err := checkMoEBorrow(c.config, c.device, owned); err != nil {
		rollback()
		return err
	}
	if b.Input.Len() != d.Hidden || b.Output.Len() != d.Intermediate || b.Completed.Len() != 1 || b.Abort.Len() != 1 || b.StartOrder.Len() != 2 {
		rollback()
		return errors.New("cudamega: typed buffer extents differ")
	}
	rs := []moeWeightRegion{{b.Input.Ptr(), uint64(b.Input.SizeBytes()), func() (func(), error) { return b.Input.Borrow(c.device) }}, {b.Output.Ptr(), uint64(b.Output.SizeBytes()), func() (func(), error) { return b.Output.Borrow(c.device) }}, {b.Completed.Ptr(), uint64(b.Completed.SizeBytes()), func() (func(), error) { return b.Completed.Borrow(c.device) }}, {b.Abort.Ptr(), uint64(b.Abort.SizeBytes()), func() (func(), error) { return b.Abort.Borrow(c.device) }}, {b.StartOrder.Ptr(), uint64(b.StartOrder.SizeBytes()), func() (func(), error) { return b.StartOrder.Borrow(c.device) }}}
	rs = append(rs, moeWeightRegion{d.WeightBase, owned.WeightBytes, nil}, moeWeightRegion{d.ScaleBase, owned.ScaleBytes, nil})
	if err := checkMoERanges(rs); err != nil {
		rollback()
		return err
	}
	for _, r := range rs {
		if r.borrow != nil {
			r, e := r.borrow()
			if e != nil {
				rollback()
				return e
			}
			releases = append(releases, r)
		}
	}
	x := make([]float32, b.Input.Len())
	if err := b.Input.CopyToHost(x); err != nil {
		rollback()
		return err
	}
	if err := checkMoEInput(x); err != nil {
		rollback()
		return err
	}
	for _, f := range []func() error{func() error { return b.Completed.CopyFromHost([]uint32{0}) }, func() error { return b.Abort.CopyFromHost([]uint32{0}) }, func() error { return b.StartOrder.CopyFromHost([]uint64{0, 0}) }} {
		if err := f(); err != nil {
			rollback()
			return err
		}
	}
	c.pending, err = cudaplan.NewStreamCompletion(stream, releases...)
	if err != nil {
		rollback()
		return err
	}
	w, s, in, out, completed, order, abort := d.WeightBase, d.ScaleBase, b.Input.Ptr(), b.Output.Ptr(), b.Completed.Ptr(), b.StartOrder.Ptr(), b.Abort.Ptr()
	layer, expert, n := int32(d.Layer), int32(d.Expert), uint32(d.Intermediate)
	args := []unsafe.Pointer{unsafe.Pointer(&w), unsafe.Pointer(&s), unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&completed), unsafe.Pointer(&order), unsafe.Pointer(&abort), unsafe.Pointer(&layer), unsafe.Pointer(&expert), unsafe.Pointer(&n), unsafe.Pointer(&epoch)}
	c.failure = stream.WithHandle(func(raw driver.Stream) error {
		return c.kernel.Raw().LaunchCooperativeKernel(driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 32, Y: 1, Z: 1}, 0, raw, args)
	})
	return nil
}

type moeWeightRegion struct {
	ptr    driver.DevicePtr
	bytes  uint64
	borrow func() (func(), error)
}

func checkMoERanges(rs []moeWeightRegion) error {
	for i, r := range rs {
		if r.ptr == 0 || r.bytes == 0 || uint64(r.ptr) > math.MaxUint64-r.bytes {
			return errors.New("cudamega: invalid typed range")
		}
		for _, q := range rs[:i] {
			if uint64(r.ptr) < uint64(q.ptr)+q.bytes && uint64(q.ptr) < uint64(r.ptr)+r.bytes {
				return errors.New("cudamega: overlapping typed ranges")
			}
		}
	}
	return nil
}

func checkMoEInput(x []float32) error {
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("cudamega: nonfinite typed input")
		}
	}
	return nil
}

func checkMoEPublication(b MoEWeightBuffers) error {
	completed := make([]uint32, 1)
	abort := make([]uint32, 1)
	order := make([]uint64, 2)
	if err := b.Completed.CopyToHost(completed); err != nil {
		return err
	}
	if err := b.Abort.CopyToHost(abort); err != nil {
		return err
	}
	if err := b.StartOrder.CopyToHost(order); err != nil {
		return err
	}
	if completed[0] != 1 || abort[0] != 0 || order[0] != 1 || order[1] != 1 {
		return errors.New("cudamega: typed task publication incomplete")
	}
	return nil
}

// Wait retries actual stream drain after Submit or an incomplete Execute. After recovery
// callers must check publication and numerical output before using the result.
func (c *MoEWeightConsumer) Wait(ctx context.Context) error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.wait(ctx)
}

// Close drains incomplete work before unloading; failures preserve retry owners.
func (c *MoEWeightConsumer) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.wait(context.Background()); err != nil {
		return err
	}
	if c.module != nil {
		if err := c.module.Close(); err != nil {
			return err
		}
		c.module = nil
		c.kernel = nil
	}
	if c.raw != 0 {
		err := c.device.WithContext(func() error {
			if err := driver.CtxSynchronize(); err != nil {
				return err
			}
			if err := c.raw.Close(); err != nil {
				return err
			}
			c.raw = 0
			return nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}
