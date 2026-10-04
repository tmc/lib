package cudamoe

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
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// MoEWeightConsumerKind distinguishes a retained range pointer from global slab
// addressing. Both kernels compute SiLU(gate matvec) times up matvec in FP32.
// Decoding E4M3FN storage is manual; neither path uses Tensor Cores.
type MoEWeightConsumerKind int

const (
	MoEWeightViewConsumer MoEWeightConsumerKind = iota + 1
	MoEWeightSlabConsumer
)

func moeWeightEntry(kind MoEWeightConsumerKind) (string, error) {
	switch kind {
	case MoEWeightViewConsumer:
		return "moe_weight_view", nil
	case MoEWeightSlabConsumer:
		return "moe_weight_slab", nil
	}
	return "", errors.New("cudamoe: unsupported weight consumer kind")
}

// MoEWeightSchema is the common ABI of both ordinary weight consumers. Pointer
// slots are naturally eight-byte aligned; pointed-to FP8 storage needs only byte
// alignment and FP32 storage needs four-byte alignment.
func MoEWeightSchema() (cudakernel.Schema, error) {
	var p []cudakernel.Param
	for _, name := range []string{"weights", "scales", "input", "output"} {
		p = append(p, cudakernel.Param{Name: name, Type: "ptr", ByteWidth: 8, Alignment: 8, IsPointer: true})
	}
	for _, name := range []string{"layer", "expert", "experts", "intermediate", "hidden"} {
		p = append(p, cudakernel.Param{Name: name, Type: "s32", ByteWidth: 4, Alignment: 4})
	}
	return cudakernel.NewSchema(p...)
}

// MoEWeightSource returns the ordinary CUDA consumer and its declared entry.
// One thread computes each output neuron in increasing hidden-column order;
// scale multiplication follows the FP32 dot sum. Slab indexing preserves global
// layer holes. Compiler options must disable contraction to match this identity.
func MoEWeightSource(kind MoEWeightConsumerKind) (source, entry string, err error) {
	entry, err = moeWeightEntry(kind)
	if err != nil {
		return "", "", err
	}
	offset := "0ULL"
	if kind == MoEWeightSlabConsumer {
		offset = "((unsigned long long)layer*experts+expert)*(2ULL*intermediate)"
	}
	source = `static __device__ float weight_decode(unsigned char raw) {
    unsigned int magnitude=raw&127, e=magnitude>>3, m=magnitude&7;
    if (magnitude==127) return __int_as_float(0x7fc00000);
    float v=e==0 ? __fmul_rn((float)m,0.001953125f) : __int_as_float(((e+120)<<23)|(m<<20));
    return __int_as_float(__float_as_int(v)|((unsigned int)(raw&128)<<24));
}
extern "C" __global__ void ` + entry + `(const unsigned char *weights,const float *scales,const float *input,float *output,int layer,int expert,int experts,int intermediate,int hidden) {
    unsigned int n=blockIdx.x*blockDim.x+threadIdx.x;
    if (gridDim.y!=1||gridDim.z!=1||blockDim.y!=1||blockDim.z!=1||n>=(unsigned int)intermediate) return;
    unsigned long long base=` + offset + `;
    unsigned long long gate_row=base+(n/8)*16+n%8, up_row=gate_row+8;
    float gate=0.0f,up=0.0f;
    for (int c=0;c<hidden;++c) {
        gate=__fadd_rn(gate,__fmul_rn(weight_decode(weights[gate_row*hidden+c]),input[c]));
        up=__fadd_rn(up,__fmul_rn(weight_decode(weights[up_row*hidden+c]),input[c]));
    }
    gate=__fmul_rn(gate,scales[gate_row]);
    up=__fmul_rn(up,scales[up_row]);
    output[n]=__fmul_rn(gate/__fadd_rn(1.0f,expf(-gate)),up);
}
`
	return source, entry, nil
}

// MoEWeightConsumer owns a loaded ordinary kernel. Its zero value is closed.
// Execute and Wait are serialized; failed or canceled waits retain all read
// leases. Keep the device and stream live until Close succeeds.
type MoEWeightConsumer struct {
	mu      sync.Mutex
	device  *cuda.Device
	kind    MoEWeightConsumerKind
	module  *cuda.Module
	kernel  *cuda.Kernel
	raw     driver.Module
	pending *cudaplan.Completion
	failure error
}

func checkedMoEWeightArtifact(kind MoEWeightConsumerKind, artifact *cudakernel.Artifact) (*cudakernel.Artifact, error) {
	source, entry, err := MoEWeightSource(kind)
	if err != nil {
		return nil, err
	}
	if err := artifact.Validate(); err != nil {
		return nil, err
	}
	schema, err := MoEWeightSchema()
	if err != nil {
		return nil, err
	}
	if artifact.EntryPoint != entry || !reflect.DeepEqual(artifact.Schema, schema) || artifact.Kind != cudakernel.KindPTX && artifact.Kind != cudakernel.KindCUBIN {
		return nil, errors.New("cudamoe: weight consumer entry, image kind or schema differs")
	}
	want := fmt.Sprintf("%x", sha256.Sum256([]byte(source)))
	count := 0
	for _, d := range artifact.Dependencies {
		if d.Path == "cudamoe_weight_storage.cu" {
			if !d.Tracked || d.Digest != want || d.ErrorMsg != "" {
				return nil, errors.New("cudamoe: weight consumer source binding differs")
			}
			count++
		}
	}
	if count != 1 {
		return nil, errors.New("cudamoe: missing or duplicate weight consumer source binding")
	}
	owned := *artifact
	owned.Raw = append([]byte(nil), artifact.Raw...)
	owned.Schema.Params = append([]cudakernel.Param(nil), artifact.Schema.Params...)
	owned.Dependencies = append([]cudakernel.Dependency(nil), artifact.Dependencies...)
	owned.CompilerOpts = append([]string(nil), artifact.CompilerOpts...)
	return &owned, nil
}

// LoadMoEWeightConsumer loads a trusted compilation artifact of MoEWeightSource.
// Its exact ABI, entry and tracked cudamoe_weight_storage.cu source digest must
// match. Receipt checks associate source; they do not authenticate a compiler or
// establish arbitrary machine-code semantics. Descriptor slices are copied;
// callers must exclude mutation during the call. Close every nonnil error result.
func LoadMoEWeightConsumer(dev *cuda.Device, kind MoEWeightConsumerKind, artifact *cudakernel.Artifact) (*MoEWeightConsumer, error) {
	owned, err := checkedMoEWeightArtifact(kind, artifact)
	if err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, errors.New("cudamoe: nil weight consumer device")
	}
	c := &MoEWeightConsumer{device: dev, kind: kind}
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
	return c, err
}

// Execute reads one retained layer/expert using the specified consumer's
// addressing mode. Input has Hidden FP32 elements and output has Intermediate
// FP32 elements; their active ranges must be disjoint from each other and both
// slabs. Callers must exclude mutation, Free and other uses until Wait succeeds.
// A canceled or failed wait is incomplete work, not an admitted output.
func (c *MoEWeightConsumer) Execute(ctx context.Context, stream *cuda.Stream, view *MoEWeightView, input, output cudamem.Slice[float32]) error {
	if c == nil {
		return errors.New("cudamoe: nil weight consumer")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.wait(ctx); err != nil {
		return err
	}
	if c.kernel == nil || stream == nil || stream.Device() != c.device {
		return errors.New("cudamoe: closed consumer or different stream device")
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	d, releaseView, err := view.retain(c.device)
	if err != nil {
		return err
	}
	releases := []func(){releaseView}
	rollback := func() {
		for _, release := range releases {
			release()
		}
	}
	for _, s := range []cudamem.Slice[float32]{input, output} {
		release, err := s.Borrow(c.device)
		if err != nil {
			rollback()
			return err
		}
		releases = append(releases, release)
	}
	if input.Len() != d.Hidden || output.Len() != d.Intermediate {
		rollback()
		return errors.New("cudamoe: weight consumer input or output extent differs")
	}
	view.owner.mu.Lock()
	wBytes, sBytes, experts := view.owner.layout.weightBytes, view.owner.layout.scaleRows*4, view.owner.layout.shape.Experts
	view.owner.mu.Unlock()
	ranges := [][2]uint64{{uint64(d.WeightBase), uint64(wBytes)}, {uint64(d.ScaleBase), uint64(sBytes)}, {uint64(input.Ptr()), uint64(input.SizeBytes())}, {uint64(output.Ptr()), uint64(output.SizeBytes())}}
	for i, a := range ranges {
		if a[0] == 0 || a[1] == 0 || a[0] > math.MaxUint64-a[1] {
			rollback()
			return errors.New("cudamoe: invalid weight consumer range")
		}
		for _, b := range ranges[:i] {
			if a[0] < b[0]+b[1] && b[0] < a[0]+a[1] {
				rollback()
				return errors.New("cudamoe: overlapping weight consumer ranges")
			}
		}
	}
	x := make([]float32, input.Len())
	if err := input.CopyToHost(x); err != nil {
		rollback()
		return err
	}
	if err := validateMoEWeightPayload(nil, x); err != nil {
		rollback()
		return err
	}
	completion, err := cudaplan.NewStreamCompletion(stream, releases...)
	if err != nil {
		rollback()
		return err
	}
	c.pending = completion
	w, scale := d.WeightBase, d.ScaleBase
	layer, expert, nExperts := int32(d.Layer), int32(d.Expert), int32(experts)
	if c.kind == MoEWeightViewConsumer {
		w += driver.DevicePtr(d.WeightOffsetBytes)
		scale += driver.DevicePtr(d.ScaleOffsetBytes)
		layer, expert, nExperts = 0, 0, 1
	}
	in, out := input.Ptr(), output.Ptr()
	intermediate, hidden := int32(d.Intermediate), int32(d.Hidden)
	args := []unsafe.Pointer{unsafe.Pointer(&w), unsafe.Pointer(&scale), unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&layer), unsafe.Pointer(&expert), unsafe.Pointer(&nExperts), unsafe.Pointer(&intermediate), unsafe.Pointer(&hidden)}
	c.failure = stream.WithHandle(func(raw driver.Stream) error {
		return c.kernel.Launch([3]uint32{uint32((d.Intermediate + 31) / 32), 1, 1}, [3]uint32{32, 1, 1}, 0, raw, args)
	})
	if err := c.wait(ctx); err != nil {
		return err
	}
	y := make([]float32, output.Len())
	if err := output.CopyToHost(y); err != nil {
		return err
	}
	return validateMoEWeightPayload(nil, y)
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

// Wait retries a pending actual stream drain. A successful return releases
// consumer leases; cancellation or failure retains them for another attempt.
// After retry, callers must independently validate any device output.
func (c *MoEWeightConsumer) Wait(ctx context.Context) error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.wait(ctx)
}

// Close drains pending work before unloading the owner module. Cleanup errors
// retain the remaining module or work for retry. Caller allocations are not freed.
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
		c.module, c.kernel = nil, nil
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
