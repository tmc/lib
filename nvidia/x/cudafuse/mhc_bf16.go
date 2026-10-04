package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// MHCBF16Inputs uses BF16 bit storage for X and Residual, with FP32 Logits and
// Post. Shapes and disjoint-view requirements are identical to MHCInputs.
// Every BF16 value expands exactly by shifting its bits left sixteen places.
type MHCBF16Inputs struct {
	Logits, Post cudamem.Slice[float32]
	X, Residual  cudamem.Slice[uint16]
}

func mhcBF16Expand(x uint16) float32 { return math.Float32frombits(uint32(x) << 16) }

func mhcValidateBF16Ranges(c MHCConfig, ranges []mhcRange) error {
	if err := c.validate(); err != nil {
		return err
	}
	if len(ranges) != 4 {
		return errors.New("cudafuse: invalid bf16 mhc binding count")
	}
	lengths := []int{c.Batch * 16, c.Batch * c.Width, c.Batch * 4, c.Batch * 4 * c.Width}
	units := []uint64{4, 2, 4, 2}
	for i, r := range ranges {
		if r.length != lengths[i] || r.bytes != uint64(lengths[i])*units[i] || r.ptr == 0 || r.ptr%units[i] != 0 || r.bytes > math.MaxUint64-r.ptr {
			return fmt.Errorf("cudafuse: invalid bf16 mhc binding %d", i)
		}
		for _, s := range ranges[:i] {
			if r.ptr < s.ptr+s.bytes && s.ptr < r.ptr+r.bytes {
				return errors.New("cudafuse: aliased bf16 mhc bindings")
			}
		}
	}
	return nil
}

// MHCBF16Source returns ordinary two-stage source with FP32 normalization and
// accumulation and a single BF16 round-to-nearest, ties-to-even output cast.
// It uses integer bit conversion, without requiring a BF16 hardware feature.
func MHCBF16Source(c MHCConfig) ([2]FusedKernel, error) {
	s, err := MHCSource(c)
	if err != nil {
		return s, err
	}
	s[1].Name = "mhc_post_bf16"
	s[1].Source = fmt.Sprintf("#define MHC_WIDTH %d\n", c.Width) + mhcBF16PostSource
	return s, nil
}

// ExecuteMHCBF16 runs the two-stage storage variant. Normalization scratch is
// FP32 and final output storage is BF16. Execution.Readback expands each final
// BF16 bit pattern exactly to float32; it does not add an FP32 output cast.
// Close, completion, borrowing, cancellation and guard rules match ExecuteMHC.
// The final conversion policy is separate from FP32 arithmetic error.
func ExecuteMHCBF16(ctx context.Context, stream *cuda.Stream, c MHCConfig, in MHCBF16Inputs) (*Execution, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	if stream == nil || stream.Device() == nil {
		return nil, errors.New("cudafuse: nil mhc stream or device")
	}
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	dev := stream.Device()
	var releases []func()
	rollback := func() {
		for _, r := range releases {
			r()
		}
	}
	ptrs := []driver.DevicePtr{in.Logits.Ptr(), in.X.Ptr(), in.Post.Ptr(), in.Residual.Ptr()}
	lengths := []int{in.Logits.Len(), in.X.Len(), in.Post.Len(), in.Residual.Len()}
	bytes := []uint64{uint64(in.Logits.SizeBytes()), uint64(in.X.SizeBytes()), uint64(in.Post.SizeBytes()), uint64(in.Residual.SizeBytes())}
	var ranges []mhcRange
	for i := range ptrs {
		ranges = append(ranges, mhcRange{uint64(ptrs[i]), bytes[i], lengths[i]})
	}
	if err := mhcValidateBF16Ranges(c, ranges); err != nil {
		return nil, err
	}
	for _, b := range []cudamem.Slice[float32]{in.Logits, in.Post} {
		r, err := b.Borrow(dev)
		if err != nil {
			rollback()
			return nil, err
		}
		releases = append(releases, r)
	}
	for _, b := range []cudamem.Slice[uint16]{in.X, in.Residual} {
		r, err := b.Borrow(dev)
		if err != nil {
			rollback()
			return nil, err
		}
		releases = append(releases, r)
	}
	for _, b := range []cudamem.Slice[float32]{in.Logits, in.Post} {
		xs := make([]float32, b.Len())
		if err := b.CopyToHost(xs); err != nil {
			rollback()
			return nil, err
		}
		if err := mhcValidateValues(xs); err != nil {
			rollback()
			return nil, err
		}
	}
	for _, b := range []cudamem.Slice[uint16]{in.X, in.Residual} {
		xs := make([]uint16, b.Len())
		if err := b.CopyToHost(xs); err != nil {
			rollback()
			return nil, err
		}
		for _, x := range xs {
			if !mhcFinite(mhcBF16Expand(x)) {
				rollback()
				return nil, errors.New("cudafuse: nonfinite bf16 mhc input")
			}
		}
	}
	sources, _ := MHCBF16Source(c)
	return mhcExecute(ctx, c, ptrs, mhcBF16Ops(dev, stream, sources), releases)
}

func mhcBF16Ops(dev *cuda.Device, stream *cuda.Stream, sources [2]FusedKernel) executionOps {
	ops := nativeExecutionOps(dev, stream)
	alloc := ops.alloc
	compile := ops.compile
	allocations := 0
	ops.alloc = func(n int) (*executionBuffer, error) {
		allocations++
		if allocations == 1 {
			return alloc(n)
		}
		storage, err := cudamem.Alloc[uint16](dev, n+2)
		if storage.Ptr() == 0 {
			return nil, err
		}
		b := &executionBuffer{ptr: storage.Slice(1, n+1).Ptr(), free: storage.Free, freed: func() bool { return storage.Ptr() == 0 }, read: func(out []float32) error {
			all := make([]uint16, n+2)
			if err := storage.CopyToHost(all); err != nil {
				return err
			}
			if all[0] != 0x7fc1 || all[n+1] != 0x7fc1 {
				return ErrOutputGuard
			}
			for i, v := range all[1 : n+1] {
				if v == 0x7fc1 {
					return ErrOutputUnwritten
				}
				out[i] = mhcBF16Expand(v)
			}
			return nil
		}}
		if err != nil {
			return b, err
		}
		poison := make([]uint16, n+2)
		for i := range poison {
			poison[i] = 0x7fc1
		}
		return b, storage.CopyFromHost(poison)
	}
	ops.compile = func(s FusedKernel) (*executionKernel, error) {
		if s.Name == sources[0].Name {
			return compile(sources[0])
		}
		return compile(sources[1])
	}
	return ops
}

const mhcBF16PostSource = `
__device__ float mhc_expand_bf16(unsigned short x) { union {unsigned int u;float f;} v;v.u=((unsigned int)x)<<16;return v.f; }
__device__ unsigned short mhc_round_bf16(float x) {
    union {unsigned int u;float f;} v;v.f=x;
    if ((v.u&0x7f800000u)==0x7f800000u) return (unsigned short)((v.u>>16)|((v.u&0x7fffffu)?0x40u:0u));
    return (unsigned short)((v.u+0x7fffu+((v.u>>16)&1u))>>16);
}
extern "C" __global__ void mhc_post_bf16(const unsigned short *x,const float *post,const unsigned short *residual,const float *mix,unsigned short *out,int count) {
    if (blockDim.y!=1||blockDim.z!=1||gridDim.y!=1||gridDim.z!=1) return;
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (count<=0||i>=(unsigned int)count) return;
    unsigned int c=i%MHC_WIDTH,n=(i/MHC_WIDTH)%4,b=i/(4*MHC_WIDTH);
    float v=mhc_expand_bf16(x[b*MHC_WIDTH+c])*post[b*4+n];
    for (int m=0;m<4;++m) v+=mix[b*16+m*4+n]*mhc_expand_bf16(residual[b*4*MHC_WIDTH+m*MHC_WIDTH+c]);
    out[i]=mhc_round_bf16(v);
}
`
