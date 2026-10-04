package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// MHCFusedSource emits the same FP32 math as MHCSource in one ordinary CUDA
// kernel. Each output thread computes its batch's 4x4 normalization locally.
// This deliberately simple alternative makes no performance claim.
func MHCFusedSource(c MHCConfig) (FusedKernel, error) {
	s, err := MHCSource(c)
	if err != nil {
		return FusedKernel{}, err
	}
	head, body, ok := strings.Cut(s[0].Source, "extern \"C\" __global__ void mhc_sinkhorn_fp32")
	if !ok {
		return FusedKernel{}, errors.New("cudafuse: missing mhc normalization entry")
	}
	start := strings.Index(body, "    float a[16];")
	end := strings.Index(body, "    for (int k=0;k<16;++k) mix[b*16+k]=a[k];")
	if start < 0 || end < start {
		return FusedKernel{}, errors.New("cudafuse: missing mhc normalization body")
	}
	source := head + `extern "C" __global__ void mhc_fused_fp32(const float *logits,const float *x,const float *post,const float *residual,float *out,int count) {
    if (blockDim.y!=1||blockDim.z!=1||gridDim.y!=1||gridDim.z!=1) return;
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (count<=0||i>=(unsigned int)count) return;
    unsigned int c=i%MHC_WIDTH,n=(i/MHC_WIDTH)%4,b=i/(4*MHC_WIDTH);
` + body[start:end] + `
    float v=x[b*MHC_WIDTH+c]*post[b*4+n];
    for (int m=0;m<4;++m) v+=a[m*4+n]*residual[b*4*MHC_WIDTH+m*MHC_WIDTH+c];
    out[i]=v;
}
`
	return FusedKernel{Name: "mhc_fused_fp32", Source: source, Inputs: []string{"logits", "x", "post", "residual"}, Outputs: []string{"out"}}, nil
}

func mhcBorrowFP32(dev *cuda.Device, c MHCConfig, in MHCInputs) ([]driver.DevicePtr, []func(), error) {
	var ptrs []driver.DevicePtr
	var releases []func()
	rollback := func() {
		for _, r := range releases {
			r()
		}
	}
	var ranges []mhcRange
	inputs := []cudamem.Slice[float32]{in.Logits, in.X, in.Post, in.Residual}
	for _, b := range inputs {
		r, err := b.Borrow(dev)
		if err != nil {
			rollback()
			return nil, nil, err
		}
		releases = append(releases, r)
		ptrs = append(ptrs, b.Ptr())
		ranges = append(ranges, mhcRange{uint64(b.Ptr()), uint64(b.SizeBytes()), b.Len()})
	}
	if err := mhcValidateRanges(c, ranges); err != nil {
		rollback()
		return nil, nil, err
	}
	for _, b := range inputs {
		xs := make([]float32, b.Len())
		if err := b.CopyToHost(xs); err != nil {
			rollback()
			return nil, nil, err
		}
		if err := mhcValidateValues(xs); err != nil {
			rollback()
			return nil, nil, err
		}
	}
	return ptrs, releases, nil
}

// ExecuteMHCFused runs the single-kernel FP32 alternative using the same inputs,
// normalization order, arithmetic mode, guards and completion contract as
// ExecuteMHC. Execution owns one output allocation and one loaded module.
func ExecuteMHCFused(ctx context.Context, stream *cuda.Stream, c MHCConfig, in MHCInputs) (*Execution, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	if stream == nil || stream.Device() == nil {
		return nil, errors.New("cudafuse: nil mhc stream or device")
	}
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	ptrs, releases, err := mhcBorrowFP32(stream.Device(), c, in)
	if err != nil {
		return nil, err
	}
	return mhcFusedExecute(ctx, c, ptrs, nativeExecutionOps(stream.Device(), stream), releases)
}

func mhcFusedExecute(ctx context.Context, c MHCConfig, ptrs []driver.DevicePtr, ops executionOps, releases []func()) (*Execution, error) {
	e := &Execution{ops: ops, releases: releases}
	fail := func(err error) (*Execution, error) {
		e.failure = err
		if !e.attempted {
			return e, errors.Join(err, e.Close())
		}
		return e, err
	}
	s, err := MHCFusedSource(c)
	if err != nil {
		return fail(err)
	}
	e.n = c.Batch * 4 * c.Width
	if len(ptrs) != 4 {
		return fail(errors.New("cudafuse: invalid fused mhc pointer count"))
	}
	if ctx != nil && ctx.Err() != nil {
		return fail(ctx.Err())
	}
	b, err := ops.alloc(e.n)
	if b != nil {
		e.resources = append(e.resources, b)
		e.output = b
		e.stats.Allocations++
	}
	if err != nil {
		return fail(fmt.Errorf("cudafuse: allocate fused mhc: %w", err))
	}
	if ctx != nil && ctx.Err() != nil {
		return fail(ctx.Err())
	}
	k, err := ops.compile(s)
	if k != nil {
		e.kernels = append(e.kernels, k)
	}
	if err != nil {
		var pending io.Closer
		if errors.As(err, &pending) {
			e.pending = append(e.pending, pending)
		}
		return fail(fmt.Errorf("cudafuse: compile fused mhc: %w", err))
	}
	e.receipts = append(e.receipts, k.receipt)
	if ctx != nil && ctx.Err() != nil {
		return fail(ctx.Err())
	}
	e.attempted = true
	if err := k.launch(append(append([]driver.DevicePtr(nil), ptrs...), b.ptr), e.n); err != nil {
		return fail(fmt.Errorf("cudafuse: launch fused mhc: %w", err))
	}
	e.stats.Launches++
	if err := e.Wait(ctx); err != nil {
		return e, err
	}
	return e, nil
}
