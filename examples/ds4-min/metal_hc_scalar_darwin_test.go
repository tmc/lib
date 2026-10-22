//go:build darwin && arm64

package main

import (
	"context"
	"fmt"
	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

// Historical scalar HC implementation. Retained only as a regression control;
// its earlier failures are not relabeled when production arithmetic changes.
type metalHCScalarOps struct {
	dev     *libmetal.Device
	library apple.MTLLibrary
	kernels struct {
		HCPre  apple.MTLComputePipelineState `metal:"hc_pre"`
		HCPost apple.MTLComputePipelineState `metal:"hc_post"`
	}
}

func newMetalHCScalarOps(dev *libmetal.Device) (o *metalHCScalarOps, err error) {
	o = &metalHCScalarOps{dev: dev}
	owned := o
	defer func() {
		if err != nil {
			owned.Close()
		}
	}()
	o.library, err = dev.CompileLibrary(metalPinnedMSL + metalHCScalarMSL)
	if err != nil {
		return nil, err
	}
	if err = metalx.Bind(dev, o.library, &o.kernels); err != nil {
		return nil, err
	}
	return o, nil
}
func (o *metalHCScalarOps) Close() {
	if o == nil {
		return
	}
	for _, p := range []apple.MTLComputePipelineState{o.kernels.HCPre, o.kernels.HCPost} {
		if p != nil && p.GetID() != 0 {
			objectivec.ObjectFromID(p.GetID()).Release()
		}
	}
	if o.library != nil && o.library.GetID() != 0 {
		objectivec.ObjectFromID(o.library.GetID()).Release()
	}
	o.library = nil
	o.kernels.HCPre, o.kernels.HCPost = nil, nil
}
func (o *metalHCScalarOps) HCPre(state []float32, hcFn, hcScale, hcBase, norm boundMetalTensor) ([]float32, []float32, error) {
	if len(state) != 4*4096 || hcFn.typeID != 1 || hcScale.typeID != 0 || hcBase.typeID != 0 || norm.typeID != 0 {
		return nil, nil, fmt.Errorf("HC pre: invalid tensor profile")
	}
	in, err := metalmem.AllocFromSlice(o.dev, state)
	if err != nil {
		return nil, nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](o.dev, 2*4096)
	if err != nil {
		return nil, nil, err
	}
	defer out.Free()
	cb := o.dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.HCPre)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	for i, tensor := range []boundMetalTensor{hcFn, hcScale, hcBase, norm} {
		enc.SetBufferWithOffsetAtIndex(tensor.data.Buffer(), tensor.data.Offset(), uint(i+1))
	}
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	cb.Commit()
	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, nil, fmt.Errorf("HC pre: %w", err)
	}
	host := out.HostSlice()
	return append([]float32(nil), host[:4096]...), append([]float32(nil), host[4096:]...), nil
}

func (o *metalHCScalarOps) HCPost(state, block []float32, hcFn, hcScale, hcBase boundMetalTensor) ([]float32, error) {
	if len(state) != 4*4096 || len(block) != 4096 || hcFn.typeID != 1 || hcScale.typeID != 0 || hcBase.typeID != 0 {
		return nil, fmt.Errorf("HC post: invalid tensor profile")
	}
	stateBuffer, err := metalmem.AllocFromSlice(o.dev, state)
	if err != nil {
		return nil, err
	}
	defer stateBuffer.Free()
	blockBuffer, err := metalmem.AllocFromSlice(o.dev, block)
	if err != nil {
		return nil, err
	}
	defer blockBuffer.Free()
	out, err := metalmem.Alloc[float32](o.dev, len(state))
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.HCPost)
	enc.SetBufferWithOffsetAtIndex(stateBuffer.Buffer(), stateBuffer.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(blockBuffer.Buffer(), blockBuffer.Offset(), 1)
	for i, tensor := range []boundMetalTensor{hcFn, hcScale, hcBase} {
		enc.SetBufferWithOffsetAtIndex(tensor.data.Buffer(), tensor.data.Offset(), uint(i+2))
	}
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	cb.Commit()
	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, fmt.Errorf("HC post: %w", err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

const metalHCScalarMSL = `
kernel void hc_pre(
    device const float *state [[buffer(0)]], device const half *hc_fn [[buffer(1)]],
    device const float *scale [[buffer(2)]], device const float *base [[buffer(3)]],
    device const float *norm_weight [[buffer(4)]], device float *out [[buffer(5)]]) {
  float ss=0.0f;for(uint i=0;i<16384;i++)ss+=state[i]*state[i];
  float inv=rsqrt(ss/16384.0f+1.0e-6f),mix[24];
  for(uint row=0;row<24;row++){float sum=0.0f;const device half *wr=hc_fn+uint64_t(row)*16384;for(uint i=0;i<16384;i++)sum+=float(wr[i])*state[i]*inv;mix[row]=sum;}
  float pre[4],post[4],comb[16];hc_split4(mix,scale,base,pre,post,comb);
  float cur_ss=0.0f;
  for(uint d=0;d<4096;d++){float v=0.0f;for(uint h=0;h<4;h++)v+=state[h*4096+d]*pre[h];out[d]=v;cur_ss+=v*v;}
  float cur_inv=rsqrt(cur_ss/4096.0f+1.0e-6f);
  for(uint d=0;d<4096;d++)out[4096+d]=out[d]*cur_inv*norm_weight[d];
}

kernel void hc_post(
    device const float *state [[buffer(0)]], device const float *block [[buffer(1)]],
    device const half *hc_fn [[buffer(2)]], device const float *scale [[buffer(3)]],
    device const float *base [[buffer(4)]], device float *out [[buffer(5)]]) {
  float ss=0.0f;for(uint i=0;i<16384;i++)ss+=state[i]*state[i];
  float inv=rsqrt(ss/16384.0f+1.0e-6f),mix[24];
  for(uint row=0;row<24;row++){float sum=0.0f;const device half *wr=hc_fn+uint64_t(row)*16384;for(uint i=0;i<16384;i++)sum+=float(wr[i])*state[i]*inv;mix[row]=sum;}
  float pre[4],post[4],comb[16];hc_split4(mix,scale,base,pre,post,comb);
  for(uint dst=0;dst<4;dst++)for(uint d=0;d<4096;d++){float v=block[d]*post[dst];for(uint src=0;src<4;src++)v+=comb[dst+src*4]*state[src*4096+d];out[dst*4096+d]=v;}
}

`
