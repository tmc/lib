//go:build darwin && arm64

package main

import (
	"fmt"
	"math"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// metalMXFP4Experts is an independently owned selected projection pipeline.
// The caller must keep it alive until its batches have completed.
type metalMXFP4Experts struct {
	library  apple.MTLLibrary
	pipeline apple.MTLComputePipelineState
}

func newMetalMXFP4Experts(dev *libmetal.Device, source string) (*metalMXFP4Experts, error) {
	if dev == nil {
		return nil, fmt.Errorf("MXFP4 experts: nil device")
	}
	q := new(metalMXFP4Experts)
	var err error
	q.library, err = dev.CompileLibrary(source)
	if err != nil {
		return nil, fmt.Errorf("compile MXFP4 experts: %w", err)
	}
	fn := q.library.NewFunctionWithName("mxfp4_selected_f32")
	if fn == nil || fn.GetID() == 0 {
		q.Close()
		return nil, fmt.Errorf("MXFP4 experts: missing kernel")
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	q.pipeline, err = dev.Raw().NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		q.Close()
		return nil, fmt.Errorf("compile MXFP4 pipeline: %w", err)
	}
	return q, nil
}

func (q *metalMXFP4Experts) Close() {
	if q == nil {
		return
	}
	if q.pipeline != nil {
		objectivec.ObjectFromID(q.pipeline.GetID()).Release()
		q.pipeline = nil
	}
	if q.library != nil {
		objectivec.ObjectFromID(q.library.GetID()).Release()
		q.library = nil
	}
}

func checkMetalMXFP4Selected(t boundMetalTensor, slots, input, output int, slotted bool) error {
	if t.typeID != typeMXFP4 || len(t.dimensions) != 3 || slots < 1 || slots > 6 || input < 0 || output < 0 {
		return fmt.Errorf("MXFP4 experts: invalid tensor shape")
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32/uint64(slots) || experts == 0 || experts > math.MaxInt32 {
		return fmt.Errorf("MXFP4 experts: invalid geometry")
	}
	n := uint64(1)
	if slotted {
		n = uint64(slots)
	}
	if cols/32 > math.MaxUint64/17/rows/experts || uint64(t.data.Len()) != cols/32*17*rows*experts || uint64(input) != cols*n || uint64(output) != rows*uint64(slots) {
		return fmt.Errorf("MXFP4 experts: invalid buffer shape")
	}
	return nil
}

func (b *metalBatch) EncodeMatvecMXFP4Selected(q *metalMXFP4Experts, t boundMetalTensor, ids metalmem.Slice[int32], input, output metalmem.Slice[float32], slotted bool) error {
	if err := b.ready(); err != nil {
		return err
	}
	if q == nil || q.pipeline == nil {
		return fmt.Errorf("MXFP4 experts: closed pipeline")
	}
	if err := checkMetalMXFP4Selected(t, ids.Len(), input.Len(), output.Len(), slotted); err != nil {
		return err
	}
	enc := b.encoder()
	enc.SetComputePipelineState(q.pipeline)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(ids.Buffer(), ids.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(input.Buffer(), input.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(output.Buffer(), output.Offset(), 3)
	libmetal.SetScalar(enc, uint32(t.dimensions[0]), 4)
	libmetal.SetScalar(enc, uint32(t.dimensions[1]), 5)
	libmetal.SetScalar(enc, uint32(t.dimensions[2]), 6)
	var perSlot uint32
	if slotted {
		perSlot = 1
	}
	libmetal.SetScalar(enc, perSlot, 7)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(apple.MTLSize{Width: uint(t.dimensions[1]), Height: uint(ids.Len()), Depth: 1}, libmetal.Threads(32))
	b.encoded++
	return nil
}

const metalMXFP4ExpertsMSL = `
#include <metal_stdlib>
using namespace metal;
kernel void mxfp4_selected_f32(device const uchar *weights [[buffer(0)]],
 device const int *ids [[buffer(1)]], device const float *input [[buffer(2)]],
 device float *output [[buffer(3)]], constant uint &cols [[buffer(4)]],
 constant uint &rows [[buffer(5)]], constant uint &experts [[buffer(6)]],
 constant uint &per_slot [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]]) {
 uint row=group.x,slot=group.y;
 int expert=ids[slot];
 if(expert<0 || uint(expert)>=experts) { if(lane==0)output[slot*rows+row]=NAN; return; }
 uint nb=cols/32;
 device const uchar *w=weights+(uint64_t(expert)*rows+row)*nb*17;
 device const float *x=input+(per_slot ? uint64_t(slot)*cols : 0);
 const float values[8]={0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
 float total=0;
 for(uint block=0;block<nb;block++) {
  device const uchar *p=w+block*17;
  uint scale_bits=p[0]==0 ? 0x00400000u : uint(p[0])<<23;
  float scale=as_type<float>(scale_bits);
  uint code=(p[1+(lane&15)]>>((lane>>4)*4))&15;
  float v=values[code&7];
  if(code&8)v=-v;
  total+=(scale*v)*x[block*32+lane];
 }
 total=simd_sum(total);
 if(lane==0)output[slot*rows+row]=total;
}
`
