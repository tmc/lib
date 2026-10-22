//go:build darwin && arm64

package main

import (
	"fmt"
	"math"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
)

func planMetalExpertLayers(layers []metalLayer) error {
	names := map[uint32]string{typeIQ2XXS: "IQ2_XXS", typeQ2K: "Q2_K", typeQ4K: "Q4_K"}
	records := make([]dsv4tools.ExpertLayoutRecord, len(layers))
	for i, l := range layers {
		records[i] = dsv4tools.ExpertLayoutRecord{Layer: i,
			GateType: names[l.expertGate.typeID], UpType: names[l.expertUp.typeID], DownType: names[l.expertDown.typeID],
			GateDims: l.expertGate.dimensions, UpDims: l.expertUp.dimensions, DownDims: l.expertDown.dimensions}
	}
	_, err := dsv4tools.PlanExpertDispatch(records, dsv4tools.ExpertBackendMetal)
	return err
}

// metalQ4Experts belongs to one request's operations. Q2 requests never
// compile or retain this pipeline. Inputs remain float32, including the
// independent SwiGLU vector for each selected down-projection slot.
type metalQ4Experts struct {
	library  apple.MTLLibrary
	pipeline apple.MTLComputePipelineState
}

func (q *metalQ4Experts) Close() {
	if q.pipeline != nil {
		objectivec.ObjectFromID(q.pipeline.GetID()).Release()
	}
	if q.library != nil {
		objectivec.ObjectFromID(q.library.GetID()).Release()
	}
	q.pipeline, q.library = nil, nil
}

func (o *metalPinnedOps) ensureQ4Experts() (*metalQ4Experts, error) {
	if o.q4Experts != nil {
		return o.q4Experts, nil
	}
	q := new(metalQ4Experts)
	var err error
	q.library, err = o.dev.CompileLibrary(metalQ4ExpertsMSL)
	if err != nil {
		return nil, fmt.Errorf("Metal Q4 experts library: %w", err)
	}
	fn := q.library.NewFunctionWithName("q4_selected_f32")
	if fn == nil || fn.GetID() == 0 {
		q.Close()
		return nil, fmt.Errorf("Metal Q4 experts: missing kernel")
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	q.pipeline, err = o.dev.Raw().NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		q.Close()
		return nil, fmt.Errorf("Metal Q4 experts pipeline: %w", err)
	}
	o.q4Experts = q
	return q, nil
}

func checkMetalQ4Selected(t boundMetalTensor, ids, input, output int, slotted bool) error {
	if t.typeID != typeQ4K || len(t.dimensions) != 3 {
		return fmt.Errorf("Metal Q4 experts: invalid tensor type or rank")
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	if cols == 0 || cols%256 != 0 || rows == 0 || experts == 0 || ids <= 0 || ids > 6 || rows > math.MaxInt32/uint64(ids) || cols > math.MaxUint32 || experts > math.MaxUint32 {
		return fmt.Errorf("Metal Q4 experts: invalid geometry")
	}
	n := uint64(1)
	if slotted {
		n = uint64(ids)
	}
	if uint64(input) != cols*n || uint64(output) != rows*uint64(ids) || cols/256 > math.MaxUint64/144/rows/experts || uint64(t.data.Len()) != cols/256*144*rows*experts {
		return fmt.Errorf("Metal Q4 experts: invalid buffer shape")
	}
	return nil
}

func (b *metalBatch) EncodeMatvecQ4Selected(t boundMetalTensor, ids metalmem.Slice[int32], input, output metalmem.Slice[float32], slotted bool) error {
	if err := b.ready(); err != nil {
		return err
	}
	if err := checkMetalQ4Selected(t, ids.Len(), input.Len(), output.Len(), slotted); err != nil {
		return err
	}
	q, err := b.ops.ensureQ4Experts()
	if err != nil {
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

// Q4_K uses 144-byte blocks: two half super-scales, twelve packed group
// scale/min bytes, then 128 packed nibbles. Each simdgroup owns one row/slot.
// No invalid selected ID leaves stale output: it writes a NaN for the caller's
// finite-state gate. This is not admission for an unregistered model.
const metalQ4ExpertsMSL = `
#include <metal_stdlib>
using namespace metal;
kernel void q4_selected_f32(device const uchar *weights [[buffer(0)]],
 device const int *ids [[buffer(1)]], device const float *input [[buffer(2)]],
 device float *output [[buffer(3)]], constant uint &cols [[buffer(4)]],
 constant uint &rows [[buffer(5)]], constant uint &experts [[buffer(6)]],
 constant uint &per_slot [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]]) {
 uint row=group.x,slot=group.y;
 int expert=ids[slot];
 if(expert<0 || uint(expert)>=experts) { if(lane==0)output[slot*rows+row]=NAN; return; }
 uint nb=cols/256;
 device const uchar *w=weights+(uint64_t(expert)*rows+row)*nb*144;
 device const float *x=input+(per_slot ? uint64_t(slot)*cols : 0);
 float total=0;
 for(uint block=0;block<nb;block++) {
  device const uchar *p=w+block*144;
  float d=float(*reinterpret_cast<device const half *>(p));
  float dm=float(*reinterpret_cast<device const half *>(p+2));
  device const uchar *s=p+4;
  for(uint g=0;g<8;g++) {
   uint sc=g<4 ? (s[g]&63) : ((s[g+4]&15)|((s[g-4]>>6)<<4));
   uint mn=g<4 ? (s[g+4]&63) : ((s[g+4]>>4)|((s[g]>>6)<<4));
   uint q=(p[16+(g/2)*32+lane]>>((g&1)*4))&15;
   total+=(d*float(sc)*float(q)-dm*float(mn))*x[block*256+g*32+lane];
  }
 }
 total=simd_sum(total);
 if(lane==0)output[slot*rows+row]=total;
}
`
