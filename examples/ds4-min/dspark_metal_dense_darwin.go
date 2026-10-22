//go:build darwin && arm64

package main

import (
	"fmt"

	applemetal "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Native support matrices only. Q8_0 weights retain their original GGUF
// layout; inputs remain F32. No target evaluation or CPU decode fallback.
const metalDSparkDenseMSL = `
#include <metal_stdlib>
using namespace metal;
kernel void dspark_dense(device const uchar *weights [[buffer(0)]],
 device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
 constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]],
 constant uint &batch [[buffer(5)]],constant uint &type [[buffer(6)]],
 uint index [[thread_position_in_grid]]) {
 if(index>=rows*batch)return;
 uint row=index%rows,b=index/rows;
 float sum=0.0f;
 for(uint col=0;col<cols;col++) {
   float value;
   if(type==0)value=reinterpret_cast<device const float *>(weights)[row*cols+col];
   else if(type==1)value=float(reinterpret_cast<device const half *>(weights)[row*cols+col]);
   else {
     uint offset=(row*(cols/32)+col/32)*34;
     float d=float(*reinterpret_cast<device const half *>(weights+offset));
     value=d*float(reinterpret_cast<device const char *>(weights+offset+2)[col%32]);
   }
   sum+=value*input[b*cols+col];
 }
 output[index]=sum;
}
`

type metalDSparkDense struct {
	weights  *metalDSparkWeights
	library  applemetal.MTLLibrary
	pipeline applemetal.MTLComputePipelineState
	batches  []*metalBatch
	closed   bool
}

func newMetalDSparkDense(dev *libmetal.Device, weights *metalDSparkWeights) (d *metalDSparkDense, err error) {
	if dev == nil || weights == nil || weights.device != dev {
		return nil, fmt.Errorf("metal dspark dense: device and support required")
	}
	if err := weights.acquire(); err != nil {
		return nil, err
	}
	d = &metalDSparkDense{weights: weights}
	owner := d
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, owner)
		}
	}()
	d.library, err = dev.CompileLibrary(metalDSparkDenseMSL)
	if err != nil {
		return nil, err
	}
	d.pipeline, err = dev.NewComputePipeline(d.library, "dspark_dense")
	if err != nil {
		return nil, err
	}
	return d, nil
}

func (d *metalDSparkDense) Encode(batch *metalBatch, name string, input, output metalmem.Slice[float32], count int) error {
	if d == nil || d.closed || d.weights == nil || batch == nil || batch.ops == nil || batch.ops.dev != d.weights.device || count < 1 || count > 17 {
		return fmt.Errorf("metal dspark dense: invalid owner or batch")
	}
	t, err := d.weights.tensor(name)
	if err != nil {
		return err
	}
	return d.encodeTensor(batch, t, input, output, count)
}

// EncodeRows selects physical matrix rows, not activation columns. Attention
// output A uses one contiguous 128-row slice per head group.
func (d *metalDSparkDense) EncodeRows(batch *metalBatch, name string, start, rows int, input, output metalmem.Slice[float32], count int) error {
	if d == nil || d.closed || d.weights == nil {
		return fmt.Errorf("metal dspark dense: closed owner")
	}
	t, err := d.weights.tensor(name)
	if err != nil {
		return err
	}
	t, err = t.Rows(start, rows)
	if err != nil {
		return err
	}
	return d.encodeTensor(batch, t, input, output, count)
}

func (d *metalDSparkDense) encodeTensor(batch *metalBatch, t boundMetalTensor, input, output metalmem.Slice[float32], count int) error {
	if d == nil || d.closed || d.weights == nil || d.weights.model == nil || batch == nil || batch.ops == nil || batch.ops.dev != d.weights.device || count < 1 || count > 17 {
		return fmt.Errorf("metal dspark dense: invalid owner or batch")
	}
	if len(t.dimensions) != 2 || t.dimensions[0] == 0 || t.dimensions[1] == 0 || t.dimensions[0] > 1<<20 || t.dimensions[1] > 1<<20 {
		return fmt.Errorf("metal dspark dense: invalid matrix shape")
	}
	cols, rows := int(t.dimensions[0]), int(t.dimensions[1])
	if input.Len() != count*cols || output.Len() != count*rows || (t.typeID != 0 && t.typeID != 1 && t.typeID != 8) || (t.typeID == 8 && cols%32 != 0) {
		return fmt.Errorf("metal dspark dense: unsupported format or extent")
	}
	if err := batch.ready(); err != nil {
		return err
	}
	enc := batch.encoder()
	enc.SetComputePipelineState(d.pipeline)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(input.Buffer(), input.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(output.Buffer(), output.Offset(), 2)
	libmetal.SetScalar(enc, uint32(cols), 3)
	libmetal.SetScalar(enc, uint32(rows), 4)
	libmetal.SetScalar(enc, uint32(count), 5)
	libmetal.SetScalar(enc, t.typeID, 6)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(count*rows), libmetal.Threads(256))
	batch.encoded++
	d.batches = retainMetalDSparkBatch(d.batches, batch)
	return nil
}

func (d *metalDSparkDense) Close() error {
	if d == nil || d.closed {
		return nil
	}
	for _, b := range d.batches {
		if !b.done || !metalGroupedTerminal(b) {
			return fmt.Errorf("metal dspark dense: live consumer retains pipeline and weights")
		}
	}
	if err := d.weights.release(); err != nil {
		return err
	}
	if d.pipeline != nil && d.pipeline.GetID() != 0 {
		objectivec.ObjectFromID(d.pipeline.GetID()).Release()
	}
	if d.library != nil && d.library.GetID() != 0 {
		objectivec.ObjectFromID(d.library.GetID()).Release()
	}
	d.closed = true
	return nil
}

// Completed consumers no longer own resources. Keep each live producer only
// once, so repeated operators and support cycles cannot grow this journal.
func retainMetalDSparkBatch(batches []*metalBatch, batch *metalBatch) []*metalBatch {
	live := batches[:0]
	found := false
	for _, b := range batches {
		if b.done && metalGroupedTerminal(b) {
			continue
		}
		live = append(live, b)
		found = found || b == batch
	}
	if !found {
		live = append(live, batch)
	}
	return live
}
