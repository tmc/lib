//go:build darwin && arm64

package main

import (
	"fmt"

	applemetal "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

const metalDSparkFeatureMSL = `
#include <metal_stdlib>
using namespace metal;
kernel void dspark_capture_feature(
 device const float *hc [[buffer(0)]], device float *features [[buffer(1)]],
 device atomic_uint *status [[buffer(2)]], constant uint &slot [[buffer(3)]],
 uint d [[thread_position_in_grid]]) {
 if(d>=4096)return;
 float mean=0.0f;
 for(uint lane=0;lane<4;lane++)mean+=hc[lane*4096+d]*0.25f;
 features[slot*4096+d]=mean;
 if(!isfinite(mean))atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
}
`

// metalDSparkFeatures owns only actual target captures. The caller must encode
// Capture after the selected layer's post-FFN HC and Seal only after completion.
// It is not a support prediction cache and does not run target Eval as a drafter.
type metalDSparkFeatures struct {
	request         *metalRequestState
	weights         *metalDSparkWeights
	journal         *dsparkFeatureJournal
	targets         []int
	captured        []bool
	position, token int
	epoch           uint64
	batch           *metalBatch
	groupRows       []dsparkFeatureRow
	arena           *metalmem.Arena
	data            metalmem.Slice[float32]
	status          metalmem.Slice[uint32]
	library         applemetal.MTLLibrary
	pipeline        applemetal.MTLComputePipelineState
	closed          bool
	support         *metalDSparkNative
}

func newMetalDSparkFeatures(request *metalRequestState, weights *metalDSparkWeights, epoch uint64) (f *metalDSparkFeatures, err error) {
	return newMetalDSparkFeaturesWithFactory(request, weights, epoch, metalDSparkFeatureFactory{
		allocate: func(f *metalDSparkFeatures, count int) {
			f.arena = metalmem.NewArena(request.runtime.dev)
			f.data = metalmem.AllocIn[float32](f.arena, "dspark-actual-features", count)
			f.status = metalmem.AllocIn[uint32](f.arena, "dspark-feature-status", 1)
		},
		compile: func(f *metalDSparkFeatures) (err error) {
			f.library, err = request.runtime.dev.CompileLibrary(metalDSparkFeatureMSL)
			if err != nil {
				return err
			}
			f.pipeline, err = request.runtime.dev.NewComputePipeline(f.library, "dspark_capture_feature")
			return err
		},
	})
}

type metalDSparkFeatureFactory struct {
	allocate func(*metalDSparkFeatures, int)
	compile  func(*metalDSparkFeatures) error
}

func newMetalDSparkFeaturesWithFactory(request *metalRequestState, weights *metalDSparkWeights, epoch uint64, factory metalDSparkFeatureFactory) (f *metalDSparkFeatures, err error) {
	if request == nil || request.closed || request.runtime == nil || request.runtime.dev == nil || weights == nil || weights.device != request.runtime.dev || weights.model == nil || epoch == 0 {
		return nil, fmt.Errorf("metal dspark features: owned request and support required")
	}
	if err := weights.acquire(); err != nil {
		return nil, err
	}
	f = &metalDSparkFeatures{request: request, weights: weights, targets: append([]int(nil), weights.model.targets...), epoch: epoch, position: -1, token: -1}
	owner := f
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, owner)
		}
	}()
	rows, count, err := dsparkFeatureJournalShape(request.runtime.ctx, weights.model.block, len(f.targets), Flash.Embed)
	if err != nil {
		return nil, err
	}
	f.journal, err = newDSparkFeatureJournal(rows, epoch)
	if err != nil {
		return nil, err
	}
	f.captured = make([]bool, len(f.targets))
	factory.allocate(f, count)
	if err = f.arena.Err(); err != nil {
		return nil, fmt.Errorf("metal dspark features allocate: %w", err)
	}
	if err = factory.compile(f); err != nil {
		return nil, err
	}
	return f, nil
}

func attachMetalDSparkFeatures(request *metalRequestState, weights *metalDSparkWeights, epoch uint64) (*metalDSparkFeatures, error) {
	if request == nil || request.runtime == nil || request.runtime.streaming != nil || request.runtime.pos != 0 || request.runtime.dsparkFeatures != nil {
		return nil, fmt.Errorf("metal dspark features: attach requires fresh uncaptured request")
	}
	f, err := newMetalDSparkFeatures(request, weights, epoch)
	if err != nil {
		return nil, err
	}
	request.runtime.dsparkFeatures = f
	return f, nil
}

func (f *metalDSparkFeatures) begin(position, token int, epoch uint64) error {
	if f == nil || f.closed || f.request.closed || f.request.runtime.failed != nil || epoch != f.epoch || position != f.journal.frontier || position != f.request.runtime.pos || token < 0 || token >= Flash.Vocab {
		return fmt.Errorf("metal dspark features: foreign or noncontiguous frontier")
	}
	if f.batch != nil && !metalGroupedTerminal(f.batch) {
		return fmt.Errorf("metal dspark features: previous capture remains live")
	}
	f.position, f.token, f.batch = position, token, nil
	f.groupRows = nil
	clear(f.captured)
	f.status.HostSlice()[0] = 0
	return nil
}

func (f *metalDSparkFeatures) beginGrouped(position, token int, batch *metalBatch) error {
	if f == nil || f.closed || batch == nil || batch.ops != f.request.runtime.ops || token < 0 || token >= Flash.Vocab || len(f.groupRows) >= 8 || position != f.journal.frontier+len(f.groupRows) {
		return fmt.Errorf("metal dspark grouped capture: invalid row frontier")
	}
	if len(f.groupRows) == 0 {
		if f.batch != nil {
			return fmt.Errorf("metal dspark grouped capture: pending prior producer")
		}
		f.status.HostSlice()[0] = 0
	}
	f.position, f.token = position, token
	clear(f.captured)
	return nil
}

func (f *metalDSparkFeatures) finishGroupedRow() error {
	if f == nil || f.batch == nil {
		return fmt.Errorf("metal dspark grouped capture: missing producer")
	}
	for _, done := range f.captured {
		if !done {
			return fmt.Errorf("metal dspark grouped capture: selected layer missing")
		}
	}
	f.groupRows = append(f.groupRows, dsparkFeatureRow{position: f.position, token: f.token, epoch: f.epoch})
	return nil
}

func (f *metalDSparkFeatures) sealGrouped(count int) error {
	if f == nil || count < 1 || len(f.groupRows) != count || f.batch == nil || !f.batch.done || f.batch.buffer.Status() != applemetal.MTLCommandBufferStatusCompleted || f.status.HostSlice()[0] != 0 {
		return fmt.Errorf("metal dspark grouped capture: invalid terminal group")
	}
	j := f.journal.clone()
	for _, row := range f.groupRows {
		if err := j.append(row.position, row.token, row.epoch); err != nil {
			return err
		}
	}
	f.journal = j
	f.batch = nil
	f.groupRows = nil
	return nil
}

func (f *metalDSparkFeatures) capture(batch *metalBatch, layer, position int) error {
	if f == nil || f.closed || batch == nil || batch.ops != f.request.runtime.ops || position != f.position {
		return fmt.Errorf("metal dspark features: capture ownership mismatch")
	}
	slot := -1
	for i, want := range f.targets {
		if layer == want {
			slot = i
			break
		}
	}
	if slot < 0 {
		return nil
	}
	if f.captured[slot] {
		return fmt.Errorf("metal dspark features: duplicate selected layer")
	}
	if f.batch != nil && f.batch != batch && !metalGroupedTerminal(f.batch) {
		return fmt.Errorf("metal dspark features: previous producer remains live")
	}
	if err := batch.ready(); err != nil {
		return err
	}
	enc := batch.encoder()
	enc.SetComputePipelineState(f.pipeline)
	hc := f.request.runtime.state.hc[0]
	enc.SetBufferWithOffsetAtIndex(hc.Buffer(), hc.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(f.data.Buffer(), f.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(f.status.Buffer(), f.status.Offset(), 2)
	libmetal.SetScalar(enc, uint32((position%len(f.journal.rows))*len(f.targets)+slot), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(4096), libmetal.Threads(256))
	batch.encoded++
	f.batch = batch
	f.captured[slot] = true
	return nil
}

func (f *metalDSparkFeatures) seal(epoch uint64) error {
	if f == nil || f.closed || epoch != f.epoch || f.batch == nil || !metalGroupedTerminal(f.batch) || f.batch.buffer.Status() != applemetal.MTLCommandBufferStatusCompleted {
		return fmt.Errorf("metal dspark features: capture incomplete")
	}
	for _, done := range f.captured {
		if !done {
			return fmt.Errorf("metal dspark features: selected layer missing")
		}
	}
	if f.status.HostSlice()[0] != 0 {
		return fmt.Errorf("metal dspark features: nonfinite actual target capture")
	}
	if err := f.journal.append(f.position, f.token, epoch); err != nil {
		return err
	}
	f.batch = nil
	return nil
}

func (f *metalDSparkFeatures) Close() error {
	if f == nil || f.closed {
		return nil
	}
	if f.support != nil {
		if err := f.support.Close(); err != nil {
			return err
		}
	}
	if f.batch != nil {
		// The target owns this buffer: a feature consumer cannot discard its
		// still-unsubmitted encoder or release captures it may later submit.
		if !f.batch.done || !metalGroupedTerminal(f.batch) {
			return fmt.Errorf("metal dspark features: live producer retains buffers")
		}
	}
	if err := f.weights.release(); err != nil {
		return err
	}
	if f.pipeline != nil && f.pipeline.GetID() != 0 {
		objectivec.ObjectFromID(f.pipeline.GetID()).Release()
	}
	if f.library != nil && f.library.GetID() != 0 {
		objectivec.ObjectFromID(f.library.GetID()).Release()
	}
	if f.arena != nil {
		f.arena.Free()
	}
	f.closed = true
	if f.request != nil && f.request.runtime != nil && f.request.runtime.dsparkFeatures == f {
		f.request.runtime.dsparkFeatures = nil
	}
	return nil
}
