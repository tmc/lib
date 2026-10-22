//go:build darwin && arm64

package main

import "fmt"

type metalDSparkFeatureSnapshot struct {
	owner          *metalDSparkFeatures
	journal        *dsparkFeatureJournal
	data           []float32
	modelSHA       string
	targets        []int
	context, block int
}

func (f *metalDSparkFeatures) snapshot() (*metalDSparkFeatureSnapshot, error) {
	if f == nil || f.closed || f.request == nil || f.request.closed || f.journal == nil || f.journal.frontier != f.request.runtime.pos {
		return nil, fmt.Errorf("metal dspark snapshot: incomplete request frontier")
	}
	if f.batch != nil {
		return nil, fmt.Errorf("metal dspark snapshot: unpublished producer")
	}
	if f.support != nil && f.support.batch != nil {
		return nil, fmt.Errorf("metal dspark snapshot: live support consumer")
	}
	if err := f.request.runtime.synchronizeState(); err != nil {
		return nil, err
	}
	if err := f.journal.validate(f.request.runtime.pos); err != nil {
		return nil, err
	}
	data := append([]float32(nil), f.data.HostSlice()...)
	if !allFinite(data) {
		return nil, fmt.Errorf("metal dspark snapshot: nonfinite feature ring")
	}
	return &metalDSparkFeatureSnapshot{owner: f, journal: f.journal.clone(), data: data, modelSHA: f.weights.model.sha256, targets: append([]int(nil), f.targets...), context: f.request.runtime.ctx, block: f.weights.model.block}, nil
}

// Validate precedes the target Restore write. Publication below cannot fail:
// both the complete device extent and replayable journal were checked here.
func (f *metalDSparkFeatures) validateSnapshot(v *metalDSparkFeatureSnapshot, position int) error {
	if f == nil || f.closed || v == nil || v.owner != f || v.journal == nil || f.journal == nil || v.journal.epoch == 0 || len(v.data) != f.data.Len() || len(v.journal.rows) != len(f.journal.rows) || !allFinite(v.data) {
		return fmt.Errorf("metal dspark restore: foreign feature owner or extent")
	}
	if f.batch != nil && (!f.batch.done || !metalGroupedTerminal(f.batch)) {
		return fmt.Errorf("metal dspark restore: unpublished producer")
	}
	if f.support != nil && f.support.batch != nil {
		return fmt.Errorf("metal dspark restore: live support consumer")
	}
	return v.journal.validate(position)
}

func (f *metalDSparkFeatures) publishSnapshot(v *metalDSparkFeatureSnapshot) {
	copy(f.data.HostSlice(), v.data)
	f.journal = v.journal.clone()
	f.epoch = f.journal.epoch // replayable capture generation, not State epoch
	f.position, f.token = -1, -1
	f.batch = nil
	f.groupRows = nil
	clear(f.captured)
	f.status.HostSlice()[0] = 0
}
