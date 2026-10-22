//go:build linux

package main

import (
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Target snapshots retain the derived selected-layer features, not the
// drafter's future/private KV. The drafter lifecycle owns that separate cache.
type cudaDSparkFeatureSnapshot struct {
	owner      *cudaDSparkFeatures
	features   cudamem.Slice[float32]
	status     cudamem.Slice[uint32]
	captured   []bool
	position   int
	token      int
	generation uint64
	published  bool
	modelSHA   string
	context    int
	block      int
	targets    []int
	history    cudamem.Slice[float32]
	journal    *dsparkFeatureJournal
}

func (f *cudaDSparkFeatures) snapshot(a *cudamem.Arena) (*cudaDSparkFeatureSnapshot, error) {
	if f == nil {
		return nil, nil
	}
	if !f.bound() || f.r.dspark != nil {
		return nil, fmt.Errorf("cuda dspark snapshot: active or unbound capture")
	}
	values, err := cloneCUDAFloat(a, "dspark-selected-features", f.features)
	if err != nil {
		return nil, err
	}
	status, err := cloneCUDAUint32(a, "dspark-selected-status", f.status)
	if err != nil {
		return nil, err
	}
	v := &cudaDSparkFeatureSnapshot{owner: f, features: values, status: status, captured: append([]bool(nil), f.captured...), position: f.position, token: f.token, generation: f.generation, published: f.published, modelSHA: f.modelSHA, context: f.r.ctx, block: f.block, targets: append([]int(nil), f.targets...)}
	if f.history != nil {
		v.history, err = cloneCUDAFloat(a, "dspark-feature-history", f.history.values)
		if err != nil {
			return nil, err
		}
		v.journal = f.history.journal.clone()
	}
	return v, nil
}

func (f *cudaDSparkFeatures) validateSnapshot(v *cudaDSparkFeatureSnapshot, position int, generation uint64) error {
	if f == nil && v == nil {
		return nil
	}
	if f == nil || v == nil || !f.bound() || v.owner != f || v.modelSHA != f.modelSHA || !slices.Equal(v.targets, f.targets) || f.r.dspark != nil || v.features.Len() != f.features.Len() || v.status.Len() != f.status.Len() || len(v.captured) != len(f.captured) {
		return fmt.Errorf("cuda dspark restore: feature snapshot owner or shape mismatch")
	}
	if position < 0 || position > f.r.ctx || v.generation > generation {
		return fmt.Errorf("cuda dspark restore: invalid feature generation or frontier")
	}
	if f.history != nil {
		if v.history.Len() != f.history.values.Len() || v.journal == nil || len(v.journal.rows) != len(f.history.journal.rows) || v.journal.epoch == 0 || v.journal.frontier != position {
			return fmt.Errorf("cuda dspark restore: journal shape or frontier mismatch")
		}
		if err := v.journal.validate(position); err != nil {
			return fmt.Errorf("cuda dspark restore: incomplete trusted journal: %w", err)
		}
	}
	if v.published {
		if v.position < 0 || v.position != position-1 || v.token < 0 || v.token >= Flash.Vocab || v.generation == ^uint64(0) || v.generation+1 != generation {
			return fmt.Errorf("cuda dspark restore: published frontier mismatch")
		}
		if f.history != nil {
			row, err := v.journal.row(v.position, v.journal.epoch)
			if err != nil || row.token != v.token {
				return fmt.Errorf("cuda dspark restore: published token and journal mismatch")
			}
		}
		for _, captured := range v.captured {
			if !captured {
				return fmt.Errorf("cuda dspark restore: published feature incomplete")
			}
		}
	} else {
		if v.position != -1 || v.token != -1 {
			return fmt.Errorf("cuda dspark restore: pending feature snapshot")
		}
		for _, captured := range v.captured {
			if captured {
				return fmt.Errorf("cuda dspark restore: unpublished feature captured")
			}
		}
	}
	return nil
}

func (f *cudaDSparkFeatures) restore(v *cudaDSparkFeatureSnapshot, position int, generation uint64) error {
	if err := f.validateSnapshot(v, position, generation); err != nil {
		return err
	}
	if f == nil {
		return nil
	}
	if err := f.features.CopyDeviceToDevice(v.features); err != nil {
		return err
	}
	if err := f.status.CopyDeviceToDevice(v.status); err != nil {
		return err
	}
	if f.history != nil {
		if err := f.history.values.CopyDeviceToDevice(v.history); err != nil {
			return err
		}
		f.history.journal = v.journal.clone()
	}
	copy(f.captured, v.captured)
	f.position, f.generation, f.published = v.position, v.generation, v.published
	f.token = v.token
	return nil
}
