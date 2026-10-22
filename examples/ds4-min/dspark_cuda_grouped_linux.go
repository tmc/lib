//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type cudaDSparkGroupedCapture struct {
	start      int
	tokens     []int
	captured   []bool
	generation uint64
}

func (f *cudaDSparkFeatures) beginGrouped(tokens []int, position int) error {
	if !f.bound() || position < 0 || position > f.r.ctx || len(tokens) > f.r.ctx-position || f.history == nil || f.history.journal == nil || f.r.dspark != nil || f.r.failed != nil || position != f.r.pos || position != f.history.journal.frontier || len(tokens) < 2 || len(tokens) > len(f.history.journal.rows) || uint64(len(tokens)) > ^uint64(0)-f.request.generation {
		return fmt.Errorf("cuda dspark grouped: invalid target frontier")
	}
	if err := dsparkRequestOpen(f.r); err != nil {
		return err
	}
	for _, token := range tokens {
		if token < 0 || token >= Flash.Vocab {
			return fmt.Errorf("cuda dspark grouped: invalid target token")
		}
	}
	f.grouped = &cudaDSparkGroupedCapture{start: position, tokens: append([]int(nil), tokens...), captured: make([]bool, len(tokens)*len(f.targets)), generation: f.request.generation}
	f.published = false
	if err := f.status.CopyFromHost([]uint32{0}); err != nil {
		f.invalidate()
		return err
	}
	return nil
}

func (f *cudaDSparkFeatures) captureGrouped(layer, position int, rows []*cudaTokenState) error {
	if !f.bound() || f.r.dspark != f || f.grouped == nil || f.history == nil || position != f.grouped.start || f.request.generation != f.grouped.generation || len(rows) != len(f.grouped.tokens) {
		return fmt.Errorf("cuda dspark grouped: inactive or foreign capture")
	}
	if err := dsparkRequestOpen(f.r); err != nil {
		return err
	}
	selected := -1
	for slot, target := range f.targets {
		if layer == target {
			selected = slot
			break
		}
	}
	if selected < 0 {
		return nil
	}
	for row, values := range rows {
		index := row*len(f.targets) + selected
		if values == nil || values.hc[0].Len() != Flash.HC*Flash.Embed || f.grouped.captured[index] {
			return fmt.Errorf("cuda dspark grouped: duplicate or malformed row")
		}
		_, slot, err := f.history.journal.groupedRow(position, row, len(rows), f.history.journal.epoch)
		if err != nil {
			return err
		}
		deviceSlot := slot*len(f.targets) + selected
		if err := f.r.withContext(func() error {
			return cudalaunch.Kernel(f.function).Grid(16, 1, 1).Block(256, 1, 1).On(f.r.stream()).Launch(values.hc[0], f.history.values, f.status, uint32(deviceSlot))
		}); err != nil {
			return err
		}
		f.grouped.captured[index] = true
	}
	return nil
}

func (f *cudaDSparkFeatures) validateGroupedComplete() error {
	if !f.bound() || f.r.dspark != f || f.grouped == nil || f.history == nil || f.r.failed != nil || f.r.pos != f.grouped.start+len(f.grouped.tokens) || f.request.generation != f.grouped.generation {
		return fmt.Errorf("cuda dspark grouped: incomplete target Eval")
	}
	if len(f.grouped.captured) != len(f.grouped.tokens)*len(f.targets) {
		return fmt.Errorf("cuda dspark grouped: malformed capture map")
	}
	for _, captured := range f.grouped.captured {
		if !captured {
			return fmt.Errorf("cuda dspark grouped: selected row missing")
		}
	}
	return nil
}

func (f *cudaDSparkFeatures) sealGrouped() error {
	if err := f.validateGroupedComplete(); err != nil {
		return err
	}
	if err := syncStream(f.r.stream()); err != nil {
		return err
	}
	var status [1]uint32
	if err := f.status.CopyToHost(status[:]); err != nil {
		return err
	}
	if status[0] != 0 {
		return fmt.Errorf("cuda dspark grouped: nonfinite selected feature")
	}
	g := f.grouped
	for row, token := range g.tokens {
		if err := f.history.journal.append(g.start+row, token, f.history.journal.epoch); err != nil {
			return err
		}
	}
	f.position, f.token = g.start+len(g.tokens)-1, g.tokens[len(g.tokens)-1]
	f.generation = g.generation + uint64(len(g.tokens)) - 1
	f.request.generation = f.generation + 1
	values, err := f.history.row(f.position)
	if err != nil {
		return err
	}
	if err := f.features.CopyDeviceToDeviceAsync(values, f.r.stream()); err != nil {
		return err
	}
	if err := syncStream(f.r.stream()); err != nil {
		return err
	}
	for i := range f.captured {
		f.captured[i] = true
	}
	f.grouped, f.published = nil, true
	return nil
}
