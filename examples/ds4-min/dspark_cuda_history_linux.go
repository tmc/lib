//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaDSparkHistory owns trusted target features, separate from support KV.
// The selected-layer capture's temporary row is copied here only after a
// complete successful target step. Rows are lane-mean feature vectors.
type cudaDSparkHistory struct {
	owner   *cudaDSparkFeatures
	values  cudamem.Slice[float32]
	journal *dsparkFeatureJournal
	width   int
}

func newCUDADSparkHistory(f *cudaDSparkFeatures, block int) (*cudaDSparkHistory, error) {
	if f == nil || !f.bound() || f.r.pos != 0 {
		return nil, fmt.Errorf("cuda dspark history: fresh target frontier required")
	}
	if err := dsparkRequestOpen(f.r); err != nil {
		return nil, err
	}
	rows, values, err := dsparkFeatureJournalShape(f.r.ctx, block, len(f.targets), Flash.Embed)
	if err != nil {
		return nil, err
	}
	j, err := newDSparkFeatureJournal(rows, 1)
	if err != nil {
		return nil, err
	}
	storage, err := cudamem.Alloc[float32](f.r.device, values)
	h := &cudaDSparkHistory{owner: f, values: storage, journal: j, width: len(f.targets) * Flash.Embed}
	if err != nil {
		return nil, finishBackendCleanup(err, h)
	}
	return h, nil
}

func (h *cudaDSparkHistory) record(f *cudaDSparkFeatures) error {
	if h == nil || h.owner != f || !f.bound() || !f.published || h.journal == nil || f.position != h.journal.frontier || f.features.Len() != h.width {
		return fmt.Errorf("cuda dspark history: untrusted or discontinuous feature")
	}
	if err := dsparkRequestOpen(f.r); err != nil {
		return err
	}
	slot := f.position % len(h.journal.rows)
	if err := h.values.Slice(slot*h.width, (slot+1)*h.width).CopyDeviceToDeviceAsync(f.features, f.r.stream()); err != nil {
		return err
	}
	if err := syncStream(f.r.stream()); err != nil {
		return err
	}
	return h.journal.append(f.position, f.token, h.journal.epoch)
}

func (h *cudaDSparkHistory) row(position int) (cudamem.Slice[float32], error) {
	if h == nil || h.owner == nil || !h.owner.bound() || h.journal == nil {
		return cudamem.Slice[float32]{}, fmt.Errorf("cuda dspark history: unavailable")
	}
	if _, err := h.journal.row(position, h.journal.epoch); err != nil {
		return cudamem.Slice[float32]{}, err
	}
	slot := position % len(h.journal.rows)
	return h.values.Slice(slot*h.width, (slot+1)*h.width), nil
}

func (h *cudaDSparkHistory) reset() error {
	if h == nil || h.journal == nil {
		return nil
	}
	if h.journal.epoch == ^uint64(0) {
		return fmt.Errorf("cuda dspark history: epoch exhausted")
	}
	j, err := newDSparkFeatureJournal(len(h.journal.rows), h.journal.epoch+1)
	if err != nil {
		return err
	}
	h.journal = j
	return nil
}

func (h *cudaDSparkHistory) close() error {
	if h == nil {
		return nil
	}
	if err := h.values.Free(); err != nil {
		return err
	}
	h.owner, h.journal = nil, nil
	h.values = cudamem.Slice[float32]{}
	return nil
}

func (h *cudaDSparkHistory) Close() error { return h.close() }
