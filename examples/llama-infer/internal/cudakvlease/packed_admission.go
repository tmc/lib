package cudakvlease

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/lm/kvcache"
)

// AdmissionStats reports migrations owned by one packed admission attempt.
type AdmissionStats struct {
	AllocatedPages int
	OffloadedPages int
	RestoredPages  int
	MigrationBytes int64
}

// Admit makes every page required by requests device-resident before
// publishing packed metadata. New allocations are returned by PackedBatch.
func (m *PackedMetadata) Admit(ctx context.Context, requests []PackedRequest, blockSize int, completion Completion) (*PackedBatch, AdmissionStats, error) {
	if m == nil || ctx == nil || completion == nil || blockSize <= 0 || len(requests) == 0 {
		return nil, AdmissionStats{}, ErrInvalid
	}
	m.admitMu.Lock()
	defer m.admitMu.Unlock()
	adapter := m.requests.adapter
	got := make([]PackedRequest, len(requests))
	copy(got, requests)
	exclude := make([]Allocation, 0)
	restores := make([]Allocation, 0)
	missing := 0
	for i := range got {
		got[i].Allocations = append([]Allocation(nil), got[i].Allocations...)
		if got[i].Length <= 0 {
			return nil, AdmissionStats{}, ErrInvalid
		}
		blocks := (got[i].Length + blockSize - 1) / blockSize
		if blocks > m.requests.maxBlocks || len(got[i].Allocations) > blocks {
			return nil, AdmissionStats{}, ErrInvalid
		}
		missing += blocks - len(got[i].Allocations)
		for _, allocation := range got[i].Allocations {
			exclude = append(exclude, allocation)
			residency, err := adapter.Residency(allocation)
			if err != nil {
				return nil, AdmissionStats{}, err
			}
			if residency.Tier == "pinned-host" {
				restores = append(restores, allocation)
			} else if residency.Tier != "device" {
				return nil, AdmissionStats{}, ErrInvalid
			}
		}
	}
	required := missing + len(restores)
	free := adapter.NumFreePages()
	var victims []Allocation
	if required > free {
		set, err := adapter.ReserveVictims(required-free, exclude...)
		if err != nil {
			return nil, AdmissionStats{}, fmt.Errorf("cudakvlease: reserve packed admission victims: %w", err)
		}
		for _, reservation := range set.reservations {
			victims = append(victims, Allocation{adapter: adapter, lease: reservation.lease})
		}
		if err := set.Offload(ctx, completion); err != nil {
			return nil, AdmissionStats{}, fmt.Errorf("cudakvlease: offload packed admission victims: %w", err)
		}
	}
	newAllocations := make([]Allocation, 0, missing)
	restored := make([]Allocation, 0, len(restores))
	rollback := func(cause error) (*PackedBatch, AdmissionStats, error) {
		rollbackCtx := context.WithoutCancel(ctx)
		var rollbackErrs []error
		for i := len(newAllocations) - 1; i >= 0; i-- {
			if err := adapter.Retire(rollbackCtx, newAllocations[i], immediateCompletion{}); err != nil {
				rollbackErrs = append(rollbackErrs, fmt.Errorf("retire new page %d: %w", i, err))
			}
		}
		for i := len(restored) - 1; i >= 0; i-- {
			if err := adapter.Offload(rollbackCtx, restored[i], immediateCompletion{}); err != nil {
				rollbackErrs = append(rollbackErrs, fmt.Errorf("re-offload requested page %d: %w", i, err))
			}
		}
		for i := len(victims) - 1; i >= 0; i-- {
			if err := adapter.Restore(rollbackCtx, victims[i]); err != nil {
				rollbackErrs = append(rollbackErrs, fmt.Errorf("restore victim %d: %w", i, err))
			}
		}
		return nil, AdmissionStats{}, errors.Join(cause, errors.Join(rollbackErrs...))
	}
	for _, allocation := range restores {
		if err := ctx.Err(); err != nil {
			return rollback(err)
		}
		if err := adapter.Restore(ctx, allocation); err != nil {
			return rollback(fmt.Errorf("cudakvlease: restore packed page: %w", err))
		}
		restored = append(restored, allocation)
	}
	for i := range got {
		blocks := (got[i].Length + blockSize - 1) / blockSize
		for len(got[i].Allocations) < blocks {
			if err := ctx.Err(); err != nil {
				return rollback(err)
			}
			start := len(got[i].Allocations) * blockSize
			end := min(start+blockSize, got[i].Length)
			allocation, err := adapter.Allocate(got[i].Key, kvcache.Range{Start: start, End: end})
			if err != nil {
				return rollback(fmt.Errorf("cudakvlease: allocate packed page: %w", err))
			}
			got[i].Allocations = append(got[i].Allocations, allocation)
			newAllocations = append(newAllocations, allocation)
		}
	}
	batch, err := m.Populate(ctx, got)
	if err != nil {
		return rollback(fmt.Errorf("cudakvlease: publish packed admission: %w", err))
	}
	batch.allocations = make(map[string][]Allocation, len(got))
	for i := range got {
		batch.allocations[got[i].Key] = append([]Allocation(nil), got[i].Allocations...)
	}
	pageBytes := int64(2 * adapter.config.NumLayers * adapter.valuesPerPage() * 4)
	stats := AdmissionStats{
		AllocatedPages: len(newAllocations), OffloadedPages: len(victims), RestoredPages: len(restored),
		MigrationBytes: int64(len(victims)+len(restored)) * pageBytes,
	}
	return batch, stats, nil
}
