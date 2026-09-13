package cuptipcsampling

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// read drains an already enabled, exclusively owned sampling context once.
// The caller must finish processing previous output before calling read, and
// must not pass the concurrently written configured output buffer here.
// GetData does not synchronize the device. Even an empty successful read with
// no remaining PCs is not evidence that workload submission has completed.
func (b *samplingBuffer) read(ctx cupti.Context, version uint32, names *samplingNames) (samplingBatch, error) {
	return b.drainOwned(ctx, version, cupti.PCSamplingGetData, names)
}

func (b *samplingBuffer) drain(ctx cupti.Context, version uint32, query func(*cupti.PCSamplingGetDataParams) error, copyName func(uintptr) (string, error)) (samplingBatch, error) {
	out, err := b.fetch(ctx, version, query)
	if err != nil {
		return out, err
	}
	return b.snapshot(copyName)
}

func (b *samplingBuffer) drainOwned(ctx cupti.Context, version uint32, query func(*cupti.PCSamplingGetDataParams) error, names *samplingNames) (samplingBatch, error) {
	if names == nil {
		return samplingBatch{}, errors.New("cuptipcsampling: missing native name owner")
	}
	if err := names.reserve(len(b.pcs)); err != nil {
		return samplingBatch{}, err
	}
	out, err := b.fetch(ctx, version, query)
	if err != nil {
		return out, err
	}
	return b.snapshotOwned(names)
}

func (b *samplingBuffer) snapshotOwned(names *samplingNames) (samplingBatch, error) {
	out := b.diagnostics()
	if names == nil || b.released {
		return out, errors.New("cuptipcsampling: invalid owned snapshot")
	}
	if err := names.reserve(len(b.pcs)); err != nil {
		return out, err
	}
	if err := b.checkOutput(); err != nil {
		return out, err
	}
	for i := uint(0); i < b.data.TotalNumPcs; i++ {
		if err := b.checkPC(i); err != nil {
			return out, err
		}
	}
	// Capture all names before snapshot can stop at a duplicate reason or a
	// failed name copy. Otherwise later allocations would be abandoned.
	var nameErrors []error
	for i := uint(0); i < b.data.TotalNumPcs; i++ {
		if _, err := names.copy(b.pcs[i].FunctionName); err != nil {
			nameErrors = append(nameErrors, err)
		}
	}
	if err := errors.Join(nameErrors...); err != nil {
		return out, err
	}
	return b.snapshot(names.copy)
}

func (b *samplingBuffer) fetch(ctx cupti.Context, version uint32, query func(*cupti.PCSamplingGetDataParams) error) (samplingBatch, error) {
	if b.released || ctx == 0 || query == nil {
		return samplingBatch{}, errors.New("cuptipcsampling: invalid drain request")
	}
	// This complete output layout has a retained C-header witness for every
	// version qualifiedSamplingVersion admits. Admitting another version
	// requires the same output-layout witness, not merely the version gates of
	// unrelated reason or CRC query functions.
	if unsafe.Sizeof(uintptr(0)) != 8 || !qualifiedSamplingVersion(version) {
		return samplingBatch{}, fmt.Errorf("cuptipcsampling: unqualified drain layout for CUPTI %d", version)
	}
	b.reset()
	ptr := uintptr(unsafe.Pointer(&b.data))
	p := cupti.PCSamplingGetDataParams{Size: getDataParamsSize(), Ctx: ctx, PcSamplingData: ptr}
	err := query(&p)
	out := b.diagnostics()
	if err != nil {
		// Never follow returned pointers on native failure. In particular,
		// hardware-buffer exhaustion can return zero PCs and nonzero loss.
		return out, fmt.Errorf("cuptipcsampling: drain PC data: %w", err)
	}
	if p.Size != 32 || p.PPriv != 0 || p.Ctx != ctx || p.PcSamplingData != ptr {
		return out, errors.New("cuptipcsampling: native drain changed input storage")
	}
	if out.hardwareFull != 0 {
		return out, errors.New("cuptipcsampling: native sampling hardware buffer full")
	}
	return out, nil
}

func (b *samplingBuffer) diagnostics() samplingBatch {
	d := b.data
	return samplingBatch{total: d.TotalSamples, dropped: d.DroppedSamples, nonUser: d.NonUsrKernelsTotalSamples, rangeID: d.RangeId, remaining: d.RemainingNumPcs, hardwareFull: d.HardwareBufferFull}
}
