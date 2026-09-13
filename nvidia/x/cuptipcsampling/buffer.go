package cuptipcsampling

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// samplingBuffer owns the arrays supplied to CUPTI. Its owner must disable
// sampling and end native access before release. Names belong to the session's
// shared native-name registry, not to this buffer.
type samplingBuffer struct {
	data     cupti.PCSamplingData
	pcs      []cupti.PCSamplingPCData
	reasons  [][]cupti.PCSamplingStallReason
	pin      runtime.Pinner
	released bool
}

type pcSamples struct {
	crc, offset                uint64
	functionIndex, correlation uint32
	name                       string
	reasons                    []cupti.PCSamplingStallReason
}

type samplingBatch struct {
	valid                            bool
	total, dropped, nonUser, rangeID uint64
	remaining                        uint
	hardwareFull                     uint8
	pcs                              []pcSamples
}

// The budget includes the output header, PC records and reason arrays. It
// excludes Go bookkeeping and native buffers configured separately by CUPTI.
func newSamplingBuffer(pcCount, reasonCount int, maxBytes uint64) (*samplingBuffer, error) {
	if unsafe.Sizeof(uintptr(0)) != 8 || pcCount < 1 || reasonCount < 1 || reasonCount > 65536 {
		return nil, errors.New("cuptipcsampling: invalid sampling buffer dimensions")
	}
	stride := uint64(unsafe.Sizeof(cupti.PCSamplingPCData{})) + uint64(reasonCount)*uint64(unsafe.Sizeof(cupti.PCSamplingStallReason{}))
	header := uint64(unsafe.Sizeof(cupti.PCSamplingData{}))
	if maxBytes < header || uint64(pcCount) > (maxBytes-header)/stride {
		return nil, errors.New("cuptipcsampling: sampling buffer exceeds byte limit")
	}
	b := &samplingBuffer{pcs: make([]cupti.PCSamplingPCData, pcCount), reasons: make([][]cupti.PCSamplingStallReason, pcCount)}
	b.pin.Pin(&b.data)
	b.pin.Pin(&b.pcs[0])
	for i := range b.pcs {
		b.reasons[i] = make([]cupti.PCSamplingStallReason, reasonCount)
		b.pin.Pin(&b.reasons[i][0])
	}
	b.reset()
	return b, nil
}

// reset is legal only when no native operation is using the buffer.
func (b *samplingBuffer) reset() {
	b.data = cupti.PCSamplingData{Size: uint(unsafe.Sizeof(b.data)), CollectNumPcs: uint(len(b.pcs)), PPcData: uintptr(unsafe.Pointer(&b.pcs[0]))}
	for i := range b.pcs {
		clear(b.reasons[i])
		b.pcs[i] = cupti.PCSamplingPCData{Size: uint(unsafe.Sizeof(b.pcs[i])), StallReason: uintptr(unsafe.Pointer(&b.reasons[i][0]))}
	}
}

// snapshot copies a quiescent output buffer. copyName must use the session's
// bounded native-name registry; it must not free a shared name. Metadata is
// returned on invalid output so the caller can retain loss diagnostics, but
// there are no partial PC results. Native call success is checked by the owner.
func (b *samplingBuffer) snapshot(copyName func(uintptr) (string, error)) (samplingBatch, error) {
	if b.released {
		return samplingBatch{}, errors.New("cuptipcsampling: sampling buffer released")
	}
	out := b.diagnostics()
	if err := b.checkOutput(); err != nil {
		return out, err
	}
	d := b.data
	pcs := make([]pcSamples, 0, d.TotalNumPcs)
	for i := uint(0); i < d.TotalNumPcs; i++ {
		p := b.pcs[i]
		if err := b.checkPC(i); err != nil {
			return out, err
		}
		if p.FunctionName == 0 || copyName == nil {
			return out, errors.New("cuptipcsampling: missing native PC name")
		}
		name, err := copyName(p.FunctionName)
		if err != nil {
			return out, fmt.Errorf("cuptipcsampling: copy PC name: %w", err)
		}
		if name == "" {
			return out, errors.New("cuptipcsampling: empty native PC name")
		}
		reasons := append([]cupti.PCSamplingStallReason(nil), b.reasons[i][:p.StallReasonCount]...)
		seen := make(map[uint32]bool, len(reasons))
		for _, r := range reasons {
			if seen[r.PcSamplingStallReasonIndex] {
				return out, errors.New("cuptipcsampling: duplicate native PC reason")
			}
			seen[r.PcSamplingStallReasonIndex] = true
		}
		pcs = append(pcs, pcSamples{crc: p.CubinCrc, offset: p.PcOffset, functionIndex: p.FunctionIndex, correlation: p.CorrelationId, name: name, reasons: reasons})
	}
	out.pcs = pcs
	out.valid = true
	return out, nil
}

func (b *samplingBuffer) checkPC(i uint) error {
	p := b.pcs[i]
	if p.Size != uint(unsafe.Sizeof(p)) || p.StallReason != uintptr(unsafe.Pointer(&b.reasons[i][0])) || p.StallReasonCount > uint(len(b.reasons[i])) {
		return fmt.Errorf("cuptipcsampling: invalid native PC buffer %d", i)
	}
	return nil
}

func (b *samplingBuffer) checkOutput() error {
	d := b.data
	if d.Size != uint(unsafe.Sizeof(d)) || d.CollectNumPcs != uint(len(b.pcs)) || d.PPcData != uintptr(unsafe.Pointer(&b.pcs[0])) || d.TotalNumPcs > uint(len(b.pcs)) || d.HardwareBufferFull > 1 {
		return errors.New("cuptipcsampling: invalid native sampling buffer")
	}
	return nil
}

func (b *samplingBuffer) release() {
	if !b.released {
		b.data = cupti.PCSamplingData{}
		for i := range b.pcs {
			b.pcs[i] = cupti.PCSamplingPCData{}
			clear(b.reasons[i])
		}
		b.pin.Unpin()
		b.released = true
	}
}
