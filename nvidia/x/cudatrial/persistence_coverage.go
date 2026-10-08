package cudatrial

import "fmt"

// Persistent notification values match cuFuzz's persistent-loop delimiters.
// They do not request CUDA device reset.
const (
	SignalStart = 1234
	SignalEnd   = 5678
)

// PersistentCoverage retains completed traces across loop iterations. A device
// fault makes this owner permanently spent; no coverage is assigned to that
// faulting iteration. The last completed trace remains separately available.
type PersistentCoverage struct {
	current *ThreadEdgeMap
	merged  *EdgeBitmap
	last    *EdgeBitmap
	active  bool
	spent   bool
}

// NewPersistentCoverage creates bounded per-thread and merged host maps.
func NewPersistentCoverage(size, threads int) (*PersistentCoverage, error) {
	m, err := NewThreadEdgeMap(size, threads)
	if err != nil {
		return nil, err
	}
	b, err := NewEdgeBitmap(size)
	if err != nil {
		return nil, err
	}
	return &PersistentCoverage{current: m, merged: b}, nil
}

// Notify starts or completes an iteration. Out-of-order signals are refused
// without clearing existing evidence.
func (p *PersistentCoverage) Notify(signal int) error {
	if p == nil || p.current == nil || p.spent {
		return fmt.Errorf("persistent coverage: unavailable or spent")
	}
	switch signal {
	case SignalStart:
		if p.active {
			return fmt.Errorf("persistent coverage: iteration already active")
		}
		p.current.Reset()
		p.active = true
	case SignalEnd:
		if !p.active {
			return fmt.Errorf("persistent coverage: no active iteration")
		}
		p.last = p.current.Snapshot()
		if err := p.merged.Merge(p.last); err != nil {
			return err
		}
		p.active = false
	default:
		return fmt.Errorf("persistent coverage: unknown signal")
	}
	return nil
}

// Record appends a caller-supplied CFG block identity to an active trace.
func (p *PersistentCoverage) Record(thread int, block uint64) (int, error) {
	if p == nil || !p.active || p.spent {
		return 0, fmt.Errorf("persistent coverage: no active iteration")
	}
	return p.current.Record(thread, block)
}

// RecordHardwareTrace records raw hardware trace bytes into the active iteration.
// It requires StatusHardwareAvailable; passing StatusPendingHardware or any
// non-available status strictly returns an error wrapping cudaerrors.ErrHardwarePending,
// forbidding faked mock coverage generation.
func (p *PersistentCoverage) RecordHardwareTrace(raw []byte, status HardwareStatus) error {
	if p == nil || !p.active || p.spent {
		return fmt.Errorf("persistent coverage: no active iteration")
	}
	if err := ValidateHardwareCoverage(raw, status); err != nil {
		return fmt.Errorf("persistent coverage: %w", err)
	}
	if len(raw) != len(p.current.bitmap.counts) {
		return fmt.Errorf("persistent coverage: mismatched hardware trace size")
	}
	for i, n := range raw {
		if n > p.current.bitmap.counts[i] {
			p.current.bitmap.counts[i] = n
		}
	}
	return nil
}

// Fault permanently retires this owner and discards the incomplete trace.
// It does not call cudaDeviceReset or return the faulting trace as coverage.
func (p *PersistentCoverage) Fault() {
	if p == nil {
		return
	}
	p.spent = true
	p.active = false
	if p.current != nil {
		p.current.Reset()
	}
}

// LastComplete returns the preceding completed trace, or nil if there is none.
func (p *PersistentCoverage) LastComplete() *EdgeBitmap {
	if p == nil || p.last == nil {
		return nil
	}
	return &EdgeBitmap{counts: p.last.Bytes()}
}

// Merged returns the union of completed traces, excluding incomplete iterations.
func (p *PersistentCoverage) Merged() *EdgeBitmap {
	if p == nil || p.merged == nil {
		return nil
	}
	return &EdgeBitmap{counts: p.merged.Bytes()}
}
