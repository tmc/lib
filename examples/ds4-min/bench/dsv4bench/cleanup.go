package dsv4bench

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// CleanupError retains benchmark owners whose Close failed. Call Close to
// retry; successful releases are never retried. Results remain unavailable.
type CleanupError struct {
	cause     error
	state     dsv4.State
	snapshots []dsv4.Snapshot
}

func (e *CleanupError) Error() string { return fmt.Sprintf("benchmark cleanup: %v", e.cause) }
func (e *CleanupError) Unwrap() error { return e.cause }

// Close retries retained owners, preserving each failed owner for another try.
func (e *CleanupError) Close() error {
	if e == nil {
		return nil
	}
	var err error
	var retained []dsv4.Snapshot
	for _, s := range e.snapshots {
		if closeErr := s.Close(); closeErr != nil {
			retained = append(retained, s)
			err = errors.Join(err, closeErr)
		}
	}
	e.snapshots = retained
	if e.state != nil {
		if closeErr := e.state.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		} else {
			e.state = nil
		}
	}
	return err
}

// MemoryStats reports backing storage still owned after failed cleanup.
// A missing retained owner's reporter fails closed instead of undercounting.
func (e *CleanupError) MemoryStats() (dsv4.ContextMemory, error) {
	var total dsv4.ContextMemory
	var owners []any
	if e != nil {
		if e.state != nil {
			owners = append(owners, e.state)
		}
		for _, s := range e.snapshots {
			owners = append(owners, s)
		}
	}
	for _, owner := range owners {
		r, ok := owner.(dsv4.MemoryReporter)
		if !ok {
			return dsv4.ContextMemory{}, dsv4.ErrUnimplemented
		}
		m, err := r.MemoryStats()
		if err != nil {
			return dsv4.ContextMemory{}, err
		}
		if _, err = m.Total(); err != nil {
			return dsv4.ContextMemory{}, err
		}
		// Validate the combined total before adding components.
		n, _ := m.Total()
		old, _ := total.Total()
		if _, err = (dsv4.ContextMemory{Raw: old, Compressed: n}).Total(); err != nil {
			return dsv4.ContextMemory{}, err
		}
		total.Raw += m.Raw
		total.Compressed += m.Compressed
		total.Scratch += m.Scratch
	}
	return total, nil
}
