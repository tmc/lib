package cudatrace

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// Clock places CUDA events on the host clock timeline.
//
// CUDA events measure intervals between one another but carry no host-domain
// timestamp, so event durations alone cannot be positioned on a timeline built
// from time.Now. Calibrate records an anchor event on an idle stream and
// brackets its completion with host clock reads. HostNs then maps any completed
// event on the same device onto that timeline, which is what RecordSpan needs.
//
// A Clock is valid only while its anchor event is alive. Recalibrate for long
// runs: the host and device clocks drift apart.
type Clock struct {
	anchor        *cuda.Event
	device        *cuda.Device
	hostNs        int64
	uncertaintyNs int64
}

// Calibrate drains s, records anchor on it, and captures the host clock on
// either side of the anchor's completion. The caller keeps ownership of anchor,
// which must not be destroyed while the returned Clock is in use.
func Calibrate(ctx context.Context, s *cuda.Stream, anchor *cuda.Event) (*Clock, error) {
	if s == nil {
		return nil, errors.New("cudatrace: nil stream")
	}
	if anchor == nil {
		return nil, errors.New("cudatrace: nil anchor event")
	}
	if !sameDevice(anchor.Device(), s.Device()) {
		return nil, errors.New("cudatrace: anchor and stream must share a device context")
	}
	if err := s.Synchronize(ctx); err != nil {
		return nil, fmt.Errorf("cudatrace: drain stream: %w", err)
	}
	before := time.Now().UnixNano()
	if err := anchor.Record(s); err != nil {
		return nil, fmt.Errorf("cudatrace: record anchor: %w", err)
	}
	// A cancellable context can widen the bracket through polling. Preserve
	// that uncertainty rather than silently ignoring cancellation.
	if err := anchor.Synchronize(ctx); err != nil {
		return nil, fmt.Errorf("cudatrace: synchronize anchor: %w", err)
	}
	after := time.Now().UnixNano()
	if after < before {
		return nil, errors.New("cudatrace: host clock went backwards during calibration")
	}
	return &Clock{
		anchor:        anchor,
		device:        s.Device(),
		hostNs:        before + (after-before)/2,
		uncertaintyNs: (after - before) / 2,
	}, nil
}

// UncertaintyNs reports half the host-clock window within which the anchor
// event completed. Timestamps from HostNs carry at least this much error.
// It does not bound later clock drift or host wall-clock adjustments.
func (c *Clock) UncertaintyNs() int64 {
	if c == nil {
		return 0
	}
	return c.uncertaintyNs
}

// AnchorNs reports the host-domain timestamp assigned to the anchor event.
func (c *Clock) AnchorNs() uint64 {
	if c == nil {
		return 0
	}
	return uint64(c.hostNs)
}

// HostNs returns the host-domain timestamp of e, which must have completed.
func (c *Clock) HostNs(e *cuda.Event) (uint64, error) {
	if c == nil {
		return 0, errors.New("cudatrace: nil clock")
	}
	if e == nil {
		return 0, errors.New("cudatrace: nil event")
	}
	if c.anchor == nil || c.device == nil {
		return 0, errors.New("cudatrace: uninitialized clock")
	}
	if !sameDevice(e.Device(), c.device) {
		return 0, errors.New("cudatrace: event belongs to a different device context")
	}
	d, err := e.ElapsedSince(c.anchor)
	if err != nil {
		return 0, fmt.Errorf("cudatrace: offset from anchor: %w", err)
	}
	ns := c.hostNs + int64(d)
	if ns < 0 {
		return 0, fmt.Errorf("cudatrace: event precedes clock epoch by %v", time.Duration(-ns))
	}
	return uint64(ns), nil
}

// cuda.Device retains the primary context, so independently opened handles
// for the same driver device share a context. Event methods check lifetimes.
func sameDevice(a, b *cuda.Device) bool {
	return a != nil && b != nil && a.Raw() == b.Raw()
}
