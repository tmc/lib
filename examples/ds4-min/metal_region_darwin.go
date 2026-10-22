//go:build darwin && arm64

package main

import "sync"

// A metalRegion is one command buffer's GPU-time span, named for the part
// of the layer that buffer carried. Start and End are the values
// MTLCommandBuffer reports, in seconds on the GPU clock.
//
// The span is device time for a whole command buffer, not for a dispatch.
// This device permits counter sampling only at stage boundaries, so a
// buffer is the finest region available without splitting encoders; see
// docs/metal-phase-profile.md.
type metalRegion struct {
	Region   string  `json:"region"`
	Layer    int     `json:"layer"`
	Position int     `json:"position"`
	Start    float64 `json:"start_s"`
	End      float64 `json:"end_s"`
}

// Seconds reports the span's duration.
func (r metalRegion) Seconds() float64 { return r.End - r.Start }

// A metalRegionLog collects the spans of completed command buffers. Its
// zero value is ready to use.
//
// Recording happens after the buffer has completed: it reads two
// properties of a finished object and encodes nothing, so a run with a
// log installed submits the same dispatches in the same order as a run
// without one. That is what makes bit-identity the right gate here.
type metalRegionLog struct {
	mu      sync.Mutex
	regions []metalRegion
}

func (l *metalRegionLog) add(r metalRegion) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.regions = append(l.regions, r)
}

// all returns the recorded spans in completion order.
func (l *metalRegionLog) all() []metalRegion {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]metalRegion(nil), l.regions...)
}
