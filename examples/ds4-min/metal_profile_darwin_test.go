//go:build darwin && arm64

package main

import (
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalprof"
)

// Phase attribution for the equivalence sweep. Most of the sweep's wall clock
// is spent before the first tensor comparison — reading the gguf, mapping the
// model, checking every tensor view, compiling the shader libraries — and that
// work is all host-side, so a GPU capture cannot see it. phaseTimer measures it
// directly and feeds a metalprof.FlightRecorder, which also collects the GPU
// dispatches, so both ends land in one pprof profile.
//
// Set DS4_METAL_PROFILE to a path to write that profile. The phase table is
// logged either way, under -v.
type phaseTimer struct {
	t     *testing.T
	fr    *metalprof.FlightRecorder
	start time.Time
	last  time.Time
	names []string
	durs  []time.Duration
}

func newPhaseTimer(t *testing.T) *phaseTimer {
	now := time.Now()
	return &phaseTimer{t: t, fr: metalprof.NewFlightRecorder(4096), start: now, last: now}
}

// mark closes the phase that ended here and names it.
func (p *phaseTimer) mark(name string) {
	if p == nil {
		return
	}
	now := time.Now()
	d := now.Sub(p.last)
	p.last = now
	p.names = append(p.names, name)
	p.durs = append(p.durs, d)
	p.fr.Record(name, d, [3]int{1, 1, 1}, [3]int{1, 1, 1}, map[string]string{"kind": "host"})
}

// report logs the phase table and, when DS4_METAL_PROFILE is set, writes a
// pprof profile covering both host phases and recorded dispatches.
func (p *phaseTimer) report() {
	if p == nil {
		return
	}
	total := time.Since(p.start)
	p.t.Logf("phase attribution (total %v):", total.Round(time.Millisecond))
	var named time.Duration
	for i, n := range p.names {
		d := p.durs[i]
		named += d
		p.t.Logf("  %-28s %8v  %5.1f%%", n, d.Round(time.Millisecond), 100*float64(d)/float64(total))
	}
	p.t.Logf("  %-28s %8v  %5.1f%%", "(unattributed)", (total - named).Round(time.Millisecond),
		100*float64(total-named)/float64(total))

	path := os.Getenv("DS4_METAL_PROFILE")
	if path == "" {
		return
	}
	f, err := os.Create(path)
	if err != nil {
		p.t.Errorf("create profile: %v", err)
		return
	}
	defer f.Close()
	if err := p.fr.SnapshotPprof(f); err != nil {
		p.t.Errorf("write profile: %v", err)
		return
	}
	p.t.Logf("wrote pprof profile to %s", path)
}
