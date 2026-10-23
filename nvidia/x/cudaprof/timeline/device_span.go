package timeline

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// A KernelInterval is one kernel execution on the device clock. Selected says
// whether the caller's classifier counts it in the work being measured;
// everything else is other work.
type KernelInterval struct {
	Name       string
	Start, End uint64 // nanoseconds in one qualified CUPTI clock domain
	Selected   bool
}

// KernelIntervalsFromActivity returns the kernel executions in records,
// classifying each by selected(name). Records of other kinds are ignored.
// A kernel record that is not Complete or ends before it starts is an error
// rather than a row left out: dropping it would shrink the span without
// saying so.
//
// The records must come from one region of one session on one device, so
// that their timestamps share a clock. KernelIntervalsFromActivity does not
// check where they came from or whether the collection lost any.
func KernelIntervalsFromActivity(records []cuptiactivity.Record, selected func(name string) bool) ([]KernelInterval, error) {
	if selected == nil {
		return nil, errors.New("nil kernel classifier")
	}
	var out []KernelInterval
	for i, r := range records {
		if r.Kind != cupti.ActivityKindConcurrentKernel && r.Kind != cupti.ActivityKindKernel {
			continue
		}
		if !r.Complete || r.End < r.Start {
			return nil, fmt.Errorf("kernel record %d (%s) has no valid interval", i, r.Name)
		}
		out = append(out, KernelInterval{Name: r.Name, Start: r.Start, End: r.End, Selected: selected(r.Name)})
	}
	return out, nil
}

// A DeviceSpan partitions one region's device span among the selected
// kernels, the other kernels, and idle time.
//
// SpanNs is the first kernel start to the last kernel end. It is not the time
// a collection or a profiler range was open, nor the sum of kernel durations:
// a region that ran no kernel has a span of zero, however long it was
// observed. Each nanosecond of the span is counted exactly once, in
// SelectedNs when only selected kernels were running, OtherNs when only other
// kernels were, OverlapNs when both were, and IdleNs when none was, so
//
//	SelectedNs + OtherNs + OverlapNs + IdleNs == SpanNs
//
// holds exactly. Launches counts kernel executions, including those of zero
// duration, and SelectedLaunches + OtherLaunches == Launches.
type DeviceSpan struct {
	Launches         int
	SelectedLaunches int
	OtherLaunches    int

	SpanNs     uint64
	SelectedNs uint64
	OtherNs    uint64
	OverlapNs  uint64
	IdleNs     uint64
}

// AccountDeviceSpan partitions the span of kernels. The order of kernels does
// not matter, and kernels is not modified. A kernel whose End precedes its
// Start is an error. No kernels give a zero DeviceSpan.
func AccountDeviceSpan(kernels []KernelInterval) (DeviceSpan, error) {
	var a DeviceSpan
	if len(kernels) == 0 {
		return a, nil
	}
	all := make([]Interval, 0, len(kernels))
	var selected, other []Interval
	first, last := kernels[0].Start, kernels[0].End
	for i, k := range kernels {
		if k.End < k.Start {
			return DeviceSpan{}, fmt.Errorf("kernel %d (%s) ends at %d before it starts at %d", i, k.Name, k.End, k.Start)
		}
		iv := Interval{Start: k.Start, End: k.End}
		all = append(all, iv)
		if k.Selected {
			selected = append(selected, iv)
		} else {
			other = append(other, iv)
		}
		first, last = min(first, k.Start), max(last, k.End)
	}
	a.Launches = len(kernels)
	a.SelectedLaunches = len(selected)
	a.OtherLaunches = len(other)

	// Each union is at most the span and at least either of its parts, so
	// every subtraction is exact. Adding the selected and other unions could
	// overflow; subtracting from the union of all cannot.
	busy, s, o := Union(all), Union(selected), Union(other)
	a.SpanNs = last - first
	a.SelectedNs = busy - o
	a.OtherNs = busy - s
	a.OverlapNs = s - a.SelectedNs
	a.IdleNs = a.SpanNs - busy
	return a, nil
}

// SpanFractions are a DeviceSpan's parts divided by its totals: the launch
// fractions by Launches, the span fractions by SpanNs.
type SpanFractions struct {
	SelectedLaunches, OtherLaunches                float64
	SelectedSpan, OtherSpan, OverlapSpan, IdleSpan float64
}

// Fractions divides the account by its totals. It refuses an account with no
// launches, and one whose kernels span no time, before dividing by either.
func (a DeviceSpan) Fractions() (SpanFractions, error) {
	if a.Launches <= 0 {
		return SpanFractions{}, errors.New("region has no kernel launches; no fraction has a denominator")
	}
	if a.SpanNs == 0 {
		return SpanFractions{}, fmt.Errorf("region's %d launches span 0 ns; no span fraction has a denominator", a.Launches)
	}
	n, s := float64(a.Launches), float64(a.SpanNs)
	return SpanFractions{
		SelectedLaunches: float64(a.SelectedLaunches) / n,
		OtherLaunches:    float64(a.OtherLaunches) / n,
		SelectedSpan:     float64(a.SelectedNs) / s,
		OtherSpan:        float64(a.OtherNs) / s,
		OverlapSpan:      float64(a.OverlapNs) / s,
		IdleSpan:         float64(a.IdleNs) / s,
	}, nil
}
