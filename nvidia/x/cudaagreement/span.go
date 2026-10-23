package cudaagreement

import (
	"fmt"

	proftimeline "github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// A Kernel is one kernel execution on the device clock. Chain says whether it
// is counted inside the chains being measured; everything else is counted
// outside them.
type Kernel struct {
	Name       string
	Start, End uint64 // nanoseconds, CUPTI timestamp domain
	Chain      bool
}

// KernelsFromActivity returns the kernel executions in records, classifying
// each by chain(name). Records of other kinds are ignored. A kernel record
// without valid timestamps is an error rather than a row left out: dropping
// it would shrink the span without saying so.
//
// It is [proftimeline.KernelIntervalsFromActivity] with Selected reported as
// Chain.
func KernelsFromActivity(records []cuptiactivity.Record, chain func(name string) bool) ([]Kernel, error) {
	ivs, err := proftimeline.KernelIntervalsFromActivity(records, chain)
	if err != nil {
		return nil, fmt.Errorf("cudaagreement: %w", err)
	}
	var out []Kernel
	for _, iv := range ivs {
		out = append(out, Kernel{Name: iv.Name, Start: iv.Start, End: iv.End, Chain: iv.Selected})
	}
	return out, nil
}

// A SpanAccount partitions one region's device span among the chains, the
// rest of the work, and idle time.
//
// SpanNs is the first kernel start to the last kernel end. It is not the time
// a collection or a profiler range was open: a region that ran no kernel has
// a span of zero, however long it was observed. Each nanosecond of the span
// is counted exactly once, in ChainNs when only chain kernels were running,
// OtherNs when only other kernels were, OverlapNs when both were, and IdleNs
// when none was, so
//
//	ChainNs + OtherNs + OverlapNs + IdleNs == SpanNs
//
// holds exactly. Launches counts kernel executions, and ChainLaunches +
// OtherLaunches == Launches.
type SpanAccount struct {
	Launches      int
	ChainLaunches int
	OtherLaunches int

	SpanNs    uint64
	ChainNs   uint64
	OtherNs   uint64
	OverlapNs uint64
	IdleNs    uint64
}

// Account partitions the span of kernels. The order of kernels does not
// matter. A kernel whose End precedes its Start is an error.
//
// It is [proftimeline.AccountDeviceSpan] with the chain kernels selected.
func Account(kernels []Kernel) (SpanAccount, error) {
	ivs := make([]proftimeline.KernelInterval, len(kernels))
	for i, k := range kernels {
		ivs[i] = proftimeline.KernelInterval{Name: k.Name, Start: k.Start, End: k.End, Selected: k.Chain}
	}
	d, err := proftimeline.AccountDeviceSpan(ivs)
	if err != nil {
		return SpanAccount{}, fmt.Errorf("cudaagreement: %w", err)
	}
	return SpanAccount{
		Launches:      d.Launches,
		ChainLaunches: d.SelectedLaunches,
		OtherLaunches: d.OtherLaunches,
		SpanNs:        d.SpanNs,
		ChainNs:       d.SelectedNs,
		OtherNs:       d.OtherNs,
		OverlapNs:     d.OverlapNs,
		IdleNs:        d.IdleNs,
	}, nil
}

// Fractions are a SpanAccount's parts divided by its totals: the launch
// fractions by Launches, the span fractions by SpanNs.
type Fractions struct {
	ChainLaunches, OtherLaunches                float64
	ChainSpan, OtherSpan, OverlapSpan, IdleSpan float64
}

// Fractions divides the account by its totals. It refuses an account with no
// launches, and one whose kernels span no time, before dividing by either.
//
// It is [proftimeline.DeviceSpan.Fractions] with the chains selected.
func (a SpanAccount) Fractions() (Fractions, error) {
	d := proftimeline.DeviceSpan{
		Launches:         a.Launches,
		SelectedLaunches: a.ChainLaunches,
		OtherLaunches:    a.OtherLaunches,
		SpanNs:           a.SpanNs,
		SelectedNs:       a.ChainNs,
		OtherNs:          a.OtherNs,
		OverlapNs:        a.OverlapNs,
		IdleNs:           a.IdleNs,
	}
	f, err := d.Fractions()
	if err != nil {
		return Fractions{}, fmt.Errorf("cudaagreement: %w", err)
	}
	return Fractions{
		ChainLaunches: f.SelectedLaunches,
		OtherLaunches: f.OtherLaunches,
		ChainSpan:     f.SelectedSpan,
		OtherSpan:     f.OtherSpan,
		OverlapSpan:   f.OverlapSpan,
		IdleSpan:      f.IdleSpan,
	}, nil
}

func (a SpanAccount) String() string {
	return fmt.Sprintf("launches chain=%d other=%d total=%d; span_ns chain=%d other=%d overlap=%d idle=%d total=%d",
		a.ChainLaunches, a.OtherLaunches, a.Launches, a.ChainNs, a.OtherNs, a.OverlapNs, a.IdleNs, a.SpanNs)
}
