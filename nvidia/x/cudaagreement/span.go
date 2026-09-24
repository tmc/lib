package cudaagreement

import (
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/cupti"
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
func KernelsFromActivity(records []cuptiactivity.Record, chain func(name string) bool) ([]Kernel, error) {
	if chain == nil {
		return nil, errors.New("cudaagreement: nil chain classifier")
	}
	var out []Kernel
	for i, r := range records {
		if r.Kind != cupti.ActivityKindConcurrentKernel && r.Kind != cupti.ActivityKindKernel {
			continue
		}
		if !r.Complete || r.End < r.Start {
			return nil, fmt.Errorf("cudaagreement: kernel record %d (%s) has no valid interval", i, r.Name)
		}
		out = append(out, Kernel{Name: r.Name, Start: r.Start, End: r.End, Chain: chain(r.Name)})
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
func Account(kernels []Kernel) (SpanAccount, error) {
	var a SpanAccount
	if len(kernels) == 0 {
		return a, nil
	}
	// A sweep over interval edges. At each edge the number of running chain
	// and other kernels changes; the time since the previous edge belongs
	// to whichever class was running over it.
	type edge struct {
		t     uint64
		chain bool
		delta int
	}
	edges := make([]edge, 0, 2*len(kernels))
	for i, k := range kernels {
		if k.End < k.Start {
			return SpanAccount{}, fmt.Errorf("cudaagreement: kernel %d (%s) ends at %d before it starts at %d", i, k.Name, k.End, k.Start)
		}
		a.Launches++
		if k.Chain {
			a.ChainLaunches++
		} else {
			a.OtherLaunches++
		}
		edges = append(edges, edge{k.Start, k.Chain, +1}, edge{k.End, k.Chain, -1})
	}
	slices.SortFunc(edges, func(x, y edge) int {
		switch {
		case x.t < y.t:
			return -1
		case x.t > y.t:
			return 1
		}
		return 0
	})
	first, last := edges[0].t, edges[len(edges)-1].t
	a.SpanNs = last - first
	var chain, other int
	prev := first
	for _, e := range edges {
		if d := e.t - prev; d > 0 {
			switch {
			case chain > 0 && other > 0:
				a.OverlapNs += d
			case chain > 0:
				a.ChainNs += d
			case other > 0:
				a.OtherNs += d
			default:
				a.IdleNs += d
			}
			prev = e.t
		}
		if e.chain {
			chain += e.delta
		} else {
			other += e.delta
		}
	}
	return a, nil
}

// Fractions are a SpanAccount's parts divided by its totals: the launch
// fractions by Launches, the span fractions by SpanNs.
type Fractions struct {
	ChainLaunches, OtherLaunches                float64
	ChainSpan, OtherSpan, OverlapSpan, IdleSpan float64
}

// Fractions divides the account by its totals. It refuses an account with no
// launches, and one whose kernels span no time, before dividing by either.
func (a SpanAccount) Fractions() (Fractions, error) {
	if a.Launches <= 0 {
		return Fractions{}, errors.New("cudaagreement: region has no kernel launches; no fraction has a denominator")
	}
	if a.SpanNs == 0 {
		return Fractions{}, fmt.Errorf("cudaagreement: region's %d launches span 0 ns; no span fraction has a denominator", a.Launches)
	}
	n, s := float64(a.Launches), float64(a.SpanNs)
	return Fractions{
		ChainLaunches: float64(a.ChainLaunches) / n,
		OtherLaunches: float64(a.OtherLaunches) / n,
		ChainSpan:     float64(a.ChainNs) / s,
		OtherSpan:     float64(a.OtherNs) / s,
		OverlapSpan:   float64(a.OverlapNs) / s,
		IdleSpan:      float64(a.IdleNs) / s,
	}, nil
}

func (a SpanAccount) String() string {
	return fmt.Sprintf("launches chain=%d other=%d total=%d; span_ns chain=%d other=%d overlap=%d idle=%d total=%d",
		a.ChainLaunches, a.OtherLaunches, a.Launches, a.ChainNs, a.OtherNs, a.OverlapNs, a.IdleNs, a.SpanNs)
}
