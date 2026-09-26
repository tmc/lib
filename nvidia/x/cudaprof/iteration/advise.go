package iteration

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// A Class is a kind of host API call that costs something when repeated
// every iteration.
type Class string

const (
	Alloc            Class = "alloc"             // cuMemAlloc, cudaMalloc, cudaHostAlloc
	Free             Class = "free"              // cuMemFree, cudaFree
	GraphBuild       Class = "graph-build"       // stream capture, cuGraphAdd*Node, cuGraphCreate
	GraphInstantiate Class = "graph-instantiate" // cuGraphInstantiate
	GraphUpdate      Class = "graph-update"      // cuGraphExecUpdate, cuGraphExec*SetParams
	GraphLaunch      Class = "graph-launch"      // cuGraphLaunch
	Sync             Class = "sync"              // cu*Synchronize, cudaDeviceSynchronize
)

// Classify returns the class of a driver or runtime API call given its
// name, or "" when it is none of them.
func Classify(name string) Class {
	has := func(s string) bool { return strings.Contains(name, s) }
	switch {
	case has("GraphExec") && !has("Destroy"):
		return GraphUpdate
	case has("GraphInstantiate"):
		return GraphInstantiate
	case has("GraphLaunch"):
		return GraphLaunch
	case has("GraphAdd"), has("GraphCreate"), has("GraphClone"), has("BeginCapture"), has("EndCapture"):
		return GraphBuild
	case has("Synchronize"):
		return Sync
	case has("Pool"), has("Destroy"):
		return ""
	case has("MemAlloc"), has("Malloc"), has("HostAlloc"), has("MemCreate"):
		return Alloc
	case has("MemFree"), has("cudaFree"), has("MemRelease"):
		return Free
	}
	return ""
}

// A Finding is one cost [Report.Advise] found in the steady state.
type Finding struct {
	Rule    string `json:"rule"`
	Finding string `json:"finding"`
	Action  string `json:"action"`

	// Nanos is the finding's cost per steady iteration: host time in the
	// calls for a call rule, device idle time for the idle rule.
	Nanos int64 `json:"ns_per_iteration"`
}

// IdleShare is the fraction of a steady iteration's window the device may
// sit idle before [Report.Advise] reports it.
const IdleShare = 0.10

// Advise returns the costs repeated in every steady iteration, largest
// first. It returns nothing for a report with no steady iterations.
func (r *Report) Advise() []Finding {
	n := r.Steady.Iterations
	if n == 0 {
		return nil
	}
	// every returns the steady totals of the classes and how many steady
	// iterations called any of them.
	every := func(classes ...Class) (CallStat, int) {
		var sum CallStat
		in := 0
		for _, it := range r.Iterations {
			if !it.Steady {
				continue
			}
			hit := false
			for _, c := range classes {
				s := it.Classes[c]
				sum.Count += s.Count
				sum.Nanos += s.Nanos
				hit = hit || s.Count > 0
			}
			if hit {
				in++
			}
		}
		return sum, in
	}
	perIter := func(s CallStat) (float64, int64) {
		return float64(s.Count) / float64(n), s.Nanos / int64(n)
	}
	// Repeated means in at least half the steady iterations; a cost paid
	// once in a long steady state is not the loop's.
	repeated := func(in int) bool { return in > 0 && 2*in >= n }

	var out []Finding
	if s, in := every(Alloc, Free); repeated(in) {
		c, ns := perIter(s)
		a, f := r.Steady.Classes[Alloc].Count, r.Steady.Classes[Free].Count
		out = append(out, Finding{
			Rule: "steady-alloc",
			Finding: fmt.Sprintf("%.1f device allocations and frees per iteration (%d allocs, %d frees in %d of %d iterations), %s host time each iteration",
				c, a, f, in, n, dur(ns)),
			Action: "Keep buffers across iterations: size a pool or cache at warm-up so the steady state allocates nothing.",
			Nanos:  ns,
		})
	}
	if s, in := every(GraphBuild, GraphInstantiate); repeated(in) {
		c, ns := perIter(s)
		u := r.Steady.Classes[GraphUpdate].Count
		finding := fmt.Sprintf("CUDA graphs are rebuilt every iteration: %.1f capture, node-add and instantiate calls per iteration in %d of %d iterations, %s host time each iteration",
			c, in, n, dur(ns))
		if u > 0 {
			finding += fmt.Sprintf("; %d exec updates", u)
		}
		out = append(out, Finding{
			Rule:    "graph-rebuild",
			Finding: finding,
			Action:  "Instantiate once and replay: update an existing executable graph with cuGraphExecUpdate or the node SetParams calls instead of building a new one.",
			Nanos:   ns,
		})
	}
	if s, in := every(Sync); repeated(in) {
		c, ns := perIter(s)
		out = append(out, Finding{
			Rule: "host-sync",
			Finding: fmt.Sprintf("the host synchronizes %.1f times per iteration in %d of %d iterations, waiting %s each iteration",
				c, in, n, dur(ns)),
			Action: "Remove the synchronization or move it off the critical path: read results back asynchronously, or synchronize once per several iterations.",
			Nanos:  ns,
		})
	}
	if w := r.Steady.WindowNanos; w > 0 {
		share := float64(r.Steady.IdleNanos) / float64(w)
		if share >= IdleShare {
			ns := r.Steady.IdleNanos / int64(n)
			out = append(out, Finding{
				Rule: "idle",
				Finding: fmt.Sprintf("the device is idle %.1f%% of each iteration (%s of %s)",
					100*share, dur(ns), dur(w/int64(n))),
				Action: "The device waits on the host: see the host calls above, launch fewer and larger kernels, or replay the iteration as a CUDA graph.",
				Nanos:  ns,
			})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Nanos > out[j].Nanos })
	return out
}

func dur(ns int64) string {
	return time.Duration(ns).Round(time.Microsecond).String()
}
