package iteration

import (
	"errors"
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/ianlancetaylor/demangle"
	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Options configures [Analyze].
type Options struct {
	// Marker matches the kernel whose launch begins an iteration. It is
	// tested against the kernel's name as [KernelName] gives it, such as
	// "mlx::core::cu::gather<float, int, 1, 2, int>", and against the name
	// as recorded.
	Marker *regexp.Regexp

	// NVTX makes Marker match the names of NVTX ranges and instantaneous
	// markers instead of kernels. An iteration then begins at the host time
	// the range was pushed.
	NVTX bool

	// Skip is the number of leading iterations excluded as warm-up.
	Skip int

	// Expect, when positive, is the number of times the marker must fire.
	Expect int

	// Group returns the op group of a kernel given its name as recorded.
	// When nil, a kernel's group is its qualified name without template
	// arguments.
	Group func(name string) string
}

// A Report is a capture measured iteration by iteration.
type Report struct {
	Marker  string `json:"marker"`
	NVTX    bool   `json:"nvtx,omitempty"`
	Markers int    `json:"markers"`
	Expect  int    `json:"expected_markers,omitempty"`
	Skip    int    `json:"skip"`

	// Iterations is every iteration between two markers, in order.
	Iterations []Iteration `json:"iterations"`

	// Steady sums the iterations after the first Skip. Groups, Kernels and
	// Calls break it down, each sorted by time, largest first.
	Steady  Totals      `json:"steady"`
	Groups  []Group     `json:"groups"`
	Kernels []Kernel    `json:"kernels"`
	Calls   []CallTotal `json:"calls"`

	Gate Gate `json:"gate"`
}

// An Iteration is the window from one marker to the next.
type Iteration struct {
	Index  int   `json:"index"`
	Start  int64 `json:"start"`
	End    int64 `json:"end"`
	Steady bool  `json:"steady"`
	Totals
}

// Totals is what one iteration or a sum of iterations held. Times are
// integer nanoseconds. A device record counts in the iterations its
// interval overlaps, clipped to each; it is launched in the one it starts
// in. A host API call counts in the iteration it starts in.
type Totals struct {
	// Iterations is how many iterations were summed.
	Iterations int `json:"iterations"`

	WindowNanos int64 `json:"window_ns"`
	BusyNanos   int64 `json:"busy_ns"`
	IdleNanos   int64 `json:"idle_ns"`

	// Exclusive time by class. Each busy instant is charged to one record:
	// a kernel before a memcpy before a memset, then the earliest started.
	// They sum to BusyNanos.
	KernelNanos int64 `json:"kernel_exclusive_ns"`
	MemcpyNanos int64 `json:"memcpy_exclusive_ns"`
	MemsetNanos int64 `json:"memset_exclusive_ns"`

	Launches      int    `json:"launches"`
	GraphLaunches int    `json:"graph_node_launches"` // kernels run as a CUDA graph node
	Memcpys       int    `json:"memcpys"`
	MemcpyBytes   uint64 `json:"memcpy_bytes"`
	Memsets       int    `json:"memsets"`

	// Calls counts every host API call; Classes counts only calls no other
	// call on the same thread encloses, so a cudaMalloc and the cuMemAlloc
	// it makes are one allocation.
	// SyncIdleNanos is the device idle time that spans the return of a
	// synchronizing call: the device drained, the host waited for it, and
	// nothing ran until the host launched again.
	SyncIdleNanos int64 `json:"sync_idle_ns"`

	Calls     int                `json:"calls"`
	CallNanos int64              `json:"call_ns"`
	Classes   map[Class]CallStat `json:"call_classes,omitempty"`
}

func (t *Totals) add(u *Totals) {
	t.Iterations += u.Iterations
	t.WindowNanos += u.WindowNanos
	t.BusyNanos += u.BusyNanos
	t.IdleNanos += u.IdleNanos
	t.KernelNanos += u.KernelNanos
	t.MemcpyNanos += u.MemcpyNanos
	t.MemsetNanos += u.MemsetNanos
	t.Launches += u.Launches
	t.GraphLaunches += u.GraphLaunches
	t.Memcpys += u.Memcpys
	t.MemcpyBytes += u.MemcpyBytes
	t.Memsets += u.Memsets
	t.SyncIdleNanos += u.SyncIdleNanos
	t.Calls += u.Calls
	t.CallNanos += u.CallNanos
	for c, s := range u.Classes {
		t.addClass(c, s)
	}
}

func (t *Totals) addClass(c Class, s CallStat) {
	if t.Classes == nil {
		t.Classes = make(map[Class]CallStat)
	}
	u := t.Classes[c]
	t.Classes[c] = CallStat{Count: u.Count + s.Count, Nanos: u.Nanos + s.Nanos}
}

// A CallStat is a number of host API calls and the host time spent in them.
type CallStat struct {
	Count int   `json:"count"`
	Nanos int64 `json:"ns"`
}

// A Group is the steady state's time in one op group.
type Group struct {
	Group     string `json:"group"`
	Launches  int    `json:"launches"`
	Exclusive int64  `json:"exclusive_ns"`
	// Summed is the raw launch durations, clipped to the steady window;
	// overlapping launches count more than once.
	Summed int64 `json:"summed_ns"`
}

// A Kernel is the steady state's time in one kernel.
type Kernel struct {
	Kernel string `json:"kernel"`
	Group
}

// A CallTotal is the steady state's calls of one host API.
type CallTotal struct {
	Name  string `json:"name"` // domain:name, such as driver:cuMemAlloc_v2
	Class Class  `json:"class,omitempty"`
	CallStat
}

// A Gate is the completeness check of a report.
type Gate struct {
	Pass     bool     `json:"pass"`
	Failures []string `json:"failures,omitempty"`
}

// Record classes, in the order an instant's owner is chosen.
const (
	classKernel = iota
	classMemcpy
	classMemset
)

func recordClass(subtype string) int {
	switch subtype {
	case "kernel":
		return classKernel
	case "memcpy":
		return classMemcpy
	case "memset":
		return classMemset
	}
	return -1
}

// A record is one device record.
type record struct {
	start, end int64
	class      int
	kernel     int // index into names, for a kernel
	graph      bool
	bytes      uint64
}

// Analyze measures records iteration by iteration.
func Analyze(records timeline.Records, opts Options) (*Report, error) {
	if opts.Marker == nil {
		return nil, errors.New("no iteration marker")
	}
	if opts.Skip < 0 {
		return nil, fmt.Errorf("invalid skip %d", opts.Skip)
	}
	r := &Report{Marker: opts.Marker.String(), NVTX: opts.NVTX, Expect: opts.Expect, Skip: opts.Skip}

	var names []string // kernel names as recorded
	index := make(map[string]int)
	var recs []record
	for _, a := range records.Activities {
		class := recordClass(a.Subtype)
		if class < 0 || a.End < a.Start {
			continue
		}
		rec := record{start: a.Start, end: a.End, class: class, kernel: -1}
		switch class {
		case classKernel:
			i, ok := index[a.Name]
			if !ok {
				i = len(names)
				index[a.Name] = i
				names = append(names, a.Name)
			}
			rec.kernel = i
			rec.graph = a.GraphID != 0
		case classMemcpy:
			rec.bytes = a.Bytes
		}
		recs = append(recs, rec)
	}
	sort.SliceStable(recs, func(i, j int) bool {
		if recs[i].start != recs[j].start {
			return recs[i].start < recs[j].start
		}
		return recs[i].end < recs[j].end
	})
	display := make([]string, len(names))
	for i, n := range names {
		display[i] = KernelName(n)
	}

	var marks []int64
	if opts.NVTX {
		for _, m := range records.Markers {
			if m.Flags&(timeline.MarkerStart|timeline.MarkerInstantaneous) != 0 && opts.Marker.MatchString(m.Name) {
				marks = append(marks, m.Time)
			}
		}
		slices.Sort(marks)
	} else {
		match := make([]bool, len(names))
		for i, n := range names {
			match[i] = opts.Marker.MatchString(display[i]) || opts.Marker.MatchString(n)
		}
		for _, rec := range recs {
			if rec.class == classKernel && match[rec.kernel] {
				marks = append(marks, rec.start)
			}
		}
	}
	r.Markers = len(marks)
	r.gate(len(marks))
	if len(marks) < 2 {
		return r, nil
	}

	group := make([]string, len(names))
	for i, n := range names {
		switch {
		case opts.Group != nil:
			group[i] = opts.Group(n)
		default:
			group[i] = cudalowering.MatchKey(n)
		}
		if group[i] == "" {
			group[i] = "(unnamed)"
		}
	}

	nIter := len(marks) - 1
	r.Iterations = make([]Iteration, nIter)
	for i := range r.Iterations {
		it := &r.Iterations[i]
		it.Index, it.Start, it.End = i, marks[i], marks[i+1]
		it.Steady = i >= opts.Skip
		it.Iterations = 1
		it.WindowNanos = it.End - it.Start
	}
	// iterationAt returns the iteration holding time t, or -1.
	iterationAt := func(t int64) int {
		k := sort.Search(len(marks), func(i int) bool { return marks[i] > t }) - 1
		if k < 0 || k >= nIter {
			return -1
		}
		return k
	}

	groups := make(map[string]*Group)
	kernels := make([]Kernel, len(names))
	for i := range kernels {
		kernels[i].Kernel, kernels[i].Group.Group = display[i], group[i]
	}
	groupOf := func(g string) *Group {
		p := groups[g]
		if p == nil {
			p = &Group{Group: g}
			groups[g] = p
		}
		return p
	}

	// Launches and summed time, by the iteration each record starts in.
	for _, rec := range recs {
		k := iterationAt(rec.start)
		if k < 0 {
			continue
		}
		it := &r.Iterations[k]
		switch rec.class {
		case classKernel:
			it.Launches++
			if rec.graph {
				it.GraphLaunches++
			}
			if it.Steady {
				d := min(rec.end, it.End) - rec.start
				kernels[rec.kernel].Launches++
				kernels[rec.kernel].Summed += d
				g := groupOf(group[rec.kernel])
				g.Launches++
				g.Summed += d
			}
		case classMemcpy:
			it.Memcpys++
			it.MemcpyBytes += rec.bytes
		case classMemset:
			it.Memsets++
		}
	}

	// Exclusive time: sweep the elementary intervals between record and
	// iteration edges, charging each to the record that owns it.
	lo, hi := marks[0], marks[nIter]
	bounds := append([]int64(nil), marks...)
	for _, rec := range recs {
		if rec.end > lo && rec.start < hi {
			bounds = append(bounds, max(rec.start, lo), min(rec.end, hi))
		}
	}
	slices.Sort(bounds)
	bounds = slices.Compact(bounds)
	var active []int
	next, k := 0, 0
	for i := 0; i+1 < len(bounds); i++ {
		t, u := bounds[i], bounds[i+1]
		kept := active[:0]
		for _, j := range active {
			if recs[j].end > t {
				kept = append(kept, j)
			}
		}
		active = kept
		for ; next < len(recs) && recs[next].start <= t; next++ {
			if recs[next].end > t {
				active = append(active, next)
			}
		}
		if len(active) == 0 {
			continue
		}
		owner := active[0]
		for _, j := range active[1:] {
			if recs[j].class < recs[owner].class || recs[j].class == recs[owner].class && j < owner {
				owner = j
			}
		}
		for k+1 < nIter && t >= marks[k+1] {
			k++
		}
		it := &r.Iterations[k]
		d := u - t
		it.BusyNanos += d
		rec := recs[owner]
		switch rec.class {
		case classKernel:
			it.KernelNanos += d
			if it.Steady {
				kernels[rec.kernel].Exclusive += d
				groupOf(group[rec.kernel]).Exclusive += d
			}
		case classMemcpy:
			it.MemcpyNanos += d
			if it.Steady {
				groupOf("memcpy").Exclusive += d
			}
		case classMemset:
			it.MemsetNanos += d
			if it.Steady {
				groupOf("memset").Exclusive += d
			}
		}
	}
	for _, rec := range recs {
		if rec.class == classKernel {
			continue
		}
		if k := iterationAt(rec.start); k >= 0 && r.Iterations[k].Steady {
			name := "memcpy"
			if rec.class == classMemset {
				name = "memset"
			}
			g := groupOf(name)
			g.Launches++
			g.Summed += min(rec.end, r.Iterations[k].End) - rec.start
		}
	}

	// Host API calls, by the iteration each starts in.
	calls := append([]timeline.Call(nil), records.Calls...)
	sort.SliceStable(calls, func(i, j int) bool { return calls[i].Start < calls[j].Start })
	outer := outermost(calls)
	// reach[i] is the latest end of recs[:i+1], for finding the idle gap
	// a call returns into.
	reach := make([]int64, len(recs))
	for i, rec := range recs {
		reach[i] = rec.end
		if i > 0 {
			reach[i] = max(reach[i], reach[i-1])
		}
	}
	byName := make(map[string]*CallTotal)
	syncGaps := make(map[int64]bool)
	for i, c := range calls {
		k := iterationAt(c.Start)
		if k < 0 {
			continue
		}
		it := &r.Iterations[k]
		d := c.End - c.Start
		it.Calls++
		it.CallNanos += d
		name := c.Name()
		class := Classify(name)
		if outer[i] && class != "" {
			it.addClass(class, CallStat{1, d})
		}
		if outer[i] && class == Sync {
			// Syncs returning into one gap share it.
			if from, to := idleAt(recs, reach, c.End, it.Start, it.End); to > from && !syncGaps[from] {
				syncGaps[from] = true
				it.SyncIdleNanos += to - from
			}
		}
		if it.Steady {
			key := c.Domain + ":" + name
			t := byName[key]
			if t == nil {
				t = &CallTotal{Name: key, Class: class}
				byName[key] = t
			}
			t.Count++
			t.Nanos += d
		}
	}

	for i := range r.Iterations {
		it := &r.Iterations[i]
		it.IdleNanos = it.WindowNanos - it.BusyNanos
		if it.Steady {
			r.Steady.add(&it.Totals)
		}
	}
	for _, g := range groups {
		r.Groups = append(r.Groups, *g)
	}
	sort.Slice(r.Groups, func(i, j int) bool {
		a, b := r.Groups[i], r.Groups[j]
		if a.Exclusive != b.Exclusive {
			return a.Exclusive > b.Exclusive
		}
		return a.Group < b.Group
	})
	for _, kn := range kernels {
		if kn.Launches > 0 || kn.Exclusive > 0 {
			r.Kernels = append(r.Kernels, kn)
		}
	}
	sort.Slice(r.Kernels, func(i, j int) bool {
		a, b := r.Kernels[i], r.Kernels[j]
		if a.Exclusive != b.Exclusive {
			return a.Exclusive > b.Exclusive
		}
		return a.Kernel < b.Kernel
	})
	for _, t := range byName {
		r.Calls = append(r.Calls, *t)
	}
	sort.Slice(r.Calls, func(i, j int) bool {
		a, b := r.Calls[i], r.Calls[j]
		if a.Nanos != b.Nanos {
			return a.Nanos > b.Nanos
		}
		return a.Name < b.Name
	})
	r.gateLaunches()
	return r, nil
}

// gate records the marker checks, which need no measurement.
func (r *Report) gate(marks int) {
	r.Gate.Pass = true
	if r.Expect > 0 && marks != r.Expect {
		r.fail("marker fired %d times, want %d", marks, r.Expect)
	}
	if steady := marks - 1 - r.Skip; steady < 1 {
		r.fail("marker fired %d times: %d iterations, none left after skipping %d", marks, max(marks-1, 0), r.Skip)
	}
}

// gateLaunches checks that every steady iteration launched the same number
// of kernels.
func (r *Report) gateLaunches() {
	counts := make(map[int][]int)
	var order []int
	for _, it := range r.Iterations {
		if !it.Steady {
			continue
		}
		if _, ok := counts[it.Launches]; !ok {
			order = append(order, it.Launches)
		}
		counts[it.Launches] = append(counts[it.Launches], it.Index)
	}
	if len(order) <= 1 {
		return
	}
	var parts []string
	for _, n := range order {
		parts = append(parts, fmt.Sprintf("%d in %s", n, formatIndexes(counts[n])))
	}
	r.fail("steady iterations launched different numbers of kernels: %s", strings.Join(parts, "; "))
}

func (r *Report) fail(format string, args ...any) {
	r.Gate.Pass = false
	r.Gate.Failures = append(r.Gate.Failures, fmt.Sprintf(format, args...))
}

// formatIndexes formats iteration indexes, collapsing runs: "1-4,7".
func formatIndexes(ix []int) string {
	var parts []string
	for i := 0; i < len(ix); {
		j := i
		for j+1 < len(ix) && ix[j+1] == ix[j]+1 {
			j++
		}
		if j == i {
			parts = append(parts, fmt.Sprint(ix[i]))
		} else {
			parts = append(parts, fmt.Sprintf("%d-%d", ix[i], ix[j]))
		}
		i = j + 1
	}
	return "iteration " + strings.Join(parts, ",")
}

// idleAt returns the device idle gap holding time t, clipped to [lo, hi),
// or an empty one when the device is busy at t.
func idleAt(recs []record, reach []int64, t, lo, hi int64) (from, to int64) {
	n := sort.Search(len(recs), func(i int) bool { return recs[i].start > t })
	from, to = lo, hi
	if n > 0 {
		if reach[n-1] > t {
			return 0, 0
		}
		from = max(from, reach[n-1])
	}
	if n < len(recs) {
		to = min(to, recs[n].start)
	}
	return from, to
}

// outermost reports, for calls sorted by start, whether no earlier call on
// the same thread encloses each one.
func outermost(calls []timeline.Call) []bool {
	out := make([]bool, len(calls))
	end := make(map[uint32]int64) // end of each thread's current outermost call
	for i, c := range calls {
		if e, ok := end[c.Thread]; ok && c.End <= e {
			continue
		}
		out[i] = true
		end[c.Thread] = c.End
	}
	return out
}

// KernelName returns a kernel's demangled name without its parameter
// list, keeping template arguments: "mlx::core::cu::gather<float, int, 1,
// 2, int>". A name that is not mangled is returned as is.
func KernelName(name string) string {
	name = strings.TrimSpace(name)
	if s, err := demangle.ToString(name, demangle.NoParams); err == nil && s != "" {
		return s
	}
	return name
}
