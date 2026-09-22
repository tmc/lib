package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalowering"
)

// Compare answers "why is target slower than base" for two captures of one
// workload, typically from two implementations whose kernel names differ.
//
// Everything is exact integer nanoseconds over one window per capture, and
// every total is a partition: each instant of the window is idle, unresolved
// (graph replay no resolved record covers), or busy, and each busy instant is
// charged to exactly one record. Overlap is resolved by a fixed rule: the
// instant goes to the record of the highest class present -- kernel, then
// memcpy, then memset -- and among records of that class to the one that
// started first, ties broken by journal order. The per-op exclusive column is
// therefore additive, and
//
//	window = Σ op exclusive + memcpy + memset + idle + unresolved
//
// holds on each side and so for the deltas. The identity is checked before
// anything is printed; a comparison that fails it is an error, not a result.
// The summed column is the raw duration total, which counts overlapping
// launches more than once and does not add up to anything.

const (
	classKernel = iota
	classMemcpy
	classMemset
)

const unmatchedOp = "(unmatched)"

type compareOptions struct {
	phase           int
	opsPath         string
	json            bool
	allowUnresolved bool
}

// runCompareCommand implements "cudaprof compare".
func runCompareCommand(args []string, stdout io.Writer) error {
	opts, paths, err := parseCompareArgs(args, stdout)
	if err != nil || paths == nil {
		return err
	}
	var rules opRules
	if opts.opsPath != "" {
		f, err := os.Open(opts.opsPath)
		if err != nil {
			return fmt.Errorf("compare: %w", err)
		}
		rules, err = parseOpRules(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("compare: %s: %w", opts.opsPath, err)
		}
	}
	var caps [2]*cudacapture.Capture
	for i, path := range paths {
		abs, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		caps[i], err = readCapturePathLimits(abs, cudacapture.Limits{})
		if err != nil {
			return fmt.Errorf("compare: read capture %s: %w", path, err)
		}
		paths[i] = abs
	}
	c, err := compareCaptures(caps[0], caps[1], rules, opts)
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}
	c.Base.Capture, c.Target.Capture = paths[0], paths[1]
	if opts.json {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(c)
	}
	return c.write(stdout)
}

const compareUsage = "usage: cudaprof compare [-phase i] [-ops ops.txt] [-allow-unresolved] [-json] base.pprof target.pprof"

// parseCompareArgs returns nil paths and a nil error when help was requested
// and printed.
func parseCompareArgs(args []string, stdout io.Writer) (compareOptions, []string, error) {
	var opts compareOptions
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.IntVar(&opts.phase, "phase", -1, "index of the capture-range phase to compare, required when a capture has several")
	fs.StringVar(&opts.opsPath, "ops", "", "file of `regex op` lines mapping kernel names to ops")
	fs.BoolVar(&opts.json, "json", false, "write the comparison as JSON")
	fs.BoolVar(&opts.allowUnresolved, "allow-unresolved", false, "report window totals even when graph replay hides the per-op census")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, compareUsage)
			fs.SetOutput(stdout)
			fs.PrintDefaults()
			return opts, nil, nil
		}
		return opts, nil, fmt.Errorf("compare: %w\n%s", err, compareUsage)
	}
	if fs.NArg() != 2 {
		return opts, nil, errors.New(compareUsage)
	}
	return opts, fs.Args(), nil
}

// opRules maps kernel names to ops. The first rule whose pattern matches the
// kernel's demangled name wins.
type opRules []opRule

type opRule struct {
	Line    int    `json:"line"`
	Pattern string `json:"pattern"`
	Op      string `json:"op"`
	re      *regexp.Regexp
}

// parseOpRules reads lines of the form "regex op". A pattern ends at the
// first run of spaces or tabs, so it cannot contain one; use \s. Blank lines
// and lines starting with # are ignored.
func parseOpRules(r io.Reader) (opRules, error) {
	var rules opRules
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		i := strings.IndexAny(line, " \t")
		if i < 0 {
			return nil, fmt.Errorf("line %d: want \"regex op\", have %q", n, line)
		}
		pattern, op := line[:i], strings.TrimSpace(line[i:])
		if op == unmatchedOp {
			return nil, fmt.Errorf("line %d: op name %s is reserved", n, unmatchedOp)
		}
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", n, err)
		}
		rules = append(rules, opRule{Line: n, Pattern: pattern, Op: op, re: re})
	}
	return rules, sc.Err()
}

// kernelOp returns the op of a kernel and the index of the rule that named it,
// or -1. Without rules every kernel is its own op.
func (rules opRules) kernelOp(name string) (string, int) {
	key := cudalowering.MatchKey(name)
	if key == "" {
		key = "(unnamed)"
	}
	if rules == nil {
		return key, -1
	}
	for i, r := range rules {
		if r.re.MatchString(key) {
			return r.Op, i
		}
	}
	return unmatchedOp, -1
}

// Comparison is the whole result of one compare run. Terms and Ops are in
// integer nanoseconds; a delta is target minus base.
type Comparison struct {
	Base   *CompareSide  `json:"base"`
	Target *CompareSide  `json:"target"`
	Terms  []GapIntDelta `json:"terms"`

	// OpStatus is UNREAD when graph replay hid device time from the census;
	// Ops is then empty rather than a floor that reads like a total.
	OpStatus string      `json:"op_status"`
	Ops      []CompareOp `json:"ops,omitempty"`

	// UnmatchedShare is |Δ(unmatched)| over the sum of |Δ| of every term of
	// the identity: how much of the movement the ops file did not name.
	UnmatchedShare float64  `json:"unmatched_share"`
	Notes          []string `json:"notes,omitempty"`
}

// CompareSide is one capture measured over its window.
type CompareSide struct {
	Capture           string `json:"capture"`
	Window            string `json:"window"`
	WindowNote        string `json:"window_note,omitempty"`
	PhaseIndex        int    `json:"phase_index"`
	Device            string `json:"device,omitempty"`
	ComputeCapability string `json:"compute_capability,omitempty"`
	// GraphTraceMode is how the agent traced graph replays, or empty when
	// the capture predates the key and the mode is UNREAD.
	GraphTraceMode graphTraceMode `json:"graph_trace_mode,omitempty"`

	StartTimestamp  int64 `json:"start_timestamp"`
	StopTimestamp   int64 `json:"stop_timestamp"`
	WindowNanos     int64 `json:"window_ns"`
	LeadInNanos     int64 `json:"lead_in_ns"`
	TailNanos       int64 `json:"tail_ns"`
	BusyNanos       int64 `json:"busy_ns"`
	IdleNanos       int64 `json:"idle_ns"`
	UnresolvedNanos int64 `json:"unresolved_ns"`

	// Exclusive time by class; they partition BusyNanos.
	KernelNanos int64 `json:"kernel_exclusive_ns"`
	MemcpyNanos int64 `json:"memcpy_exclusive_ns"`
	MemsetNanos int64 `json:"memset_exclusive_ns"`

	KernelCount int            `json:"kernel_count"`
	MemcpyCount int            `json:"memcpy_count"`
	MemcpyBytes uint64         `json:"memcpy_bytes"`
	MemcpyKinds map[string]int `json:"memcpy_kinds,omitempty"`
	MemsetCount int            `json:"memset_count"`

	// EdgeStraddlers counts device records that cross a phase edge. They are
	// outside the window, by the same rule the phase receipt uses.
	EdgeStraddlers int `json:"edge_straddlers"`

	// UnusedRules are the ops rules that matched no kernel in this window.
	UnusedRules []opRule `json:"unused_rules,omitempty"`

	ops map[string]*opTotals
}

type opTotals struct {
	launches  int
	exclusive int64
	summed    int64
}

// CompareOp is one op in both captures.
type CompareOp struct {
	Op                   string `json:"op"`
	BaseLaunches         int    `json:"base_launches"`
	TargetLaunches       int    `json:"target_launches"`
	BaseExclusiveNanos   int64  `json:"base_exclusive_ns"`
	TargetExclusiveNanos int64  `json:"target_exclusive_ns"`
	DeltaExclusiveNanos  int64  `json:"delta_exclusive_ns"`
	BaseSummedNanos      int64  `json:"base_summed_ns"`
	TargetSummedNanos    int64  `json:"target_summed_ns"`
	DeltaSummedNanos     int64  `json:"delta_summed_ns"`
	BaseMeanNanos        int64  `json:"base_mean_ns"`
	TargetMeanNanos      int64  `json:"target_mean_ns"`
}

// compareCaptures measures both captures and joins them by op.
func compareCaptures(base, target *cudacapture.Capture, rules opRules, opts compareOptions) (*Comparison, error) {
	b, err := measureCompareSide(base, rules, opts)
	if err != nil {
		return nil, fmt.Errorf("base: %w", err)
	}
	t, err := measureCompareSide(target, rules, opts)
	if err != nil {
		return nil, fmt.Errorf("target: %w", err)
	}
	if b.Device != "" && t.Device != "" && b.Device != t.Device {
		return nil, fmt.Errorf("captures ran on different devices: base %q, target %q", b.Device, t.Device)
	}
	if b.ComputeCapability != "" && t.ComputeCapability != "" && b.ComputeCapability != t.ComputeCapability {
		return nil, fmt.Errorf("captures ran on different compute capabilities: base %s, target %s", b.ComputeCapability, t.ComputeCapability)
	}
	c := &Comparison{Base: b, Target: t, OpStatus: "MEASURED"}
	c.Terms = []GapIntDelta{
		intDelta("window", b.WindowNanos, t.WindowNanos),
		intDelta("lead_in", b.LeadInNanos, t.LeadInNanos),
		intDelta("tail", b.TailNanos, t.TailNanos),
		intDelta("busy", b.BusyNanos, t.BusyNanos),
		intDelta("kernel", b.KernelNanos, t.KernelNanos),
		intDelta("memcpy", b.MemcpyNanos, t.MemcpyNanos),
		intDelta("memset", b.MemsetNanos, t.MemsetNanos),
		intDelta("idle", b.IdleNanos, t.IdleNanos),
		intDelta("unresolved", b.UnresolvedNanos, t.UnresolvedNanos),
	}
	c.Ops = joinOps(b.ops, t.ops)
	if err := checkIdentity(c); err != nil {
		return nil, err
	}
	var gross, unmatched int64
	for _, o := range c.Ops {
		gross += abs64(o.DeltaExclusiveNanos)
		if o.Op == unmatchedOp {
			unmatched = abs64(o.DeltaExclusiveNanos)
		}
	}
	for _, name := range []string{"memcpy", "memset", "idle", "unresolved"} {
		gross += abs64(c.term(name).Delta)
	}
	if gross > 0 {
		c.UnmatchedShare = float64(unmatched) / float64(gross)
	}
	if rules == nil {
		c.Notes = append(c.Notes, "no -ops file: every kernel is its own op, so kernels that differ in name between the captures do not join")
	}
	if b.Window != t.Window {
		c.Notes = append(c.Notes, fmt.Sprintf("base window is the %s and target window is the %s, so the windows need not hold the same work", b.Window, t.Window))
	}
	if b.UnresolvedNanos > 0 || t.UnresolvedNanos > 0 {
		c.OpStatus = "UNREAD"
		c.Ops = nil
		c.UnmatchedShare = 0
		c.Notes = append(c.Notes, "graph replay hides device time from the census: busy and its classes cover resolved records only")
	}
	return c, nil
}

func (c *Comparison) term(name string) GapIntDelta {
	for _, t := range c.Terms {
		if t.Name == name {
			return t
		}
	}
	return GapIntDelta{Name: name}
}

// checkIdentity reports an error unless each window is exactly the sum of its
// parts, on each side and in the deltas. The ops rows are summed as joined, so
// a row lost or duplicated by the join fails here too.
func checkIdentity(c *Comparison) error {
	for _, s := range []struct {
		name string
		side *CompareSide
	}{{"base", c.Base}, {"target", c.Target}} {
		var ops int64
		for _, o := range s.side.ops {
			ops += o.exclusive
		}
		if ops != s.side.KernelNanos {
			return fmt.Errorf("%s: ops exclusive %d ns != kernel exclusive %d ns", s.name, ops, s.side.KernelNanos)
		}
		sum := ops + s.side.MemcpyNanos + s.side.MemsetNanos + s.side.IdleNanos + s.side.UnresolvedNanos
		if sum != s.side.WindowNanos {
			return fmt.Errorf("%s: window %d ns != ops %d + memcpy %d + memset %d + idle %d + unresolved %d = %d ns",
				s.name, s.side.WindowNanos, ops, s.side.MemcpyNanos, s.side.MemsetNanos, s.side.IdleNanos, s.side.UnresolvedNanos, sum)
		}
	}
	var sum int64
	for _, o := range c.Ops {
		sum += o.DeltaExclusiveNanos
	}
	for _, name := range []string{"memcpy", "memset", "idle", "unresolved"} {
		sum += c.term(name).Delta
	}
	if w := c.term("window").Delta; sum != w {
		return fmt.Errorf("Δwindow %d ns != Σ of its terms %d ns", w, sum)
	}
	return nil
}

func joinOps(base, target map[string]*opTotals) []CompareOp {
	names := make(map[string]bool)
	for n := range base {
		names[n] = true
	}
	for n := range target {
		names[n] = true
	}
	rows := make([]CompareOp, 0, len(names))
	for n := range names {
		b, t := base[n], target[n]
		if b == nil {
			b = &opTotals{}
		}
		if t == nil {
			t = &opTotals{}
		}
		rows = append(rows, CompareOp{
			Op:                   n,
			BaseLaunches:         b.launches,
			TargetLaunches:       t.launches,
			BaseExclusiveNanos:   b.exclusive,
			TargetExclusiveNanos: t.exclusive,
			DeltaExclusiveNanos:  t.exclusive - b.exclusive,
			BaseSummedNanos:      b.summed,
			TargetSummedNanos:    t.summed,
			DeltaSummedNanos:     t.summed - b.summed,
			BaseMeanNanos:        b.mean(),
			TargetMeanNanos:      t.mean(),
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		di, dj := abs64(rows[i].DeltaExclusiveNanos), abs64(rows[j].DeltaExclusiveNanos)
		if di != dj {
			return di > dj
		}
		return rows[i].Op < rows[j].Op
	})
	return rows
}

func (o *opTotals) mean() int64 {
	if o.launches == 0 {
		return 0
	}
	return o.summed / int64(o.launches)
}

// measureCompareSide refuses a capture whose records are incomplete, picks its
// window, and partitions the window.
func measureCompareSide(cap *cudacapture.Capture, rules opRules, opts compareOptions) (*CompareSide, error) {
	data, ok := cap.Files["capture-status.json"]
	if !ok {
		return nil, errors.New("capture has no capture-status.json, so its completeness is unknown")
	}
	var envelope struct {
		Status captureResult `json:"status"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, fmt.Errorf("read capture-status.json: %w", err)
	}
	status := envelope.Status
	completeness := status.RunCompleteness
	if completeness == "" {
		completeness = runCompletenessOf(status.TransportReady, status.CollectorFinalized, status.UnqualifiedGenerations, status.DroppedRecords, status.ActivityRecords)
	}
	if completeness != runComplete {
		return nil, fmt.Errorf("run_completeness is %q, not %q: records the window would hold may be missing", completeness, runComplete)
	}
	if status.DroppedRecords != 0 {
		return nil, fmt.Errorf("capture dropped %d record(s)", status.DroppedRecords)
	}
	if l := status.RecordLoss; l != nil && l.Offered != l.Retained {
		return nil, fmt.Errorf("capture lost records: %d offered, %d retained", l.Offered, l.Retained)
	}
	journals, _, err := requireJournals(cap)
	if err != nil {
		return nil, err
	}
	records, err := decodeJournals(journals, controllerMaxJournalRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return nil, err
	}
	side := &CompareSide{PhaseIndex: -1}
	if data, ok := cap.Files["capture-provenance.json"]; ok {
		var prov deviceProvenance
		if json.Unmarshal(data, &prov) == nil {
			side.Device, side.ComputeCapability = prov.Device, prov.ComputeCapability
		}
	}
	side.GraphTraceMode, _ = captureGraphTraceMode(status.Capability)
	if err := side.selectWindow(status.PhaseReceipt, opts.phase, records); err != nil {
		return nil, err
	}
	if err := side.measure(records, rules); err != nil {
		return nil, err
	}
	if side.UnresolvedNanos > 0 && !opts.allowUnresolved {
		return nil, fmt.Errorf("%d ns of graph replay in the window is covered by no resolved record, so the per-op census is hidden (graph trace mode %s); capture with --graph-trace=node or graphs disabled, or pass -allow-unresolved for window totals only", side.UnresolvedNanos, side.graphTraceModeString())
	}
	return side, nil
}

// selectWindow takes the window from the phase receipt, or from the device
// span of the whole capture when there is no qualified receipt.
func (s *CompareSide) selectWindow(r *capturePhaseReceipt, phase int, records gapRecords) error {
	switch {
	case r != nil && len(r.Phases) > 0:
		if phase < 0 {
			return fmt.Errorf("capture has %d phases; select one with -phase", len(r.Phases))
		}
		if phase >= len(r.Phases) {
			return fmt.Errorf("-phase %d out of range: capture has %d phases", phase, len(r.Phases))
		}
		r = &r.Phases[phase]
	case r != nil && r.Status == "MEASURED":
		if phase > 0 {
			return fmt.Errorf("-phase %d out of range: capture has 1 phase", phase)
		}
		phase = 0
	default:
		if phase >= 0 {
			return fmt.Errorf("-phase %d out of range: capture has no qualified phase (%s)", phase, r)
		}
		s.Window = "capture span"
		s.WindowNote = fmt.Sprintf("phase receipt %s; window is first to last device activity", r)
		first := true
		for _, a := range append(records.activities, records.graphs...) {
			if activityClass(a.subtype) < 0 && a.subtype != "graph_trace" {
				continue
			}
			if first || a.start < s.StartTimestamp {
				s.StartTimestamp = a.start
			}
			if first || a.end > s.StopTimestamp {
				s.StopTimestamp = a.end
			}
			first = false
		}
		if first {
			return errors.New("capture holds no device activity")
		}
		return nil
	}
	s.Window, s.PhaseIndex = "phase", phase
	s.StartTimestamp, s.StopTimestamp = int64(r.StartTimestamp), int64(r.StopTimestamp)
	return nil
}

func activityClass(subtype string) int {
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

// compareRecord is one device record inside the window.
type compareRecord struct {
	start, end int64
	class      int
	op         string
}

// measure partitions the window. Device records count only when they fall
// entirely inside it and graph replay envelopes are clipped to it, the rule
// measurePhase applies to the phase receipt.
func (s *CompareSide) measure(records gapRecords, rules opRules) error {
	lo, hi := s.StartTimestamp, s.StopTimestamp
	s.WindowNanos = hi - lo
	s.ops = make(map[string]*opTotals)
	used := make([]bool, len(rules))
	var recs []compareRecord
	for _, a := range records.activities {
		class := activityClass(a.subtype)
		if class < 0 || a.end < a.start {
			continue
		}
		if a.start < lo || a.end > hi {
			if a.end > lo && a.start < hi {
				s.EdgeStraddlers++
			}
			continue
		}
		r := compareRecord{start: a.start, end: a.end, class: class}
		switch class {
		case classKernel:
			op, rule := rules.kernelOp(a.name)
			if rule >= 0 {
				used[rule] = true
			}
			t := s.ops[op]
			if t == nil {
				t = &opTotals{}
				s.ops[op] = t
			}
			t.launches++
			t.summed += a.end - a.start
			r.op = op
			s.KernelCount++
		case classMemcpy:
			s.MemcpyCount++
			s.MemcpyBytes += a.bytes
			if s.MemcpyKinds == nil {
				s.MemcpyKinds = make(map[string]int)
			}
			s.MemcpyKinds[strings.TrimPrefix(cupti.ActivityMemcpyKind(a.copyKind).String(), "ActivityMemcpyKind")]++
		case classMemset:
			s.MemsetCount++
		}
		recs = append(recs, r)
	}
	for i, r := range rules {
		if !used[i] {
			s.UnusedRules = append(s.UnusedRules, r)
		}
	}
	if len(recs) == 0 {
		return fmt.Errorf("no kernel, memcpy or memset lies inside the window [%d,%d)", lo, hi)
	}
	sort.SliceStable(recs, func(i, j int) bool {
		if recs[i].start != recs[j].start {
			return recs[i].start < recs[j].start
		}
		return recs[i].end < recs[j].end
	})
	owned := partitionBusy(recs)
	busy := make([]interval, len(recs))
	first, last := recs[0].start, recs[0].end
	for i, r := range recs {
		busy[i] = interval{uint64(r.start), uint64(r.end)}
		last = max(last, r.end)
		switch r.class {
		case classKernel:
			s.ops[r.op].exclusive += owned[i]
			s.KernelNanos += owned[i]
		case classMemcpy:
			s.MemcpyNanos += owned[i]
		case classMemset:
			s.MemsetNanos += owned[i]
		}
	}
	s.BusyNanos = int64(unionNanos(busy))
	if parts := s.KernelNanos + s.MemcpyNanos + s.MemsetNanos; parts != s.BusyNanos {
		return fmt.Errorf("partition charged %d ns, but the busy union is %d ns", parts, s.BusyNanos)
	}
	var replays []interval
	for _, g := range records.graphs {
		if l, h := max(g.start, lo), min(g.end, hi); l < h {
			replays = append(replays, interval{uint64(l), uint64(h)})
		}
	}
	covered := int64(unionNanos(append(busy, replays...)))
	s.UnresolvedNanos = covered - s.BusyNanos
	s.IdleNanos = s.WindowNanos - covered
	s.LeadInNanos, s.TailNanos = first-lo, hi-last
	return nil
}

// partitionBusy returns the time each record owns. recs must be sorted by
// start. Every instant covered by at least one record is owned by exactly one:
// the one of lowest class, and among those the one with the lowest index, which
// is the earliest started. The result therefore sums to the union of recs.
func partitionBusy(recs []compareRecord) []int64 {
	bounds := make([]int64, 0, 2*len(recs))
	for _, r := range recs {
		bounds = append(bounds, r.start, r.end)
	}
	sort.Slice(bounds, func(i, j int) bool { return bounds[i] < bounds[j] })
	owned := make([]int64, len(recs))
	var active []int
	next := 0
	for i := 0; i+1 < len(bounds); i++ {
		t, u := bounds[i], bounds[i+1]
		if t == u {
			continue
		}
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
		owned[owner] += u - t
	}
	return owned
}

func abs64(n int64) int64 {
	if n < 0 {
		return -n
	}
	return n
}

func (c *Comparison) write(w io.Writer) error {
	b, t := c.Base, c.Target
	fmt.Fprintf(w, "base    %s\n        %s\n", b.Capture, b.windowLine())
	fmt.Fprintf(w, "target  %s\n        %s\n", t.Capture, t.windowLine())
	for _, n := range c.Notes {
		fmt.Fprintf(w, "note: %s\n", n)
	}
	fmt.Fprintln(w)

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "ns\tbase\ttarget\tdelta\t")
	for _, d := range c.Terms {
		name := d.Name
		switch name {
		case "kernel", "memcpy", "memset":
			name = "  " + name + " (exclusive)"
		case "lead_in":
			name = "lead-in (to first activity)"
		case "tail":
			name = "tail (after last activity)"
		}
		fmt.Fprintf(tw, "%s\t%d\t%d\t%+d\t\n", name, d.Base, d.Target, d.Delta)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "memcpy: count %d -> %d, bytes %d -> %d, kinds %s -> %s\n",
		b.MemcpyCount, t.MemcpyCount, b.MemcpyBytes, t.MemcpyBytes, formatKinds(b.MemcpyKinds), formatKinds(t.MemcpyKinds))
	fmt.Fprintln(w, "memcpy synchronous/pageable/legacy-stream: UNREAD (records carry copy kind and stream id only)")
	fmt.Fprintf(w, "memset: count %d -> %d\n", b.MemsetCount, t.MemsetCount)
	if b.EdgeStraddlers+t.EdgeStraddlers > 0 {
		fmt.Fprintf(w, "records crossing a phase edge, excluded: %d -> %d\n", b.EdgeStraddlers, t.EdgeStraddlers)
	}
	for _, s := range []struct {
		name string
		side *CompareSide
	}{{"base", b}, {"target", t}} {
		for _, r := range s.side.UnusedRules {
			fmt.Fprintf(w, "ops line %d %q (%s) matched no kernel in %s\n", r.Line, r.Pattern, r.Op, s.name)
		}
	}
	fmt.Fprintln(w)

	rest := fmt.Sprintf("%+d %+d %+d %+d", c.term("memcpy").Delta, c.term("memset").Delta, c.term("idle").Delta, c.term("unresolved").Delta)
	if c.OpStatus != "MEASURED" {
		fmt.Fprintf(w, "identity: Δwindow = Δkernel + Δmemcpy + Δmemset + Δidle + Δunresolved\n          %+d = %+d %s (exact)\n",
			c.term("window").Delta, c.term("kernel").Delta, rest)
		fmt.Fprintln(w, "per-op: UNREAD (graph replay hides the per-op census)")
		return nil
	}
	var named, unmatched int64
	for _, o := range c.Ops {
		if o.Op == unmatchedOp {
			unmatched = o.DeltaExclusiveNanos
			continue
		}
		named += o.DeltaExclusiveNanos
	}
	fmt.Fprintf(w, "identity: Δwindow = ΣΔop + Δunmatched + Δmemcpy + Δmemset + Δidle + Δunresolved\n          %+d = %+d %+d %s (exact)\n",
		c.term("window").Delta, named, unmatched, rest)
	fmt.Fprintf(w, "coverage: (unmatched) is %.1f%% of Σ|Δ| over the identity's terms\n\n", 100*c.UnmatchedShare)

	fmt.Fprintln(w, "per op, ns. exclusive: each busy instant charged once (kernel > memcpy > memset, then earliest start); it sums to busy.")
	fmt.Fprintln(w, "summed: raw launch durations; overlapping launches count more than once.")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "launches\t\texclusive\t\t\tsummed\t\t\tmean\t\t")
	fmt.Fprintln(tw, "base\ttarget\tbase\ttarget\tdelta\tbase\ttarget\tdelta\tbase\ttarget\t  op")
	for _, o := range c.Ops {
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%+d\t%d\t%d\t%+d\t%d\t%d\t  %s\n",
			o.BaseLaunches, o.TargetLaunches,
			o.BaseExclusiveNanos, o.TargetExclusiveNanos, o.DeltaExclusiveNanos,
			o.BaseSummedNanos, o.TargetSummedNanos, o.DeltaSummedNanos,
			o.BaseMeanNanos, o.TargetMeanNanos, o.Op)
	}
	return tw.Flush()
}

func (s *CompareSide) windowLine() string {
	line := fmt.Sprintf("window: %s [%d,%d)", s.Window, s.StartTimestamp, s.StopTimestamp)
	if s.PhaseIndex >= 0 {
		line = fmt.Sprintf("window: phase %d [%d,%d)", s.PhaseIndex, s.StartTimestamp, s.StopTimestamp)
	}
	if s.Device != "" || s.ComputeCapability != "" {
		line += fmt.Sprintf("  device %q cc %s", s.Device, s.ComputeCapability)
	}
	line += "  graph trace " + s.graphTraceModeString()
	if s.WindowNote != "" {
		line += "\n        " + s.WindowNote
	}
	return line
}

func (s *CompareSide) graphTraceModeString() string {
	if s.GraphTraceMode == "" {
		return "UNREAD"
	}
	return string(s.GraphTraceMode)
}

func formatKinds(kinds map[string]int) string {
	if len(kinds) == 0 {
		return "none"
	}
	names := make([]string, 0, len(kinds))
	for k := range kinds {
		names = append(names, k)
	}
	sort.Strings(names)
	parts := make([]string, len(names))
	for i, k := range names {
		parts[i] = fmt.Sprintf("%s=%d", k, kinds[k])
	}
	return strings.Join(parts, ",")
}
