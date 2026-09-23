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
//
// Ops join by name, which misleads when the two implementations split the
// same work differently: one side's fused kernel and the other's several
// kernels are separate rows, and each reads as a gap. The -stages file names
// the work by position instead. Each line is a boundary, "regex stage", and
// the boundaries are matched in file order, cyclically: a kernel that matches
// the next expected boundary starts that boundary's stage, and every kernel
// until the one after is charged to it. A layer of a model is the usual cycle,
// with one boundary per stage of the layer, and a kernel both implementations
// launch once per stage makes a good anchor. A stage row counts how often the
// stage was entered; a count that differs between the captures says the
// boundaries did not align, and its rows then hold different work. Stages
// partition kernel exclusive time just as ops do.
//
// -warmup-until and -warmup-ms cut a prologue off the front of each window
// by the rule gaps uses for its steady state: the window then starts at the
// first kernel matching the pattern, or at the first device record more than
// N ms after the window's first. Weight loading and warmup launches then
// drop out of every op and stage total. A window in which the warmup never
// ends is refused, never compared whole.

const (
	classKernel = iota
	classMemcpy
	classMemset
)

const unmatchedOp = "(unmatched)"

// stageUnstarted is the stage of kernels that start before the first boundary.
const stageUnstarted = "(before first boundary)"

type compareOptions struct {
	phase           int
	opsPath         string
	stagesPath      string
	json            bool
	allowUnresolved bool

	// warmup, when set, cuts the prologue off the front of each window the
	// way gaps cuts its steady state; its maxRecords and journalBytes are
	// unused.
	warmup *gapOptions
}

// runCompareCommand implements "cudaprof compare".
func runCompareCommand(args []string, stdout io.Writer) error {
	opts, paths, err := parseCompareArgs(args, stdout)
	if err != nil || paths == nil {
		return err
	}
	rules, err := readOpRules(opts.opsPath)
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}
	boundaries, err := readOpRules(opts.stagesPath)
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}
	for _, b := range boundaries {
		if b.Op == stageUnstarted {
			return fmt.Errorf("compare: %s: line %d: stage name %s is reserved", opts.stagesPath, b.Line, stageUnstarted)
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
	c, err := compareCaptures(caps[0], caps[1], rules, stageBoundaries(boundaries), opts)
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

const compareUsage = "usage: cudaprof compare [-phase i] [-warmup-until=RE | -warmup-ms=N] [-ops ops.txt] [-stages stages.txt] [-allow-unresolved] [-json] base.pprof target.pprof"

// parseCompareArgs returns nil paths and a nil error when help was requested
// and printed.
func parseCompareArgs(args []string, stdout io.Writer) (compareOptions, []string, error) {
	var opts compareOptions
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.IntVar(&opts.phase, "phase", -1, "index of the capture-range phase to compare, required when a capture has several")
	fs.StringVar(&opts.opsPath, "ops", "", "file of `regex op` lines mapping kernel names to ops")
	fs.StringVar(&opts.stagesPath, "stages", "", "file of `regex stage` lines naming the boundaries of a repeating sequence of stages")
	fs.BoolVar(&opts.json, "json", false, "write the comparison as JSON")
	fs.BoolVar(&opts.allowUnresolved, "allow-unresolved", false, "report window totals even when graph replay hides the per-op census")
	var warmup gapOptions
	fs.Float64Var(&warmup.warmupMS, "warmup-ms", 0, "start each window `N` ms after its first device activity, as gaps does")
	fs.StringVar(&warmup.warmupUntil, "warmup-until", "", "start each window at its first kernel matching `RE`, as gaps does")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, compareUsage)
			fs.SetOutput(stdout)
			fs.PrintDefaults()
			return opts, nil, nil
		}
		return opts, nil, fmt.Errorf("compare: %w\n%s", err, compareUsage)
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	switch {
	case set["warmup-ms"] && set["warmup-until"]:
		return opts, nil, errors.New("compare: -warmup-ms and -warmup-until are exclusive")
	case set["warmup-ms"] && warmup.warmupMS < 0:
		return opts, nil, fmt.Errorf("compare: invalid -warmup-ms %g", warmup.warmupMS)
	case set["warmup-until"]:
		if warmup.warmupUntil == "" {
			return opts, nil, errors.New("compare: empty -warmup-until pattern")
		}
		if _, err := regexp.Compile(warmup.warmupUntil); err != nil {
			return opts, nil, fmt.Errorf("compare: -warmup-until: %w", err)
		}
	}
	if set["warmup-ms"] || set["warmup-until"] {
		opts.warmup = &warmup
	}
	if fs.NArg() != 2 {
		return opts, nil, errors.New(compareUsage)
	}
	return opts, fs.Args(), nil
}

// readOpRules reads a rules file, or returns nil rules for an empty path.
func readOpRules(path string) (opRules, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	rules, err := parseOpRules(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if rules == nil {
		return nil, fmt.Errorf("%s: no rules", path)
	}
	return rules, nil
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

// stageBoundaries are the boundaries of a repeating sequence of stages, in
// the order they are expected. Each rule's Op is the stage it starts.
type stageBoundaries []opRule

// stageTotals is one stage on one side.
type stageTotals struct {
	entries   int
	launches  int
	exclusive int64
	summed    int64
}

// assign charges each kernel of recs, which are sorted by start, to its stage
// and returns the totals by stage. owned is the exclusive time of each record.
func (b stageBoundaries) assign(recs []compareRecord, owned []int64) map[string]*stageTotals {
	out := make(map[string]*stageTotals)
	get := func(stage string) *stageTotals {
		t := out[stage]
		if t == nil {
			t = &stageTotals{}
			out[stage] = t
		}
		return t
	}
	stage, next := stageUnstarted, 0
	for i, r := range recs {
		if r.class != classKernel {
			continue
		}
		if b[next].re.MatchString(r.key) {
			stage = b[next].Op
			next = (next + 1) % len(b)
			get(stage).entries++
		}
		t := get(stage)
		t.launches++
		t.exclusive += owned[i]
		t.summed += r.end - r.start
	}
	if t := out[stageUnstarted]; t != nil {
		t.entries = 1
	}
	return out
}

// order returns the stage names in the order a table lists them: kernels
// before the first boundary, then each stage as the file first names it.
func (b stageBoundaries) order() []string {
	names := []string{stageUnstarted}
	seen := map[string]bool{stageUnstarted: true}
	for _, r := range b {
		if !seen[r.Op] {
			seen[r.Op] = true
			names = append(names, r.Op)
		}
	}
	return names
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

	// Stages is the kernel exclusive time by stage when -stages was given.
	// It is empty, like Ops, when graph replay hid device time.
	Stages []CompareStage `json:"stages,omitempty"`

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

	// Warmup describes what -warmup-ms or -warmup-until cut off the front of
	// the window, and WarmupNanos how long that was. StartTimestamp is the
	// start of what remains.
	Warmup      string `json:"warmup,omitempty"`
	WarmupNanos int64  `json:"warmup_ns,omitempty"`

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

	ops    map[string]*opTotals
	stages map[string]*stageTotals
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

// CompareStage is one stage in both captures. Entries counts how often the
// stage began; it differs between the captures when the boundaries did not
// align.
type CompareStage struct {
	Stage                string `json:"stage"`
	BaseEntries          int    `json:"base_entries"`
	TargetEntries        int    `json:"target_entries"`
	BaseLaunches         int    `json:"base_launches"`
	TargetLaunches       int    `json:"target_launches"`
	BaseExclusiveNanos   int64  `json:"base_exclusive_ns"`
	TargetExclusiveNanos int64  `json:"target_exclusive_ns"`
	DeltaExclusiveNanos  int64  `json:"delta_exclusive_ns"`
	BaseSummedNanos      int64  `json:"base_summed_ns"`
	TargetSummedNanos    int64  `json:"target_summed_ns"`
}

// compareCaptures measures both captures and joins them by op.
func compareCaptures(base, target *cudacapture.Capture, rules opRules, stages stageBoundaries, opts compareOptions) (*Comparison, error) {
	b, err := measureCompareSide(base, rules, stages, opts)
	if err != nil {
		return nil, fmt.Errorf("base: %w", err)
	}
	t, err := measureCompareSide(target, rules, stages, opts)
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
	if stages != nil {
		c.Stages = joinStages(stages.order(), b.stages, t.stages)
		for _, st := range c.Stages {
			if st.BaseEntries != st.TargetEntries {
				c.Notes = append(c.Notes, fmt.Sprintf("stage %s began %d times in base and %d in target: the boundaries did not align, so its rows hold different work", st.Stage, st.BaseEntries, st.TargetEntries))
			}
		}
	}
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
		c.Stages = nil
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
	if c.Stages == nil {
		return nil
	}
	var base, target int64
	for _, st := range c.Stages {
		base += st.BaseExclusiveNanos
		target += st.TargetExclusiveNanos
	}
	if base != c.Base.KernelNanos || target != c.Target.KernelNanos {
		return fmt.Errorf("stages exclusive %d, %d ns != kernel exclusive %d, %d ns", base, target, c.Base.KernelNanos, c.Target.KernelNanos)
	}
	return nil
}

// joinStages lists every stage either side entered, in order.
func joinStages(order []string, base, target map[string]*stageTotals) []CompareStage {
	var rows []CompareStage
	for _, n := range order {
		b, t := base[n], target[n]
		if b == nil && t == nil {
			continue
		}
		if b == nil {
			b = &stageTotals{}
		}
		if t == nil {
			t = &stageTotals{}
		}
		rows = append(rows, CompareStage{
			Stage:                n,
			BaseEntries:          b.entries,
			TargetEntries:        t.entries,
			BaseLaunches:         b.launches,
			TargetLaunches:       t.launches,
			BaseExclusiveNanos:   b.exclusive,
			TargetExclusiveNanos: t.exclusive,
			DeltaExclusiveNanos:  t.exclusive - b.exclusive,
			BaseSummedNanos:      b.summed,
			TargetSummedNanos:    t.summed,
		})
	}
	return rows
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
func measureCompareSide(cap *cudacapture.Capture, rules opRules, stages stageBoundaries, opts compareOptions) (*CompareSide, error) {
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
	if opts.warmup != nil {
		if err := side.cutWarmup(records, *opts.warmup); err != nil {
			return nil, err
		}
	}
	if err := side.measure(records, rules, stages); err != nil {
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

// cutWarmup moves the window's start to the first activity of its steady
// state, chosen by steadyActivities from the device records inside the
// window. A record that began before the new start is then an edge
// straddler or outside the window, so the prologue drops out of every total.
func (s *CompareSide) cutWarmup(records gapRecords, opts gapOptions) error {
	var until *regexp.Regexp
	if opts.warmupUntil != "" {
		re, err := regexp.Compile(opts.warmupUntil)
		if err != nil {
			return fmt.Errorf("warmup kernel pattern: %w", err)
		}
		until = re
	}
	var acts []gapActivity
	for _, a := range records.activities {
		if activityClass(a.subtype) < 0 || a.end < a.start || a.start < s.StartTimestamp || a.end > s.StopTimestamp {
			continue
		}
		acts = append(acts, a)
	}
	if len(acts) == 0 {
		return fmt.Errorf("no kernel, memcpy or memset lies inside the window [%d,%d)", s.StartTimestamp, s.StopTimestamp)
	}
	sort.SliceStable(acts, func(i, j int) bool {
		if acts[i].start != acts[j].start {
			return acts[i].start < acts[j].start
		}
		return acts[i].end < acts[j].end
	})
	s.Warmup = describeWarmup(opts)
	steady := steadyActivities(acts, int64(opts.warmupMS*1e6), until)
	if len(steady) == 0 {
		return fmt.Errorf("no steady state in the window [%d,%d): the warmup (%s) never ended", s.StartTimestamp, s.StopTimestamp, s.Warmup)
	}
	s.WarmupNanos = steady[0].start - s.StartTimestamp
	s.StartTimestamp = steady[0].start
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
	key        string // the kernel's match key, which stage boundaries test
}

// measure partitions the window. Device records count only when they fall
// entirely inside it and graph replay envelopes are clipped to it, the rule
// measurePhase applies to the phase receipt.
func (s *CompareSide) measure(records gapRecords, rules opRules, stages stageBoundaries) error {
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
			r.op, r.key = op, cudalowering.MatchKey(a.name)
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
	if stages != nil {
		s.stages = stages.assign(recs, owned)
	}
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

	fmt.Fprint(w, c.identity())
	if c.OpStatus != "MEASURED" {
		fmt.Fprintln(w, "per-op: UNREAD (graph replay hides the per-op census)")
		return nil
	}
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
	if err := tw.Flush(); err != nil {
		return err
	}
	if c.Stages == nil {
		return nil
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "per stage, ns. exclusive as above; it sums to kernel exclusive. entries: how often the stage began.")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "entries\t\tlaunches\t\texclusive\t\t\tsummed\t\t")
	fmt.Fprintln(tw, "base\ttarget\tbase\ttarget\tbase\ttarget\tdelta\tbase\ttarget\t  stage")
	for _, st := range c.Stages {
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%d\t%d\t%+d\t%d\t%d\t  %s\n",
			st.BaseEntries, st.TargetEntries, st.BaseLaunches, st.TargetLaunches,
			st.BaseExclusiveNanos, st.TargetExclusiveNanos, st.DeltaExclusiveNanos,
			st.BaseSummedNanos, st.TargetSummedNanos, st.Stage)
	}
	return tw.Flush()
}

// identity returns the two lines stating the identity and its deltas.
func (c *Comparison) identity() string {
	rest := fmt.Sprintf("%+d %+d %+d %+d", c.term("memcpy").Delta, c.term("memset").Delta, c.term("idle").Delta, c.term("unresolved").Delta)
	if c.OpStatus != "MEASURED" {
		return fmt.Sprintf("identity: Δwindow = Δkernel + Δmemcpy + Δmemset + Δidle + Δunresolved\n          %+d = %+d %s (exact)\n",
			c.term("window").Delta, c.term("kernel").Delta, rest)
	}
	var named, unmatched int64
	for _, o := range c.Ops {
		if o.Op == unmatchedOp {
			unmatched = o.DeltaExclusiveNanos
			continue
		}
		named += o.DeltaExclusiveNanos
	}
	return fmt.Sprintf("identity: Δwindow = ΣΔop + Δunmatched + Δmemcpy + Δmemset + Δidle + Δunresolved\n          %+d = %+d %+d %s (exact)\n",
		c.term("window").Delta, named, unmatched, rest)
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
	if s.Warmup != "" {
		line += fmt.Sprintf("\n        warmup: %d ns cut, %s", s.WarmupNanos, s.Warmup)
	}
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
