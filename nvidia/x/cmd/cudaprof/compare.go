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
	"slices"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/ianlancetaylor/demangle"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudaprof/gaps"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
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
// -stage-entries also reports each entry of a stage on its own row: entry 0
// is the first time the stage began, entry 1 the second, so with one cycle
// per layer, entry i is layer i. Kernels before the first boundary are one
// entry. Entries join by position, so a stage that began a different number
// of times on the two sides is refused rather than joined out of step. The
// entries of a stage sum to its row.
//
// With both -ops and -stages, the JSON output breaks each stage row, and
// each entry row with -stage-entries, down by op, so an op can be read per
// layer. The cells partition kernel exclusive time both ways: a row's cells
// sum to the row, and an op's cells over the rows sum to the op's row. Both
// sums are checked with the identity.
//
// Ops join by qualified name, and template arguments are not part of it: every
// instantiation of mul_mat_q<...> is one op, and every CUTLASS kernel is
// cutlass::Kernel2. -templates keeps the arguments, so each instantiation is
// its own op and an -ops or -stages pattern is tested against the name with
// its arguments, "mul_mat_q<(ggml_type)8, 128, false>". It is not the default
// because a pattern anchored at the end of a bare name stops matching.
//
// -warmup-until and -warmup-ms cut a prologue off the front of each window
// by the rule gaps uses for its steady state: the window then starts at the
// first kernel matching the pattern, or at the first device record more than
// N ms after the window's first. Weight loading and warmup launches then
// drop out of every op and stage total. A window in which the warmup never
// ends is refused, never compared whole.
//
// A delta says the time moved, not why. Three more sections separate
// different work from different conditions. Each op lists the set of launch
// configurations -- grid, block and dynamic shared memory, from the kernel
// records -- it ran with on each side, and an op whose sets differ is
// flagged CHANGED with both sets and their launch counts. A launch whose
// record carries no configuration makes its op UNREAD, never SAME. Ops whose
// launch counts differ by a ratio of 1.5 or more, or that ran on one side
// only, are listed as launch count changes. The p10, p50 and p90 of
// per-launch exclusive time are given for every op in JSON and for the ops
// of largest |Δ| in text: a p50 that moved with a tight spread is a slower
// kernel, a p90 that moved alone is a few slow launches. The header gives the
// SM and memory clocks, temperature and power NVML reported at each side's
// capture-range edges, or says they were not recorded; two sides at
// different clocks differ by that ratio before any code does.

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
	// stageEntries reports each entry of a stage on its own row.
	stageEntries bool
	// templates keeps template arguments in the name ops join by.
	templates bool
	// maxRecords bounds the records decoded from each capture's journals;
	// zero means controllerMaxJournalRecords.
	maxRecords int64
	// journalBytes bounds each journal's decompressed size; zero means
	// cudacapture.MaxCaptureInputBytes.
	journalBytes int64
	// limits bounds the artifacts read from each capture, as capture
	// --capture-file-bytes and --capture-total-bytes allowed it to write.
	limits cudacapture.Limits

	// warmup, when set, cuts the prologue off the front of each window the
	// way gaps cuts its steady state; its maxRecords and journalBytes are
	// unused.
	warmup *gapOptions

	// iterations, when set, narrows each window to whole iterations.
	iterations *iterationWindow
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
	stages, err := readStageBoundaries(opts.stagesPath)
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}
	var caps [2]*cudacapture.Capture
	for i, path := range paths {
		abs, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		caps[i], err = readCapturePathLimits(abs, opts.limits)
		if err != nil {
			return fmt.Errorf("compare: read capture %s: %w", path, err)
		}
		paths[i] = abs
	}
	c, err := compareCaptures(caps[0], caps[1], rules, stages, opts)
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

const compareUsage = "usage: cudaprof compare [-phase i] [-warmup-until=RE | -warmup-ms=N | -iteration-marker=RE [-skip-iterations=N] [-iteration-count=N]] [-ops ops.txt] [-stages stages.txt [-stage-entries]] [-templates] [-allow-unresolved] [-max-records=N] [-journal-bytes=N] [--capture-file-bytes=N] [--capture-total-bytes=N] [-json] base.pprof target.pprof"

// parseCompareArgs returns nil paths and a nil error when help was requested
// and printed.
func parseCompareArgs(rawArgs []string, stdout io.Writer) (compareOptions, []string, error) {
	var opts compareOptions
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return opts, nil, fmt.Errorf("compare: %w", err)
	}
	opts.limits = limits
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.IntVar(&opts.phase, "phase", -1, "index of the capture-range phase to compare, required when a capture has several")
	fs.StringVar(&opts.opsPath, "ops", "", "file of `regex op` lines mapping kernel names to ops, or \"default\" for the built-in groups")
	fs.StringVar(&opts.stagesPath, "stages", "", "file of `regex stage` lines naming the boundaries of a repeating sequence of stages")
	fs.BoolVar(&opts.stageEntries, "stage-entries", false, "also report each entry of each stage, in the order entered; refused when a stage's entry counts differ")
	fs.BoolVar(&opts.json, "json", false, "write the comparison as JSON")
	fs.BoolVar(&opts.templates, "templates", false, "keep template arguments in kernel names, so each instantiation is its own op")
	fs.BoolVar(&opts.allowUnresolved, "allow-unresolved", false, "report window totals even when graph replay hides the per-op census")
	fs.Int64Var(&opts.maxRecords, "max-records", controllerMaxJournalRecords, "maximum journal records to decode per capture, as capture --max-records allowed it to write")
	fs.Int64Var(&opts.journalBytes, "journal-bytes", cudacapture.MaxCaptureInputBytes, "maximum decompressed bytes per journal")
	var warmup gapOptions
	fs.Float64Var(&warmup.warmupMS, "warmup-ms", 0, "start each window `N` ms after its first device activity, as gaps does")
	fs.StringVar(&warmup.warmupUntil, "warmup-until", "", "start each window at its first kernel matching `RE`, as gaps does")
	var iters iterationFlags
	iters.register(fs)
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
	if opts.iterations, err = iters.window(set); err != nil {
		return opts, nil, fmt.Errorf("compare: %w", err)
	}
	if opts.maxRecords <= 0 {
		return opts, nil, fmt.Errorf("compare: invalid -max-records %d", opts.maxRecords)
	}
	if opts.journalBytes <= 0 {
		return opts, nil, fmt.Errorf("compare: invalid -journal-bytes %d", opts.journalBytes)
	}
	if opts.stageEntries && opts.stagesPath == "" {
		return opts, nil, errors.New("compare: -stage-entries needs -stages")
	}
	if fs.NArg() != 2 {
		return opts, nil, errors.New(compareUsage)
	}
	return opts, fs.Args(), nil
}

// readOpRules reads a rules file, or returns nil rules for an empty path and
// the built-in rules for "default".
func readOpRules(path string) (opRules, error) {
	if path == "" {
		return nil, nil
	}
	if path == defaultOps {
		return parseDefaultOpRules()
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

// readStageBoundaries reads a -stages file, or returns nil boundaries for an
// empty path.
func readStageBoundaries(path string) (stageBoundaries, error) {
	rules, err := readOpRules(path)
	if err != nil {
		return nil, err
	}
	for _, r := range rules {
		if r.Op == stageUnstarted {
			return nil, fmt.Errorf("%s: line %d: stage name %s is reserved", path, r.Line, stageUnstarted)
		}
	}
	return stageBoundaries(rules), nil
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
// first run of spaces or tabs, unless it is a Go quoted string ("..." or
// `...`), which may hold spaces. Blank lines and lines starting with # are
// ignored.
func parseOpRules(r io.Reader) (opRules, error) {
	var rules opRules
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		pattern, op, err := splitOpRule(line)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", n, err)
		}
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

// splitOpRule splits a trimmed rule line into its pattern and op.
func splitOpRule(line string) (pattern, op string, err error) {
	if line[0] == '"' || line[0] == '`' {
		q, err := strconv.QuotedPrefix(line)
		if err != nil {
			return "", "", fmt.Errorf("bad quoted pattern in %q", line)
		}
		pattern, _ = strconv.Unquote(q)
		op = strings.TrimSpace(line[len(q):])
		if op == "" || op == line[len(q):] {
			return "", "", fmt.Errorf("want \"regex op\", have %q", line)
		}
		return pattern, op, nil
	}
	i := strings.IndexAny(line, " \t")
	if i < 0 {
		return "", "", fmt.Errorf("want \"regex op\", have %q", line)
	}
	return line[:i], strings.TrimSpace(line[i:]), nil
}

// opKey is the name a kernel joins by: its qualified name, as
// [cudalowering.MatchKey] gives it, or with templates the qualified name
// with its template arguments.
func opKey(name string, templates bool) string {
	if !templates {
		return cudalowering.MatchKey(name)
	}
	name = strings.TrimSpace(name)
	if key, err := demangle.ToString(name, demangle.NoParams); err == nil && key != "" && key != name {
		return key
	}
	if i := strings.IndexByte(name, '('); i >= 0 {
		name = name[:i]
	}
	return strings.TrimSpace(name)
}

// kernelOp returns the op of a kernel, given its opKey, and the index of the
// rule that named it, or -1. Without rules every kernel is its own op.
func (rules opRules) kernelOp(key string) (string, int) {
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

// stageTotals is one stage on one side: one entry of it, or the sum of
// its entries.
type stageTotals struct {
	stage     string
	entries   int
	launches  int
	exclusive int64
	summed    int64
	ops       map[string]*opTotals
}

// add charges a kernel of op to t.
func (t *stageTotals) add(op string, exclusive, summed int64) {
	t.launches++
	t.exclusive += exclusive
	t.summed += summed
	if t.ops == nil {
		t.ops = make(map[string]*opTotals)
	}
	o := t.ops[op]
	if o == nil {
		o = &opTotals{}
		t.ops[op] = o
	}
	o.launches++
	o.exclusive += exclusive
	o.summed += summed
}

// assign charges each kernel of recs, which are sorted by start, to its stage
// and returns each entry of a stage in the order entered. Kernels before the
// first boundary are one entry of stageUnstarted. owned is the exclusive time
// of each record.
func (b stageBoundaries) assign(recs []compareRecord, owned []int64) []*stageTotals {
	var out []*stageTotals
	var cur *stageTotals
	next := 0
	for i, r := range recs {
		if r.class != classKernel {
			continue
		}
		if b[next].re.MatchString(r.key) {
			cur = &stageTotals{stage: b[next].Op, entries: 1}
			out = append(out, cur)
			next = (next + 1) % len(b)
		} else if cur == nil {
			cur = &stageTotals{stage: stageUnstarted, entries: 1}
			out = append(out, cur)
		}
		cur.add(r.op, owned[i], r.end-r.start)
	}
	return out
}

// byStage sums entries by stage name.
func byStage(entries []*stageTotals) map[string]*stageTotals {
	out := make(map[string]*stageTotals)
	for _, e := range entries {
		t := out[e.stage]
		if t == nil {
			t = &stageTotals{stage: e.stage}
			out[e.stage] = t
		}
		t.entries += e.entries
		t.launches += e.launches
		t.exclusive += e.exclusive
		t.summed += e.summed
		for op, o := range e.ops {
			if t.ops == nil {
				t.ops = make(map[string]*opTotals)
			}
			m := t.ops[op]
			if m == nil {
				m = &opTotals{}
				t.ops[op] = m
			}
			m.launches += o.launches
			m.exclusive += o.exclusive
			m.summed += o.summed
		}
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

	// StageEntries is each entry of each stage, in the order entered, when
	// -stage-entries was given.
	StageEntries []CompareStageEntry `json:"stage_entries,omitempty"`

	// UnmatchedShare is |Δ(unmatched)| over the sum of |Δ| of every term of
	// the identity: how much of the movement the ops file did not name.
	UnmatchedShare float64  `json:"unmatched_share"`
	Notes          []string `json:"notes,omitempty"`

	// Cudaprof is the build that computed the comparison.
	Cudaprof buildStamp `json:"cudaprof"`
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
	// Memsets reports whether the agent collected memset activity, or is nil
	// when the capture does not say. Without it a memset counts as idle.
	Memsets *bool `json:"memsets,omitempty"`

	// Warmup describes what -warmup-ms or -warmup-until cut off the front of
	// the window, and WarmupNanos how long that was. StartTimestamp is the
	// start of what remains.
	Warmup      string `json:"warmup,omitempty"`
	WarmupNanos int64  `json:"warmup_ns,omitempty"`

	// Iterations is the number of whole iterations the window holds, when
	// -iteration-marker set it.
	Iterations int `json:"iterations,omitempty"`

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

	// UnrecordedConfigs counts the kernel launches in the window whose
	// record carries no launch configuration.
	UnrecordedConfigs int `json:"unrecorded_configs"`

	// Clocks is the GPU state read at the window's capture-range edges.
	Clocks *compareClocks `json:"clocks"`

	ops          map[string]*opTotals
	launches     map[string]*opLaunches
	stages       map[string]*stageTotals
	stageEntries []*stageTotals
}

// compareClocks is the GPU state read at a window's capture-range edges.
// NotRecorded says why there is none; a sample's own Unavailable says why
// that one edge has none.
type compareClocks struct {
	Start       *gpuEdgeSample `json:"start,omitempty"`
	Stop        *gpuEdgeSample `json:"stop,omitempty"`
	NotRecorded string         `json:"not_recorded,omitempty"`
}

// rangeClocks returns the readings of the edges that bound [start, stop).
func rangeClocks(edges []capturePhaseEdge, start, stop uint64) *compareClocks {
	c := &compareClocks{}
	found := false
	for _, e := range edges {
		switch {
		case e.Edge == "start" && e.Timestamp == start:
			c.Start, found = e.GPU, true
		case e.Edge == "stop" && e.Timestamp == stop:
			c.Stop, found = e.GPU, true
		}
	}
	switch {
	case !found:
		c.NotRecorded = "the phase receipt holds no edges for the window"
	case c.Start == nil && c.Stop == nil:
		c.NotRecorded = "the capture predates GPU readings at range edges"
	}
	return c
}

// smClock returns the SM clock of the first device that reported one.
func (s *gpuEdgeSample) smClock() (uint32, bool) {
	if s == nil {
		return 0, false
	}
	for _, d := range s.Devices {
		if d.SMClockMHz != nil {
			return *d.SMClockMHz, true
		}
	}
	return 0, false
}

type opTotals struct {
	launches  int
	exclusive int64
	summed    int64
}

// opLaunches is what one op's launches were on one side, launch by launch.
type opLaunches struct {
	exclusive  []int64 // per launch
	configs    map[LaunchConfig]int
	unrecorded int // launches whose record carries no configuration
}

// LaunchConfig is the configuration a kernel was launched with.
type LaunchConfig struct {
	Grid         [3]uint32 `json:"grid"`
	Block        [3]uint32 `json:"block"`
	SharedMemory uint32    `json:"dynamic_shared_memory"`
}

func (c LaunchConfig) String() string {
	return fmt.Sprintf("grid %dx%dx%d block %dx%dx%d smem %d",
		c.Grid[0], c.Grid[1], c.Grid[2], c.Block[0], c.Block[1], c.Block[2], c.SharedMemory)
}

// launchConfigOf returns a kernel record's launch configuration, or false
// when the record does not carry one. No launch has a zero dimension, so a
// zero is a record without the field.
func launchConfigOf(a timeline.Activity) (LaunchConfig, bool) {
	var c LaunchConfig
	if a.SharedMemory == nil {
		return c, false
	}
	for i := range 3 {
		if a.Grid[i] <= 0 || a.Block[i] <= 0 {
			return c, false
		}
		c.Grid[i], c.Block[i] = uint32(a.Grid[i]), uint32(a.Block[i])
	}
	c.SharedMemory = *a.SharedMemory
	return c, true
}

// LaunchConfigCount is one launch configuration and how often an op ran
// with it.
type LaunchConfigCount struct {
	LaunchConfig
	Launches int `json:"launches"`
}

// configCounts returns the configurations of l, most launched first.
func (l *opLaunches) configCounts() []LaunchConfigCount {
	out := make([]LaunchConfigCount, 0, len(l.configs))
	for c, n := range l.configs {
		out = append(out, LaunchConfigCount{c, n})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Launches != out[j].Launches {
			return out[i].Launches > out[j].Launches
		}
		return out[i].String() < out[j].String()
	})
	return out
}

// Launch configuration status of an op.
const (
	configSame     = "SAME"      // both sides ran the same set of configurations
	configChanged  = "CHANGED"   // the sets differ
	configUnread   = "UNREAD"    // some launch's record carries no configuration
	configOneSided = "ONE_SIDED" // the op ran on one side only
)

// configStatus compares the sets of configurations, not their counts: a
// count that moved is a launch count change.
func configStatus(b, t *opLaunches) string {
	switch {
	case len(b.exclusive) == 0 || len(t.exclusive) == 0:
		return configOneSided
	case b.unrecorded > 0 || t.unrecorded > 0:
		return configUnread
	case len(b.configs) != len(t.configs):
		return configChanged
	}
	for c := range b.configs {
		if _, ok := t.configs[c]; !ok {
			return configChanged
		}
	}
	return configSame
}

// Spread is the nearest-rank p10, p50 and p90 of one op's per-launch
// exclusive time.
type Spread struct {
	P10 int64 `json:"p10_ns"`
	P50 int64 `json:"p50_ns"`
	P90 int64 `json:"p90_ns"`
}

// spreadOf returns the spread of ns, or nil when it is empty. It sorts ns.
func spreadOf(ns []int64) *Spread {
	if len(ns) == 0 {
		return nil
	}
	slices.Sort(ns)
	return &Spread{percentile(ns, 10), percentile(ns, 50), percentile(ns, 90)}
}

// percentile returns the nearest-rank p-th percentile of sorted, which must
// not be empty: the value of rank ⌈p·n/100⌉, the smallest value at least p
// percent of the values do not exceed.
func percentile(sorted []int64, p int) int64 {
	i := (p*len(sorted)+99)/100 - 1
	return sorted[max(i, 0)]
}

// launchCountChanged reports whether an op's launch count moved by a ratio
// of at least 1.5, or whether the op ran on one side only.
func launchCountChanged(base, target int) bool {
	if base == 0 || target == 0 {
		return base != target
	}
	return 2*max(base, target) >= 3*min(base, target)
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

	// LaunchCountChanged is set when the launch counts differ by a ratio
	// of at least 1.5, or one of them is zero.
	LaunchCountChanged bool `json:"launch_count_changed"`

	// ConfigStatus is SAME, CHANGED, UNREAD or ONE_SIDED; see configStatus.
	// The Unrecorded counts are launches whose record carries no
	// configuration.
	ConfigStatus            string              `json:"config_status,omitempty"`
	BaseConfigs             []LaunchConfigCount `json:"base_configs,omitempty"`
	TargetConfigs           []LaunchConfigCount `json:"target_configs,omitempty"`
	BaseUnrecordedConfigs   int                 `json:"base_unrecorded_configs,omitempty"`
	TargetUnrecordedConfigs int                 `json:"target_unrecorded_configs,omitempty"`

	// BaseSpread and TargetSpread are the spread of per-launch exclusive
	// time, absent on a side with no launches.
	BaseSpread   *Spread `json:"base_exclusive_spread,omitempty"`
	TargetSpread *Spread `json:"target_exclusive_spread,omitempty"`
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

	// Ops is the stage's time by op, when -ops was given.
	Ops []CompareOp `json:"ops,omitempty"`
}

// CompareStageEntry is one entry of a stage in both captures: Entry i is the
// stage's (i+1)th beginning.
type CompareStageEntry struct {
	Stage                string `json:"stage"`
	Entry                int    `json:"entry"`
	BaseLaunches         int    `json:"base_launches"`
	TargetLaunches       int    `json:"target_launches"`
	BaseExclusiveNanos   int64  `json:"base_exclusive_ns"`
	TargetExclusiveNanos int64  `json:"target_exclusive_ns"`
	DeltaExclusiveNanos  int64  `json:"delta_exclusive_ns"`
	BaseSummedNanos      int64  `json:"base_summed_ns"`
	TargetSummedNanos    int64  `json:"target_summed_ns"`

	// Ops is the entry's time by op, when -ops was given.
	Ops []CompareOp `json:"ops,omitempty"`
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
	c, err := compareSides(b, t, rules, stages, opts)
	if err == nil && opts.iterations != nil {
		err = c.checkIterations(opts.iterations.count > 0)
	}
	if err != nil {
		return nil, err
	}
	return c, nil
}

// compareSides joins two measured sides by op and by stage. Of opts it uses
// only stageEntries.
func compareSides(b, t *CompareSide, rules opRules, stages stageBoundaries, opts compareOptions) (*Comparison, error) {
	if b.Device != "" && t.Device != "" && b.Device != t.Device {
		return nil, fmt.Errorf("captures ran on different devices: base %q, target %q", b.Device, t.Device)
	}
	if b.ComputeCapability != "" && t.ComputeCapability != "" && b.ComputeCapability != t.ComputeCapability {
		return nil, fmt.Errorf("captures ran on different compute capabilities: base %s, target %s", b.ComputeCapability, t.ComputeCapability)
	}
	c := &Comparison{Base: b, Target: t, OpStatus: "MEASURED", Cudaprof: currentBuild()}
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
	c.Ops = joinOps(b.ops, t.ops, b.launches, t.launches)
	for _, s := range []struct {
		name string
		side *CompareSide
	}{{"base", b}, {"target", t}} {
		if m := s.side.Memsets; m != nil && !*m {
			c.Notes = append(c.Notes, fmt.Sprintf("%s: memsets not collected (capture with --memsets), so memset time is counted as idle", s.name))
		}
		switch n := s.side.UnrecordedConfigs; {
		case n == 0:
		case n == s.side.KernelCount:
			c.Notes = append(c.Notes, fmt.Sprintf("%s: launch configuration (grid, block, dynamic shared memory) not recorded, so no op's configuration is compared", s.name))
		default:
			c.Notes = append(c.Notes, fmt.Sprintf("%s: launch configuration not recorded for %d of %d kernel launches; their ops are UNREAD, not SAME", s.name, n, s.side.KernelCount))
		}
	}
	for _, e := range []struct {
		name         string
		base, target *gpuEdgeSample
	}{{"start", b.Clocks.Start, t.Clocks.Start}, {"stop", b.Clocks.Stop, t.Clocks.Stop}} {
		bc, bok := e.base.smClock()
		tc, tok := e.target.smClock()
		if bok && tok && bc != tc {
			c.Notes = append(c.Notes, fmt.Sprintf("SM clock at the %s edge was %d MHz in base and %d MHz in target: device time differs by that ratio before any code does", e.name, bc, tc))
		}
	}
	if stages != nil {
		c.Stages = joinStages(stages.order(), b.stages, t.stages, rules != nil)
		for _, st := range c.Stages {
			if st.BaseEntries != st.TargetEntries {
				c.Notes = append(c.Notes, fmt.Sprintf("stage %s began %d times in base and %d in target: the boundaries did not align, so its rows hold different work", st.Stage, st.BaseEntries, st.TargetEntries))
			}
		}
		if opts.stageEntries {
			var err error
			c.StageEntries, err = joinStageEntries(c.Stages, b.stageEntries, t.stageEntries, rules != nil)
			if err != nil {
				return nil, err
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
		c.StageEntries = nil
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
	if err := checkCells(c); err != nil {
		return err
	}
	if c.StageEntries == nil {
		return nil
	}
	base, target = 0, 0
	for _, e := range c.StageEntries {
		base += e.BaseExclusiveNanos
		target += e.TargetExclusiveNanos
	}
	if base != c.Base.KernelNanos || target != c.Target.KernelNanos {
		return fmt.Errorf("stage entries exclusive %d, %d ns != kernel exclusive %d, %d ns", base, target, c.Base.KernelNanos, c.Target.KernelNanos)
	}
	return nil
}

// checkCells reports an error unless the ops within each stage row sum to
// the row and, over the rows, to each op's row of c.Ops. It checks the
// per-entry rows too when there are any. Rows without ops pass only when no
// row has them.
func checkCells(c *Comparison) error {
	var stages, entries []cellRow
	for _, st := range c.Stages {
		stages = append(stages, cellRow{st.Stage, st.BaseExclusiveNanos, st.TargetExclusiveNanos, st.Ops})
	}
	for _, e := range c.StageEntries {
		entries = append(entries, cellRow{fmt.Sprintf("%s entry %d", e.Stage, e.Entry), e.BaseExclusiveNanos, e.TargetExclusiveNanos, e.Ops})
	}
	if err := checkCellRows(stages, c.Ops); err != nil {
		return err
	}
	return checkCellRows(entries, c.Ops)
}

// cellRow is a stage row, or a stage entry row, and its ops.
type cellRow struct {
	name         string
	base, target int64
	ops          []CompareOp
}

func checkCellRows(rows []cellRow, ops []CompareOp) error {
	if !slices.ContainsFunc(rows, func(r cellRow) bool { return r.ops != nil }) {
		return nil
	}
	byOp := make(map[string][2]int64)
	for _, r := range rows {
		var base, target int64
		for _, o := range r.ops {
			base += o.BaseExclusiveNanos
			target += o.TargetExclusiveNanos
			s := byOp[o.Op]
			byOp[o.Op] = [2]int64{s[0] + o.BaseExclusiveNanos, s[1] + o.TargetExclusiveNanos}
		}
		if base != r.base || target != r.target {
			return fmt.Errorf("stage %s: ops exclusive %d, %d ns != its exclusive %d, %d ns", r.name, base, target, r.base, r.target)
		}
	}
	for _, o := range ops {
		if s := byOp[o.Op]; s != [2]int64{o.BaseExclusiveNanos, o.TargetExclusiveNanos} {
			return fmt.Errorf("op %s: exclusive over stages %d, %d ns != its exclusive %d, %d ns", o.Op, s[0], s[1], o.BaseExclusiveNanos, o.TargetExclusiveNanos)
		}
		delete(byOp, o.Op)
	}
	for op := range byOp {
		return fmt.Errorf("op %s is charged within stages but has no row", op)
	}
	return nil
}

// joinStageEntries pairs the entries of each side by position. stages is
// the joined per-stage rows, whose entry counts must agree on both sides,
// except for kernels before the first boundary, which either side may lack.
// With cells, each row carries its ops.
func joinStageEntries(stages []CompareStage, base, target []*stageTotals, cells bool) ([]CompareStageEntry, error) {
	for _, st := range stages {
		if st.Stage != stageUnstarted && st.BaseEntries != st.TargetEntries {
			return nil, fmt.Errorf("stage %s began %d times in base and %d in target: -stage-entries cannot pair its entries", st.Stage, st.BaseEntries, st.TargetEntries)
		}
	}
	var rows []CompareStageEntry
	row := func(stage string, entry int, b, t *stageTotals) {
		var ops []CompareOp
		if cells {
			ops = joinOps(b.ops, t.ops, nil, nil)
		}
		rows = append(rows, CompareStageEntry{
			Stage:                stage,
			Entry:                entry,
			BaseLaunches:         b.launches,
			TargetLaunches:       t.launches,
			BaseExclusiveNanos:   b.exclusive,
			TargetExclusiveNanos: t.exclusive,
			DeltaExclusiveNanos:  t.exclusive - b.exclusive,
			BaseSummedNanos:      b.summed,
			TargetSummedNanos:    t.summed,
			Ops:                  ops,
		})
	}
	unstarted := func(es []*stageTotals) (*stageTotals, []*stageTotals) {
		if len(es) > 0 && es[0].stage == stageUnstarted {
			return es[0], es[1:]
		}
		return &stageTotals{}, es
	}
	bu, base := unstarted(base)
	tu, target := unstarted(target)
	if bu.entries+tu.entries > 0 {
		row(stageUnstarted, 0, bu, tu)
	}
	if len(base) != len(target) {
		return nil, fmt.Errorf("base entered %d stages and target %d", len(base), len(target))
	}
	seen := make(map[string]int)
	for i, b := range base {
		t := target[i]
		if b.stage != t.stage {
			return nil, fmt.Errorf("stage entry %d is %s in base and %s in target", i, b.stage, t.stage)
		}
		row(b.stage, seen[b.stage], b, t)
		seen[b.stage]++
	}
	return rows, nil
}

// joinStages lists every stage either side entered, in order. With cells,
// each row carries its ops.
func joinStages(order []string, base, target map[string]*stageTotals, cells bool) []CompareStage {
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
		var ops []CompareOp
		if cells {
			ops = joinOps(b.ops, t.ops, nil, nil)
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
			Ops:                  ops,
		})
	}
	return rows
}

// joinOps pairs the op totals of two sides. With launch maps, each row also
// carries its launch configurations and per-launch spread; stage cells pass
// nil, since configurations are compared once per op, not per cell.
func joinOps(base, target map[string]*opTotals, baseLaunches, targetLaunches map[string]*opLaunches) []CompareOp {
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
		row := CompareOp{
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
			LaunchCountChanged:   launchCountChanged(b.launches, t.launches),
		}
		if baseLaunches != nil {
			bl, tl := baseLaunches[n], targetLaunches[n]
			if bl == nil {
				bl = &opLaunches{}
			}
			if tl == nil {
				tl = &opLaunches{}
			}
			row.ConfigStatus = configStatus(bl, tl)
			row.BaseConfigs = bl.configCounts()
			row.TargetConfigs = tl.configCounts()
			row.BaseUnrecordedConfigs = bl.unrecorded
			row.TargetUnrecordedConfigs = tl.unrecorded
			row.BaseSpread = spreadOf(bl.exclusive)
			row.TargetSpread = spreadOf(tl.exclusive)
		}
		rows = append(rows, row)
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
	status, records, err := decodeCompleteCapture(cap, opts.maxRecords, opts.journalBytes)
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
	side.Memsets = captureMemsets(status.Capability)
	if err := side.selectWindow(status.PhaseReceipt, opts.phase, records); err != nil {
		return nil, err
	}
	if opts.warmup != nil {
		if err := side.cutWarmup(records, *opts.warmup); err != nil {
			return nil, err
		}
	}
	if opts.iterations != nil {
		if err := side.cutIterations(records, *opts.iterations); err != nil {
			return nil, err
		}
	}
	if err := side.measure(records, rules, stages, opts.templates); err != nil {
		return nil, err
	}
	if side.UnresolvedNanos > 0 && !opts.allowUnresolved {
		return nil, fmt.Errorf("%d ns of graph replay in the window is covered by no resolved record, so the per-op census is hidden (graph trace mode %s); capture with --graph-trace=node or graphs disabled, or pass -allow-unresolved for window totals only", side.UnresolvedNanos, side.graphTraceModeString())
	}
	return side, nil
}

// selectWindow takes the window from the phase receipt, or from the device
// span of the whole capture when there is no qualified receipt.
func (s *CompareSide) selectWindow(r *capturePhaseReceipt, phase int, records timeline.Records) error {
	top := r
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
		s.Clocks = &compareClocks{NotRecorded: "the window is not a capture range"}
		s.WindowNote = fmt.Sprintf("phase receipt %s; window is first to last device activity", r)
		first := true
		for _, a := range append(records.Activities, records.Graphs...) {
			if activityClass(a.Subtype) < 0 && a.Subtype != "graph_trace" {
				continue
			}
			if first || a.Start < s.StartTimestamp {
				s.StartTimestamp = a.Start
			}
			if first || a.End > s.StopTimestamp {
				s.StopTimestamp = a.End
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
	s.Clocks = rangeClocks(top.Edges, r.StartTimestamp, r.StopTimestamp)
	return nil
}

// cutWarmup moves the window's start to the first activity of its steady
// state, chosen by steadyActivities from the device records inside the
// window. A record that began before the new start is then an edge
// straddler or outside the window, so the prologue drops out of every total.
func (s *CompareSide) cutWarmup(records timeline.Records, opts gapOptions) error {
	var until *regexp.Regexp
	if opts.warmupUntil != "" {
		re, err := regexp.Compile(opts.warmupUntil)
		if err != nil {
			return fmt.Errorf("warmup kernel pattern: %w", err)
		}
		until = re
	}
	var acts []timeline.Activity
	for _, a := range records.Activities {
		if activityClass(a.Subtype) < 0 || a.End < a.Start || a.Start < s.StartTimestamp || a.End > s.StopTimestamp {
			continue
		}
		acts = append(acts, a)
	}
	if len(acts) == 0 {
		return fmt.Errorf("no kernel, memcpy or memset lies inside the window [%d,%d)", s.StartTimestamp, s.StopTimestamp)
	}
	sort.SliceStable(acts, func(i, j int) bool {
		if acts[i].Start != acts[j].Start {
			return acts[i].Start < acts[j].Start
		}
		return acts[i].End < acts[j].End
	})
	s.Warmup = describeWarmup(opts)
	steady := gaps.Steady(acts, int64(opts.warmupMS*1e6), until)
	if len(steady) == 0 {
		return fmt.Errorf("no steady state in the window [%d,%d): the warmup (%s) never ended", s.StartTimestamp, s.StopTimestamp, s.Warmup)
	}
	s.WarmupNanos = steady[0].Start - s.StartTimestamp
	s.StartTimestamp = steady[0].Start
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

	config      LaunchConfig
	configKnown bool
}

// measure partitions the window. Device records count only when they fall
// entirely inside it and graph replay envelopes are clipped to it, the rule
// measurePhase applies to the phase receipt.
func (s *CompareSide) measure(records timeline.Records, rules opRules, stages stageBoundaries, templates bool) error {
	lo, hi := s.StartTimestamp, s.StopTimestamp
	s.WindowNanos = hi - lo
	s.ops = make(map[string]*opTotals)
	s.launches = make(map[string]*opLaunches)
	used := make([]bool, len(rules))
	var recs []compareRecord
	for _, a := range records.Activities {
		class := activityClass(a.Subtype)
		if class < 0 || a.End < a.Start {
			continue
		}
		if a.Start < lo || a.End > hi {
			if a.End > lo && a.Start < hi {
				s.EdgeStraddlers++
			}
			continue
		}
		r := compareRecord{start: a.Start, end: a.End, class: class}
		switch class {
		case classKernel:
			key := opKey(a.Name, templates)
			op, rule := rules.kernelOp(key)
			if rule >= 0 {
				used[rule] = true
			}
			t := s.ops[op]
			if t == nil {
				t = &opTotals{}
				s.ops[op] = t
			}
			t.launches++
			t.summed += a.End - a.Start
			r.op, r.key = op, key
			r.config, r.configKnown = launchConfigOf(a)
			s.KernelCount++
		case classMemcpy:
			s.MemcpyCount++
			s.MemcpyBytes += a.Bytes
			if s.MemcpyKinds == nil {
				s.MemcpyKinds = make(map[string]int)
			}
			s.MemcpyKinds[strings.TrimPrefix(cupti.ActivityMemcpyKind(a.CopyKind).String(), "ActivityMemcpyKind")]++
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
		s.stageEntries = stages.assign(recs, owned)
		s.stages = byStage(s.stageEntries)
	}
	busy := make([]timeline.Interval, len(recs))
	first, last := recs[0].start, recs[0].end
	for i, r := range recs {
		busy[i] = timeline.Interval{Start: uint64(r.start), End: uint64(r.end)}
		last = max(last, r.end)
		switch r.class {
		case classKernel:
			s.ops[r.op].exclusive += owned[i]
			s.KernelNanos += owned[i]
			l := s.launches[r.op]
			if l == nil {
				l = &opLaunches{configs: make(map[LaunchConfig]int)}
				s.launches[r.op] = l
			}
			l.exclusive = append(l.exclusive, owned[i])
			if r.configKnown {
				l.configs[r.config]++
			} else {
				l.unrecorded++
				s.UnrecordedConfigs++
			}
		case classMemcpy:
			s.MemcpyNanos += owned[i]
		case classMemset:
			s.MemsetNanos += owned[i]
		}
	}
	s.BusyNanos = int64(timeline.Union(busy))
	if parts := s.KernelNanos + s.MemcpyNanos + s.MemsetNanos; parts != s.BusyNanos {
		return fmt.Errorf("partition charged %d ns, but the busy union is %d ns", parts, s.BusyNanos)
	}
	var replays []timeline.Interval
	for _, g := range records.Graphs {
		if l, h := max(g.Start, lo), min(g.End, hi); l < h {
			replays = append(replays, timeline.Interval{Start: uint64(l), End: uint64(h)})
		}
	}
	covered := int64(timeline.Union(append(busy, replays...)))
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
	if err := writeClocks(w, b.Clocks, t.Clocks); err != nil {
		return err
	}
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
	fmt.Fprintln(w, "config: launch configuration (grid, block, dynamic shared memory) SAME, CHANGED, UNREAD (not recorded) or ONE_SIDED.")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "launches\t\texclusive\t\t\tsummed\t\t\tmean\t\t\t")
	fmt.Fprintln(tw, "base\ttarget\tbase\ttarget\tdelta\tbase\ttarget\tdelta\tbase\ttarget\tconfig\t  op")
	for _, o := range c.Ops {
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%+d\t%d\t%d\t%+d\t%d\t%d\t%s\t  %s\n",
			o.BaseLaunches, o.TargetLaunches,
			o.BaseExclusiveNanos, o.TargetExclusiveNanos, o.DeltaExclusiveNanos,
			o.BaseSummedNanos, o.TargetSummedNanos, o.DeltaSummedNanos,
			o.BaseMeanNanos, o.TargetMeanNanos, o.ConfigStatus, o.Op)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if err := c.writeLaunches(w); err != nil {
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
	if err := tw.Flush(); err != nil {
		return err
	}
	if c.StageEntries == nil {
		return nil
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "per stage entry, ns. entry i is the stage's (i+1)th beginning; a stage's entries sum to its row above.")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "launches\t\texclusive\t\t\tsummed\t\t\t")
	fmt.Fprintln(tw, "base\ttarget\tbase\ttarget\tdelta\tbase\ttarget\tentry\t  stage")
	for _, e := range c.StageEntries {
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%+d\t%d\t%d\t%d\t  %s\n",
			e.BaseLaunches, e.TargetLaunches,
			e.BaseExclusiveNanos, e.TargetExclusiveNanos, e.DeltaExclusiveNanos,
			e.BaseSummedNanos, e.TargetSummedNanos, e.Entry, e.Stage)
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

// compareSpreadOps is how many ops, largest |Δ| first, the text spread table
// lists. JSON gives the spread of every op.
const compareSpreadOps = 10

// writeLaunches writes the launch configuration summary and changes, the
// launch count changes, and the spread of the ops of largest |Δ|.
func (c *Comparison) writeLaunches(w io.Writer) error {
	status := make(map[string]int)
	for _, o := range c.Ops {
		status[o.ConfigStatus]++
	}
	fmt.Fprintf(w, "\nlaunch configuration: %d ops SAME, %d CHANGED, %d UNREAD (not recorded), %d ONE_SIDED\n",
		status[configSame], status[configChanged], status[configUnread], status[configOneSided])
	for _, o := range c.Ops {
		if o.ConfigStatus != configChanged {
			continue
		}
		fmt.Fprintf(w, "CHANGED %s\n", o.Op)
		for _, cc := range o.BaseConfigs {
			fmt.Fprintf(w, "  base    %s  launches %d\n", cc.LaunchConfig, cc.Launches)
		}
		for _, cc := range o.TargetConfigs {
			fmt.Fprintf(w, "  target  %s  launches %d\n", cc.LaunchConfig, cc.Launches)
		}
	}

	var moved []CompareOp
	for _, o := range c.Ops {
		if o.LaunchCountChanged {
			moved = append(moved, o)
		}
	}
	if len(moved) > 0 {
		fmt.Fprintln(w, "\nlaunch count changes (ratio >= 1.5, or ran on one side only):")
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
		fmt.Fprintln(tw, "base\ttarget\tratio\t  op")
		for _, o := range moved {
			ratio := "-"
			if o.BaseLaunches > 0 && o.TargetLaunches > 0 {
				ratio = fmt.Sprintf("%.2fx", float64(o.TargetLaunches)/float64(o.BaseLaunches))
			}
			fmt.Fprintf(tw, "%d\t%d\t%s\t  %s\n", o.BaseLaunches, o.TargetLaunches, ratio, o.Op)
		}
		if err := tw.Flush(); err != nil {
			return err
		}
	}

	n := min(len(c.Ops), compareSpreadOps)
	fmt.Fprintf(w, "\nper-launch exclusive ns, nearest rank, for the %d ops of largest |Δ exclusive|:\n", n)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "base\t\t\ttarget\t\t\t")
	fmt.Fprintln(tw, "p10\tp50\tp90\tp10\tp50\tp90\t  op")
	for _, o := range c.Ops[:n] {
		fmt.Fprintf(tw, "%s\t%s\t  %s\n", o.BaseSpread.cells(), o.TargetSpread.cells(), o.Op)
	}
	return tw.Flush()
}

// cells returns the spread as three tab-separated cells, "-" when absent.
func (s *Spread) cells() string {
	if s == nil {
		return "-\t-\t-"
	}
	return fmt.Sprintf("%d\t%d\t%d", s.P10, s.P50, s.P90)
}

// writeClocks writes each side's GPU readings at its range edges side by
// side, or why a side has none.
func writeClocks(w io.Writer, b, t *compareClocks) error {
	fmt.Fprintln(w, "gpu at range edges (NVML, read after each edge):")
	for _, s := range []struct {
		name   string
		clocks *compareClocks
	}{{"base", b}, {"target", t}} {
		if s.clocks.NotRecorded != "" {
			fmt.Fprintf(w, "        %s: not recorded: %s\n", s.name, s.clocks.NotRecorded)
		}
	}
	if b.NotRecorded != "" && t.NotRecorded != "" {
		return nil
	}
	samples := []*gpuEdgeSample{b.Start, b.Stop, t.Start, t.Stop}
	names := []string{"base start", "base stop", "target start", "target stop"}
	var devices []uint32
	seen := make(map[uint32]bool)
	for i, s := range samples {
		switch {
		case s == nil:
			samples[i] = &gpuEdgeSample{}
		case s.Unavailable != "":
			fmt.Fprintf(w, "        %s: not recorded: %s\n", names[i], s.Unavailable)
		}
		for _, d := range samples[i].Devices {
			if !seen[d.Index] {
				seen[d.Index] = true
				devices = append(devices, d.Index)
			}
		}
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintf(tw, "\t%s\t\n", strings.Join(names, "\t"))
	fields := []struct {
		name string
		get  func(gpuReading) *uint32
	}{
		{"sm clock MHz", func(r gpuReading) *uint32 { return r.SMClockMHz }},
		{"mem clock MHz", func(r gpuReading) *uint32 { return r.MemClockMHz }},
		{"temperature C", func(r gpuReading) *uint32 { return r.TemperatureC }},
		{"power mW", func(r gpuReading) *uint32 { return r.PowerMilliwatts }},
	}
	for _, dev := range devices {
		for _, f := range fields {
			label := f.name
			if len(devices) > 1 {
				label = fmt.Sprintf("gpu %d %s", dev, f.name)
			}
			cells := make([]string, len(samples))
			for i, s := range samples {
				cells[i] = "-"
				for _, d := range s.Devices {
					if v := f.get(d); d.Index == dev && v != nil {
						cells[i] = fmt.Sprint(*v)
					}
				}
			}
			fmt.Fprintf(tw, "        %s\t%s\t\n", label, strings.Join(cells, "\t"))
		}
	}
	cells := make([]string, len(samples))
	for i, s := range samples {
		cells[i] = "-"
		if len(s.Devices) > 0 {
			cells[i] = fmt.Sprint(s.LagNanos)
		}
	}
	fmt.Fprintf(tw, "        read lag ns\t%s\t\n", strings.Join(cells, "\t"))
	return tw.Flush()
}
