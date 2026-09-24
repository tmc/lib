package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// Qualify computes a pre-registered verdict for an A/B timing campaign from
// the captures themselves, so the window and the arithmetic cannot drift
// between the harness that ran the captures and the script that scored them.
//
// A campaign is a set of blocks, each four captures run in the order C1, G1,
// G2, C2: C is the reference arm and G the arm under test. The ABBA order
// cancels a linear drift within a block. Each capture is measured by its
// steady-state device timeline span, the window gaps reports after the
// warmup, and the verdict is:
//
//	pairs          G1-C1 and G2-C2 in every block
//	block mean     the mean of a block's two pairs
//	effect         the median of the block means
//	noise floor    max(floor, MAD(C2-C1), MAD(G2-G1)) over the blocks
//	noise gate     |effect| > multiplier x noise floor
//	sign gate      every pair has the same, nonzero, sign
//
// The within-arm differences C2-C1 and G2-G1 are what the same program does
// twice in one block, so their spread is the run-to-run noise the effect has
// to clear. The floor keeps a campaign whose arms happened to repeat exactly
// from passing on an effect of nanoseconds.
//
// The ABBA arithmetic is only as good as the manifest's claims about each
// capture, so two are checked against the captures before anything is
// scored. Every capture of one arm must have run the same executable, by the
// SHA-256 the launcher recorded; a campaign in which they differ is refused.
// Within a block, the captures' measured ranges must have started in the
// order C1, G1, G2, C2 on the agent's monotonic clock; one that did not is
// refused too. Order is not compared across blocks: the clock restarts at
// boot. A capture that predates either record leaves that check unverified,
// and the report says so.
//
// -ops and -stages break the pairs down the way compare breaks down one pair:
// each capture is measured by compare's exclusive partition over its window
// with the same warmup cut, and each op and stage row reports, over every
// pair the verdict scores, the median of base (C), target (G) and delta
// (G - C) exclusive time and the range of the delta. Each median is taken
// over its own column, so a row's delta median is not its target median
// minus its base median, and the rows' medians do not sum to the median of
// the total. The breakdown is description only; it is not gated, and it is
// taken over the blocks the verdict scores.
//
// An arm that ran while some other process loaded the host measures that
// process as much as its own code; on a GB10 the CPU and GPU share memory,
// so CPU load alone slows kernels. Given a violations log, an arm with a
// violation in its wall-clock window, or whose window the capture cannot
// place, is excluded, and so is every block it belongs to: a block without
// all four arms has neither the ABBA order that cancels drift nor both
// within-arm differences. The verdict is REFUSED, not scored, when fewer
// blocks remain than -min-blocks. Identity and order are still checked over
// every capture: a wrong executable is a manifest error either way.

// qualifyArms is the run order within a block.
var qualifyArms = []string{"C1", "G1", "G2", "C2"}

// The verdicts. REFUSED means there was not enough clean evidence to score.
const (
	verdictPass    = "PASS"
	verdictFail    = "FAIL"
	verdictRefused = "REFUSED"
)

type qualifyOptions struct {
	manifest        string
	gaps            gapOptions
	noiseMultiplier float64
	noiseFloorMS    float64
	violations      string
	violationMargin time.Duration
	minBlocks       int
	json            bool
	opsPath         string
	stagesPath      string
}

const qualifyUsage = "usage: cudaprof qualify (-warmup-until=RE | -warmup-ms=N) [-noise-multiplier=3] [-noise-floor-ms=1] [-violations=FILE [-violation-margin=2s] [-min-blocks=N]] [-ops ops.txt] [-stages stages.txt] [-max-records=N] [-json] manifest.txt"

// runQualifyCommand implements "cudaprof qualify".
func runQualifyCommand(args []string, stdout io.Writer) error {
	opts, err := parseQualifyArgs(args, stdout)
	if err != nil || opts.manifest == "" {
		return err
	}
	f, err := os.Open(opts.manifest)
	if err != nil {
		return fmt.Errorf("qualify: %w", err)
	}
	entries, err := parseQualifyManifest(f, filepath.Dir(opts.manifest))
	f.Close()
	if err != nil {
		return fmt.Errorf("qualify: %s: %w", opts.manifest, err)
	}
	if blocks := len(entries) / len(qualifyArms); opts.minBlocks > blocks {
		return fmt.Errorf("qualify: -min-blocks %d exceeds the manifest's %d blocks", opts.minBlocks, blocks)
	}
	var log hostLog
	if opts.violations != "" {
		f, err := os.Open(opts.violations)
		if err != nil {
			return fmt.Errorf("qualify: %w", err)
		}
		log, err = parseHostLog(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("qualify: %s: %w", opts.violations, err)
		}
	}
	rules, err := readOpRules(opts.opsPath)
	if err != nil {
		return fmt.Errorf("qualify: %w", err)
	}
	stages, err := readStageBoundaries(opts.stagesPath)
	if err != nil {
		return fmt.Errorf("qualify: %w", err)
	}
	for i := range entries {
		e := &entries[i]
		if err := measureQualifyEntry(e, opts.gaps, rules, stages); err != nil {
			return fmt.Errorf("qualify: block %s arm %s: %w", e.Block, e.Arm, err)
		}
	}
	// Identity and order are checked over every capture, dropped or not: a
	// capture that ran the wrong executable is a manifest error either way.
	identity, order, err := checkQualifyRuns(entries)
	if err != nil {
		return fmt.Errorf("qualify: %s: %w", opts.manifest, err)
	}
	host := qualifyHostLoad{}
	if opts.violations != "" {
		host = markHostLoad(entries, log, opts.violationMargin)
		host.Log = opts.violations
	}
	q := qualify(entries, opts.noiseMultiplier, opts.noiseFloorMS, opts.minBlocks)
	q.Identity, q.Order = identity, order
	q.Window = describeWarmup(opts.gaps) + " excluded"
	q.HostLoad = host
	q.Cudaprof = currentBuild()
	if (rules != nil || stages != nil) && q.Verdict != verdictRefused {
		q.Breakdown, err = breakDownPairs(q.scored(), rules, stages)
		if err != nil {
			return fmt.Errorf("qualify: %w", err)
		}
	}
	if opts.json {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(q); err != nil {
			return err
		}
	} else {
		q.write(stdout)
	}
	switch q.Verdict {
	case verdictPass:
		return nil
	case verdictRefused:
		return fmt.Errorf("qualify: verdict %s: %s", q.Verdict, q.Reason)
	}
	return &exitCodeError{Code: 2, Err: fmt.Errorf("qualify: verdict %s: noise gate %s, sign gate %s", q.Verdict, q.NoiseGate, q.SignGate)}
}

// parseQualifyArgs returns options with an empty manifest and a nil error
// when help was requested and printed.
func parseQualifyArgs(args []string, stdout io.Writer) (qualifyOptions, error) {
	opts := qualifyOptions{gaps: gapOptions{maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes}}
	fs := flag.NewFlagSet("qualify", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.Float64Var(&opts.gaps.warmupMS, "warmup-ms", 0, "exclude the first `N` ms of each capture's device timeline")
	fs.StringVar(&opts.gaps.warmupUntil, "warmup-until", "", "exclude everything before the first kernel matching `RE`")
	fs.Float64Var(&opts.noiseMultiplier, "noise-multiplier", 3, "the noise gate passes when |effect| exceeds this many noise floors")
	fs.Float64Var(&opts.noiseFloorMS, "noise-floor-ms", 1, "the least noise floor, in `ms`")
	fs.StringVar(&opts.violations, "violations", "", "exclude arms that ran while a process in this violations `file` loaded the host")
	fs.DurationVar(&opts.violationMargin, "violation-margin", 2*time.Second, "widen each arm's window by this much on each side; at least the watcher's sampling interval")
	fs.IntVar(&opts.minBlocks, "min-blocks", 0, "the fewest blocks the verdict may be scored from after exclusions; 0 means every block")
	fs.Int64Var(&opts.gaps.maxRecords, "max-records", opts.gaps.maxRecords, "maximum journal records to decode per capture")
	fs.BoolVar(&opts.json, "json", false, "write the verdict as JSON")
	fs.StringVar(&opts.opsPath, "ops", "", "break the pairs down by op, from a file of `regex op` lines as compare reads")
	fs.StringVar(&opts.stagesPath, "stages", "", "break the pairs down by stage, from a file of `regex stage` boundaries as compare reads")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			printQualifyHelp(stdout, fs)
			return qualifyOptions{}, nil
		}
		return qualifyOptions{}, fmt.Errorf("qualify: %w\n%s", err, qualifyUsage)
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	switch {
	case set["warmup-ms"] && set["warmup-until"]:
		return qualifyOptions{}, errors.New("qualify: -warmup-ms and -warmup-until are exclusive")
	case !set["warmup-ms"] && !set["warmup-until"]:
		// The window is part of the pre-registration. A default would be a
		// window nobody chose.
		return qualifyOptions{}, fmt.Errorf("qualify: the steady-state window must be named with -warmup-until or -warmup-ms\n%s", qualifyUsage)
	case opts.gaps.warmupMS < 0:
		return qualifyOptions{}, fmt.Errorf("qualify: invalid -warmup-ms %g", opts.gaps.warmupMS)
	case set["warmup-until"] && opts.gaps.warmupUntil == "":
		return qualifyOptions{}, errors.New("qualify: empty -warmup-until pattern")
	case !(opts.noiseMultiplier > 0) || !(opts.noiseFloorMS >= 0):
		return qualifyOptions{}, errors.New("qualify: -noise-multiplier must be positive and -noise-floor-ms nonnegative")
	case opts.gaps.maxRecords <= 0:
		return qualifyOptions{}, fmt.Errorf("qualify: invalid -max-records %d", opts.gaps.maxRecords)
	case opts.violationMargin < 0:
		return qualifyOptions{}, fmt.Errorf("qualify: invalid -violation-margin %s", opts.violationMargin)
	case opts.minBlocks < 0:
		return qualifyOptions{}, fmt.Errorf("qualify: invalid -min-blocks %d", opts.minBlocks)
	case (set["violation-margin"] || set["min-blocks"]) && opts.violations == "":
		return qualifyOptions{}, errors.New("qualify: -violation-margin and -min-blocks need -violations")
	}
	if opts.gaps.warmupUntil != "" {
		if _, err := regexp.Compile(opts.gaps.warmupUntil); err != nil {
			return qualifyOptions{}, fmt.Errorf("qualify: -warmup-until: %w", err)
		}
	}
	if fs.NArg() != 1 {
		return qualifyOptions{}, errors.New(qualifyUsage)
	}
	opts.manifest = fs.Arg(0)
	return opts, nil
}

func printQualifyHelp(w io.Writer, fs *flag.FlagSet) {
	fmt.Fprintf(w, `%s

Computes the verdict of an ABBA timing campaign from its captures. The
manifest names one capture per line as "block arm path"; arm is one of C1,
G1, G2, C2, run in that order, C the reference arm and G the arm under test.
Every block names all four arms once. Relative paths are resolved against the
manifest's directory; blank lines and lines starting with # are ignored.

Before scoring, every capture of an arm must record the same executable
SHA-256, and within each block the measured ranges must have started in
C1, G1, G2, C2 order; a campaign that contradicts its manifest is refused.
A capture that records neither leaves the check unverified, as the report
says.

Each capture is measured by its steady-state device timeline span, the window
"cudaprof gaps" reports for the same -warmup option; the whole span is
reported beside it as description only. A capture whose warmup never ends --
no kernel matches -warmup-until, or the run is shorter than -warmup-ms -- is
refused. Per block the pairs are G1-C1 and G2-C2, and:

  effect       median of the block means (a positive effect: G is slower)
  noise floor  max(-noise-floor-ms, MAD(C2-C1), MAD(G2-G1)) over the blocks
  noise gate   passes when |effect| > -noise-multiplier x noise floor
  sign gate    passes when every pair has the same nonzero sign

MAD is the median absolute deviation about the median.

-ops and -stages add a breakdown of the pairs by op and by stage, read as
"cudaprof compare" reads them, with each capture measured over its window
after the same warmup cut. For each row, over every pair the verdict scores:
the median of C, of G and of G - C exclusive time, the least and greatest
G - C, and the range of launch counts. Each median is of its own column, so
the rows' medians do not sum to the total's. The breakdown is not gated.

With -violations, each arm's wall-clock window -- its first capture-range edge
to its last, widened by -violation-margin on each side -- is checked against
a log of processes that loaded the host outside the campaign. An arm with a
violation in its window is excluded and named with the offending processes;
an arm whose capture records no wall clock or no closed capture range, or
whose widened window the watcher's heartbeats do not cover, cannot be checked
and is excluded too, never passed. A block with an excluded arm is dropped
whole, from the verdict and the breakdown alike, though its captures are
still checked for identity and order; when fewer than -min-blocks blocks
remain the verdict is REFUSED. Without -violations the report says that host
load was not checked. The log has one violation or heartbeat per line:

  time pid comm cpu gpu holder
  time heartbeat holder

time RFC 3339, pid a positive integer, comm the process name without
whitespace, cpu percent of one core, gpu device memory in MiB, holder the
label of the lock holder or "-". The watcher writes a heartbeat every
sampling interval; an arm is covered when a heartbeat lies at or before the
start of its widened window, one at or after its end, and no two consecutive
heartbeats between them are further apart than -violation-margin. Fields are
separated by spaces or tabs; blank lines and lines starting with # are
ignored, and any other line that does not parse is an error.

The exit status is 0 when both gates pass, 2 when either fails, and 1 when
the verdict could not be computed or was REFUSED.

Options:
`, qualifyUsage)
	fs.SetOutput(w)
	fs.PrintDefaults()
}

// qualifyEntry is one capture of the campaign.
type qualifyEntry struct {
	Block       string `json:"block"`
	Arm         string `json:"arm"`
	Path        string `json:"path"`
	SteadyNanos int64  `json:"steady_span_ns"`
	WholeNanos  int64  `json:"whole_span_ns"`

	// side is the capture partitioned as compare partitions it, when the
	// campaign is broken down by op or stage.
	side *CompareSide

	// Executable is the SHA-256 of the program the capture ran, and Start
	// the agent's monotonic timestamp of the start of its measured range.
	// Each is empty or zero when the capture does not record it.
	Executable string `json:"executable_sha256,omitempty"`
	Start      uint64 `json:"range_start_timestamp,omitempty"`

	// Window is the arm's wall-clock window, or nil and Unchecked says why
	// the capture cannot place it. The fields after it are set only when a
	// violations log was given: HostLoad is hostClean, hostLoaded or
	// hostUnchecked, and Violations lists the log lines inside the window.
	Window     *qualifyWindow  `json:"wall_window,omitempty"`
	Unchecked  string          `json:"unchecked,omitempty"`
	HostLoad   string          `json:"host_load,omitempty"`
	Violations []hostViolation `json:"violations,omitempty"`
}

// excluded reports whether the entry must be left out of the verdict.
func (e *qualifyEntry) excluded() bool {
	return e.HostLoad == hostLoaded || e.HostLoad == hostUnchecked
}

// parseQualifyManifest reads "block arm path" lines. The result is in
// manifest block order and, within a block, in run order.
func parseQualifyManifest(r io.Reader, dir string) ([]qualifyEntry, error) {
	byBlock := map[string]map[string]string{}
	var blocks []string
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.Fields(line)
		if len(f) != 3 {
			return nil, fmt.Errorf("line %d: want \"block arm path\", have %q", n, line)
		}
		block, arm, path := f[0], f[1], f[2]
		if !slices.Contains(qualifyArms, arm) {
			return nil, fmt.Errorf("line %d: arm %q is not one of %s", n, arm, strings.Join(qualifyArms, ", "))
		}
		if byBlock[block] == nil {
			byBlock[block] = map[string]string{}
			blocks = append(blocks, block)
		}
		if _, dup := byBlock[block][arm]; dup {
			return nil, fmt.Errorf("line %d: block %s names arm %s twice", n, block, arm)
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(dir, path)
		}
		byBlock[block][arm] = path
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(blocks) == 0 {
		return nil, errors.New("no captures")
	}
	var entries []qualifyEntry
	for _, b := range blocks {
		for _, arm := range qualifyArms {
			path, ok := byBlock[b][arm]
			if !ok {
				return nil, fmt.Errorf("block %s has no arm %s", b, arm)
			}
			entries = append(entries, qualifyEntry{Block: b, Arm: arm, Path: path})
		}
	}
	return entries, nil
}

// measureQualifyEntry reads the capture e names, once, and fills in its
// spans, the executable it ran, the start of its measured range and its
// wall-clock window. With rules or stages it also partitions the capture as
// compare does, after the same warmup cut.
func measureQualifyEntry(e *qualifyEntry, opts gapOptions, rules opRules, stages stageBoundaries) error {
	path, err := filepath.Abs(e.Path)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, opts.limits)
	if err != nil {
		return fmt.Errorf("read capture %s: %w", e.Path, err)
	}
	report, err := analyzeGaps(cap, opts)
	if err != nil {
		return fmt.Errorf("analyze %s: %w", e.Path, err)
	}
	e.SteadyNanos, e.WholeNanos, err = qualifySpan(e.Path, report)
	if err != nil {
		return err
	}
	if rules != nil || stages != nil {
		warmup := opts
		e.side, err = measureCompareSide(cap, rules, stages, compareOptions{phase: -1, warmup: &warmup})
		if err != nil {
			return fmt.Errorf("partition %s: %w", e.Path, err)
		}
	}
	data, ok := cap.Files["capture-status.json"]
	if !ok {
		e.Unchecked = "capture has no capture-status.json"
		return nil
	}
	var envelope struct {
		Status captureResult `json:"status"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		e.Unchecked = fmt.Sprintf("read capture-status.json: %v", err)
		return nil
	}
	st := &envelope.Status
	if st.Target != nil && st.Target.Executable != nil {
		e.Executable = st.Target.Executable.SHA256
	}
	if st.PhaseReceipt != nil {
		e.Start = st.PhaseReceipt.StartTimestamp
	}
	e.Window, e.Unchecked = armWindow(st)
	return nil
}

// checkQualifyRuns checks that each arm ran one executable and that each
// block ran in manifest order, and describes what it could verify. An error
// is a campaign that contradicts its manifest.
func checkQualifyRuns(entries []qualifyEntry) (identity, order string, err error) {
	identity, order = "verified", "verified"
	arms := map[string]string{} // arm letter to the first executable seen
	first := map[string]qualifyEntry{}
	for _, e := range entries {
		if e.Executable == "" {
			identity = "unverified: a capture records no executable identity"
			continue
		}
		letter := e.Arm[:1]
		if prev, ok := arms[letter]; ok && prev != e.Executable {
			f := first[letter]
			return "", "", fmt.Errorf("arm %s ran two executables: block %s %s ran %.12s, block %s %s ran %.12s", letter, f.Block, f.Arm, prev, e.Block, e.Arm, e.Executable)
		}
		if _, ok := arms[letter]; !ok {
			arms[letter], first[letter] = e.Executable, e
		}
	}
	for i := 0; i+len(qualifyArms) <= len(entries); i += len(qualifyArms) {
		block := entries[i : i+len(qualifyArms)]
		for j := 1; j < len(block); j++ {
			prev, e := block[j-1], block[j]
			if prev.Start == 0 || e.Start == 0 {
				order = "unverified: a capture records no measured-range start"
				continue
			}
			if e.Start <= prev.Start {
				return "", "", fmt.Errorf("block %s ran out of manifest order: %s started at %d, before %s at %d (a reboot within a block also reads this way)", e.Block, e.Arm, e.Start, prev.Arm, prev.Start)
			}
		}
	}
	return identity, order, nil
}

// qualifySpan returns the steady-state and whole device timeline spans in
// report, the gap report of the capture at path.
func qualifySpan(path string, report *GapReport) (steady, whole int64, err error) {
	if len(report.Devices) != 1 {
		return 0, 0, fmt.Errorf("capture ran on %d devices, want 1", len(report.Devices))
	}
	d := report.Devices[0]
	switch {
	case d.SteadyActivities == 0:
		return 0, 0, fmt.Errorf("no steady-state window: the warmup (%s) never ended", report.warmup())
	case d.SteadyState == nil:
		// The warmup excluded nothing: the window is the whole timeline.
		return d.Overall.SpanNanos, d.Overall.SpanNanos, nil
	}
	return d.SteadyState.SpanNanos, d.Overall.SpanNanos, nil
}

// Qualification is the verdict of one campaign. Every duration is in
// milliseconds; a pair is G minus C, so a positive effect means G is slower.
type Qualification struct {
	Window string `json:"window"`
	// Identity and Order say whether each arm was seen to run one executable
	// and each block to run in manifest order: "verified", or "unverified"
	// and why. A campaign that contradicts either is refused, not scored.
	// Both cover every capture, including those of dropped blocks.
	Identity string          `json:"identity"`
	Order    string          `json:"order"`
	HostLoad qualifyHostLoad `json:"host_load"`
	Captures []qualifyEntry  `json:"captures"`

	// Blocks are the blocks scored. DroppedBlocks had an excluded arm, and
	// MinBlocks is the fewest Blocks the verdict may be scored from.
	Blocks        []qualifyBlock `json:"blocks"`
	DroppedBlocks []string       `json:"dropped_blocks,omitempty"`
	MinBlocks     int            `json:"min_blocks"`

	// Breakdown is the pairs of Blocks by op and stage, when -ops or -stages
	// was given and the verdict was scored. It is description only.
	Breakdown *qualifyBreakdown `json:"breakdown,omitempty"`

	// Steady is gated; Whole, over the whole device timeline of each
	// capture, is description only.
	Steady qualifyStats `json:"steady"`
	Whole  qualifyStats `json:"whole"`

	NoiseFloorMS     float64 `json:"noise_floor_ms"`
	NoiseMultiplier  float64 `json:"noise_multiplier"`
	NoiseThresholdMS float64 `json:"noise_threshold_ms"`
	NoiseGate        string  `json:"noise_gate"`
	SignGate         string  `json:"sign_gate"`

	// Verdict is PASS, FAIL or REFUSED. A REFUSED verdict computes no
	// statistics or gates, and Reason says why.
	Verdict string `json:"verdict"`
	Reason  string `json:"reason,omitempty"`

	// Cudaprof is the build that computed the verdict.
	Cudaprof buildStamp `json:"cudaprof"`
}

// scored returns the captures of the blocks the verdict scores, in run
// order: every block but those dropped for host load.
func (q *Qualification) scored() []qualifyEntry {
	var entries []qualifyEntry
	for _, e := range q.Captures {
		if !slices.Contains(q.DroppedBlocks, e.Block) {
			entries = append(entries, e)
		}
	}
	return entries
}

type qualifyBlock struct {
	Block string `json:"block"`
	// Steady and Whole are the block's pairs and within-arm differences
	// over each window.
	Steady qualifyDiffs `json:"steady"`
	Whole  qualifyDiffs `json:"whole"`
}

type qualifyDiffs struct {
	Pair1MS float64 `json:"g1_minus_c1_ms"`
	Pair2MS float64 `json:"g2_minus_c2_ms"`
	MeanMS  float64 `json:"mean_ms"`
	DiffCMS float64 `json:"c2_minus_c1_ms"`
	DiffGMS float64 `json:"g2_minus_g1_ms"`
}

type qualifyStats struct {
	MedianMS  float64 `json:"median_block_mean_ms"`
	PairMinMS float64 `json:"pair_min_ms"`
	PairMaxMS float64 `json:"pair_max_ms"`
	MADCMS    float64 `json:"mad_c_ms"`
	MADGMS    float64 `json:"mad_g_ms"`
}

// qualify computes the verdict. entries holds whole blocks of four in run
// order, as parseQualifyManifest returns them. A block with an excluded
// entry is dropped, and the verdict is REFUSED when fewer than minBlocks
// blocks remain; minBlocks 0 means every block.
func qualify(entries []qualifyEntry, multiplier, floorMS float64, minBlocks int) *Qualification {
	n := len(entries) / len(qualifyArms)
	if minBlocks == 0 {
		minBlocks = n
	}
	q := &Qualification{Captures: entries, NoiseMultiplier: multiplier, MinBlocks: minBlocks}
	var steady, whole []qualifyDiffs
	for i := 0; i+len(qualifyArms) <= len(entries); i += len(qualifyArms) {
		c1, g1, g2, c2 := entries[i], entries[i+1], entries[i+2], entries[i+3]
		if slices.ContainsFunc(entries[i:i+len(qualifyArms)], func(e qualifyEntry) bool { return e.excluded() }) {
			q.DroppedBlocks = append(q.DroppedBlocks, c1.Block)
			continue
		}
		b := qualifyBlock{
			Block:  c1.Block,
			Steady: blockDiffs(c1.SteadyNanos, g1.SteadyNanos, g2.SteadyNanos, c2.SteadyNanos),
			Whole:  blockDiffs(c1.WholeNanos, g1.WholeNanos, g2.WholeNanos, c2.WholeNanos),
		}
		q.Blocks = append(q.Blocks, b)
		steady = append(steady, b.Steady)
		whole = append(whole, b.Whole)
	}
	if len(q.Blocks) == 0 || len(q.Blocks) < minBlocks {
		q.Verdict = verdictRefused
		q.Reason = fmt.Sprintf("%d of %d blocks remain after excluding arms that ran under host load or could not be checked, and the verdict needs %d", len(q.Blocks), n, max(minBlocks, 1))
		return q
	}
	q.Steady, q.Whole = campaignStats(steady), campaignStats(whole)
	q.NoiseFloorMS = max(floorMS, q.Steady.MADCMS, q.Steady.MADGMS)
	q.NoiseThresholdMS = multiplier * q.NoiseFloorMS
	q.NoiseGate = gate(math.Abs(q.Steady.MedianMS) > q.NoiseThresholdMS)
	q.SignGate = gate(q.Steady.PairMinMS > 0 || q.Steady.PairMaxMS < 0)
	q.Verdict = gate(q.NoiseGate == "PASS" && q.SignGate == "PASS")
	return q
}

func blockDiffs(c1, g1, g2, c2 int64) qualifyDiffs {
	ms := func(ns int64) float64 { return float64(ns) / 1e6 }
	d := qualifyDiffs{
		Pair1MS: ms(g1 - c1),
		Pair2MS: ms(g2 - c2),
		DiffCMS: ms(c2 - c1),
		DiffGMS: ms(g2 - g1),
	}
	d.MeanMS = (d.Pair1MS + d.Pair2MS) / 2
	return d
}

func campaignStats(blocks []qualifyDiffs) qualifyStats {
	var means, pairs, diffC, diffG []float64
	for _, b := range blocks {
		means = append(means, b.MeanMS)
		pairs = append(pairs, b.Pair1MS, b.Pair2MS)
		diffC = append(diffC, b.DiffCMS)
		diffG = append(diffG, b.DiffGMS)
	}
	return qualifyStats{
		MedianMS:  median(means),
		PairMinMS: slices.Min(pairs),
		PairMaxMS: slices.Max(pairs),
		MADCMS:    mad(diffC),
		MADGMS:    mad(diffG),
	}
}

func gate(pass bool) string {
	if pass {
		return verdictPass
	}
	return verdictFail
}

// median returns the median of xs, the mean of the middle two when len(xs)
// is even. xs is not modified.
func median(xs []float64) float64 {
	s := slices.Clone(xs)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// mad returns the median absolute deviation of xs about its median.
func mad(xs []float64) float64 {
	m := median(xs)
	dev := make([]float64, len(xs))
	for i, x := range xs {
		dev[i] = math.Abs(x - m)
	}
	return median(dev)
}

func (q *Qualification) write(w io.Writer) {
	fmt.Fprintf(w, "cudaprof qualify: %d blocks, window: %s\n", len(q.Blocks), q.Window)
	fmt.Fprintf(w, "executable per arm: %s; run order within blocks: %s\n\n", q.Identity, q.Order)
	fmt.Fprintf(w, "%-6s %-4s %12s %12s  %s\n", "block", "arm", "steady ms", "whole ms", "capture")
	for _, e := range q.Captures {
		fmt.Fprintf(w, "%-6s %-4s %12.3f %12.3f  %s\n", e.Block, e.Arm, float64(e.SteadyNanos)/1e6, float64(e.WholeNanos)/1e6, e.Path)
	}
	fmt.Fprintln(w)
	q.writeHostLoad(w)
	if len(q.DroppedBlocks) > 0 {
		fmt.Fprintf(w, "dropped blocks: %s\n", strings.Join(q.DroppedBlocks, ", "))
	}
	if q.Verdict == verdictRefused {
		fmt.Fprintf(w, "\nverdict                %s: %s\n", q.Verdict, q.Reason)
		return
	}
	fmt.Fprintf(w, "\nsteady-state span, ms (pairs are G - C)\n")
	fmt.Fprintf(w, "%-6s %10s %10s %10s %10s %10s\n", "block", "G1-C1", "G2-C2", "mean", "C2-C1", "G2-G1")
	for _, b := range q.Blocks {
		d := b.Steady
		fmt.Fprintf(w, "%-6s %10.3f %10.3f %10.3f %10.3f %10.3f\n", b.Block, d.Pair1MS, d.Pair2MS, d.MeanMS, d.DiffCMS, d.DiffGMS)
	}
	s := q.Steady
	fmt.Fprintf(w, "\nmedian of block means  %.3f ms\n", s.MedianMS)
	fmt.Fprintf(w, "pairs                  %.3f .. %.3f ms\n", s.PairMinMS, s.PairMaxMS)
	fmt.Fprintf(w, "MAD within arm         C %.3f ms, G %.3f ms\n", s.MADCMS, s.MADGMS)
	fmt.Fprintf(w, "noise floor            %.3f ms\n", q.NoiseFloorMS)
	fmt.Fprintf(w, "noise gate             %s: |%.3f| ms > %g x %.3f = %.3f ms\n", q.NoiseGate, s.MedianMS, q.NoiseMultiplier, q.NoiseFloorMS, q.NoiseThresholdMS)
	fmt.Fprintf(w, "sign gate              %s: pairs %.3f .. %.3f ms\n", q.SignGate, s.PairMinMS, s.PairMaxMS)
	fmt.Fprintf(w, "whole span (not gated) median %.3f ms, pairs %.3f .. %.3f ms\n", q.Whole.MedianMS, q.Whole.PairMinMS, q.Whole.PairMaxMS)
	fmt.Fprintf(w, "verdict                %s\n", q.Verdict)
	if q.Breakdown != nil {
		q.Breakdown.write(w)
	}
}
