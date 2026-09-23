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

// qualifyArms is the run order within a block.
var qualifyArms = []string{"C1", "G1", "G2", "C2"}

type qualifyOptions struct {
	manifest        string
	gaps            gapOptions
	noiseMultiplier float64
	noiseFloorMS    float64
	json            bool
}

const qualifyUsage = "usage: cudaprof qualify (-warmup-until=RE | -warmup-ms=N) [-noise-multiplier=3] [-noise-floor-ms=1] [-json] manifest.txt"

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
	for i := range entries {
		e := &entries[i]
		e.SteadyNanos, e.WholeNanos, err = qualifySpan(e.Path, opts.gaps)
		if err != nil {
			return fmt.Errorf("qualify: block %s arm %s: %w", e.Block, e.Arm, err)
		}
	}
	q := qualify(entries, opts.noiseMultiplier, opts.noiseFloorMS)
	q.Window = describeWarmup(opts.gaps) + " excluded"
	if opts.json {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(q); err != nil {
			return err
		}
	} else {
		q.write(stdout)
	}
	if q.Verdict != "PASS" {
		return &exitCodeError{Code: 2, Err: fmt.Errorf("qualify: verdict %s: noise gate %s, sign gate %s", q.Verdict, q.NoiseGate, q.SignGate)}
	}
	return nil
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
	fs.Int64Var(&opts.gaps.maxRecords, "max-records", opts.gaps.maxRecords, "maximum journal records to decode per capture")
	fs.BoolVar(&opts.json, "json", false, "write the verdict as JSON")
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

Each capture is measured by its steady-state device timeline span, the window
"cudaprof gaps" reports for the same -warmup option; the whole span is
reported beside it as description only. A capture whose warmup never ends --
no kernel matches -warmup-until, or the run is shorter than -warmup-ms -- is
refused. Per block the pairs are G1-C1 and G2-C2, and:

  effect       median of the block means (a positive effect: G is slower)
  noise floor  max(-noise-floor-ms, MAD(C2-C1), MAD(G2-G1)) over the blocks
  noise gate   passes when |effect| > -noise-multiplier x noise floor
  sign gate    passes when every pair has the same nonzero sign

MAD is the median absolute deviation about the median. The exit status is 0
when both gates pass, 2 when either fails, and 1 when the verdict could not be
computed.

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

// qualifySpan returns the steady-state and whole device timeline spans of
// the capture at path, measured as gaps measures them.
func qualifySpan(path string, opts gapOptions) (steady, whole int64, err error) {
	report, err := gapReportForPath(path, opts)
	if err != nil {
		return 0, 0, err
	}
	if len(report.Devices) != 1 {
		return 0, 0, fmt.Errorf("%s: capture ran on %d devices, want 1", path, len(report.Devices))
	}
	d := report.Devices[0]
	switch {
	case d.SteadyActivities == 0:
		return 0, 0, fmt.Errorf("%s: no steady-state window: the warmup (%s) never ended", path, report.warmup())
	case d.SteadyState == nil:
		// The warmup excluded nothing: the window is the whole timeline.
		return d.Overall.SpanNanos, d.Overall.SpanNanos, nil
	}
	return d.SteadyState.SpanNanos, d.Overall.SpanNanos, nil
}

// Qualification is the verdict of one campaign. Every duration is in
// milliseconds; a pair is G minus C, so a positive effect means G is slower.
type Qualification struct {
	Window   string         `json:"window"`
	Captures []qualifyEntry `json:"captures"`
	Blocks   []qualifyBlock `json:"blocks"`

	// Steady is gated; Whole, over the whole device timeline of each
	// capture, is description only.
	Steady qualifyStats `json:"steady"`
	Whole  qualifyStats `json:"whole"`

	NoiseFloorMS     float64 `json:"noise_floor_ms"`
	NoiseMultiplier  float64 `json:"noise_multiplier"`
	NoiseThresholdMS float64 `json:"noise_threshold_ms"`
	NoiseGate        string  `json:"noise_gate"`
	SignGate         string  `json:"sign_gate"`
	Verdict          string  `json:"verdict"`
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
// order, as parseQualifyManifest returns them.
func qualify(entries []qualifyEntry, multiplier, floorMS float64) *Qualification {
	q := &Qualification{Captures: entries, NoiseMultiplier: multiplier}
	var steady, whole []qualifyDiffs
	for i := 0; i+len(qualifyArms) <= len(entries); i += len(qualifyArms) {
		c1, g1, g2, c2 := entries[i], entries[i+1], entries[i+2], entries[i+3]
		b := qualifyBlock{
			Block:  c1.Block,
			Steady: blockDiffs(c1.SteadyNanos, g1.SteadyNanos, g2.SteadyNanos, c2.SteadyNanos),
			Whole:  blockDiffs(c1.WholeNanos, g1.WholeNanos, g2.WholeNanos, c2.WholeNanos),
		}
		q.Blocks = append(q.Blocks, b)
		steady = append(steady, b.Steady)
		whole = append(whole, b.Whole)
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
		return "PASS"
	}
	return "FAIL"
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
	fmt.Fprintf(w, "cudaprof qualify: %d blocks, window: %s\n\n", len(q.Blocks), q.Window)
	fmt.Fprintf(w, "%-6s %-4s %12s %12s  %s\n", "block", "arm", "steady ms", "whole ms", "capture")
	for _, e := range q.Captures {
		fmt.Fprintf(w, "%-6s %-4s %12.3f %12.3f  %s\n", e.Block, e.Arm, float64(e.SteadyNanos)/1e6, float64(e.WholeNanos)/1e6, e.Path)
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
}
