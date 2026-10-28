package main

import (
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
)

func runAdviseCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	opts, err := parseAdviseArgs(args)
	if err != nil {
		return err
	}
	if len(opts.ranks) > 0 {
		return adviseRanks(opts, limits)
	}

	path, err := filepath.Abs(opts.capture)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}

	report, err := advise.Analyze(cap)
	if err != nil {
		return fmt.Errorf("analyze capture: %w", err)
	}
	if opts.iterationMarker != "" {
		if err := adviseIterations(report, cap, opts); err != nil {
			return fmt.Errorf("advise: %w", err)
		}
	}
	if opts.base != "" {
		if err := adviseAgainstBase(report, cap, path, opts, limits); err != nil {
			return fmt.Errorf("advise: %w", err)
		}
	}
	if opts.target != "" {
		if err := report.ProjectTo(cap, opts.target); err != nil {
			return fmt.Errorf("advise: %w", err)
		}
	}
	return printAdvice(report, opts)
}

// adviseRanks analyzes the captures of several ranks of one job together.
func adviseRanks(opts adviseOptions, limits cudacapture.Limits) error {
	var ranks []advise.Rank
	for _, name := range append([]string{opts.capture}, opts.ranks...) {
		path, err := filepath.Abs(name)
		if err != nil {
			return err
		}
		cap, err := readCapturePathLimits(path, limits)
		if err != nil {
			return fmt.Errorf("read capture %s: %w", name, err)
		}
		ranks = append(ranks, advise.Rank{Name: strings.TrimSuffix(filepath.Base(name), filepath.Ext(name)), Capture: cap})
	}
	report, err := advise.AnalyzeRanks(ranks, opts.align)
	if err != nil {
		return fmt.Errorf("advise: %w", err)
	}
	return printAdvice(report, opts)
}

func printAdvice(report *advise.Report, opts adviseOptions) error {
	if opts.format == "json" {
		// -limit governs how much the markdown report prints, not how much
		// the tool knows. The summary counts every finding, so one format has
		// to be able to reach every finding; JSON is it. Truncating here left
		// the counted remainder unreachable by any flag combination.
		out, err := report.JSON()
		if err != nil {
			return err
		}
		fmt.Println(out)
		return nil
	}

	fmt.Print(report.MarkdownWithLimit(opts.limit))
	return nil
}

const adviseUsage = "usage: cudaprof advise [-e] [-limit N] [--format=json|markdown] [-target sm_103a|gb300] [-base base.pprof [-ops ops.txt] [-phase i]] [-iteration-marker RE [-nvtx] [-skip-iterations N] [-expect-markers N]] capture.pprof\n       cudaprof advise [-e] [-limit N] [--format=json|markdown] [-align collective|clock] rank0.pprof rank1.pprof..."

type adviseOptions struct {
	format  string
	limit   int
	capture string

	// ranks are the captures after the first, which select a multi-rank
	// analysis aligned by align.
	ranks []string
	align advise.Alignment

	// target names a device to project the step onto.
	target string

	// base, ops and phase select a comparison against a base capture, as
	// compare's arguments of the same names do.
	base  string
	ops   string
	phase int

	// iterationMarker, when set, adds the findings of cudaprof iterations
	// run with the same flags.
	iterationMarker string
	nvtx            bool
	skipIterations  int
	expectMarkers   int
}

// parseAdviseArgs parses advise's flags, each spelled with one dash or two
// and taking its value as -flag=v or -flag v.
func parseAdviseArgs(args []string) (adviseOptions, error) {
	opts := adviseOptions{limit: advise.DefaultLimit, phase: -1, skipIterations: 1, align: advise.AlignCollective}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "-") {
			if opts.capture != "" {
				opts.ranks = append(opts.ranks, arg)
				continue
			}
			opts.capture = arg
			continue
		}
		name, val, hasVal := strings.Cut(strings.TrimPrefix(strings.TrimPrefix(arg, "-"), "-"), "=")
		value := func() (string, error) {
			if hasVal {
				return val, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("flag -%s requires an argument", name)
			}
			i++
			return args[i], nil
		}
		intValue := func() (int, error) {
			v, err := value()
			if err != nil {
				return 0, err
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				return 0, fmt.Errorf("invalid -%s value: %s", name, v)
			}
			return n, nil
		}
		var err error
		switch {
		case !hasVal && (name == "json" || name == "markdown"):
			opts.format = name
		case hasVal && name == "format" && (val == "json" || val == "markdown"):
			opts.format = val
		case !hasVal && (name == "e" || name == "all"):
			opts.limit = 0
		case name == "limit" || name == "n":
			opts.limit, err = intValue()
		case name == "base":
			opts.base, err = value()
		case name == "ops":
			opts.ops, err = value()
		case name == "phase":
			opts.phase, err = intValue()
		case name == "iteration-marker":
			opts.iterationMarker, err = value()
		case !hasVal && name == "nvtx":
			opts.nvtx = true
		case name == "skip-iterations":
			opts.skipIterations, err = intValue()
		case name == "expect-markers":
			opts.expectMarkers, err = intValue()
		case name == "target":
			opts.target, err = value()
		case name == "align":
			var v string
			v, err = value()
			opts.align = advise.Alignment(v)
		default:
			return opts, fmt.Errorf("unknown flag: %s", arg)
		}
		if err != nil {
			return opts, err
		}
	}
	if opts.capture == "" {
		return opts, errors.New(adviseUsage)
	}
	if opts.iterationMarker == "" && (opts.nvtx || opts.skipIterations != 1 || opts.expectMarkers != 0) {
		return opts, errors.New("-nvtx, -skip-iterations and -expect-markers need -iteration-marker")
	}
	if opts.skipIterations < 0 {
		return opts, fmt.Errorf("invalid -skip-iterations %d", opts.skipIterations)
	}
	if len(opts.ranks) > 0 && (opts.base != "" || opts.iterationMarker != "" || opts.target != "") {
		return opts, errors.New("-base, -iteration-marker and -target take one capture, not several ranks")
	}
	if opts.align != advise.AlignCollective && opts.align != advise.AlignClock {
		return opts, fmt.Errorf("invalid -align %q: want collective or clock", opts.align)
	}
	if opts.base == "" && (opts.ops != "" || opts.phase >= 0) {
		return opts, errors.New("-ops and -phase select a comparison and need -base")
	}
	return opts, nil
}

// adviseIterations adds the per-iteration findings of a complete capture
// to report.
func adviseIterations(report *advise.Report, cap *cudacapture.Capture, opts adviseOptions) error {
	marker, err := regexp.Compile(opts.iterationMarker)
	if err != nil {
		return fmt.Errorf("-iteration-marker: %w", err)
	}
	_, records, err := decodeCompleteCapture(cap, 0, 0)
	if err != nil {
		return fmt.Errorf("iterations: %w", err)
	}
	r, err := iteration.Analyze(records, iteration.Options{Marker: marker, NVTX: opts.nvtx, Skip: opts.skipIterations, Expect: opts.expectMarkers})
	if err != nil {
		return fmt.Errorf("iterations: %w", err)
	}
	report.AddIterations(r)
	return nil
}
