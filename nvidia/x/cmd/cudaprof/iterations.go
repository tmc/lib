package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

const iterationsUsage = "usage: cudaprof iterations -iteration-marker=RE [-nvtx] [-skip-iterations=N] [-expect-markers=N] [-ops ops.txt] [-kernels=N] [-max-records=N] [-journal-bytes=N] [--capture-file-bytes=N] [--capture-total-bytes=N] [-json] capture.pprof"

type iterationsOptions struct {
	marker       string
	nvtx         bool
	skip         int
	expect       int
	opsPath      string
	kernels      int
	json         bool
	maxRecords   int64
	journalBytes int64
	limits       cudacapture.Limits
}

// parseIterationsArgs returns a nil path and a nil error when help was
// requested.
func parseIterationsArgs(rawArgs []string, stdout io.Writer) (iterationsOptions, string, error) {
	var opts iterationsOptions
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return opts, "", fmt.Errorf("iterations: %w", err)
	}
	opts.limits = limits
	fs := flag.NewFlagSet("iterations", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&opts.marker, "iteration-marker", "", "begin an iteration at each kernel matching `RE`, demangled with template arguments and without parameters")
	fs.BoolVar(&opts.nvtx, "nvtx", false, "match -iteration-marker against NVTX range and marker names instead of kernels")
	fs.IntVar(&opts.skip, "skip-iterations", 1, "exclude the first `N` iterations as warm-up")
	fs.IntVar(&opts.expect, "expect-markers", 0, "fail the gate unless the marker fired exactly `N` times")
	fs.StringVar(&opts.opsPath, "ops", "", "file of `regex op` lines grouping kernels into ops, as compare reads")
	fs.IntVar(&opts.kernels, "kernels", 15, "list the `N` kernels with the most exclusive time")
	fs.BoolVar(&opts.json, "json", false, "write the report as JSON")
	fs.Int64Var(&opts.maxRecords, "max-records", controllerMaxJournalRecords, "maximum journal records to decode, as capture --max-records allowed it to write")
	fs.Int64Var(&opts.journalBytes, "journal-bytes", cudacapture.MaxCaptureInputBytes, "maximum decompressed bytes per journal")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, iterationsUsage)
			fs.SetOutput(stdout)
			fs.PrintDefaults()
			return opts, "", nil
		}
		return opts, "", fmt.Errorf("iterations: %w\n%s", err, iterationsUsage)
	}
	switch {
	case opts.marker == "":
		return opts, "", fmt.Errorf("iterations: -iteration-marker is required\n%s", iterationsUsage)
	case opts.skip < 0:
		return opts, "", fmt.Errorf("iterations: invalid -skip-iterations %d", opts.skip)
	case fs.NArg() != 1:
		return opts, "", fmt.Errorf("iterations: want one capture\n%s", iterationsUsage)
	}
	return opts, fs.Arg(0), nil
}

// An iterationsResult is what iterations writes as JSON.
type iterationsResult struct {
	Capture string              `json:"capture"`
	Report  *iteration.Report   `json:"report"`
	Advice  []iteration.Finding `json:"advice"`
}

func runIterationsCommand(args []string, stdout io.Writer) error {
	opts, path, err := parseIterationsArgs(args, stdout)
	if err != nil || path == "" {
		return err
	}
	marker, err := regexp.Compile(opts.marker)
	if err != nil {
		return fmt.Errorf("iterations: -iteration-marker: %w", err)
	}
	rules, err := readOpRules(opts.opsPath)
	if err != nil {
		return fmt.Errorf("iterations: %w", err)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(abs, opts.limits)
	if err != nil {
		return fmt.Errorf("iterations: read capture %s: %w", path, err)
	}
	res, err := measureIterations(cap, marker, rules, opts)
	if err != nil {
		return fmt.Errorf("iterations: %w", err)
	}
	res.Capture = abs
	r := res.Report
	if opts.json {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		err = enc.Encode(res)
	} else {
		err = res.write(stdout, opts.kernels)
	}
	if err != nil {
		return err
	}
	if !r.Gate.Pass {
		return &exitCodeError{Code: 2, Err: fmt.Errorf("iterations: gate failed: %s", r.Gate.Failures[0])}
	}
	return nil
}

// measureIterations measures a complete capture iteration by iteration,
// grouping kernels into ops by rules.
func measureIterations(cap *cudacapture.Capture, marker *regexp.Regexp, rules opRules, opts iterationsOptions) (*iterationsResult, error) {
	_, records, err := decodeCompleteCapture(cap, opts.maxRecords, opts.journalBytes)
	if err != nil {
		return nil, err
	}
	r, err := iteration.Analyze(records, iteration.Options{
		Marker: marker,
		NVTX:   opts.nvtx,
		Skip:   opts.skip,
		Expect: opts.expect,
		Group: func(name string) string {
			op, _ := rules.kernelOp(cudalowering.MatchKey(name))
			return op
		},
	})
	if err != nil {
		return nil, err
	}
	return &iterationsResult{Report: r, Advice: r.Advise()}, nil
}

func (res *iterationsResult) write(w io.Writer, kernels int) error {
	r := res.Report
	kind := "kernel"
	if r.NVTX {
		kind = "NVTX"
	}
	s := r.Steady
	fmt.Fprintf(w, "capture  %s\n", res.Capture)
	fmt.Fprintf(w, "marker   %s (%s): %d markers, %d iterations, skip %d, %d steady\n", r.Marker, kind, r.Markers, len(r.Iterations), r.Skip, s.Iterations)
	if r.Gate.Pass {
		fmt.Fprintf(w, "gate     pass\n")
	} else {
		for _, f := range r.Gate.Failures {
			fmt.Fprintf(w, "gate     FAIL: %s\n", f)
		}
	}
	n := s.Iterations
	if n == 0 {
		return nil
	}
	ms := func(ns int64) string { return fmt.Sprintf("%.3f", float64(ns)/float64(n)/1e6) }
	pct := func(a, b int64) string {
		if b == 0 {
			return "-"
		}
		return fmt.Sprintf("%.1f%%", 100*float64(a)/float64(b))
	}
	per := func(c int) string { return fmt.Sprintf("%.1f", float64(c)/float64(n)) }

	fmt.Fprintf(w, "\nper steady iteration (mean of %d)\n", n)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintf(tw, "  window\t%s ms\t\t\n", ms(s.WindowNanos))
	fmt.Fprintf(tw, "  busy\t%s ms\t%s\t\n", ms(s.BusyNanos), pct(s.BusyNanos, s.WindowNanos))
	fmt.Fprintf(tw, "    kernel\t%s ms\t%s\t\n", ms(s.KernelNanos), pct(s.KernelNanos, s.WindowNanos))
	fmt.Fprintf(tw, "    memcpy\t%s ms\t%s\t\n", ms(s.MemcpyNanos), pct(s.MemcpyNanos, s.WindowNanos))
	fmt.Fprintf(tw, "    memset\t%s ms\t%s\t\n", ms(s.MemsetNanos), pct(s.MemsetNanos, s.WindowNanos))
	fmt.Fprintf(tw, "  idle\t%s ms\t%s\t\n", ms(s.IdleNanos), pct(s.IdleNanos, s.WindowNanos))
	fmt.Fprintf(tw, "  host API calls\t%s ms\t%s calls\t\n", ms(s.CallNanos), per(s.Calls))
	tw.Flush()
	fmt.Fprintf(w, "  launches %s (graph nodes %s), memcpys %s (%d bytes), memsets %s\n",
		per(s.Launches), per(s.GraphLaunches), per(s.Memcpys), s.MemcpyBytes/uint64(n), per(s.Memsets))

	fmt.Fprintf(w, "\niterations\n")
	tw = tabwriter.NewWriter(w, 0, 4, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintf(tw, "  #\t\twindow ms\tbusy ms\tidle ms\tlaunches\tmemcpys\talloc\tfree\tgraph build\tinstantiate\tupdate\tsync\t\n")
	for _, it := range r.Iterations {
		warm := "warm-up"
		if it.Steady {
			warm = ""
		}
		c := it.Classes
		fmt.Fprintf(tw, "  %d\t%s\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t\n",
			it.Index, warm, float64(it.WindowNanos)/1e6, float64(it.BusyNanos)/1e6, float64(it.IdleNanos)/1e6,
			it.Launches, it.Memcpys,
			c[iteration.Alloc].Count, c[iteration.Free].Count, c[iteration.GraphBuild].Count,
			c[iteration.GraphInstantiate].Count, c[iteration.GraphUpdate].Count, c[iteration.Sync].Count)
	}
	tw.Flush()

	fmt.Fprintf(w, "\nops (per steady iteration)\n")
	tw = tabwriter.NewWriter(w, 0, 4, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintf(tw, "\texclusive ms\tof busy\tlaunches\tsummed ms\t\n")
	for _, g := range r.Groups {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t\n", g.Group, ms(g.Exclusive), pct(g.Exclusive, s.BusyNanos), per(g.Launches), ms(g.Summed))
	}
	tw.Flush()

	if kernels > 0 && len(r.Kernels) > 0 {
		fmt.Fprintf(w, "\nkernels (per steady iteration, top %d of %d)\n", min(kernels, len(r.Kernels)), len(r.Kernels))
		tw = tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprintf(tw, "  exclusive ms\tlaunches\top\tkernel\n")
		for _, k := range r.Kernels[:min(kernels, len(r.Kernels))] {
			fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\n", ms(k.Exclusive), per(k.Launches), k.Group.Group, k.Kernel)
		}
		tw.Flush()
	}

	if len(r.Calls) > 0 {
		fmt.Fprintf(w, "\nhost API (per steady iteration)\n")
		tw = tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprintf(tw, "  ms\tcalls\tclass\tcall\n")
		for _, c := range r.Calls {
			class := string(c.Class)
			if class == "" {
				class = "-"
			}
			fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\n", ms(c.Nanos), per(c.Count), class, c.Name)
		}
		tw.Flush()
	}

	if len(res.Advice) > 0 {
		fmt.Fprintf(w, "\nadvice\n")
		for _, f := range res.Advice {
			fmt.Fprintf(w, "  [%s] %s\n    %s\n", f.Rule, f.Finding, f.Action)
		}
	}
	return nil
}

// decodeCompleteCapture decodes a capture's journals after checking that
// its run is complete and lost no records: a measurement over a capture
// that did would be missing work it cannot see.
func decodeCompleteCapture(cap *cudacapture.Capture, maxRecords, journalBytes int64) (captureResult, timeline.Records, error) {
	var none timeline.Records
	data, ok := cap.Files["capture-status.json"]
	if !ok {
		return captureResult{}, none, errors.New("capture has no capture-status.json, so its completeness is unknown")
	}
	var envelope struct {
		Status captureResult `json:"status"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return captureResult{}, none, fmt.Errorf("read capture-status.json: %w", err)
	}
	status := envelope.Status
	if err := targetFailure(&status); err != nil {
		return status, none, err
	}
	completeness := status.RunCompleteness
	if completeness == "" {
		completeness = runCompletenessOf(status.TransportReady, status.CollectorFinalized, status.UnqualifiedGenerations, status.DroppedRecords, status.ActivityRecords)
	}
	if completeness != runComplete {
		return status, none, fmt.Errorf("run_completeness is %q, not %q: records the window would hold may be missing", completeness, runComplete)
	}
	if status.DroppedRecords != 0 {
		return status, none, fmt.Errorf("capture dropped %d record(s)", status.DroppedRecords)
	}
	if l := status.RecordLoss; l != nil && l.Offered != l.Retained {
		return status, none, fmt.Errorf("capture lost records: %d offered, %d retained", l.Offered, l.Retained)
	}
	journals, err := requireJournals(cap)
	if err != nil {
		return status, none, err
	}
	if maxRecords == 0 {
		maxRecords = controllerMaxJournalRecords
	}
	if journalBytes == 0 {
		journalBytes = cudacapture.MaxCaptureInputBytes
	}
	records, err := timeline.Decode(journals, maxRecords, journalBytes)
	if errors.Is(err, wire.ErrRecordLimit) {
		return status, none, fmt.Errorf("%w: raise -max-records past %d", err, maxRecords)
	}
	if err != nil {
		return status, none, err
	}
	return status, records, nil
}
