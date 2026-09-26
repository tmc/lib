package main

import (
	"cmp"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/perfscript"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

const iterationsUsage = "usage: cudaprof iterations -iteration-marker=RE [-nvtx] [-skip-iterations=N] [-expect-markers=N] [-ops ops.txt] [-kernels=N] [-gaps=DURATION] [-perf=host.txt] [-max-records=N] [-journal-bytes=N] [--capture-file-bytes=N] [--capture-total-bytes=N] [-json] capture.pprof"

type iterationsOptions struct {
	marker       string
	nvtx         bool
	skip         int
	expect       int
	opsPath      string
	kernels      int
	gaps         time.Duration
	perfPath     string
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
	fs.StringVar(&opts.opsPath, "ops", defaultOps, "file of `regex op` lines grouping kernels into ops, as compare reads; \"default\" is the built-in groups, and empty makes each kernel its own op")
	fs.IntVar(&opts.kernels, "kernels", 15, "list the `N` kernels with the most exclusive time")
	fs.DurationVar(&opts.gaps, "gaps", 0, "list device idle gaps of at least `DURATION` in steady iterations (default 1ms with -perf)")
	fs.StringVar(&opts.perfPath, "perf", "", "label idle gaps with the host stacks in `file`, the output of perf script --ns -F comm,tid,time,ip,sym for a perf record -k CLOCK_MONOTONIC of the capture")
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
	case opts.gaps < 0:
		return opts, "", fmt.Errorf("iterations: invalid -gaps %v", opts.gaps)
	case fs.NArg() != 1:
		return opts, "", fmt.Errorf("iterations: want one capture\n%s", iterationsUsage)
	}
	if opts.perfPath != "" && opts.gaps == 0 {
		opts.gaps = time.Millisecond
	}
	return opts, fs.Arg(0), nil
}

// An iterationsResult is what iterations writes as JSON.
type iterationsResult struct {
	Capture string              `json:"capture"`
	Report  *iteration.Report   `json:"report"`
	Advice  []iteration.Finding `json:"advice"`
	Notes   []string            `json:"notes,omitempty"`
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
	var host []iteration.HostSample
	if opts.perfPath != "" {
		if host, err = readHostSamples(opts.perfPath); err != nil {
			return fmt.Errorf("iterations: %w", err)
		}
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(abs, opts.limits)
	if err != nil {
		return fmt.Errorf("iterations: read capture %s: %w", path, err)
	}
	res, err := measureIterations(cap, marker, rules, opts, host)
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
// grouping kernels into ops by rules and labeling idle gaps with host.
func measureIterations(cap *cudacapture.Capture, marker *regexp.Regexp, rules opRules, opts iterationsOptions, host []iteration.HostSample) (*iterationsResult, error) {
	status, records, err := decodeCompleteCapture(cap, opts.maxRecords, opts.journalBytes)
	if err != nil {
		return nil, err
	}
	r, err := iteration.Analyze(records, iteration.Options{
		Marker: marker,
		NVTX:   opts.nvtx,
		Skip:   opts.skip,
		Expect: opts.expect,
		MinGap: opts.gaps.Nanoseconds(),
		Host:   host,
		Group: func(name string) string {
			op, _ := rules.kernelOp(cudalowering.MatchKey(name))
			return op
		},
	})
	if err != nil {
		return nil, err
	}
	res := &iterationsResult{Report: r, Advice: r.Advise()}
	if note := memsetNote(status, r); note != "" {
		res.Notes = append(res.Notes, note)
	}
	return res, nil
}

// memsetNote says when the capture did not collect memset activity, whose
// device time then counts as idle.
func memsetNote(status captureResult, r *iteration.Report) string {
	var c struct {
		Memsets *bool `json:"memsets"`
	}
	if json.Unmarshal(status.Capability, &c) != nil || c.Memsets == nil || *c.Memsets {
		return ""
	}
	note := "memset activity was not collected (capture --memsets=false), so the device time of every memset counts as idle"
	if n := r.Steady.Iterations; n > 0 {
		calls := r.Steady.Classes[iteration.Memset].Count
		note += fmt.Sprintf(": each steady iteration made %.1f memset calls, and a CUDA graph also replays the memsets captured into it without a call", float64(calls)/float64(n))
	}
	return note
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
	fmt.Fprintf(w, "  launches %s (graph nodes %s), memcpys %s (%d bytes), memsets %s (memset calls %s)\n",
		per(s.Launches), per(s.GraphLaunches), per(s.Memcpys), s.MemcpyBytes/uint64(n), per(s.Memsets), per(s.Classes[iteration.Memset].Count))
	for _, note := range res.Notes {
		fmt.Fprintf(w, "  note: %s\n", note)
	}

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

	if r.Gaps != nil || r.GapStacks != nil {
		res.writeGaps(w)
	}

	if len(res.Advice) > 0 {
		fmt.Fprintf(w, "\nadvice\n")
		for _, f := range res.Advice {
			fmt.Fprintf(w, "  [%s] %s\n    %s\n", f.Rule, f.Finding, f.Action)
		}
	}
	return nil
}

// readHostSamples reads the perf script output at path.
func readHostSamples(path string) ([]iteration.HostSample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	samples, err := perfscript.Parse(f)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	host := make([]iteration.HostSample, len(samples))
	for i, s := range samples {
		host[i] = iteration.HostSample{Time: s.Time, Thread: s.Thread, Stack: s.Stack}
	}
	return host, nil
}

// writeGaps writes the idle gaps and the host stacks sampled in them.
func (res *iterationsResult) writeGaps(w io.Writer) {
	r := res.Report
	n := float64(r.Steady.Iterations)
	var nanos int64
	var calls, samples int
	for _, g := range r.Gaps {
		nanos += g.End - g.Start
		calls += g.Calls
		samples += g.Samples
	}
	fmt.Fprintf(w, "\nidle gaps (per steady iteration): %.1f gaps, %.3f ms of %.3f ms idle, %.1f host API calls begun in them",
		float64(len(r.Gaps))/n, float64(nanos)/n/1e6, float64(r.Steady.IdleNanos)/n/1e6, float64(calls)/n)
	if r.GapStacks == nil {
		fmt.Fprintf(w, "\n")
	} else {
		fmt.Fprintf(w, ", %.1f host samples taken in them\n", float64(samples)/n)
	}
	if len(r.GapStacks) > 0 {
		fmt.Fprintf(w, "\nhost stacks sampled in idle gaps (top %d of %d)\n", min(10, len(r.GapStacks)), len(r.GapStacks))
		tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprintf(tw, "  samples\tgaps\tstack, leaf first\n")
		for _, st := range r.GapStacks[:min(10, len(r.GapStacks))] {
			fmt.Fprintf(tw, "  %.1f%%\t%d\t%s\n", 100*float64(st.Samples)/float64(samples), st.Gaps, stackLine(st.Stack))
		}
		tw.Flush()
	}
	if len(r.GapCalls) > 0 {
		fmt.Fprintf(w, "\nhost API calls begun in idle gaps (per steady iteration, top %d of %d)\n", min(10, len(r.GapCalls)), len(r.GapCalls))
		tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprintf(tw, "  ms\tcalls\tclass\tcall\n")
		for _, c := range r.GapCalls[:min(10, len(r.GapCalls))] {
			class := string(c.Class)
			if class == "" {
				class = "-"
			}
			fmt.Fprintf(tw, "  %.3f\t%.1f\t%s\t%s\n", float64(c.Nanos)/n/1e6, float64(c.Count)/n, class, c.Name)
		}
		tw.Flush()
	}
	gaps := slices.Clone(r.Gaps)
	slices.SortStableFunc(gaps, func(a, b iteration.Gap) int { return cmp.Compare(b.End-b.Start, a.End-a.Start) })
	gaps = gaps[:min(10, len(gaps))]
	fmt.Fprintf(w, "\nlongest idle gaps (top %d of %d)\n", len(gaps), len(r.Gaps))
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintf(tw, "  iteration\tat ms\tidle ms\tcalls\tsamples\ttop stack\n")
	for _, g := range gaps {
		at := g.Start - r.Iterations[g.Iteration].Start
		fmt.Fprintf(tw, "  %d\t%.3f\t%.3f\t%d\t%d\t%s\n", g.Iteration, float64(at)/1e6, float64(g.End-g.Start)/1e6, g.Calls, g.Samples, stackLine(g.Stack))
	}
	tw.Flush()
}

// stackLine formats a stack's innermost frames on one line.
func stackLine(stack []string) string {
	const frames = 6
	if len(stack) == 0 {
		return "-"
	}
	s := strings.Join(stack[:min(frames, len(stack))], " < ")
	if len(stack) > frames {
		s += " < ..."
	}
	return s
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
