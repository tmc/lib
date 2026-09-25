package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/gaps"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// The gaps command runs package gaps over one capture, or over two with
// --diff_base. This file parses its flags, adds to a refusal the remedy those
// flags can offer, and renders the report.

// gapOptions are the parsed arguments of the gaps command.
type gapOptions struct {
	capturePath  string
	diffBase     string
	warmupMS     float64
	warmupUntil  string
	maxRecords   int64
	journalBytes int64
	json         bool
	limits       cudacapture.Limits
}

// runGapsCommand implements "cudaprof gaps".
func runGapsCommand(rawArgs []string) error {
	if len(rawArgs) == 0 {
		printGapsHelp()
		return nil
	}
	opts, err := parseGapsArgs(rawArgs)
	if err != nil {
		if errors.Is(err, errGapsHelp) {
			printGapsHelp()
			return nil
		}
		return err
	}
	if opts.diffBase != "" {
		return runGapsDiff(opts)
	}
	path, err := filepath.Abs(opts.capturePath)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, opts.limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	report, err := analyzeGaps(cap, opts)
	if err != nil {
		return err
	}
	report.Capture = path
	if opts.json {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(report)
	}
	fmt.Fprint(os.Stdout, report.text())
	return nil
}

// runGapsDiff implements "cudaprof gaps --diff_base=base.pprof target.pprof".
// Both captures are analyzed under the same options, so the steady-state
// windows are cut at the same warmup and the deltas below subtract like
// windows.
func runGapsDiff(opts gapOptions) error {
	base, err := gapReportForPath(opts.diffBase, opts)
	if err != nil {
		return err
	}
	target, err := gapReportForPath(opts.capturePath, opts)
	if err != nil {
		return err
	}
	diff, err := diffGapReports(base, target)
	if err != nil {
		return err
	}
	if opts.json {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(diff)
	}
	fmt.Fprint(os.Stdout, diff.text())
	return nil
}

var errGapsHelp = errors.New("gaps: help requested")

func parseGapsArgs(rawArgs []string) (gapOptions, error) {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return gapOptions{}, err
	}
	opts := gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes, limits: limits}
	for _, arg := range args {
		switch {
		case arg == "-h" || arg == "-help" || arg == "--help":
			return gapOptions{}, errGapsHelp
		case arg == "-json" || arg == "--json" || arg == "--format=json":
			opts.json = true
		case strings.HasPrefix(arg, "--warmup-ms=") || strings.HasPrefix(arg, "-warmup-ms="):
			v, err := strconv.ParseFloat(arg[strings.IndexByte(arg, '=')+1:], 64)
			if err != nil || v < 0 {
				return gapOptions{}, fmt.Errorf("invalid warmup: %s", arg)
			}
			opts.warmupMS = v
		case strings.HasPrefix(arg, "--warmup-until=") || strings.HasPrefix(arg, "-warmup-until="):
			v := arg[strings.IndexByte(arg, '=')+1:]
			if _, err := regexp.Compile(v); err != nil || v == "" {
				return gapOptions{}, fmt.Errorf("invalid warmup kernel pattern: %s", arg)
			}
			opts.warmupUntil = v
		case strings.HasPrefix(arg, "--diff_base=") || strings.HasPrefix(arg, "-diff_base="):
			opts.diffBase = arg[strings.IndexByte(arg, '=')+1:]
			if opts.diffBase == "" {
				return gapOptions{}, fmt.Errorf("invalid diff base: %s", arg)
			}
		case strings.HasPrefix(arg, "--max-records=") || strings.HasPrefix(arg, "-max-records="):
			v, err := strconv.ParseInt(arg[strings.IndexByte(arg, '=')+1:], 10, 64)
			if err != nil || v <= 0 {
				return gapOptions{}, fmt.Errorf("invalid max records: %s", arg)
			}
			opts.maxRecords = v
		case strings.HasPrefix(arg, "--journal-bytes=") || strings.HasPrefix(arg, "-journal-bytes="):
			v, err := strconv.ParseInt(arg[strings.IndexByte(arg, '=')+1:], 10, 64)
			if err != nil || v <= 0 {
				return gapOptions{}, fmt.Errorf("invalid journal bytes: %s", arg)
			}
			opts.journalBytes = v
		case strings.HasPrefix(arg, "-"):
			return gapOptions{}, fmt.Errorf("unknown flag: %s", arg)
		case opts.capturePath == "":
			opts.capturePath = arg
		default:
			return gapOptions{}, errors.New("multiple capture files specified")
		}
	}
	if opts.capturePath == "" {
		return gapOptions{}, errors.New("usage: cudaprof gaps [--warmup-ms=N | --warmup-until=REGEX] [--max-records=N] [--journal-bytes=N] [--diff_base=base.pprof] [--json] capture.pprof")
	}
	return opts, nil
}

func printGapsHelp() {
	fmt.Fprint(os.Stdout, `Usage: cudaprof gaps [options] capture.pprof
       cudaprof gaps --diff_base=base.pprof [options] target.pprof

Reconstructs the device timeline from the capture journals attached to a
capture, merges kernel and memcpy intervals across streams per device, and
reports GPU duty cycle, the distribution of inter-activity idle gaps, and the
host concurrency state at each gap.

The analysis needs the per-record journal. pprof samples are aggregates and
carry no timestamps, so a capture whose journal was omitted by the artifact
budget, truncated, or lossy is refused instead of analyzed.

With --diff_base the two captures are compared instead: span, active, idle,
duty cycle, gap count and every gap percentile are reported as a delta. Both
captures are analyzed under the same options, so the two steady-state windows
are cut at the same warmup. This is the form to use for a before/after test.
A warmup of fixed length cuts the two runs at different points in their work
when their startups differ in length; --warmup-until cuts each at the first
kernel whose name matches, so both steady-state windows begin at the same
point in the program.
Summed per-kernel device time cannot see this axis: a change that moves work
out of inter-kernel idle and into the kernels raises device-active time while
the run gets faster, and a device-time-only diff scores that as a regression.

A --diff_base run also prints a per-kernel census in three classes. A kernel
whose launch count moved took a different path: that is the change. A kernel
that kept its count but whose launches sit in a different part of the run is
not untouched -- the census covers the whole capture, and a rate measured over
a window will disagree with these totals, because the window no longer holds
the same work in both arms. Only a kernel that kept both its count and its
place was left alone, and device time that moved there is what the change cost
elsewhere on the device. Ranked together by delta all three are
indistinguishable, and the untouched kernel is the one that sorts first
whenever the change cost more than it saved.

Options:
  --diff_base=FILE  compare against this capture and report deltas
  --warmup-ms=N     warmup window excluded from the steady-state section (default 200)
  --warmup-until=RE exclude instead everything before the first kernel matching RE
  --max-records=N   maximum journal records to decode (default 1048576)
  --journal-bytes=N maximum decompressed bytes per journal (default 536870912)
  --json            write the report as JSON
`)
}

// ---------------------------------------------------------------------------
// journal selection and decoding

// gapStatus is the part of capture-status.json that says what a refusal of
// the capture's journals can offer as a remedy.
type gapStatus struct {
	OmittedFiles []captureOmittedFile `json:"omitted_files"`
	// RecordLoss is the producer's own breakdown of what it never wrote. It
	// names the mechanism, which is what a refusal has to report: the one
	// remedy a reader can reach for, --max-records, addresses none of them.
	RecordLoss *captureLoss `json:"record_loss"`
	// Capability is the agent's own configuration block. It decides which
	// remedy the truncation message may offer, because the largest one --
	// dropping API attribution -- is only available to a run that collected it.
	Capability json.RawMessage `json:"capability"`
}

func readGapStatus(cap *cudacapture.Capture) (gapStatus, bool) {
	data, ok := cap.Files["capture-status.json"]
	if !ok {
		return gapStatus{}, false
	}
	var envelope struct {
		Status gapStatus `json:"status"`
	}
	if json.Unmarshal(data, &envelope) != nil {
		return gapStatus{}, false
	}
	return envelope.Status, true
}

// requireJournals returns the journal members of a capture as
// [timeline.Journals] does, adding to a refusal the remedy this command's
// flags can offer.
func requireJournals(cap *cudacapture.Capture) (map[string][]byte, error) {
	journals, err := timeline.Journals(cap)
	if err != nil {
		return nil, withRemedy(cap, err)
	}
	return journals, nil
}

// withRemedy adds to a refusal of [timeline.Journals] the remedy this
// command's flags can offer. Any other error is returned as is.
func withRemedy(cap *cudacapture.Capture, err error) error {
	status, _ := readGapStatus(cap)
	switch {
	case errors.Is(err, timeline.ErrJournalOmitted):
		return fmt.Errorf("%w%s", err, budgetRemedy(status.OmittedFiles))
	case errors.Is(err, timeline.ErrRecordsDropped):
		return fmt.Errorf("%w%s", err, truncationRemedy(status))
	}
	return err
}

// truncationRemedy names the mechanism that lost the records and what would
// actually address it. --max-records is deliberately not offered: it bounds
// how many journal frames this reader decodes and cannot recover a record the
// producer never wrote.
func truncationRemedy(status gapStatus) string {
	if status.RecordLoss == nil {
		return "; the producer reported no loss breakdown" + volumeRemedy(status.Capability) + "; --max-records bounds decoding only and cannot recover an unwritten record"
	}
	var b strings.Builder
	if pct, ok := status.RecordLoss.retainedFraction(); ok {
		fmt.Fprintf(&b, "; retained %.2f%% of the %d records the producer offered", pct, status.RecordLoss.Offered)
	}
	if why := status.RecordLoss.dominant(status.Capability); why != "" {
		fmt.Fprintf(&b, "; %s", why)
	}
	b.WriteString("; --max-records is not the remedy here: it bounds how many already-written frames this reader decodes, and a journal that exceeds it fails outright with \"capture journal exceeded limit\" rather than dropping silently")
	return b.String()
}

// ---------------------------------------------------------------------------
// analysis

// GapReport is the whole result of one gaps run: the analysis, stamped with
// the build that computed it.
type GapReport struct {
	gaps.Report

	// Cudaprof is the build that computed the report.
	Cudaprof buildStamp `json:"cudaprof"`
}

func analyzeGaps(cap *cudacapture.Capture, opts gapOptions) (*GapReport, error) {
	r, err := gaps.Analyze(cap, opts.analysis())
	if err != nil {
		return nil, withRemedy(cap, err)
	}
	if status, ok := readGapStatus(cap); ok {
		if m, ok := captureGraphTraceMode(status.Capability); ok {
			r.GraphTraceMode = string(m)
		}
	}
	return &GapReport{Report: *r, Cudaprof: currentBuild()}, nil
}

// analysis returns the options of the analysis itself.
func (o gapOptions) analysis() gaps.Options {
	return gaps.Options{
		WarmupMS:     o.warmupMS,
		WarmupUntil:  o.warmupUntil,
		MaxRecords:   o.maxRecords,
		JournalBytes: o.journalBytes,
	}
}

// describeWarmup describes what opts excludes from the steady state.
func describeWarmup(opts gapOptions) string {
	return (&gaps.Report{WarmupMS: opts.warmupMS, WarmupUntil: opts.warmupUntil}).Warmup()
}

// ---------------------------------------------------------------------------
// rendering

func (r *GapReport) text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "cudaprof gaps: %s\n", r.Capture)
	for _, j := range r.Journals {
		form := "plain"
		if j.Compressed {
			form = "gzip"
		}
		fmt.Fprintf(&b, "  journal %s (%s, %d bytes, %d records)", j.Name, form, j.Bytes, j.Records)
		if j.NoCUDA {
			b.WriteString(" ran no CUDA")
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "  activities %d  api records %d  correlated %.1f%%\n", r.Activities, r.APIRecords, r.Correlated)
	if r.UnresolvedSyncs > 0 {
		fmt.Fprintf(&b, "  %d synchronization record(s) totalling %s could not be placed on a device: no record named their context\n", r.UnresolvedSyncs, millis(r.UnresolvedSyncNanos))
	}
	if r.GraphTraceMode != "" {
		fmt.Fprintf(&b, "  graph trace mode %s\n", r.GraphTraceMode)
	} else {
		fmt.Fprintf(&b, "  graph trace mode UNREAD: the agent capability does not state one\n")
	}
	if r.OpaqueGraphReplays {
		fmt.Fprintf(&b, "  graph replays are opaque: the capture holds replay envelopes and no device record naming a graph id, so each replay is counted as activity from its envelope alone. The idle between nodes of a graph is not observable here and is reported as busy. Capturing with --graph-trace=node records each graph node as a kernel instead.\n")
	}
	for _, d := range r.Devices {
		if d.SteadyActivities == 0 {
			fmt.Fprintf(&b, "\ndevice %d, steady-state: none, the warmup (%s) never ended\n", d.Device, r.Warmup())
		}
		for _, s := range []*gaps.Stats{d.Overall, d.SteadyState} {
			if s == nil {
				continue
			}
			fmt.Fprintf(&b, "\ndevice %d, %s", d.Device, s.Label)
			if s.Label == "steady-state" {
				fmt.Fprintf(&b, " (%s excluded)", r.Warmup())
			}
			b.WriteString("\n")
			writeGapStats(&b, s)
		}
	}
	return b.String()
}

func writeGapStats(b *strings.Builder, s *gaps.Stats) {
	fmt.Fprintf(b, "  timeline span     %12s\n", millis(s.SpanNanos))
	fmt.Fprintf(b, "  device active     %12s  (%.2f%% of span)\n", millis(s.ActiveNanos), pct(s.ActiveNanos, s.SpanNanos))
	fmt.Fprintf(b, "  device idle       %12s  (%.2f%% of span)\n", millis(s.IdleNanos), pct(s.IdleNanos, s.SpanNanos))
	fmt.Fprintf(b, "  duty cycle        %11.2f%%\n", s.DutyCyclePct)
	fmt.Fprintf(b, "  activities %d, merged intervals %d, gaps %d\n", s.Activities, s.MergedIntervals, s.GapCount)
	d := s.Distribution
	fmt.Fprintf(b, "  gap distribution (us)\n")
	fmt.Fprintf(b, "    min %10s  p25 %10s  median %10s  p75 %10s\n", micros(d.Min), micros(d.P25), micros(d.Median), micros(d.P75))
	fmt.Fprintf(b, "    p90 %10s  p99 %10s  max    %10s  mean   %10.3f\n", micros(d.P90), micros(d.P99), micros(d.Max), d.Mean/1e3)
	fmt.Fprintf(b, "  host concurrency at each gap\n")
	for _, c := range s.Concurrency {
		fmt.Fprintf(b, "    %-12s count %7d  idle %12s  median %10s us  mean %10.3f us\n",
			c.Name, c.Count, millis(c.IdleNanos), micros(c.Median), c.Mean/1e3)
	}
	fmt.Fprintf(b, "  idle attribution\n")
	fmt.Fprintf(b, "    host behind       %12s  (%.2f%% of idle)\n", millis(s.Attribution.HostBehindNanos), pct(s.Attribution.HostBehindNanos, s.IdleNanos))
	fmt.Fprintf(b, "    driver submission %12s  (%.2f%% of idle)\n", millis(s.Attribution.DriverSubmitNanos), pct(s.Attribution.DriverSubmitNanos, s.IdleNanos))
	fmt.Fprintf(b, "    hardware queue    %12s  (%.2f%% of idle)\n", millis(s.Attribution.HardwareNanos), pct(s.Attribution.HardwareNanos, s.IdleNanos))
	fmt.Fprintf(b, "    unattributed      %12s  (%.2f%% of idle)\n", millis(s.Attribution.UnattributedNanos), pct(s.Attribution.UnattributedNanos, s.IdleNanos))
	// Printed apart from the four above, and labelled as overlapping them,
	// because they do not partition idle and must not be read as further
	// shares of it.
	fmt.Fprintf(b, "  idle overlay (overlapping, does not sum with the split above)\n")
	fmt.Fprintf(b, "    cupti overhead    %12s  (%.2f%% of idle, %d gap(s))\n", millis(s.Overlay.CuptiOverheadNanos), pct(s.Overlay.CuptiOverheadNanos, s.IdleNanos), s.Overlay.CuptiOverheadGaps)
	fmt.Fprintf(b, "    synchronization   %12s  (%.2f%% of idle, %d gap(s))\n", millis(s.Overlay.SyncNanos), pct(s.Overlay.SyncNanos, s.IdleNanos), s.Overlay.SyncGaps)
	fmt.Fprintf(b, "    graph replay      %12s  (%.2f%% of idle, %d gap(s))\n", millis(s.Overlay.GraphReplayNanos), pct(s.Overlay.GraphReplayNanos, s.IdleNanos), s.Overlay.GraphReplayGaps)
}

func millis(ns int64) string { return fmt.Sprintf("%.3f ms", float64(ns)/1e6) }
func micros(ns int64) string { return fmt.Sprintf("%.3f", float64(ns)/1e3) }

func pct(part, whole int64) float64 {
	if whole <= 0 {
		return 0
	}
	return float64(part) / float64(whole) * 100
}
