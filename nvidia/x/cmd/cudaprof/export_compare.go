package main

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"syscall"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

const exportCompareUsage = `Usage: cudaprof export-compare -iteration-marker RE [options] base.pprof target.pprof

Write a token-aligned Perfetto TAR containing both complete timelines.
The selected start markers coincide; durations are never stretched. Before
and after work stays visible, including warm-up and lookahead. This is a
comparison of independent runs, not simultaneous execution or wall-clock sync.
Completeness failures refuse export. Stationarity failures stay visible and
do not prevent diagnostic export. Open the TAR in current ui.perfetto.dev.
The hidden .alignment.json member contains capture hashes and gate receipts.
With -summary, observed hierarchy and token tracks use the same fixed offsets
and clock domain as the raw records; missing application context stays unknown.
-summary-domain selects driver (default) or runtime submission identity,
without fallback between domains. -summary-marker adds separately labeled
inferred kernel intervals; the alignment marker is not used implicitly.
` + goTraceOptionsHelp + `
`

type exportCompareOptions struct {
	detail                   string
	outputBytes              int64
	memoryBytes              int64
	maxTracks                int64
	maxSummaryReferences     int64
	memory                   bool
	limits                   cudacapture.Limits
	demangle                 bool
	summary                  bool
	telemetry                bool
	submissionLinks          bool
	graphLifecycle           bool
	summaryDomain            string
	summaryMarker            *regexp.Regexp
	window                   iterationWindow
	paths                    [2]string
	labels                   [2]string
	skips                    [2]int
	expects                  [2]int
	out                      string
	maxRecords, journalBytes int64
	goTrace                  goTraceOptions
}

type alignmentArm struct {
	MarkerTimes      []int64        `json:"marker_timestamps_ns"`
	Path             string         `json:"capture_path"`
	SHA256           string         `json:"capture_sha256"`
	Label            string         `json:"label"`
	Trace            string         `json:"trace_path"`
	Origin           int64          `json:"origin_ns"`
	Start            int64          `json:"selected_start_ns"`
	End              int64          `json:"selected_end_ns"`
	Offset           int64          `json:"offset_ns"`
	Markers          int            `json:"observed_markers"`
	Expected         int            `json:"expected_markers,omitempty"`
	MarkerVerdict    string         `json:"marker_invariant"`
	Skip             int            `json:"skip_iterations"`
	Iterations       int            `json:"selected_iterations"`
	Records          int            `json:"retained_trace_records"`
	Completeness     string         `json:"collector_completeness"`
	Stationarity     iteration.Gate `json:"selected_stationarity"`
	FullGate         iteration.Gate `json:"full_iteration_gate"`
	Status           captureResult  `json:"capture_status"`
	records          []traceRecord
	overview         *overview.Report
	telemetry        []capturePhaseEdge
	provenance       *timeline.Records
	provenanceDomain string
	showLifecycle    bool
	Isolation        isolation.Result `json:"isolation"`
	showMemory       bool
	detail           string
}

type alignmentReceipt struct {
	maxTracks            int64
	maxSummaryReferences int64
	Detail               string          `json:"detail"`
	Memory               bool            `json:"memory_requested,omitempty"`
	Telemetry            bool            `json:"telemetry_queries_requested,omitempty"`
	SubmissionLinks      bool            `json:"submission_links_requested,omitempty"`
	GraphLifecycle       bool            `json:"graph_lifecycle_requested,omitempty"`
	SummaryMarker        string          `json:"summary_inferred_marker,omitempty"`
	SummaryDomain        string          `json:"summary_api_domain,omitempty"`
	GoTraceDecoder       string          `json:"go_trace_decoder,omitempty"`
	Version              int             `json:"version"`
	Mode                 string          `json:"mode"`
	Marker               string          `json:"marker"`
	TimeScale            int             `json:"time_scale"`
	Anchor               int64           `json:"shared_anchor_ns"`
	Arms                 [2]alignmentArm `json:"arms"`
}

func parseExportCompareArgs(args []string, out io.Writer) (exportCompareOptions, error) {
	var opts exportCompareOptions
	args, limits, err := parseReadLimits(args)
	if err != nil {
		return opts, err
	}
	opts.limits = limits
	fs := flag.NewFlagSet("export-compare", flag.ContinueOnError)
	fs.SetOutput(out)
	fs.Usage = func() { fmt.Fprint(out, exportCompareUsage); fs.PrintDefaults() }
	var flags iterationFlags
	flags.register(fs)
	summaryMarker := fs.String("summary-marker", "", "optional kernel regex for separately inferred summary intervals")
	fs.StringVar(&opts.summaryDomain, "summary-domain", "driver", "submission API domain for summary: driver or runtime")
	fs.StringVar(&opts.detail, "detail", "", "raw, compact, token or operation; default raw")
	fs.Int64Var(&opts.maxTracks, "max-tracks", defaultExportTracks, "combined maximum tracks for both arms")
	fs.Int64Var(&opts.maxSummaryReferences, "max-summary-references", 2000000, "combined maximum source references for both arms")
	fs.Int64Var(&opts.memoryBytes, "memory-bytes", defaultExportMemory, "soft Go memory budget; not an RSS limit")
	fs.Int64Var(&opts.outputBytes, "output-bytes", defaultExportBytes, "maximum exported archive bytes")
	fs.BoolVar(&opts.memory, "memory", false, "append observed memory service and policy evidence")
	fs.StringVar(&opts.summaryDomain, "operation-domain", "driver", "alias for summary-domain")
	fs.BoolVar(&opts.telemetry, "telemetry", false, "append bounded NVML query receipts with the same fixed alignment")
	fs.BoolVar(&opts.submissionLinks, "submission-links", false, "append exact submission identity flows; not dependency edges")
	fs.BoolVar(&opts.graphLifecycle, "graph-lifecycle", false, "append retained graph lifecycle events and qualification")
	fs.BoolVar(&opts.summary, "summary", false, "append observed hierarchy and token overview tracks")
	fs.BoolVar(&opts.demangle, "demangle", false, "display demangled kernel names, retaining raw_symbol args")
	fs.StringVar(&opts.out, "o", "comparison.tar", "output TAR path")
	fs.StringVar(&opts.labels[0], "base-label", "base", "base arm and machine name")
	fs.StringVar(&opts.labels[1], "target-label", "target", "target arm and machine name")
	fs.IntVar(&opts.skips[0], "base-skip-iterations", -1, "override base leading marker skip")
	fs.IntVar(&opts.skips[1], "target-skip-iterations", -1, "override target leading marker skip")
	fs.IntVar(&opts.expects[0], "base-expect-markers", 0, "independent workload invariant for base, 0 means unchecked")
	fs.IntVar(&opts.expects[1], "target-expect-markers", 0, "independent workload invariant for target, 0 means unchecked")
	fs.Int64Var(&opts.maxRecords, "max-records", 0, "maximum decoded journal records; 0 uses capture budget")
	fs.Int64Var(&opts.journalBytes, "journal-bytes", 0, "maximum decompressed journal bytes; 0 uses capture budget")
	opts.goTrace.register(fs)
	if err := fs.Parse(args); err != nil {
		return opts, err
	}
	if err := opts.goTrace.check(); err != nil {
		return opts, fmt.Errorf("export-compare: %w", err)
	}
	if *summaryMarker != "" {
		marker, err := regexp.Compile(*summaryMarker)
		if err != nil {
			return opts, fmt.Errorf("export-compare: summary marker: %w", err)
		}
		opts.summaryMarker = marker
	}
	if opts.summaryDomain != "driver" && opts.summaryDomain != "runtime" {
		return opts, errors.New("export-compare: invalid summary API domain")
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	if set["operation-domain"] && set["summary-domain"] {
		return opts, errors.New("use one of operation-domain and summary-domain")
	}
	opts.detail, err = resolveExportDetail(opts.detail, opts.summaryDomain, opts.summary, set["summary-marker"], (set["summary-domain"] || set["operation-domain"]) && !opts.submissionLinks)
	if err != nil {
		return opts, err
	}
	opts.summary = opts.detail != "raw"
	w, err := flags.window(set)
	if err != nil {
		return opts, err
	}
	if w == nil {
		return opts, errors.New("export-compare: -iteration-marker is required")
	}
	opts.window = *w
	if fs.NArg() != 2 {
		return opts, errors.New("export-compare: want base and target capture paths")
	}
	for i := range 2 {
		opts.paths[i] = fs.Arg(i)
		if opts.skips[i] < -1 || opts.expects[i] < 0 {
			return opts, errors.New("export-compare: invalid marker skip or expectation")
		}
		if opts.skips[i] == -1 {
			opts.skips[i] = w.skip
		}
		if opts.labels[i] == "" {
			return opts, errors.New("export-compare: arm labels must not be empty")
		}
	}
	if opts.labels[0] == opts.labels[1] {
		return opts, errors.New("export-compare: arm labels must differ")
	}
	if opts.out == "" || opts.outputBytes <= 0 || opts.memoryBytes <= 0 || opts.maxTracks <= 0 || opts.maxSummaryReferences <= 0 || opts.maxRecords < 0 || opts.journalBytes < 0 {
		return opts, errors.New("export-compare: invalid output or decode limit")
	}
	return opts, nil
}

func runExportCompareCommand(args []string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return runExportCompare(ctx, args)
}

func runExportCompare(ctx context.Context, args []string) error {
	opts, err := parseExportCompareArgs(args, os.Stdout)
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	if err != nil {
		return err
	}
	restore := exportMemoryBudget(opts.memoryBytes)
	defer restore()
	receipt := alignmentReceipt{maxTracks: opts.maxTracks, maxSummaryReferences: opts.maxSummaryReferences, Detail: opts.detail, Memory: opts.memory, Version: 1, Mode: "independent runs aligned by selected start marker; no clock synchronization claimed", Marker: opts.window.marker.String(), TimeScale: 1, Telemetry: opts.telemetry, SubmissionLinks: opts.submissionLinks, GraphLifecycle: opts.graphLifecycle, GoTraceDecoder: opts.goTrace.decoder}
	if opts.summary || opts.submissionLinks {
		receipt.SummaryDomain = opts.summaryDomain
		if opts.summaryMarker != nil {
			receipt.SummaryMarker = opts.summaryMarker.String()
		}
	}
	for i := range 2 {
		arm, err := loadAlignmentArm(ctx, opts, i)
		if err != nil {
			return fmt.Errorf("export-compare %s: %w", opts.labels[i], err)
		}
		receipt.Arms[i] = arm
	}
	if receipt.Arms[0].Iterations != receipt.Arms[1].Iterations {
		return errors.New("export-compare: selected iteration counts differ; pass -iteration-count")
	}
	if err := alignArms(&receipt); err != nil {
		return err
	}
	err = atomicExport(opts.out, opts.paths[:], opts.outputBytes, func(w io.Writer) error { return writeComparisonBundle(w, receipt) })
	if err != nil {
		return err
	}
	fmt.Printf("wrote %s: %d aligned intervals per arm, all records retained\n", opts.out, receipt.Arms[0].Iterations)
	for _, arm := range receipt.Arms {
		fmt.Printf("%s: collector complete; marker invariant %s; selected stationarity %t; full gate %t\n", arm.Label, arm.MarkerVerdict, arm.Stationarity.Pass, arm.FullGate.Pass)
	}
	return nil
}

func loadAlignmentArm(ctx context.Context, opts exportCompareOptions, i int) (alignmentArm, error) {
	arm := alignmentArm{Path: opts.paths[i], Label: opts.labels[i], Trace: fmt.Sprintf("arm-%d.pftrace", i), Skip: opts.skips[i], Expected: opts.expects[i], MarkerVerdict: "unchecked"}
	cap, digest, err := readAlignmentCapture(arm.Path, opts.limits)
	if err != nil {
		return arm, err
	}
	cap.Profile = nil // comparison uses journal evidence, not aggregate samples
	arm.SHA256 = digest
	arm.Isolation = captureIsolation(cap)
	status, records, err := decodeCompleteCapture(cap, opts.maxRecords, opts.journalBytes)
	if err != nil {
		return arm, err
	}
	if opts.summary {
		report := detailOverview(overview.Analyze(records, opts.summaryDomain, opts.summaryMarker), opts.detail)
		arm.overview = &report
	}
	if opts.telemetry {
		arm.telemetry, err = telemetryEdges(cap)
		if err != nil {
			return arm, err
		}
	}
	if opts.submissionLinks || opts.graphLifecycle || opts.memory {
		arm.provenance = &records
		arm.provenanceDomain = opts.summaryDomain
		arm.showLifecycle = opts.graphLifecycle
		arm.showMemory = opts.memory
	}
	arm.Status = status
	arm.Completeness = "passed"
	w := opts.window
	w.skip = arm.Skip
	if err := selectAlignmentWindow(&arm, records, w); err != nil {
		return arm, err
	}
	journals, err := requireJournals(cap)
	if err != nil {
		return arm, err
	}
	maxRecords, maxBytes := decodeLimits(cap, opts.maxRecords, opts.journalBytes)
	arm.records, _, _, err = decodeTraceJournals(journals, maxRecords, maxBytes, haveLanguageStacks(cap))
	if err != nil {
		return arm, err
	}
	if err := attachLanguageStacks(cap, arm.records); err != nil {
		return arm, err
	}
	arm.records, err = appendLanguageSamples(ctx, cap, arm.records, opts.goTrace)
	if err != nil {
		return arm, err
	}
	if opts.demangle {
		demangleTraceRecords(arm.records)
	}
	arm.detail = opts.detail
	arm.Records = len(arm.records)
	arm.Origin = math.MaxInt64
	for _, r := range arm.records {
		if r.rec.Start > math.MaxInt64 || r.rec.End > math.MaxInt64 {
			return arm, errors.New("record timestamp exceeds signed comparison clock")
		}
		arm.Origin = min(arm.Origin, int64(r.rec.Start))
	}
	if opts.graphLifecycle {
		arm.Origin = int64(lifecycleOrigin(records, uint64(arm.Origin)))
	}
	if opts.memory {
		arm.Origin = int64(memoryTraceOrigin(records, uint64(arm.Origin)))
	}
	return arm, nil
}

func selectAlignmentWindow(arm *alignmentArm, records timeline.Records, w iterationWindow) error {
	side := CompareSide{StartTimestamp: math.MaxInt64}
	for _, a := range records.Activities {
		side.StartTimestamp = min(side.StartTimestamp, a.Start)
		side.StopTimestamp = max(side.StopTimestamp, a.End)
	}
	if err := side.cutIterations(records, w); err != nil {
		return err
	}
	arm.Start, arm.End, arm.Iterations = side.StartTimestamp, side.StopTimestamp, side.Iterations
	full, err := iteration.Analyze(records, iteration.Options{Marker: w.marker, Skip: w.skip, Expect: arm.Expected})
	if err != nil {
		return err
	}
	arm.Markers = full.Markers
	arm.MarkerTimes = nil
	for _, it := range full.Iterations {
		arm.MarkerTimes = append(arm.MarkerTimes, it.Start)
	}
	if len(full.Iterations) > 0 {
		arm.MarkerTimes = append(arm.MarkerTimes, full.Iterations[len(full.Iterations)-1].End)
	}
	arm.FullGate = full.Gate
	if arm.Expected > 0 {
		if arm.Expected != arm.Markers {
			return fmt.Errorf("marker invariant failed: expected %d, observed %d", arm.Expected, arm.Markers)
		}
		arm.MarkerVerdict = "passed"
	}
	selected := timeline.Records{}
	for _, a := range records.Activities {
		if a.Start >= arm.Start && a.Start <= arm.End {
			selected.Activities = append(selected.Activities, a)
		}
	}
	for _, c := range records.Calls {
		if c.Start >= arm.Start && c.Start <= arm.End {
			selected.Calls = append(selected.Calls, c)
		}
	}
	report, err := iteration.Analyze(selected, iteration.Options{Marker: w.marker})
	if err != nil {
		return err
	}
	arm.Stationarity = report.Gate
	return nil
}

// Offset each rebased trace by a nonnegative amount. Prefix work remains on
// the timeline, and only the selected start boundary (not every token) aligns.
func alignArms(r *alignmentReceipt) error {
	for _, a := range r.Arms {
		r.Anchor = max(r.Anchor, a.Start-a.Origin)
	}
	for i := range r.Arms {
		a := &r.Arms[i]
		a.Offset = r.Anchor - (a.Start - a.Origin)
		if a.Origin < 0 || a.Start < a.Origin || a.End < a.Start {
			return errors.New("invalid alignment timestamps")
		}
	}
	for _, a := range r.Arms {
		for _, rec := range a.records {
			if rec.rec.End > uint64(math.MaxInt64-r.Arms[0].Offset) {
				return errors.New("alignment timestamp shift overflows")
			}
		}
	}
	return nil
}

func writeComparisonBundle(w io.Writer, r alignmentReceipt) error {
	var tracks, references int64
	trackLimit, referenceLimit := r.maxTracks, r.maxSummaryReferences
	if trackLimit == 0 {
		trackLimit = defaultExportTracks
	}
	if referenceLimit == 0 {
		referenceLimit = 2000000
	}
	for _, arm := range r.Arms {
		references += summaryReferenceBound(arm.overview)
		tracks += projectionTrackBound(arm.records, arm.overview, arm.provenance, len(arm.telemetry), r.SubmissionLinks, arm.showLifecycle, arm.showMemory, true)
	}
	if err := requireProjectionTracks(tracks, trackLimit); err != nil {
		return err
	}
	if err := requireSummaryReferences(references, referenceLimit); err != nil {
		return err
	}
	tw := tar.NewWriter(w)
	bindings := tracecontract.Bindings{Version: 1, Schema: "cudaprof.bindings.v1"}
	member := func(name string, data []byte) error {
		if name != ".cudaprof-bindings.json" {
			kind, source := "metadata", ""
			for _, arm := range r.Arms {
				if name == arm.Trace {
					kind, source = "trace", arm.SHA256
				}
			}
			binding, err := tracecontract.Digest(name, kind, source, data)
			if err != nil {
				return err
			}
			bindings.Members = append(bindings.Members, binding)
		}
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0644, Size: int64(len(data))}); err != nil {
			return err
		}
		_, err := tw.Write(data)
		return err
	}
	files := []map[string]any{}
	attrs := map[string]any{"cudaprof.alignment": r.Mode, "cudaprof.time_scale": 1, "cudaprof.base_byte_shift_ns": r.Arms[0].Offset, "cudaprof.marker": r.Marker, "cudaprof.receipt": ".alignment.json"}
	for i, a := range r.Arms {
		file := map[string]any{"path": a.Trace, "machine": map[string]any{"name": a.Label}, "clocks": map[string]any{"clock": "BOOTTIME", "sync_to": map[string]any{"file": r.Arms[0].Trace, "clock": "BOOTTIME"}, "offset_ns": a.Offset - r.Arms[0].Offset}}
		if i == 0 {
			delete(file, "clocks")
		}
		files = append(files, file)
		attrs[fmt.Sprintf("cudaprof.arm_%d.collector", i)] = a.Completeness
		attrs[fmt.Sprintf("cudaprof.arm_%d.marker_invariant", i)] = a.MarkerVerdict
		attrs[fmt.Sprintf("cudaprof.arm_%d.stationarity", i)] = fmt.Sprintf("selected=%t full=%t; %v", a.Stationarity.Pass, a.FullGate.Pass, a.FullGate.Failures)
	}
	// Move the base trace forward when the target has a longer prefix. This
	// shift is encoded in its bytes; the relative manifest offset still aligns.
	manifest := map[string]any{"perfetto_manifest": map[string]any{"version": 1, "trace_time": map[string]any{"clock": "BOOTTIME", "file": r.Arms[0].Trace}, "files": files, "attributes": attrs}}
	data, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	if err := member("perfetto_manifest.json", data); err != nil {
		return err
	}
	for i, a := range r.Arms {
		var buf bytes.Buffer
		var output io.Writer = &limitedExportWriter{Writer: &buf, remaining: defaultExportBytes}
		records := append([]traceRecord(nil), a.records...)
		for j := range records {
			if records[j].rec.End > math.MaxUint64-uint64(r.Arms[0].Offset) {
				return errors.New("comparison timestamp overflow")
			}
			records[j].rec.Start += uint64(r.Arms[0].Offset)
			records[j].rec.End += uint64(r.Arms[0].Offset)
		}
		origin := uint64(a.Origin)
		base := uint64(i+1) << 32
		identity := traceExportIdentity{sha256: a.SHA256, path: a.Path, arm: a.Label, shift: uint64(r.Arms[0].Offset), summaryReferences: referenceLimit}
		n, err := writeTraceOrigin(output, records, origin, base, a.Label+": ", identity)
		if err != nil {
			return err
		}
		if len(a.telemetry) > 0 {
			if err := writeTelemetryTraceAt(output, a.telemetry, origin, uint64(i+1)<<50, uint64(r.Arms[0].Offset), perfetto.ClockBoottime); err != nil {
				return err
			}
		}
		if a.provenance != nil {
			if r.SubmissionLinks {
				if err := writeSubmissionTrace(output, *a.provenance, a.provenanceDomain, origin, uint64(i+1)<<52, uint64(r.Arms[0].Offset), perfetto.ClockBoottime, identity); err != nil {
					return err
				}
			}
			if a.showLifecycle {
				if err := writeGraphLifecycleTrace(output, *a.provenance, origin, uint64(i+1)<<54, uint64(r.Arms[0].Offset), perfetto.ClockBoottime); err != nil {
					return err
				}
			}
		}
		if a.provenance != nil && a.showMemory {
			if err := writeMemoryTrace(output, *a.provenance, origin, uint64(i+1)<<56, uint64(r.Arms[0].Offset), perfetto.ClockBoottime); err != nil {
				return err
			}
		}
		pw := perfetto.NewWriter(output)
		pw.TimestampClockID = perfetto.ClockBoottime
		id := base + uint64(n+1)
		verdict := fmt.Sprintf("%s alignment (collector complete; marker %s; stationarity selected=%t full=%t)", a.Label, a.MarkerVerdict, a.Stationarity.Pass, a.FullGate.Pass)
		if err := pw.WriteTrackDescriptor(id, verdict, 0); err != nil {
			return err
		}
		end := int64(0)
		for _, rec := range a.records {
			end = max(end, int64(rec.rec.End)-a.Origin)
		}
		ranges := []struct {
			name       string
			start, end int64
		}{{"before selected window", 0, a.Start - a.Origin}, {"selected comparison window", a.Start - a.Origin, a.End - a.Origin}, {"after selected window", a.End - a.Origin, end}}
		for _, span := range ranges {
			if span.end <= span.start {
				continue
			}
			if err := pw.WriteSliceBegin(id, uint64(span.start+r.Arms[0].Offset), span.name, "comparison"); err != nil {
				return err
			}
			if err := pw.WriteSliceEnd(id, uint64(span.end+r.Arms[0].Offset)); err != nil {
				return err
			}
		}

		markerID := id + 1
		if err := pw.WriteTrackDescriptor(markerID, a.Label+": selected boundaries", 0); err != nil {
			return err
		}
		boundary := 0
		for _, mark := range a.MarkerTimes {
			if mark < a.Start || mark > a.End {
				continue
			}
			if err := pw.WriteInstant(markerID, uint64(mark-a.Origin+r.Arms[0].Offset), fmt.Sprintf("selected boundary %d", boundary), "comparison-boundary"); err != nil {
				return err
			}
			boundary++
		}
		if a.overview != nil {
			if err := writeOverviewTraceAt(output, *a.overview, origin, uint64(i+1)<<48, uint64(r.Arms[0].Offset), perfetto.ClockBoottime, identity); err != nil {
				return err
			}
		}
		if a.SHA256 != "" && a.Path != "" {
			receipt := exportReceipt("trace compare", identity, a.records, a.Status)
			receipt.Options["detail"] = r.Detail
			receipt.Limits = map[string]int64{"max_tracks": trackLimit, "max_summary_references": referenceLimit}
			receipt.Gates["isolation"] = isolationExportGate(a.Isolation)
			receipt.Gates["workload"] = tracecontract.Gate{Disposition: a.MarkerVerdict, Reason: "independent expected marker count", Provenance: "alignment receipt"}
			verdict := "failed"
			if a.Stationarity.Pass {
				verdict = "passed"
			}
			receipt.Gates["stationarity"] = tracecontract.Gate{Disposition: verdict, Reason: "selected iteration gate", Provenance: "alignment receipt"}
			if err := writeExportReceipt(pw, receipt, uint64(i+1)<<58, uint64(r.Arms[0].Offset)); err != nil {
				return err
			}
		}
		if a.SHA256 != "" {
			if err := writeIsolationTrace(output, a.Isolation, identity, uint64(i+1)<<60, perfetto.ClockBoottime); err != nil {
				return err
			}
		}
		if err := member(a.Trace, buf.Bytes()); err != nil {
			return err
		}
	}
	data, err = json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err := member(".alignment.json", data); err != nil {
		return err
	}
	data, err = bindings.JSON()
	if err != nil {
		return err
	}
	if err := member(".cudaprof-bindings.json", data); err != nil {
		return err
	}
	return tw.Close()
}

func readAlignmentCapture(path string, limits cudacapture.Limits) (*cudacapture.Capture, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, "", err
	}
	hash := sha256.New()
	reader := io.TeeReader(f, hash)
	cap, err := cudacapture.ReadLimits(reader, trustedLimits(limits, fi.Size()))
	if err != nil {
		return nil, "", readLimitRemedy(err)
	}
	if _, err := io.Copy(io.Discard, reader); err != nil {
		return nil, "", err
	}
	return cap, hex.EncodeToString(hash.Sum(nil)), nil
}

func checkComparisonOutput(out string, inputs [2]string) error {
	path, err := filepath.Abs(out)
	if err != nil {
		return err
	}
	fi, statErr := os.Stat(out)
	if statErr != nil && !errors.Is(statErr, os.ErrNotExist) {
		return statErr
	}
	for _, input := range inputs {
		abs, err := filepath.Abs(input)
		if err != nil {
			return err
		}
		same := abs == path
		if fi != nil {
			other, err := os.Stat(input)
			if err != nil {
				return err
			}
			same = same || os.SameFile(fi, other)
		}
		if same {
			return errors.New("comparison output must differ from input captures")
		}
	}
	return nil
}
