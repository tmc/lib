package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	languagestack "github.com/tmc/lib/nvidia/x/cudaprof/stack"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// cudaprof export writes a capture's device timeline as a Perfetto trace.
//
// The per-record journal is the only tier that carries timestamps: pprof
// samples are aggregates, so a trace cannot be built from the projection. This
// command therefore refuses exactly the captures gaps refuses, and for the same
// reason -- a timeline assembled from a journal that dropped records has holes
// no viewer displays, and a hole is indistinguishable from device idle.
//
// The trace is standalone. Timestamps are CUPTI's own domain rebased on the
// earliest record, and no clock-synchronization packet is emitted, so it must
// not be merged with a host trace taken from another clock.

type exportOptions struct {
	detail               string
	outputBytes          int64
	memoryBytes          int64
	maxTracks            int64
	maxSummaryReferences int64
	demangle             bool
	summary              bool
	telemetry            bool
	findings             bool
	submissionLinks      bool
	graphLifecycle       bool
	memory               bool
	summaryDomain        string
	summaryMarker        *regexp.Regexp
	capturePath          string
	outPath              string
	maxRecords           int64
	journalBytes         int64
	limits               cudacapture.Limits
	goTrace              goTraceOptions
}

var errExportHelp = errors.New("export: help requested")

func runExportCommand(args []string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return runExport(ctx, args)
}

func runExport(ctx context.Context, args []string) error {
	if len(args) == 0 {
		printExportHelp()
		return nil
	}
	opts, err := parseExportArgs(args)
	if err != nil {
		if errors.Is(err, errExportHelp) {
			return nil
		}
		return err
	}
	restore := exportMemoryBudget(opts.memoryBytes)
	defer restore()
	path, err := filepath.Abs(opts.capturePath)
	if err != nil {
		return err
	}
	cap, sha, err := readAlignmentCapture(path, opts.limits)
	if err != nil {
		return fmt.Errorf("read capture %s: %w", opts.capturePath, err)
	}
	// Export uses retained journals, not aggregate pprof samples. Release the
	// profile and its embedded attachment strings before decoding the timeline.
	cap.Profile = nil
	journals, err := requireJournals(cap)
	if err != nil {
		return err
	}
	maxRecords, journalBytes := decodeLimits(cap, opts.maxRecords, opts.journalBytes)
	records, nvtx, ext, err := decodeTraceJournals(journals, maxRecords, journalBytes, haveLanguageStacks(cap))
	if err != nil {
		return decodeRemedy(cap, err, maxRecords, journalBytes)
	}
	if err := attachLanguageStacks(cap, records); err != nil {
		return err
	}
	groups, err := readLanguageSamples(ctx, cap, opts.goTrace)
	if err != nil {
		return err
	}
	// Detach the report so the groups' samples are not retained through
	// decode and encode.
	unavailable, omitted := unavailableLanguageGroups(groups)
	records, err = appendLanguageGroups(records, groups)
	if err != nil {
		return err
	}
	var telemetry []capturePhaseEdge
	if opts.telemetry {
		telemetry, err = telemetryEdges(cap)
		if err != nil {
			return err
		}
	}
	var decoded timeline.Records
	if opts.summary || opts.submissionLinks || opts.graphLifecycle || opts.memory {
		decoded, err = timeline.Decode(journals, maxRecords, journalBytes)
		if err != nil {
			return err
		}
	}
	origin := traceOrigin(records)
	if opts.graphLifecycle {
		origin = lifecycleOrigin(decoded, origin)
	}
	if opts.memory {
		origin = memoryTraceOrigin(decoded, origin)
	}
	var summary *overview.Report
	if opts.summary {
		report := detailOverview(overview.Analyze(decoded, opts.summaryDomain, opts.summaryMarker), opts.detail)
		summary = &report
	}
	trackBound := projectionTrackBound(records, summary, &decoded, len(telemetry), opts.submissionLinks, opts.graphLifecycle, opts.memory, false)
	if opts.findings {
		trackBound++
	}
	if err := requireProjectionTracks(trackBound, opts.maxTracks); err != nil {
		return err
	}
	if err := requireSummaryReferences(summaryReferenceBound(summary), opts.maxSummaryReferences); err != nil {
		return err
	}
	var findings []tracecontract.Finding
	if opts.findings {
		findings, err = exportRecordFindings(journals, traceExportIdentity{sha256: sha, path: path}, captureExportStatus(cap), maxRecords, journalBytes)
		if err != nil {
			return err
		}
	}
	var n int
	err = atomicExport(opts.outPath, []string{path}, opts.outputBytes, func(f io.Writer) error {
		if opts.demangle {
			demangleTraceRecords(records)
		}
		identity := traceExportIdentity{sha256: sha, path: path, summaryReferences: opts.maxSummaryReferences}
		var err error
		n, err = writeTraceOrigin(f, records, origin, 0, "", identity)
		if err == nil && opts.summary {
			err = writeOverviewTraceAt(f, *summary, origin, 1<<48, 0, 0, identity)
		}
		if err == nil && opts.submissionLinks {
			err = writeSubmissionTrace(f, decoded, opts.summaryDomain, origin, 3<<48, 0, 0, identity)
		}
		if err == nil && opts.graphLifecycle {
			err = writeGraphLifecycleTrace(f, decoded, origin, 4<<48, 0, 0)
		}
		if err == nil && opts.telemetry {
			err = writeTelemetryTrace(f, telemetry, origin, 2<<48)
		}
		if err == nil && opts.memory {
			err = writeMemoryTrace(f, decoded, origin, 5<<48, 0, 0)
		}
		if err == nil {
			pw := perfetto.NewWriter(f)
			receipt := exportReceipt("trace export", identity, records, captureExportStatus(cap))
			receipt.Gates["isolation"] = isolationExportGate(captureIsolation(cap))
			receipt.Options["detail"] = opts.detail
			receipt.Limits = map[string]int64{"output_bytes": opts.outputBytes, "soft_go_memory_bytes": opts.memoryBytes, "max_tracks": opts.maxTracks, "max_summary_references": opts.maxSummaryReferences}
			err = writeExportReceipt(pw, receipt, 6<<48, 0)
			if err == nil {
				err = writeIsolationTrace(f, captureIsolation(cap), identity, 7<<48, 0)
			}
		}
		if err == nil && opts.findings {
			err = writeRecordFindings(f, findings, identity, 8<<48)
		}
		return err
	})
	if err != nil {
		return err
	}
	fmt.Printf("wrote %s: %d record(s) on %d track(s)\n", opts.outPath, len(records), n)
	if nvtx != (nvtxSummary{}) {
		fmt.Printf("nvtx: %d range(s), %d instantaneous marker(s)\n", nvtx.ranges, nvtx.points)
		if nvtx.unpaired > 0 {
			fmt.Printf("nvtx: %d marker(s) drawn as nothing: a START with no END is a range still open when the capture ended, and an END with no START opened before it began\n", nvtx.unpaired)
		}
		if nvtx.unknown > 0 {
			fmt.Printf("nvtx: %d marker(s) carried none of the start/end/instantaneous flags and were skipped\n", nvtx.unknown)
		}
	}
	for _, line := range externalSummaryLines(ext) {
		fmt.Println(line)
	}
	for _, u := range unavailable {
		fmt.Println(u.line())
	}
	if omitted > 0 {
		fmt.Println(goTraceOmittedLine(omitted))
	}
	return nil
}

func parseExportArgs(rawArgs []string) (exportOptions, error) {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return exportOptions{}, err
	}
	opts := exportOptions{limits: limits, summaryDomain: "driver", outputBytes: defaultExportBytes, memoryBytes: defaultExportMemory, maxTracks: defaultExportTracks, maxSummaryReferences: 2000000}
	var domainSet bool
	var legacyDomain, operationDomain bool
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if (arg == "-max-tracks" || arg == "-max-summary-references" || arg == "-memory-bytes" || arg == "-output-bytes") && i+1 < len(args) {
			i++
			arg += "=" + args[i]
		}
		switch {
		case arg == "-detail" && i+1 < len(args):
			i++
			opts.detail = args[i]
		case strings.HasPrefix(arg, "-detail="):
			opts.detail = strings.TrimPrefix(arg, "-detail=")
		case arg == "-operation-domain" && i+1 < len(args):
			i++
			opts.summaryDomain = args[i]
			domainSet = true
			operationDomain = true
		case strings.HasPrefix(arg, "-operation-domain="):
			opts.summaryDomain = strings.TrimPrefix(arg, "-operation-domain=")
			domainSet = true
			operationDomain = true
		case strings.HasPrefix(arg, "-max-tracks="):
			opts.maxTracks, err = strconv.ParseInt(strings.TrimPrefix(arg, "-max-tracks="), 10, 64)
			if err != nil || opts.maxTracks <= 0 {
				return exportOptions{}, fmt.Errorf("invalid maximum tracks")
			}
		case strings.HasPrefix(arg, "-max-summary-references="):
			opts.maxSummaryReferences, err = strconv.ParseInt(strings.TrimPrefix(arg, "-max-summary-references="), 10, 64)
			if err != nil || opts.maxSummaryReferences <= 0 {
				return exportOptions{}, fmt.Errorf("invalid maximum summary references")
			}
		case strings.HasPrefix(arg, "-memory-bytes="):
			opts.memoryBytes, err = strconv.ParseInt(strings.TrimPrefix(arg, "-memory-bytes="), 10, 64)
			if err != nil || opts.memoryBytes <= 0 {
				return exportOptions{}, fmt.Errorf("invalid memory byte budget")
			}
		case strings.HasPrefix(arg, "-output-bytes="):
			opts.outputBytes, err = strconv.ParseInt(strings.TrimPrefix(arg, "-output-bytes="), 10, 64)
			if err != nil || opts.outputBytes <= 0 {
				return exportOptions{}, fmt.Errorf("invalid output byte limit")
			}
		case arg == "-summary-marker" && i+1 < len(args):
			i++
			opts.summaryMarker, err = regexp.Compile(args[i])
			if err != nil {
				return exportOptions{}, fmt.Errorf("summary marker: %w", err)
			}
		case strings.HasPrefix(arg, "-summary-marker="):
			opts.summaryMarker, err = regexp.Compile(strings.TrimPrefix(arg, "-summary-marker="))
			if err != nil {
				return exportOptions{}, fmt.Errorf("summary marker: %w", err)
			}
		case arg == "-summary-domain" && i+1 < len(args):
			i++
			opts.summaryDomain = args[i]
			domainSet = true
			legacyDomain = true
		case strings.HasPrefix(arg, "-summary-domain="):
			opts.summaryDomain = strings.TrimPrefix(arg, "-summary-domain=")
			domainSet = true
			legacyDomain = true
		case arg == "-submission-links":
			opts.submissionLinks = true
		case arg == "-graph-lifecycle":
			opts.graphLifecycle = true
		case arg == "-memory":
			opts.memory = true
		case arg == "-findings" || arg == "--findings":
			opts.findings = true
		case arg == "-telemetry" || arg == "--telemetry":
			opts.telemetry = true
		case arg == "-summary" || arg == "--summary":
			opts.summary = true
		case arg == "-demangle" || arg == "--demangle":
			opts.demangle = true
		case arg == "-h" || arg == "--help" || arg == "-help":
			printExportHelp()
			return exportOptions{}, errExportHelp
		case arg == "-o" && i+1 < len(args):
			i++
			opts.outPath = args[i]
		case strings.HasPrefix(arg, "-o="), strings.HasPrefix(arg, "--out="):
			opts.outPath = arg[strings.Index(arg, "=")+1:]
		case strings.HasPrefix(arg, "--max-records="), strings.HasPrefix(arg, "-max-records="):
			v, err := strconv.ParseInt(arg[strings.Index(arg, "=")+1:], 10, 64)
			if err != nil || v <= 0 {
				return exportOptions{}, fmt.Errorf("invalid max records: %s", arg)
			}
			opts.maxRecords = v
		case strings.HasPrefix(arg, "--journal-bytes="), strings.HasPrefix(arg, "-journal-bytes="):
			v, err := strconv.ParseInt(arg[strings.Index(arg, "=")+1:], 10, 64)
			if err != nil || v <= 0 {
				return exportOptions{}, fmt.Errorf("invalid journal bytes: %s", arg)
			}
			opts.journalBytes = v
		case (arg == "-go-trace-decoder" || arg == "-go-trace-scratch") && i+1 < len(args):
			i++
			if err := opts.goTrace.set(arg[1:], args[i]); err != nil {
				return exportOptions{}, err
			}
		case strings.HasPrefix(arg, "-go-trace-decoder="), strings.HasPrefix(arg, "-go-trace-scratch="):
			name, value, _ := strings.Cut(arg[1:], "=")
			if err := opts.goTrace.set(name, value); err != nil {
				return exportOptions{}, err
			}
		case strings.HasPrefix(arg, "-"):
			return exportOptions{}, fmt.Errorf("unknown flag: %s", arg)
		case opts.capturePath == "":
			opts.capturePath = arg
		default:
			return exportOptions{}, fmt.Errorf("unexpected argument: %s", arg)
		}
	}
	if err := opts.goTrace.check(); err != nil {
		return exportOptions{}, err
	}
	if opts.summaryDomain != "driver" && opts.summaryDomain != "runtime" {
		return exportOptions{}, fmt.Errorf("invalid summary API domain %q", opts.summaryDomain)
	}
	if legacyDomain && operationDomain {
		return exportOptions{}, errors.New("use one of operation-domain and summary-domain")
	}
	opts.detail, err = resolveExportDetail(opts.detail, opts.summaryDomain, opts.summary, opts.summaryMarker != nil, domainSet && !opts.submissionLinks)
	if err != nil {
		return exportOptions{}, err
	}
	opts.summary = opts.detail != "raw"
	if opts.capturePath == "" {
		return exportOptions{}, errors.New("usage: cudaprof export [-o out.pftrace] [--max-records=N] [--journal-bytes=N] capture.pprof")
	}
	if opts.outPath == "" {
		opts.outPath = strings.TrimSuffix(opts.capturePath, ".pprof") + ".pftrace"
	}
	return opts, nil
}

func printExportHelp() {
	fmt.Fprint(os.Stdout, `Usage: cudaprof export [options] capture.pprof

Writes the capture's device timeline as a Perfetto trace (.pftrace), openable
at https://ui.perfetto.dev.

Kernels and memcpys go on one track per device and stream; driver and runtime
API calls go on one track per thread. Activities that overlap on the same
stream are split across numbered lanes rather than nested, so no slice is
drawn inside another it did not contain.

The timeline comes from the per-record journal, not from the pprof samples,
which are aggregates and carry no timestamps. A capture whose journal dropped
records is refused: a hole in a timeline is indistinguishable from device idle
once drawn.

Timestamps are CUPTI's domain rebased on the earliest record. No clock
synchronization packet is emitted, so this trace must not be merged with a
host trace taken from another clock.

Options:
  -detail raw|compact|token|operation  summary detail (default raw)
  -max-tracks=N      combined track bound (default 100000)
  -max-summary-references=N  combined source reference bound (default 2000000)
  -memory-bytes=N   soft Go memory budget (default 805306368; not RSS)
  -output-bytes=N   maximum output bytes (default 536870912)
  -memory           append observed memory service and policy evidence
  -submission-links  append exact API/activity identity flows (not dependencies)
  -graph-lifecycle   append observed lifecycle events and coverage
  -findings         append admitted and withheld raw-record count findings
  -telemetry        append bounded NVML query intervals, without interpolation
  -summary          append observed hierarchy and token overview tracks
  -summary-domain driver|runtime  API domain for explicit context (default driver)
  -summary-marker RE  append separately labeled inferred kernel intervals
  -demangle         display demangled kernel names; retain raw_symbol args
  `+goTraceOptionsHelp+`
  -o <path>          output file (default: the capture path with .pftrace)
  --max-records=N    maximum journal records to decode (default: the larger of
                     1048576 and what the capture records)
  --journal-bytes=N  maximum decompressed bytes per journal (default: the
                     larger of 536870912 and what the capture records)
`)
}

// tracePayload is the typed journal payload, in full. gapPayload reads the
// four fields a duty cycle needs; a timeline draws the rest.
type tracePayload struct {
	sequence     uint64
	Kind         string `json:"kind"`
	Subtype      string `json:"subtype"`
	Domain       string `json:"domain"`
	Name         string `json:"name"`
	Valid        bool   `json:"valid"`
	Start        uint64 `json:"start"`
	End          uint64 `json:"end"`
	Bytes        uint64 `json:"bytes"`
	Queued       uint64 `json:"queued"`
	Submitted    uint64 `json:"submitted"`
	CopyKind     uint32 `json:"copy_kind"`
	RuntimeCorr  uint32 `json:"runtime_correlation"`
	Device       uint32 `json:"device"`
	Context      uint32 `json:"context"`
	Stream       uint32 `json:"stream"`
	Correlation  uint32 `json:"correlation"`
	GridX        uint32 `json:"grid_x"`
	GridY        uint32 `json:"grid_y"`
	GridZ        uint32 `json:"grid_z"`
	BlockX       uint32 `json:"block_x"`
	BlockY       uint32 `json:"block_y"`
	BlockZ       uint32 `json:"block_z"`
	Registers    uint32 `json:"registers_per_thread"`
	DynShared    uint32 `json:"dynamic_shared_memory"`
	GraphID      uint32 `json:"graph_id"`
	GraphNodeID  uint64 `json:"graph_node_id"`
	OverheadKind uint32 `json:"overhead_kind"`
	ObjectKind   uint32 `json:"object_kind"`
	SyncType     uint32 `json:"sync_type"`
	CudaEventID  uint32 `json:"cuda_event_id"`
	ProcessID    uint32 `json:"process_id"`
	ThreadID     uint32 `json:"thread_id"`
	CallbackID   uint32 `json:"callback_id"`
	ReturnValue  uint32 `json:"return_value"`
	Dropped      uint64 `json:"dropped"`

	// External correlation. The producer emits these in their own frame kind:
	// ExternalID is the framework scope -- a PyTorch operator, an NCCL
	// collective -- that was on the stack while Correlation was the live
	// native ID. The record has no interval and draws no slice; it is what
	// lets the join say which scope a kernel ran under.
	ExternalKind uint32 `json:"external_kind"`
	ExternalID   uint64 `json:"external_id"`

	// NVTX. A marker record carries one timestamp in both Start and End; the
	// duration of a range lives in the pair of records sharing MarkerID, not
	// in either one.
	MarkerDomain string `json:"marker_domain"`
	MarkerID     uint32 `json:"marker_id"`
	MarkerFlags  uint32 `json:"marker_flags"`
}

// NVTX marker flags, mirroring [cupti.ActivityFlagMarkerStart] and its
// siblings. They are tested as bits: CUPTI may set others alongside.
const (
	markerFlagInstantaneous = 1 << 0
	markerFlagStart         = 1 << 1
	markerFlagEnd           = 1 << 2
)

// trackKind is the namespace a record's group belongs to. It exists because a
// device stream id, a host thread id and an NVTX thread id are all small
// integers: one shared key namespace would silently merge a device stream 1
// with a host thread 1 onto a single track.
type trackKind uint32

const (
	trackDevice      trackKind = iota // device: group is (device, stream)
	trackHost                         // host API: group is (pid, thread)
	trackNVTX                         // NVTX range: group is (pid, thread)
	trackStackSample                  // observed host stack samples, not CUDA calls
)

// traceRecord is one activity with the track it belongs on. The track key is
// separate from the record because a device stream and a host thread are
// different namespaces that would otherwise collide on small integers.
type traceRecord struct {
	journal       string
	sequence      uint64
	annotations   []perfetto.DebugAnnotation
	session       string
	process       uint64
	generation    uint64
	external      map[uint32]cuptiactivity.ExternalMatch
	stack         *languagestack.Record
	sample        *languagestack.Sample
	clocksUnified bool

	rec   cuptiactivity.Record
	kind  trackKind
	group [2]uint32
}

// decodeTraceJournals decodes every journal member into activities and API
// calls. The refusals mirror timeline.Decode: a journal that never finalized,
// lost records, or decoded short leaves a timeline whose missing spans read as
// idle device, which is a wrong answer rather than a partial one.
func decodeTraceJournals(journals map[string][]byte, maxRecords, maxBytes int64, stacks bool) ([]traceRecord, nvtxSummary, externalSummary, error) {
	names := make([]string, 0, len(journals))
	for name := range journals {
		names = append(names, name)
	}
	sort.Strings(names)
	var out []traceRecord
	var nvtx nvtxSummary
	var ext externalSummary
	for _, name := range names {
		reader, err := timeline.OpenJournal(journals[name], maxBytes)
		if err != nil {
			return nil, nvtx, ext, fmt.Errorf("read journal %s: %w", name, err)
		}
		err = decodeTraceJournal(name, reader, maxRecords, &out, &nvtx, &ext, stacks)
		if cerr := reader.Close(); err == nil && cerr != nil {
			err = fmt.Errorf("close journal %s: %w", name, cerr)
		}
		if err != nil {
			return nil, nvtx, ext, err
		}
	}
	if len(out) == 0 {
		return nil, nvtx, ext, errors.New("journals hold no complete activity records, so there is no timeline to draw")
	}
	annotateTraceCorrelations(out)
	return out, nvtx, ext, nil
}

// externalSummary is what the external-correlation tier contributed: how much
// of the device work in the trace carries a framework scope. It is reported
// because attribution coverage is not visible in the timeline itself -- a
// kernel nothing pushed an ID for is drawn exactly like one that has a scope.
type externalSummary struct {
	kind                        cupti.ExternalCorrelationKind
	records, matched, ambiguous uint64
	duration, matchedNanos      uint64
	// kinds counts the external-correlation records by namespace, including
	// the namespaces that were not joined. Without it the coverage line
	// reports the namespace it used and says nothing about the one it
	// discarded, so a kernel we chose not to join reads as a kernel nothing
	// was pushed for.
	kinds externalKindCounts
}

// externalSummaryLines is what the export reports about attribution coverage.
// It is a function rather than a run of Printf calls so a test can assert the
// lines a user reads: the omission note below is worth nothing if it is
// computed and never printed.
func externalSummaryLines(ext externalSummary) []string {
	if ext.records == 0 {
		return nil
	}
	line := fmt.Sprintf("external: %d of %d device record(s) carry a framework scope in namespace %s", ext.matched, ext.records, ext.kind)
	if ext.duration > 0 {
		line += fmt.Sprintf(" (%d%% of device time)", 100*ext.matchedNanos/ext.duration)
	}
	lines := []string{line}
	if ext.ambiguous > 0 {
		lines = append(lines, fmt.Sprintf("external: %d device record(s) matched more than one external id and were left unattributed", ext.ambiguous))
	}
	if note := ext.kinds.note(); note != "" {
		lines = append(lines, "external: "+note)
	}
	return lines
}

func (s *externalSummary) add(o externalSummary) {
	if s.kind == cupti.ExternalCorrelationKindInvalid {
		s.kind = o.kind
	}
	s.kinds = s.kinds.add(o.kinds)
	s.records += o.records
	s.matched += o.matched
	s.ambiguous += o.ambiguous
	s.duration += o.duration
	s.matchedNanos += o.matchedNanos
}

// externalCoverage joins one journal's device records to its
// external-correlation records and reports how much of the device work the
// join covered. The join is cuptiactivity.ExternalCoverage, which applies
// MatchExternal's exact rules; this function only supplies the scope.
//
// Scope: the capture session is both session and run, and the journal member
// is the process generation -- one journal is written by one generation, which
// is the identity a native correlation ID is unique within.
//
// One namespace is reported, the lowest-numbered one present, because
// MatchExternal joins one namespace per call; externalKindCounts.chosen says
// why the lowest. A target pushing into two -- as PyTorch does -- has the
// second left out, and externalKindCounts.note states that it was, so a
// record we did not join is not read as a record nothing was pushed for.
func externalCoverage(session, process string, devices []traceRecord, externals []cuptiactivity.Record) externalSummary {
	var counts externalKindCounts
	for _, e := range externals {
		counts.observe(e.ExternalKind)
	}
	kind := counts.chosen()
	if kind == cupti.ExternalCorrelationKindInvalid {
		return externalSummary{}
	}
	scoped := make([]cuptiactivity.ScopedRecord, 0, len(devices)+len(externals))
	for _, d := range devices {
		scoped = append(scoped, cuptiactivity.ScopedRecord{Session: session, Run: session, Process: process, Record: d.rec})
	}
	for _, e := range externals {
		scoped = append(scoped, cuptiactivity.ScopedRecord{Session: session, Run: session, Process: process, Record: e})
	}
	rows, err := cuptiactivity.ExternalCoverage(scoped, kind)
	if err != nil {
		// Coverage is a report about the trace, never a condition on writing
		// it. The identity above is supplied by this function, so the only
		// way here is an overflow in the totals, which costs the summary line
		// and nothing else.
		return externalSummary{}
	}
	s := externalSummary{kind: kind, kinds: counts}
	for _, r := range rows {
		s.records += r.Total.Records
		s.duration += r.Total.Duration
		s.matched += r.Matched.Records
		s.matchedNanos += r.Matched.Duration
		s.ambiguous += r.Ambiguous.Records
	}
	return s
}

// nvtxSummary is what the NVTX tier contributed, reported alongside the trace
// because an unpaired marker is drawn as nothing and would otherwise be an
// invisible loss.
type nvtxSummary struct {
	ranges   int // START/END pairs drawn as ranges
	points   int // instantaneous markers drawn as zero-length slices
	unpaired int // halves with no partner, drawn as nothing
	unknown  int // markers carrying none of the three flags
}

func (s *nvtxSummary) add(o nvtxSummary) {
	s.ranges += o.ranges
	s.points += o.points
	s.unpaired += o.unpaired
	s.unknown += o.unknown
}

// markerKey identifies the range an NVTX marker belongs to. The id alone is
// not enough: NVTX ids are unique within a domain, and two threads pushing
// ranges concurrently reuse them freely.
type markerKey struct {
	pid, tid, id uint32
	domain       string
}

// pairMarkers turns one journal's marker records into NVTX slices.
//
// CUPTI never reports a range. It reports a START record and an END record
// that share an id, each carrying one timestamp, so a range exists only as a
// pair and neither half carries a duration. Reading a duration off a single
// marker reads zero, which is the truth about that record and a lie about the
// range it belongs to.
//
// Pairing is LIFO within a key because NVTX ranges nest: an inner range
// closing must close the innermost open range, not the outermost one that
// happens to share its id.
//
// An unpaired half is counted and not drawn. A START whose END never arrived
// is a range that was still open when the process ended: extending it to any
// chosen end would state a duration nobody measured, and drawing it as a point
// would state that it closed immediately. Both are wrong answers where a
// counted absence is a partial one.
func pairMarkers(markers []tracePayload, out *[]traceRecord) nvtxSummary {
	var s nvtxSummary
	// Journal order is drain order, which several producer threads interleave.
	// Pairing walks time, so it has to sort by it first.
	sort.SliceStable(markers, func(i, j int) bool { return markers[i].Start < markers[j].Start })
	open := map[markerKey][]tracePayload{}
	for _, m := range markers {
		k := markerKey{pid: m.ProcessID, tid: m.ThreadID, id: m.MarkerID, domain: m.MarkerDomain}
		switch {
		case m.MarkerFlags&markerFlagStart != 0:
			open[k] = append(open[k], m)
		case m.MarkerFlags&markerFlagEnd != 0:
			stack := open[k]
			if len(stack) == 0 {
				s.unpaired++
				continue
			}
			start := stack[len(stack)-1]
			open[k] = stack[:len(stack)-1]
			paired := markerRange(start, m.End)
			paired.annotations = append(paired.annotations, perfetto.NewUintAnnotation("raw_end_record_sequence", m.sequence))
			*out = append(*out, paired)
			s.ranges++
		case m.MarkerFlags&markerFlagInstantaneous != 0:
			*out = append(*out, markerRange(m, m.Start))
			s.points++
		default:
			s.unknown++
		}
	}
	for _, stack := range open {
		s.unpaired += len(stack)
	}
	return s
}

// markerRange builds the slice for a range that opened at start and closed at
// end. The NVTX domain qualifies the name because two domains may use the same
// range name for unrelated work.
func markerRange(start tracePayload, end uint64) traceRecord {
	name := start.Name
	if start.MarkerDomain != "" {
		name = start.MarkerDomain + "/" + name
	}
	return traceRecord{
		sequence: start.sequence,
		rec: cuptiactivity.Record{
			Kind:      cupti.ActivityKindMarker,
			Name:      name,
			Start:     start.Start,
			End:       end,
			Complete:  true,
			ProcessID: start.ProcessID,
			ThreadID:  start.ThreadID,
		},
		kind:  trackNVTX,
		group: [2]uint32{start.ProcessID, start.ThreadID},
	}
}

func decodeTraceJournal(name string, reader io.Reader, maxRecords int64, out *[]traceRecord, nvtx *nvtxSummary, ext *externalSummary, stacks bool) error {
	limit := uint32(math.MaxUint32)
	if maxRecords > 0 && maxRecords < math.MaxUint32 {
		limit = uint32(maxRecords)
	}
	d := wire.NewDecoder(reader, wire.Limits{MaxRecords: limit})
	first, finalized, decoded := true, false, 0
	var noCUDA timeline.NoCUDA
	var markers []tracePayload
	var externals []cuptiactivity.Record
	var session string
	var process, generation uint64
	var clocksUnified bool
	base := len(*out)
	for {
		r, err := d.Next()
		if err != nil {
			switch {
			case errors.Is(err, io.EOF):
				if decoded == 0 {
					return fmt.Errorf("journal %s is empty, so no timeline was recorded", name)
				}
				if !finalized {
					if noCUDA.Proven() {
						// Host records are all it can hold, and they
						// belong to no device timeline.
						*out = (*out)[:base]
						return nil
					}
					return fmt.Errorf("journal %s decoded %d record(s) but never finalized, so the timeline has no known end", name, decoded)
				}
				if stats := d.Stats(); stats.Gaps > 0 || stats.Dropped > 0 {
					return fmt.Errorf("journal %s lost records (%d sequence gap(s), %d dropped), so the timeline is missing spans that would draw as idle", name, stats.Gaps, stats.Dropped)
				}
				// Before pairMarkers, so the slice holds this journal's
				// device work and nothing else.
				ext.add(externalCoverage(session, name, (*out)[base:], externals))
				nvtx.add(pairMarkers(markers, out))
				if maxRecords > 0 && int64(len(*out)) > maxRecords {
					return fmt.Errorf("total trace records exceed limit %d: %w", maxRecords, wire.ErrRecordLimit)
				}
				for i := base; i < len(*out); i++ {
					(*out)[i].journal = name
					(*out)[i].session = session
					(*out)[i].process = process
					(*out)[i].generation = generation
					(*out)[i].clocksUnified = clocksUnified
				}
				if stacks {
					if err := retainStackCorrelations((*out)[base:], externals); err != nil {
						return err
					}
				}
				return nil
			case errors.Is(err, io.ErrUnexpectedEOF):
				return fmt.Errorf("journal %s is truncated after %d record(s): the timeline ends at an unknown point", name, decoded)
			default:
				return fmt.Errorf("decode journal %s after %d record(s): %w", name, decoded, err)
			}
		}
		if first {
			d.ExpectSession(r.Session)
			session = hex.EncodeToString(r.Session[:])
			process, generation = r.Process, r.Generation
			first = false
		}
		decoded++
		if maxRecords > 0 && int64(len(*out)) > maxRecords {
			return fmt.Errorf("total trace records exceed limit %d: %w", maxRecords, wire.ErrRecordLimit)
		}
		var p tracePayload
		if len(r.Payload) > 0 && json.Unmarshal(r.Payload, &p) != nil {
			return fmt.Errorf("journal %s record %d is not a typed capture record", name, r.Sequence)
		}
		noCUDA.Observe(r.Kind, p.Subtype, r.Payload)
		p.sequence = r.Sequence
		switch r.Kind {
		case wire.KindCapability:
			var capability struct {
				ClocksUnified bool `json:"clocks_unified"`
			}
			if err := json.Unmarshal(r.Payload, &capability); err != nil {
				return err
			}
			clocksUnified = capability.ClocksUnified
		case wire.KindActivity:
			if p.Dropped > 0 {
				return fmt.Errorf("journal %s reports %d dropped record(s), so the timeline is missing activities", name, p.Dropped)
			}
			if !p.Valid || p.Start == 0 || p.End < p.Start {
				continue
			}
			if p.Subtype == "capture_range" {
				// The producer's own bookkeeping: when it started and
				// stopped collecting. It is not device work and it is not an
				// NVTX marker, so it neither becomes a slice nor enters the
				// pairing below, where its edge would be matched against a
				// marker id it does not have.
				continue
			}
			if p.Subtype == "marker" {
				// Held back: a marker alone is a point in time, and the range
				// it belongs to does not exist until its partner arrives.
				markers = append(markers, p)
				continue
			}
			if p.Subtype == "overhead" {
				// CUPTI's own cost is paid on a host thread, and the record
				// says so: its object is a process or a thread, not a device.
				// Drawing it on a device track would put a slice the device
				// never ran beside the kernels it did, which is the same
				// claim the lane splitting above exists to avoid making.
				*out = append(*out, traceRecord{sequence: uint64(r.Sequence), rec: activityRecord(p), kind: trackHost, group: [2]uint32{p.ProcessID, p.ThreadID}})
				continue
			}
			*out = append(*out, traceRecord{sequence: uint64(r.Sequence), rec: activityRecord(p), kind: trackDevice, group: [2]uint32{p.Device, p.Stream}})
		case wire.KindAPI:
			if p.Kind == "graph_lifecycle" || p.Kind == "dependency" || p.Kind == "memory_policy" {
				continue
			}
			if p.Start == 0 || p.End < p.Start {
				continue
			}
			*out = append(*out, traceRecord{sequence: uint64(r.Sequence), rec: apiRecord(p), kind: trackHost, group: [2]uint32{p.ProcessID, p.ThreadID}})
		case wire.KindExternal:
			// No interval, so no slice: this record is a link, not work. It
			// is kept for the join below, which is the only thing that can
			// say which framework scope the device records ran under.
			if p.Correlation == 0 {
				continue
			}
			externals = append(externals, cuptiactivity.Record{
				Kind:          cupti.ActivityKindExternalCorrelation,
				CorrelationID: p.Correlation,
				ExternalKind:  cupti.ExternalCorrelationKind(p.ExternalKind),
				ExternalID:    p.ExternalID,
			})
		case wire.KindFinalized:
			finalized = true
			if p.Dropped > 0 {
				return fmt.Errorf("journal %s finalized with %d dropped record(s), so the timeline is missing activities", name, p.Dropped)
			}
		}
	}
}

func activityRecord(p tracePayload) cuptiactivity.Record {
	r := cuptiactivity.Record{
		Name:          p.Name,
		Start:         p.Start,
		End:           p.End,
		Complete:      true,
		CorrelationID: p.Correlation,
		DeviceID:      p.Device,
		ContextID:     p.Context,
		StreamID:      p.Stream,
		Bytes:         p.Bytes,
		GraphID:       p.GraphID,
		GraphNodeID:   p.GraphNodeID,
	}
	switch p.Subtype {
	case "memcpy":
		r.Kind = cupti.ActivityKindMemcpy
		r.CopyKind = cupti.ActivityMemcpyKind(p.CopyKind)
		r.RuntimeCorrelationID = p.RuntimeCorr
	case "overhead":
		// Each of the three arms below exists for the reason the memset arm
		// below it does: the default is the kernel arm, so a subtype it does
		// not name is not rejected but relabelled, and the export is then
		// complete, well formed, and wrong in a way no count shows.
		r.Kind = cupti.ActivityKindOverhead
		r.ProcessID, r.ThreadID = p.ProcessID, p.ThreadID
		if r.Name == "" {
			r.Name = cupti.ActivityOverheadKind(p.OverheadKind).String()
		}
	case "graph_trace":
		r.Kind = cupti.ActivityKindGraphTrace
		if r.Name == "" {
			r.Name = "cudaGraphLaunch"
		}
	case "sync":
		r.Kind = cupti.ActivityKindSynchronization
		r.Synchronization = cupti.ActivitySynchronizationType(p.SyncType)
		r.CudaEventID = p.CudaEventID
		if r.Name == "" {
			r.Name = r.Synchronization.String()
		}
	case "memset":
		// Without this arm a memset falls to the default below and is
		// exported as a kernel, carrying whatever the zero grid and block
		// decode to. The default is deliberately permissive -- it is the
		// kernel arm -- so a new device subtype has to claim itself here.
		r.Kind = cupti.ActivityKindMemset
	default:
		r.Kind = cupti.ActivityKindConcurrentKernel
		r.Queued, r.Submitted = p.Queued, p.Submitted
		r.Grid = [3]int32{int32(p.GridX), int32(p.GridY), int32(p.GridZ)}
		r.Block = [3]int32{int32(p.BlockX), int32(p.BlockY), int32(p.BlockZ)}
		r.RegistersPerThread = uint16(p.Registers)
		r.DynamicSharedMemory = int32(p.DynShared)
	}
	return r
}

func apiRecord(p tracePayload) cuptiactivity.Record {
	r := cuptiactivity.Record{
		Kind:          cupti.ActivityKindRuntime,
		Name:          p.Domain,
		Start:         p.Start,
		End:           p.End,
		Complete:      true,
		CorrelationID: p.Correlation,
		ProcessID:     p.ProcessID,
		ThreadID:      p.ThreadID,
		CallbackID:    p.CallbackID,
		ReturnValue:   p.ReturnValue,
	}
	if p.Domain == "driver" {
		r.Kind = cupti.ActivityKindDriver
	}
	if r.Name == "" {
		r.Name = r.Kind.String()
	}
	return r
}

// writeTrace emits descriptors and slices, returning the track count.
//
// Lanes exist because a Perfetto track is a stack: two slices that overlap on
// one track are drawn as one containing the other. CUPTI reports concurrent
// kernels, and a memcpy can overlap a kernel on the same stream id, so an
// overlapping activity goes on a numbered lane of the same group instead. That
// is a drawing decision, not a reinterpretation: no record moves in time.
func writeTrace(w io.Writer, records []traceRecord) (int, error) {
	origin := uint64(math.MaxUint64)
	for _, t := range records {
		if t.rec.Start < origin {
			origin = t.rec.Start
		}
	}
	return writeTraceOrigin(w, records, origin, 0, "")
}

// writeTraceOrigin keeps each record on its original clock relative to origin.
// uuidBase separates independent exports; label makes their arm visible.
func writeTraceOrigin(w io.Writer, records []traceRecord, origin, uuidBase uint64, label string, identities ...traceExportIdentity) (int, error) {
	uuids, names := assignTracks(records)
	pw := perfetto.NewWriter(w)
	pw.SequenceID = 100 + (uuidBase >> 32)
	if label != "" {
		pw.TimestampClockID = perfetto.ClockBoottime
	}
	identity := traceExportIdentity{}
	if len(identities) > 0 {
		identity = identities[0]
	}
	first := make(map[uint64]traceRecord)
	for i, t := range records {
		if _, ok := first[uuids[i]]; !ok {
			first[uuids[i]] = t
		}
	}
	for i, name := range names {
		if err := rawTrack(pw, uuidBase+uint64(i+1), label+name, first[uint64(i+1)], identity); err != nil {
			return 0, fmt.Errorf("write track descriptor: %w", err)
		}
	}
	for i, t := range records {
		annotations := traceAnnotations(t)
		if identity.sha256 != "" {
			annotations = append(annotations, perfetto.NewStringAnnotation("cudaprof_capture_sha256", identity.sha256), perfetto.NewStringAnnotation("cudaprof_arm", identity.arm))
		}
		annotations = append(annotations, perfetto.NewStringAnnotation("cudaprof_activity_kind", traceRecordKind(t)))
		stackID, err := internLanguageStack(pw, t.stack)
		if t.sample != nil {
			stackID, err = internLanguageFrames(pw, t.sample.Frames)
		}
		if err != nil {
			return len(names), fmt.Errorf("intern record %d stack: %w", i, err)
		}
		annotations = append(annotations, perfetto.NewUintAnnotation("cudaprof_stack_iid", stackID))
		if t.sample != nil {
			if err := pw.WriteInstantWithCallstack(uuidBase+uuids[i], t.rec.Start-origin, t.rec.Name, "host_stack_sample", annotations, stackID); err != nil {
				return len(names), fmt.Errorf("write sample %d: %w", i, err)
			}
			continue
		}
		if err := cuptiactivity.WriteRecordWithCallstack(pw, uuidBase+uuids[i], origin, t.rec, stackID, annotations...); err != nil {
			return len(names), fmt.Errorf("write record %d: %w", i, err)
		}
	}
	return len(names), nil
}

// assignTracks sorts records into drawing order and gives each one a track,
// returning the per-record track uuid and the track names in uuid order
// (uuid i+1 is names[i]).
//
// Lanes exist because a Perfetto track is a stack: two slices that overlap on
// one track are drawn as one CONTAINING the other, which asserts a nesting the
// hardware never reported. CUPTI reports concurrent kernels, and a memcpy can
// overlap a kernel carrying the same stream id, so an overlapping activity
// goes on a numbered lane of the same group instead. That is a drawing
// decision and not a reinterpretation: no record moves in time, and a record
// stays in its own device/stream or pid/thread group.
func assignTracks(records []traceRecord) ([]uint64, []string) {
	sort.SliceStable(records, func(i, j int) bool {
		a, b := records[i], records[j]
		if a.kind != b.kind {
			return a.kind < b.kind
		}
		if a.group != b.group {
			if a.group[0] != b.group[0] {
				return a.group[0] < b.group[0]
			}
			return a.group[1] < b.group[1]
		}
		if a.journal != b.journal {
			return a.journal < b.journal
		}
		if traceRole(a) != traceRole(b) {
			return traceRole(a) < traceRole(b)
		}
		if a.rec.ContextID != b.rec.ContextID {
			return a.rec.ContextID < b.rec.ContextID
		}
		return a.rec.Start < b.rec.Start
	})
	type lane struct {
		uuid uint64
		end  uint64
	}
	var (
		lanes = map[struct {
			kind          trackKind
			group         [2]uint32
			journal, role string
			context       uint32
		}][]lane{}
		uuids = make([]uint64, len(records))
		names []string
	)
	for i, t := range records {
		key := struct {
			kind          trackKind
			group         [2]uint32
			journal, role string
			context       uint32
		}{t.kind, t.group, t.journal, traceRole(t), t.rec.ContextID}
		group := lanes[key]
		slot := -1
		for j := range group {
			if group[j].end <= t.rec.Start {
				slot = j
				break
			}
		}
		if slot < 0 {
			names = append(names, trackName(t.kind, t.group, len(group)))
			group = append(group, lane{uuid: uint64(len(names))})
			slot = len(group) - 1
		}
		group[slot].end = t.rec.End
		lanes[key] = group
		uuids[i] = group[slot].uuid
	}
	return uuids, names
}

func trackName(kind trackKind, group [2]uint32, lane int) string {
	var name string
	switch kind {
	case trackHost:
		name = fmt.Sprintf("pid %d thread %d", group[0], group[1])
	case trackNVTX:
		name = fmt.Sprintf("nvtx pid %d thread %d", group[0], group[1])
	case trackStackSample:
		name = fmt.Sprintf("host stack samples pid %d thread %d", group[0], group[1])
	default:
		name = fmt.Sprintf("device %d stream %d", group[0], group[1])
	}
	if lane > 0 {
		name += fmt.Sprintf(" (overlap %d)", lane)
	}
	return name
}

// demangleTraceRecords changes display labels only, caching each unique symbol.
func demangleTraceRecords(records []traceRecord) {
	names := make(map[string]string)
	for i := range records {
		r := &records[i].rec
		if r.Kind != cupti.ActivityKindConcurrentKernel {
			continue
		}
		name, ok := names[r.Name]
		if !ok {
			name = iteration.KernelName(r.Name)
			names[r.Name] = name
		}
		r.DisplayName = name
	}
}
