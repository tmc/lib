package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Gap analysis reconstructs the device timeline a pprof profile cannot carry.
// pprof samples are aggregates: they sum durations and drop the start and stop
// timestamps, so inter-kernel idle time is not recoverable from them. The
// per-record detail with timestamps lives in the capture journals attached to
// the capture, and every number below is computed from those records alone.
//
// A journal that is absent, omitted by the artifact budget, truncated, or
// lossy makes the timeline unreconstructable. Such a capture is refused rather
// than analyzed, because a duty cycle computed from a surviving prefix reads
// exactly like one computed from the whole run.

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
		status, _ := readGapStatus(cap)
		switch {
		case errors.Is(err, timeline.ErrJournalOmitted):
			return nil, fmt.Errorf("%w%s", err, budgetRemedy(status.OmittedFiles))
		case errors.Is(err, timeline.ErrRecordsDropped):
			return nil, fmt.Errorf("%w%s", err, truncationRemedy(status))
		}
		return nil, err
	}
	return journals, nil
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

// GapReport is the whole result of one gaps run.
type GapReport struct {
	Capture    string             `json:"capture"`
	Journals   []timeline.Journal `json:"journals"`
	Activities int                `json:"activities"`
	APIRecords int                `json:"api_records"`
	Correlated float64            `json:"correlation_match_pct"`
	WarmupMS   float64            `json:"warmup_ms"`
	// WarmupUntil, when set, replaces WarmupMS: the steady-state window
	// begins at the first kernel whose name matches it.
	WarmupUntil string      `json:"warmup_until,omitempty"`
	Devices     []GapDevice `json:"devices"`

	// Synchronization spans whose context no device-carrying record named, so
	// no device timeline could claim them. They are stated here rather than
	// folded into device 0: an explanation placed on the wrong device is
	// worse than one that says it has no device.
	UnresolvedSyncs     int   `json:"unresolved_sync_records"`
	UnresolvedSyncNanos int64 `json:"unresolved_sync_ns"`

	// Kernels is the per-kernel census: how many times each kernel ran and
	// how much device time it held, both taken from the same records.
	Kernels []KernelCensus `json:"kernels,omitempty"`

	// UnnamedKernelRecords counts kernel activities the journal carried
	// without a name. They are stated rather than dropped: a census whose
	// rows silently omit a population reports a total that does not add up
	// to the device-active time beside it.
	UnnamedKernelRecords int `json:"unnamed_kernel_records,omitempty"`

	// OpaqueGraphReplays reports that the capture held graph replay envelopes
	// and no device record naming a graph id, so the replays are counted as
	// device activity from their envelopes alone. The idle inside them is not
	// observable and is reported as busy.
	OpaqueGraphReplays bool `json:"opaque_graph_replays,omitempty"`

	// GraphTraceMode is the graph trace mode the agent reported: node records
	// per graph node, or one envelope per replay. Empty is UNREAD: the agent
	// predates the field or the capture carries no capability record.
	GraphTraceMode graphTraceMode `json:"graph_trace_mode,omitempty"`

	// Cudaprof is the build that computed the report.
	Cudaprof buildStamp `json:"cudaprof"`
}

// GapDevice holds the whole-run and steady-state timelines of one device.
type GapDevice struct {
	Device      uint32    `json:"device"`
	Overall     *GapStats `json:"overall"`
	SteadyState *GapStats `json:"steady_state,omitempty"`

	// SteadyActivities counts the activities past the warmup. SteadyState is
	// omitted both when the warmup never ended -- zero here -- and when it
	// excluded nothing, and those two must not read alike: the first means
	// the steady-state window does not exist.
	SteadyActivities int `json:"steady_state_activities"`
}

// GapStats describes one merged device timeline.
type GapStats struct {
	Label           string          `json:"label"`
	Activities      int             `json:"activities"`
	MergedIntervals int             `json:"merged_intervals"`
	SpanNanos       int64           `json:"span_ns"`
	ActiveNanos     int64           `json:"active_ns"`
	IdleNanos       int64           `json:"idle_ns"`
	DutyCyclePct    float64         `json:"duty_cycle_pct"`
	GapCount        int             `json:"gap_count"`
	Distribution    GapDistribution `json:"distribution_ns"`
	Concurrency     []GapClass      `json:"host_concurrency"`
	Attribution     GapAttribution  `json:"idle_attribution_ns"`
	Overlay         GapOverlay      `json:"idle_overlay_ns"`
}

// GapOverlay names the idle time that something in the journal explains
// directly: CUPTI was flushing, the host was waiting on a synchronize, or the
// device was between the nodes of a graph replay.
//
// It overlays GapAttribution rather than extending it. The three numbers there
// partition idle and sum to IdleNanos; these do not. One gap can be both host
// behind and spent inside a cudaStreamSynchronize, one overhead span can cover
// several gaps, and a graph replay envelope can contain both. Adding an
// overlay figure to an attribution figure therefore double counts, and adding
// two overlay figures to each other can exceed idle.
type GapOverlay struct {
	// CuptiOverheadNanos is idle time covered by a CUPTI overhead span: time
	// the tool cost the run, which the tool would otherwise report as the
	// program leaving the device idle. Overhead records carry a host thread
	// rather than a device, so the same spans are applied to every device.
	CuptiOverheadNanos int64 `json:"cupti_overhead"`
	CuptiOverheadGaps  int   `json:"cupti_overhead_gap_count"`
	// SyncNanos is idle covered by a synchronization span on this device.
	SyncNanos int64 `json:"synchronization"`
	SyncGaps  int   `json:"synchronization_gap_count"`
	// GraphReplayNanos is idle inside a graph replay envelope: the device was
	// between nodes of a launched graph, which is graph scheduling cost and
	// not the host failing to keep up.
	GraphReplayNanos int64 `json:"graph_replay"`
	GraphReplayGaps  int   `json:"graph_replay_gap_count"`
}

// GapDistribution reports the shape of the idle gaps, not only their total. A
// duty cycle built from a few long synchronization stalls and the same duty
// cycle built from thousands of uniform dispatch gaps are different
// engineering problems, and only the distribution separates them.
type GapDistribution struct {
	Min    int64   `json:"min"`
	P25    int64   `json:"p25"`
	Median int64   `json:"median"`
	P75    int64   `json:"p75"`
	P90    int64   `json:"p90"`
	P99    int64   `json:"p99"`
	Max    int64   `json:"max"`
	Mean   float64 `json:"mean"`
}

// GapClass is one host concurrency class with its count and its own idle time
// and idle-gap distribution, so a class that is rare but long is not hidden by
// a class that is frequent and short.
type GapClass struct {
	Name      string  `json:"name"`
	Meaning   string  `json:"meaning"`
	Count     int     `json:"count"`
	IdleNanos int64   `json:"idle_ns"`
	Median    int64   `json:"median_ns"`
	Mean      float64 `json:"mean_ns"`
}

// GapAttribution splits idle time by where it was spent, as distinct from
// which class the gap fell into.
type GapAttribution struct {
	HostBehindNanos   int64 `json:"host_behind"`
	DriverSubmitNanos int64 `json:"driver_submission"`
	HardwareNanos     int64 `json:"hardware_queue"`
	// UnattributedNanos is idle before an activity whose correlation id
	// matched no API record. It is its own arm because there is no mechanism
	// to name: the launch that would say where the time went was not found.
	// It was previously added to HardwareNanos, which reported "we could not
	// find the launch" as "hardware queue latency" -- a specific mechanism
	// claim, and the one a reader acts on. Any future record kind arriving
	// without a correlation id lands here rather than inventing a cause.
	UnattributedNanos int64 `json:"unattributed"`
}

const (
	classHostAhead  = "Host Ahead"
	classHostDuring = "Host During"
	classHostBehind = "Host Behind"
	classUnmatched  = "Unmatched"
)

var classMeaning = map[string]string{
	classHostAhead:  "launch returned before the previous op ended: idle is hardware queue latency",
	classHostDuring: "previous op ended inside the launch call: the GPU waited on the driver",
	classHostBehind: "launch started after the previous op ended: the GPU was starved by the host",
	classUnmatched:  "no API record correlated to the next op: idle is unattributed",
}

func analyzeGaps(cap *cudacapture.Capture, opts gapOptions) (*GapReport, error) {
	journals, err := requireJournals(cap)
	if err != nil {
		return nil, err
	}
	journalBytes := opts.journalBytes
	if journalBytes == 0 {
		journalBytes = cudacapture.MaxCaptureInputBytes
	}
	records, err := timeline.Decode(journals, opts.maxRecords, journalBytes)
	if err != nil {
		return nil, err
	}
	// A graph replay envelope is excluded from the device timeline because it
	// spans the records of the nodes replayed inside it, and counting both
	// would charge that device time twice. When CUPTI emitted the envelopes
	// and no record inside them -- no device record names a graph id -- the
	// envelope spans nothing, so excluding it deletes the whole replay from
	// the timeline and reports its duration as device idle. On one capture
	// that read 14.54% duty cycle where the work was about 71%.
	//
	// The capture carries its own discriminator, so this is decided rather
	// than assumed. With no node records the envelope is the only evidence
	// the graph ran and is counted as the activity it is. What is then lost
	// is the idle BETWEEN nodes, which is counted as busy -- a bounded error
	// in the honest direction, unlike deleting the replay outright.
	//
	// The size of that error is measured, not argued. Two captures of one
	// workload from a single lock hold, one with CUDA graphs disabled so the
	// same work resolves into real kernel records and no promotion happens:
	//
	//	graphs on    duty 72.24%   active 3845.843 ms   169 envelopes
	//	graphs off   duty 70.83%   active 3738.689 ms     0 envelopes
	//
	// 1.4 points apart, the promoted arm high by 107 ms, which is the idle
	// between nodes that this counts as busy. The control needs a second arm
	// of the same workload and so is not reproducible from any one capture.
	graphsAreOpaque := len(records.Graphs) != 0 && records.GraphNodes == 0
	if graphsAreOpaque {
		records.Activities = append(records.Activities, records.Graphs...)
		records.Graphs = nil
	}
	sort.SliceStable(records.Activities, func(i, j int) bool {
		a, b := records.Activities[i], records.Activities[j]
		if a.Start != b.Start {
			return a.Start < b.Start
		}
		return a.End < b.End
	})
	matched := 0
	for _, a := range records.Activities {
		if _, ok := records.APIs[a.Correlation]; ok {
			matched++
		}
	}
	report := &GapReport{
		Journals:            records.Journals,
		Activities:          len(records.Activities),
		APIRecords:          len(records.APIs),
		Correlated:          float64(matched) / float64(len(records.Activities)) * 100,
		WarmupMS:            opts.warmupMS,
		WarmupUntil:         opts.warmupUntil,
		OpaqueGraphReplays:  graphsAreOpaque,
		Cudaprof:            currentBuild(),
		UnresolvedSyncs:     records.UnresolvedSyncs,
		UnresolvedSyncNanos: records.UnresolvedNanos,
	}
	if status, ok := readGapStatus(cap); ok {
		if m, ok := captureGraphTraceMode(status.Capability); ok {
			report.GraphTraceMode = m
		}
	}
	report.Kernels, report.UnnamedKernelRecords = kernelCensus(records.Activities)
	byDevice := make(map[uint32][]timeline.Activity)
	for _, a := range records.Activities {
		byDevice[a.Device] = append(byDevice[a.Device], a)
	}
	devices := make([]uint32, 0, len(byDevice))
	for d := range byDevice {
		devices = append(devices, d)
	}
	sort.Slice(devices, func(i, j int) bool { return devices[i] < devices[j] })
	warmupNanos := int64(opts.warmupMS * 1e6)
	var warmupUntil *regexp.Regexp
	if opts.warmupUntil != "" {
		re, err := regexp.Compile(opts.warmupUntil)
		if err != nil {
			return nil, fmt.Errorf("warmup kernel pattern: %w", err)
		}
		warmupUntil = re
	}
	for _, d := range devices {
		acts := byDevice[d]
		// Overhead spans carry no device, so every device is offered all of
		// them; the sync and graph spans are filtered to this one.
		spans := gapSpans{
			overhead: records.Overheads,
			sync:     spansForDevice(records.Syncs, d),
			graph:    graphSpansForDevice(records.Graphs, d),
		}
		steady := steadyActivities(acts, warmupNanos, warmupUntil)
		entry := GapDevice{Device: d, Overall: computeGapStats(acts, records.APIs, "overall", spans), SteadyActivities: len(steady)}
		if len(steady) != 0 && len(steady) != len(acts) {
			entry.SteadyState = computeGapStats(steady, records.APIs, "steady-state", spans)
		}
		report.Devices = append(report.Devices, entry)
	}
	return report, nil
}

// steadyActivities returns the activities past the warmup: those starting
// at or after the first kernel matching until, or, when until is nil, those
// starting more than warmupNanos after the first activity. acts is sorted by
// start.
func steadyActivities(acts []timeline.Activity, warmupNanos int64, until *regexp.Regexp) []timeline.Activity {
	if until != nil {
		for i, a := range acts {
			if strings.EqualFold(a.Subtype, "kernel") && until.MatchString(a.Name) {
				return acts[i:]
			}
		}
		return nil
	}
	t0 := acts[0].Start
	for i, a := range acts {
		if a.Start-t0 > warmupNanos {
			return acts[i:]
		}
	}
	return nil
}

// describeWarmup describes what opts excludes from the steady state.
func describeWarmup(opts gapOptions) string {
	return (&GapReport{WarmupMS: opts.warmupMS, WarmupUntil: opts.warmupUntil}).warmup()
}

// warmup describes what the steady-state window excludes.
func (r *GapReport) warmup() string {
	if r.WarmupUntil != "" {
		return fmt.Sprintf("activity before the first kernel matching %q", r.WarmupUntil)
	}
	return fmt.Sprintf("first %g ms", r.WarmupMS)
}

// KernelCensus is one kernel's launch count, device time, and place in the
// timeline in one capture.
type KernelCensus struct {
	Name        string `json:"name"`
	Launches    int    `json:"launches"`
	DeviceNanos int64  `json:"device_ns"`

	// MedianOrderFrac is where this kernel's median launch sat in the ORDER
	// of the capture's kernel launches: 0 at the first launch of the run, 1
	// at the last.
	//
	// A launch count says whether the work ran, never when. A change that
	// relocates work -- the same launches, in a different phase of the run --
	// leaves every count identical, and a census reading counts alone reports
	// the kernel as untouched.
	//
	// Order, not elapsed time. Position measured as a fraction of the device
	// span moves whenever any duration changes: a kernel that got slower
	// stretches the span it is divided by, and its own median start slides
	// backwards through a run it never moved in. The launch order cannot be
	// perturbed by a duration at all, so what remains is what the census is
	// asking about -- whether the same work sits in the same part of the run.
	MedianOrderFrac float64 `json:"median_launch_order_frac"`
}

// kernelCensus counts launches and device time per kernel name.
//
// Both numbers come from the same records on purpose. A census that takes
// counts from one source and durations from another can report a kernel whose
// count held still while its time moved when what actually happened is that
// the two sources disagree about which records exist -- and that reading is
// the one a caller acts on, because a launch count that did not move says the
// kernel itself did not change.
func kernelCensus(acts []timeline.Activity) ([]KernelCensus, int) {
	byName := make(map[string]*KernelCensus)
	ranks := make(map[string][]int)
	unnamed := 0
	// acts arrives sorted by start, so the running count is the launch order.
	// Unnamed kernel records advance it too: they are launches, and leaving
	// them out would move every later kernel's rank by however many of them
	// the capture happened to carry.
	launches := 0
	for _, a := range acts {
		if !strings.EqualFold(a.Subtype, "kernel") {
			continue
		}
		rank := launches
		launches++
		if a.Name == "" {
			unnamed++
			continue
		}
		c := byName[a.Name]
		if c == nil {
			c = &KernelCensus{Name: a.Name}
			byName[a.Name] = c
		}
		c.Launches++
		c.DeviceNanos += a.End - a.Start
		ranks[a.Name] = append(ranks[a.Name], rank)
	}
	out := make([]KernelCensus, 0, len(byName))
	for _, c := range byName {
		// The median, not the mean: one launch at the far end of the run
		// moves a mean past the shift being measured and leaves the median
		// where the bulk of the launches actually are.
		if launches > 1 {
			r := ranks[c.Name]
			c.MedianOrderFrac = float64(r[len(r)/2]) / float64(launches-1)
		}
		out = append(out, *c)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].DeviceNanos != out[j].DeviceNanos {
			return out[i].DeviceNanos > out[j].DeviceNanos
		}
		return out[i].Name < out[j].Name
	})
	return out, unnamed
}

// mergedInterval is one span of contiguous device occupancy. first is the
// activity that opened the interval; its correlation is what the next gap is
// classified against.
type mergedInterval struct {
	start, end int64
	first      timeline.Activity
}

// mergeIntervals collapses overlapping and touching activity intervals into
// the spans during which the device was busy. Activities must be sorted by
// start then end. A new activity that starts at or before the running end
// extends it; anything later opens a new interval.
func mergeIntervals(acts []timeline.Activity) []mergedInterval {
	var merged []mergedInterval
	for _, a := range acts {
		if len(merged) == 0 {
			merged = append(merged, mergedInterval{start: a.Start, end: a.End, first: a})
			continue
		}
		last := &merged[len(merged)-1]
		if a.Start <= last.end {
			if a.End > last.end {
				last.end = a.End
			}
			continue
		}
		merged = append(merged, mergedInterval{start: a.Start, end: a.End, first: a})
	}
	return merged
}

// gapSpans are the non-device spans offered to one device's timeline.
type gapSpans struct {
	overhead, sync, graph []timeline.Span
}

// graphSpansForDevice is spansForDevice over the graph envelopes, which are
// kept as activities so they can be promoted onto the timeline whole.
func graphSpansForDevice(spans []timeline.Activity, device uint32) []timeline.Span {
	var out []timeline.Span
	for _, s := range spans {
		if s.Device == device {
			out = append(out, timeline.Span{Device: s.Device, Start: s.Start, End: s.End})
		}
	}
	return out
}

func spansForDevice(spans []timeline.Span, device uint32) []timeline.Span {
	var out []timeline.Span
	for _, s := range spans {
		if s.Device == device {
			out = append(out, s)
		}
	}
	return out
}

// overlayIdle reports how much of the idle in gaps is covered by spans, and
// how many gaps at least one span touched.
//
// Overlapping spans are counted once: the answer is the measure of the union
// of the intersections, not the sum of them. A CUPTI buffer flush nested
// inside a graph replay would otherwise be charged twice to one gap, and a
// figure derived that way can exceed the idle it is a part of.
func overlayIdle(gaps, spans []timeline.Span) (int64, int) {
	if len(gaps) == 0 || len(spans) == 0 {
		return 0, 0
	}
	var total int64
	var touched int
	for _, g := range gaps {
		var pieces []timeline.Span
		for _, s := range spans {
			start, end := max64(g.Start, s.Start), min64(g.End, s.End)
			if end > start {
				pieces = append(pieces, timeline.Span{Start: start, End: end})
			}
		}
		if len(pieces) == 0 {
			continue
		}
		touched++
		sort.Slice(pieces, func(i, j int) bool { return pieces[i].Start < pieces[j].Start })
		cur := pieces[0]
		for _, p := range pieces[1:] {
			if p.Start <= cur.End {
				if p.End > cur.End {
					cur.End = p.End
				}
				continue
			}
			total += cur.End - cur.Start
			cur = p
		}
		total += cur.End - cur.Start
	}
	return total, touched
}

func computeGapStats(acts []timeline.Activity, apis map[uint32]timeline.Span, label string, spans gapSpans) *GapStats {
	if len(acts) == 0 {
		return nil
	}
	merged := mergeIntervals(acts)
	stats := &GapStats{Label: label, Activities: len(acts), MergedIntervals: len(merged)}
	// The span runs to the greatest end, which is not the end of the last
	// activity by start order: acts is sorted by (start, end), so a long
	// activity containing a shorter one sorts first and its end is the larger.
	// Taking acts[len-1].end there reports a span shorter than the busy time
	// measured inside it, and "% of span" then exceeds 100. merged already
	// carries a running maximum, so its last end is the true one.
	stats.SpanNanos = merged[len(merged)-1].end - merged[0].start
	for _, m := range merged {
		stats.ActiveNanos += m.end - m.start
	}
	classGaps := map[string][]int64{}
	var gaps []int64
	var idle []timeline.Span
	for i := 0; i+1 < len(merged); i++ {
		prevEnd, nextStart := merged[i].end, merged[i+1].start
		gap := nextStart - prevEnd
		if gap <= 0 {
			continue
		}
		gaps = append(gaps, gap)
		idle = append(idle, timeline.Span{Start: prevEnd, End: nextStart})
		api, ok := apis[merged[i+1].first.Correlation]
		if !ok {
			classGaps[classUnmatched] = append(classGaps[classUnmatched], gap)
			stats.Attribution.UnattributedNanos += gap
			continue
		}
		switch {
		case api.End <= prevEnd:
			classGaps[classHostAhead] = append(classGaps[classHostAhead], gap)
			stats.Attribution.HardwareNanos += gap
		case api.Start <= prevEnd:
			classGaps[classHostDuring] = append(classGaps[classHostDuring], gap)
			stats.Attribution.DriverSubmitNanos += min64(api.End-prevEnd, gap)
			stats.Attribution.HardwareNanos += max64(0, nextStart-api.End)
		default:
			classGaps[classHostBehind] = append(classGaps[classHostBehind], gap)
			starved := min64(api.Start-prevEnd, gap)
			stats.Attribution.HostBehindNanos += starved
			stats.Attribution.DriverSubmitNanos += min64(api.End-api.Start, max64(0, gap-starved))
			stats.Attribution.HardwareNanos += max64(0, nextStart-api.End)
		}
	}
	for _, g := range gaps {
		stats.IdleNanos += g
	}
	stats.GapCount = len(gaps)
	stats.Distribution = describeGaps(gaps)
	stats.Overlay.CuptiOverheadNanos, stats.Overlay.CuptiOverheadGaps = overlayIdle(idle, spans.overhead)
	stats.Overlay.SyncNanos, stats.Overlay.SyncGaps = overlayIdle(idle, spans.sync)
	stats.Overlay.GraphReplayNanos, stats.Overlay.GraphReplayGaps = overlayIdle(idle, spans.graph)
	if total := stats.ActiveNanos + stats.IdleNanos; total > 0 {
		stats.DutyCyclePct = float64(stats.ActiveNanos) / float64(total) * 100
	}
	for _, name := range []string{classHostAhead, classHostDuring, classHostBehind, classUnmatched} {
		g := classGaps[name]
		class := GapClass{Name: name, Meaning: classMeaning[name], Count: len(g)}
		for _, v := range g {
			class.IdleNanos += v
		}
		if len(g) != 0 {
			sorted := append([]int64(nil), g...)
			sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
			class.Median = percentileNanos(sorted, 50)
			class.Mean = float64(class.IdleNanos) / float64(len(g))
		}
		stats.Concurrency = append(stats.Concurrency, class)
	}
	return stats
}

// describeGaps reports the gap distribution. The median is reported alongside
// the mean everywhere: on sub-millisecond work one scheduling hiccup moves the
// mean past the effect being measured and leaves the median where it was.
func describeGaps(gaps []int64) GapDistribution {
	if len(gaps) == 0 {
		return GapDistribution{}
	}
	sorted := append([]int64(nil), gaps...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	var sum int64
	for _, g := range sorted {
		sum += g
	}
	return GapDistribution{
		Min:    sorted[0],
		P25:    percentileNanos(sorted, 25),
		Median: percentileNanos(sorted, 50),
		P75:    percentileNanos(sorted, 75),
		P90:    percentileNanos(sorted, 90),
		P99:    percentileNanos(sorted, 99),
		Max:    sorted[len(sorted)-1],
		Mean:   float64(sum) / float64(len(sorted)),
	}
}

// percentileNanos indexes a sorted sample. It is the nearest-rank position
// used by the reference analyzer: no interpolation, so every reported value is
// an observed gap rather than one no gap ever took.
func percentileNanos(sorted []int64, p float64) int64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(p / 100 * float64(len(sorted)-1))
	if i < 0 {
		i = 0
	}
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
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
			fmt.Fprintf(&b, "\ndevice %d, steady-state: none, the warmup (%s) never ended\n", d.Device, r.warmup())
		}
		for _, s := range []*GapStats{d.Overall, d.SteadyState} {
			if s == nil {
				continue
			}
			fmt.Fprintf(&b, "\ndevice %d, %s", d.Device, s.Label)
			if s.Label == "steady-state" {
				fmt.Fprintf(&b, " (%s excluded)", r.warmup())
			}
			b.WriteString("\n")
			writeGapStats(&b, s)
		}
	}
	return b.String()
}

func writeGapStats(b *strings.Builder, s *GapStats) {
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
