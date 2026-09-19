package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
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
		return gapOptions{}, errors.New("usage: cudaprof gaps [--warmup-ms=N] [--max-records=N] [--journal-bytes=N] [--diff_base=base.pprof] [--json] capture.pprof")
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
Summed per-kernel device time cannot see this axis: a change that moves work
out of inter-kernel idle and into the kernels raises device-active time while
the run gets faster, and a device-time-only diff scores that as a regression.

Options:
  --diff_base=FILE  compare against this capture and report deltas
  --warmup-ms=N     warmup window excluded from the steady-state section (default 200)
  --max-records=N   maximum journal records to decode (default 1048576)
  --journal-bytes=N maximum decompressed bytes per journal (default 536870912)
  --json            write the report as JSON
`)
}

// ---------------------------------------------------------------------------
// journal selection and decoding

// gapJournal is one decoded capture journal member of the capture.
type gapJournal struct {
	Name       string `json:"name"`
	Compressed bool   `json:"compressed"`
	Bytes      int    `json:"bytes"`
	Records    int    `json:"records"`
}

// gapStatus is the part of capture-status.json that says whether the
// per-record detail of this capture is present at all.
type gapStatus struct {
	OmittedFiles    []captureOmittedFile `json:"omitted_files"`
	DetailedRecords string               `json:"detailed_records"`
	RunCompleteness string               `json:"run_completeness"`
	DroppedRecords  uint64               `json:"dropped_records"`
	ActivityRecords uint64               `json:"activity_records"`
	ActivitySamples int                  `json:"activity_samples"`
	// RecordLoss is the producer's own breakdown of what it never wrote. It
	// names the mechanism, which is what a refusal has to report: the one
	// remedy a reader can reach for, --max-records, addresses none of them.
	RecordLoss *captureLoss `json:"record_loss"`
	// UnqualifiedGenerations counts generations that never finalized. Their
	// activities are absent from the journal entirely, so a timeline built
	// from what remains has holes no drop count reports.
	UnqualifiedGenerations uint64 `json:"unqualified_generations"`
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

// requireJournals returns the journal members of a capture, refusing every
// capture whose per-record detail is not wholly present. The refusal names the
// missing artifact and why the analysis cannot proceed; it never falls back to
// an estimate, and never analyzes the part that survived.
func requireJournals(cap *cudacapture.Capture) (map[string][]byte, gapStatus, error) {
	status, haveStatus := readGapStatus(cap)
	var omitted []string
	for _, f := range status.OmittedFiles {
		if cudacapture.IsJournalName(f.Name) {
			omitted = append(omitted, fmt.Sprintf("%s (%d bytes, sha256 %s: %s)", f.Name, f.Bytes, f.SHA256, f.Reason))
		}
	}
	if len(omitted) != 0 {
		sort.Strings(omitted)
		return nil, status, fmt.Errorf("capture omits the per-record journal, so the device timeline cannot be reconstructed: %s; capture-status.json reports run_completeness=%q and detailed_records=%q, and gap analysis needs the omitted journal itself, not the pprof projection%s",
			strings.Join(omitted, ", "), status.RunCompleteness, status.DetailedRecords, budgetRemedy(status.OmittedFiles))
	}
	journals := make(map[string][]byte)
	for name, data := range cap.Files {
		if cudacapture.IsJournalName(name) {
			journals[name] = data
		}
	}
	if len(journals) == 0 {
		detail := "capture-status.json is absent"
		if haveStatus {
			detail = fmt.Sprintf("capture-status.json reports run_completeness=%q and detailed_records=%q", status.RunCompleteness, status.DetailedRecords)
		}
		return nil, status, fmt.Errorf("capture contains no capture.journal member, so the device timeline cannot be reconstructed: %s; pprof samples are aggregates and carry no timestamps", detail)
	}
	if status.DroppedRecords > 0 {
		return nil, status, fmt.Errorf("capture dropped %d record(s) and retained %d, so the journal is missing activities the gaps would span: a duty cycle computed from the remainder would be wrong, not approximate%s", status.DroppedRecords, status.ActivityRecords, truncationRemedy(status))
	}
	// A run can lose whole generations without dropping a record: a generation
	// that never finalized contributes nothing, and its absence is invisible in
	// a drop count. capture_success qualifies the target generation alone and
	// stays true through both, which is why this reads run_completeness.
	if status.RunCompleteness == runPartial {
		return nil, status, fmt.Errorf("capture reports run_completeness=%q with %d unqualified generation(s), so activities the idle gaps would span were never recorded: capture_success qualifies the target generation alone and does not report this", status.RunCompleteness, status.UnqualifiedGenerations)
	}
	return journals, status, nil
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

// gapRecords is everything gap analysis reads out of the journals.
type gapRecords struct {
	activities []gapActivity
	apis       map[uint32]gapAPI
	journals   []gapJournal

	// Spans that explain idle rather than occupy the device. None of the
	// three is in activities: see the exclusions in decodeJournal.
	overheads []gapInterval
	graphs    []gapActivity
	syncs     []gapInterval
	rawSyncs  []rawSync

	// graphNodes counts device records that name the graph they ran as a
	// node of. It is what decides whether excluding the replay envelope is
	// sound: the exclusion exists because the envelope spans those records,
	// and with none of them present it spans nothing.
	graphNodes int

	// contextDevice maps a CUPTI context to the device the records that carry
	// both reported it on. A synchronization record carries a context and no
	// device, so this is the only thing that can place one on a device
	// timeline. Context 0 is never entered: it is the value a record that
	// named no context leaves behind, and treating it as a context would
	// resolve every unattributable sync onto whichever device happened to
	// leave a zero there.
	contextDevice map[uint32]uint32

	// Syncs whose context no device-carrying record ever named. They are
	// reported, not placed: assigning them to device 0 would put the
	// explanation for one device's idle on another device's timeline.
	unresolvedSyncs int
	unresolvedNanos int64
}

type gapActivity struct {
	device      uint32
	start, end  int64
	correlation uint32

	// Carried for consumers that need the launch itself rather than only its
	// place on the timeline. They are unset for the spans that are not kernel
	// launches, and nothing in the gap analysis reads them.
	subtype     string
	name        string
	stream      uint32
	graphID     uint32
	grid, block [3]int
}

// gapInterval is one span on one device's timeline. Overhead spans carry no
// device -- CUPTI's cost is charged to a host thread -- and are held with
// device left at zero and applied to every device, which the overlay below
// states rather than hides.
type gapInterval struct {
	device     uint32
	start, end int64
}

// rawSync is a synchronization span before its device is known.
type rawSync struct {
	context    uint32
	start, end int64
}

type gapAPI struct{ start, end int64 }

// gapPayload is the subset of a typed journal payload gap analysis reads.
type gapPayload struct {
	Kind         string `json:"kind"`
	Subtype      string `json:"subtype"`
	Name         string `json:"name"`
	Stream       uint32 `json:"stream"`
	GridX        uint32 `json:"grid_x"`
	GridY        uint32 `json:"grid_y"`
	GridZ        uint32 `json:"grid_z"`
	BlockX       uint32 `json:"block_x"`
	BlockY       uint32 `json:"block_y"`
	BlockZ       uint32 `json:"block_z"`
	State        string `json:"state"`
	Domain       string `json:"domain"`
	Valid        bool   `json:"valid"`
	Start        uint64 `json:"start"`
	End          uint64 `json:"end"`
	Device       uint32 `json:"device"`
	Context      uint32 `json:"context"`
	Correlation  uint32 `json:"correlation"`
	GraphID      uint32 `json:"graph_id"`
	OverheadKind uint32 `json:"overhead_kind"`
	SyncType     uint32 `json:"sync_type"`
	Dropped      uint64 `json:"dropped"`
}

// decodeJournals decodes every journal member. A journal whose frames do not
// decode to completion, or that never finalized, is refused: both leave a
// timeline whose end is unknown, and a truncated prefix is indistinguishable
// from a short run once the records are counted.
func decodeJournals(journals map[string][]byte, maxRecords, maxBytes int64) (gapRecords, error) {
	out := gapRecords{apis: make(map[uint32]gapAPI), contextDevice: make(map[uint32]uint32)}
	names := make([]string, 0, len(journals))
	for name := range journals {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		reader, compressed, err := decompressJournal(journals[name], maxBytes)
		if err != nil {
			return gapRecords{}, fmt.Errorf("read journal %s: %w", name, err)
		}
		summary := gapJournal{Name: name, Compressed: compressed}
		if err := decodeJournal(name, reader, maxRecords, &out, &summary); err != nil {
			_ = reader.Close()
			return gapRecords{}, err
		}
		if err := reader.Close(); err != nil {
			return gapRecords{}, fmt.Errorf("close journal %s: %w", name, err)
		}
		summary.Bytes = int(reader.n)
		if summary.Bytes == 0 {
			return gapRecords{}, fmt.Errorf("journal %s is empty, so no device timeline was recorded", name)
		}
		out.journals = append(out.journals, summary)
	}
	if len(out.activities) == 0 {
		return gapRecords{}, errors.New("journals hold no valid activity records, so there is no device timeline to analyze")
	}
	// Resolution runs once every journal has been read, not per journal: the
	// kernel that names a context and the sync that shares it need not have
	// been drained into the same journal member.
	out.resolveSyncDevices()
	return out, nil
}

// resolveSyncDevices places each synchronization span on the device its
// context belongs to.
//
// A sync that cannot be resolved is counted and dropped rather than defaulted.
// Device 0 is the wrong answer twice over: on a single-device run it is
// indistinguishable from a correct one, so the defect never shows; on a
// multi-device run it explains one device's idle with another device's wait.
func (r *gapRecords) resolveSyncDevices() {
	for _, y := range r.rawSyncs {
		device, ok := r.contextDevice[y.context]
		if !ok {
			r.unresolvedSyncs++
			r.unresolvedNanos += y.end - y.start
			continue
		}
		r.syncs = append(r.syncs, gapInterval{device: device, start: y.start, end: y.end})
	}
}

// decompressJournal returns the journal frames, transparently decompressing a
// gzip-stored member. Detection is by content, so a journal is read the same
// way whether or not its name carries the .gz suffix.
func decompressJournal(data []byte, maxBytes int64) (*boundedJournalReader, bool, error) {
	if maxBytes <= 0 {
		return nil, false, fmt.Errorf("invalid journal byte limit %d", maxBytes)
	}
	if len(data) < 2 || data[0] != 0x1f || data[1] != 0x8b {
		return newBoundedJournalReader(io.NopCloser(bytes.NewReader(data)), maxBytes), false, nil
	}
	zr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, true, fmt.Errorf("open gzip journal: %w", err)
	}
	return newBoundedJournalReader(zr, maxBytes), true, nil
}

type boundedJournalReader struct {
	io.ReadCloser
	limit int64
	n     int64
}

func newBoundedJournalReader(r io.ReadCloser, limit int64) *boundedJournalReader {
	return &boundedJournalReader{ReadCloser: r, limit: limit}
}

func (r *boundedJournalReader) Read(p []byte) (int, error) {
	remaining := r.limit - r.n
	if remaining < 0 {
		return 0, fmt.Errorf("journal exceeds %d bytes decompressed", r.limit)
	}
	if remaining == 0 {
		var one [1]byte
		n, err := r.ReadCloser.Read(one[:])
		if n > 0 {
			return 0, fmt.Errorf("journal exceeds %d bytes decompressed", r.limit)
		}
		return 0, err
	}
	if int64(len(p)) > remaining {
		p = p[:remaining]
	}
	n, err := r.ReadCloser.Read(p)
	r.n += int64(n)
	return n, err
}

func decodeJournal(name string, reader io.Reader, maxRecords int64, out *gapRecords, summary *gapJournal) error {
	limitsMaxRecords := uint32(math.MaxUint32)
	if maxRecords > 0 && maxRecords < math.MaxUint32 {
		limitsMaxRecords = uint32(maxRecords)
	}
	d := wire.NewDecoder(reader, wire.Limits{MaxRecords: limitsMaxRecords})
	first, finalized := true, false
	for {
		r, err := d.Next()
		if err != nil {
			switch {
			case errors.Is(err, io.EOF):
				if summary.Records == 0 {
					return fmt.Errorf("journal %s is empty, so no device timeline was recorded", name)
				}
				if !finalized {
					return fmt.Errorf("journal %s decoded %d record(s) but never finalized, so the recorded timeline has no known end", name, summary.Records)
				}
				if stats := d.Stats(); stats.Gaps > 0 || stats.Dropped > 0 {
					return fmt.Errorf("journal %s lost records (%d sequence gap(s), %d dropped), so activities the idle gaps would span are absent", name, stats.Gaps, stats.Dropped)
				}
				return nil
			case errors.Is(err, io.ErrUnexpectedEOF):
				return fmt.Errorf("journal %s is truncated after %d record(s): its final frame is incomplete, so the timeline ends at an unknown point", name, summary.Records)
			default:
				return fmt.Errorf("decode journal %s after %d record(s): %w", name, summary.Records, err)
			}
		}
		if first {
			d.ExpectSession(r.Session)
			first = false
		}
		summary.Records++
		var payload gapPayload
		if len(r.Payload) > 0 && json.Unmarshal(r.Payload, &payload) != nil {
			return fmt.Errorf("journal %s record %d is not a typed capture record", name, r.Sequence)
		}
		switch r.Kind {
		case recordActivity:
			if payload.Dropped > 0 {
				return fmt.Errorf("journal %s reports %d dropped record(s), so the timeline is missing activities", name, payload.Dropped)
			}
			if !payload.Valid || payload.Start == 0 || payload.End < payload.Start || payload.End > math.MaxInt64 {
				continue
			}
			// Neither an NVTX marker nor a capture-range edge is device
			// work: the first is a host annotation, the second is the
			// producer stating when it started and stopped collecting.
			// Counting either would inflate Activities, stretch SpanNanos to
			// whichever one sits furthest from the device timeline, and split
			// a real device gap in two at its timestamp.
			if payload.Subtype == "marker" || payload.Subtype == "capture_range" {
				continue
			}
			// The same exclusion, for three more kinds and a sharper reason.
			// This filter is a denylist -- anything it does not name counts
			// as device busy -- so a span that merely explains idle, added
			// upstream and not named here, would be subtracted from the idle
			// it was added to explain, and nothing would fail. That is the
			// whole trap:
			//
			//   overhead    CUPTI's own cost, paid on a host thread. Counting
			//               it as busy makes the tool's headline idle number
			//               shrink in proportion to how much the tool cost.
			//   sync        the host waiting. This is the reason the device
			//               was idle; charging it to busy erases the idle it
			//               explains, exactly.
			//   graph_trace the replay envelope, which spans the kernel
			//               records of the nodes replayed inside it. It would
			//               both double-count their device time and paper
			//               over the scheduling gaps between them.
			//
			// Each is kept as a span instead and intersected with the idle
			// gaps below, which is attribution rather than occupancy.
			switch payload.Subtype {
			case "overhead":
				out.overheads = append(out.overheads, gapInterval{start: int64(payload.Start), end: int64(payload.End)})
				continue
			case "graph_trace":
				out.graphs = append(out.graphs, gapActivity{
					device:      payload.Device,
					start:       int64(payload.Start),
					end:         int64(payload.End),
					correlation: payload.Correlation,
					subtype:     payload.Subtype,
					name:        payload.Name,
					stream:      payload.Stream,
					graphID:     payload.GraphID,
				})
				continue
			case "sync":
				out.rawSyncs = append(out.rawSyncs, rawSync{context: payload.Context, start: int64(payload.Start), end: int64(payload.End)})
				continue
			}
			// A record that carries both a context and the device it ran on
			// is the only source for the map a synchronization record has to
			// be resolved through. First writer wins: a context belongs to
			// one device for its lifetime, and a later disagreement is a
			// reused id, not a migration.
			if payload.GraphID != 0 {
				out.graphNodes++
			}
			if payload.Context != 0 {
				if _, ok := out.contextDevice[payload.Context]; !ok {
					out.contextDevice[payload.Context] = payload.Device
				}
			}
			out.activities = append(out.activities, gapActivity{
				device:      payload.Device,
				start:       int64(payload.Start),
				end:         int64(payload.End),
				correlation: payload.Correlation,
				subtype:     payload.Subtype,
				name:        payload.Name,
				stream:      payload.Stream,
				graphID:     payload.GraphID,
				grid:        [3]int{int(payload.GridX), int(payload.GridY), int(payload.GridZ)},
				block:       [3]int{int(payload.BlockX), int(payload.BlockY), int(payload.BlockZ)},
			})
		case recordAPI:
			if payload.End < payload.Start || payload.End > math.MaxInt64 {
				continue
			}
			out.apis[payload.Correlation] = gapAPI{start: int64(payload.Start), end: int64(payload.End)}
		case recordFinalized:
			finalized = true
			if payload.Dropped > 0 {
				return fmt.Errorf("journal %s finalized with %d dropped record(s), so the timeline is missing activities", name, payload.Dropped)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// analysis

// GapReport is the whole result of one gaps run.
type GapReport struct {
	Capture    string       `json:"capture"`
	Journals   []gapJournal `json:"journals"`
	Activities int          `json:"activities"`
	APIRecords int          `json:"api_records"`
	Correlated float64      `json:"correlation_match_pct"`
	WarmupMS   float64      `json:"warmup_ms"`
	Devices    []GapDevice  `json:"devices"`

	// Synchronization spans whose context no device-carrying record named, so
	// no device timeline could claim them. They are stated here rather than
	// folded into device 0: an explanation placed on the wrong device is
	// worse than one that says it has no device.
	UnresolvedSyncs     int   `json:"unresolved_sync_records"`
	UnresolvedSyncNanos int64 `json:"unresolved_sync_ns"`

	// OpaqueGraphReplays reports that the capture held graph replay envelopes
	// and no device record naming a graph id, so the replays are counted as
	// device activity from their envelopes alone. The idle inside them is not
	// observable and is reported as busy.
	OpaqueGraphReplays bool `json:"opaque_graph_replays,omitempty"`
}

// GapDevice holds the whole-run and steady-state timelines of one device.
type GapDevice struct {
	Device      uint32    `json:"device"`
	Overall     *GapStats `json:"overall"`
	SteadyState *GapStats `json:"steady_state,omitempty"`
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
	journals, _, err := requireJournals(cap)
	if err != nil {
		return nil, err
	}
	journalBytes := opts.journalBytes
	if journalBytes == 0 {
		journalBytes = cudacapture.MaxCaptureInputBytes
	}
	records, err := decodeJournals(journals, opts.maxRecords, journalBytes)
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
	graphsAreOpaque := len(records.graphs) != 0 && records.graphNodes == 0
	if graphsAreOpaque {
		records.activities = append(records.activities, records.graphs...)
		records.graphs = nil
	}
	sort.SliceStable(records.activities, func(i, j int) bool {
		a, b := records.activities[i], records.activities[j]
		if a.start != b.start {
			return a.start < b.start
		}
		return a.end < b.end
	})
	matched := 0
	for _, a := range records.activities {
		if _, ok := records.apis[a.correlation]; ok {
			matched++
		}
	}
	report := &GapReport{
		Journals:            records.journals,
		Activities:          len(records.activities),
		APIRecords:          len(records.apis),
		Correlated:          float64(matched) / float64(len(records.activities)) * 100,
		WarmupMS:            opts.warmupMS,
		OpaqueGraphReplays:  graphsAreOpaque,
		UnresolvedSyncs:     records.unresolvedSyncs,
		UnresolvedSyncNanos: records.unresolvedNanos,
	}
	byDevice := make(map[uint32][]gapActivity)
	for _, a := range records.activities {
		byDevice[a.device] = append(byDevice[a.device], a)
	}
	devices := make([]uint32, 0, len(byDevice))
	for d := range byDevice {
		devices = append(devices, d)
	}
	sort.Slice(devices, func(i, j int) bool { return devices[i] < devices[j] })
	warmupNanos := int64(opts.warmupMS * 1e6)
	for _, d := range devices {
		acts := byDevice[d]
		// Overhead spans carry no device, so every device is offered all of
		// them; the sync and graph spans are filtered to this one.
		spans := gapSpans{
			overhead: records.overheads,
			sync:     spansForDevice(records.syncs, d),
			graph:    graphSpansForDevice(records.graphs, d),
		}
		entry := GapDevice{Device: d, Overall: computeGapStats(acts, records.apis, "overall", spans)}
		t0 := acts[0].start
		var steady []gapActivity
		for _, a := range acts {
			if a.start-t0 > warmupNanos {
				steady = append(steady, a)
			}
		}
		if len(steady) != 0 && len(steady) != len(acts) {
			entry.SteadyState = computeGapStats(steady, records.apis, "steady-state", spans)
		}
		report.Devices = append(report.Devices, entry)
	}
	return report, nil
}

// mergedInterval is one span of contiguous device occupancy. first is the
// activity that opened the interval; its correlation is what the next gap is
// classified against.
type mergedInterval struct {
	start, end int64
	first      gapActivity
}

// mergeIntervals collapses overlapping and touching activity intervals into
// the spans during which the device was busy. Activities must be sorted by
// start then end. A new activity that starts at or before the running end
// extends it; anything later opens a new interval.
func mergeIntervals(acts []gapActivity) []mergedInterval {
	var merged []mergedInterval
	for _, a := range acts {
		if len(merged) == 0 {
			merged = append(merged, mergedInterval{start: a.start, end: a.end, first: a})
			continue
		}
		last := &merged[len(merged)-1]
		if a.start <= last.end {
			if a.end > last.end {
				last.end = a.end
			}
			continue
		}
		merged = append(merged, mergedInterval{start: a.start, end: a.end, first: a})
	}
	return merged
}

// gapSpans are the non-device spans offered to one device's timeline.
type gapSpans struct {
	overhead, sync, graph []gapInterval
}

// graphSpansForDevice is spansForDevice over the graph envelopes, which are
// kept as activities so they can be promoted onto the timeline whole.
func graphSpansForDevice(spans []gapActivity, device uint32) []gapInterval {
	var out []gapInterval
	for _, s := range spans {
		if s.device == device {
			out = append(out, gapInterval{device: s.device, start: s.start, end: s.end})
		}
	}
	return out
}

func spansForDevice(spans []gapInterval, device uint32) []gapInterval {
	var out []gapInterval
	for _, s := range spans {
		if s.device == device {
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
func overlayIdle(gaps, spans []gapInterval) (int64, int) {
	if len(gaps) == 0 || len(spans) == 0 {
		return 0, 0
	}
	var total int64
	var touched int
	for _, g := range gaps {
		var pieces []gapInterval
		for _, s := range spans {
			start, end := max64(g.start, s.start), min64(g.end, s.end)
			if end > start {
				pieces = append(pieces, gapInterval{start: start, end: end})
			}
		}
		if len(pieces) == 0 {
			continue
		}
		touched++
		sort.Slice(pieces, func(i, j int) bool { return pieces[i].start < pieces[j].start })
		cur := pieces[0]
		for _, p := range pieces[1:] {
			if p.start <= cur.end {
				if p.end > cur.end {
					cur.end = p.end
				}
				continue
			}
			total += cur.end - cur.start
			cur = p
		}
		total += cur.end - cur.start
	}
	return total, touched
}

func computeGapStats(acts []gapActivity, apis map[uint32]gapAPI, label string, spans gapSpans) *GapStats {
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
	var idle []gapInterval
	for i := 0; i+1 < len(merged); i++ {
		prevEnd, nextStart := merged[i].end, merged[i+1].start
		gap := nextStart - prevEnd
		if gap <= 0 {
			continue
		}
		gaps = append(gaps, gap)
		idle = append(idle, gapInterval{start: prevEnd, end: nextStart})
		api, ok := apis[merged[i+1].first.correlation]
		if !ok {
			classGaps[classUnmatched] = append(classGaps[classUnmatched], gap)
			stats.Attribution.UnattributedNanos += gap
			continue
		}
		switch {
		case api.end <= prevEnd:
			classGaps[classHostAhead] = append(classGaps[classHostAhead], gap)
			stats.Attribution.HardwareNanos += gap
		case api.start <= prevEnd:
			classGaps[classHostDuring] = append(classGaps[classHostDuring], gap)
			stats.Attribution.DriverSubmitNanos += min64(api.end-prevEnd, gap)
			stats.Attribution.HardwareNanos += max64(0, nextStart-api.end)
		default:
			classGaps[classHostBehind] = append(classGaps[classHostBehind], gap)
			starved := min64(api.start-prevEnd, gap)
			stats.Attribution.HostBehindNanos += starved
			stats.Attribution.DriverSubmitNanos += min64(api.end-api.start, max64(0, gap-starved))
			stats.Attribution.HardwareNanos += max64(0, nextStart-api.end)
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
		fmt.Fprintf(&b, "  journal %s (%s, %d bytes, %d records)\n", j.Name, form, j.Bytes, j.Records)
	}
	fmt.Fprintf(&b, "  activities %d  api records %d  correlated %.1f%%\n", r.Activities, r.APIRecords, r.Correlated)
	if r.UnresolvedSyncs > 0 {
		fmt.Fprintf(&b, "  %d synchronization record(s) totalling %s could not be placed on a device: no record named their context\n", r.UnresolvedSyncs, millis(r.UnresolvedSyncNanos))
	}
	if r.OpaqueGraphReplays {
		fmt.Fprintf(&b, "  graph replays are opaque: the capture holds replay envelopes and no device record naming a graph id, so each replay is counted as activity from its envelope alone. The idle between nodes of a graph is not observable here and is reported as busy.\n")
	}
	for _, d := range r.Devices {
		for _, s := range []*GapStats{d.Overall, d.SteadyState} {
			if s == nil {
				continue
			}
			fmt.Fprintf(&b, "\ndevice %d, %s", d.Device, s.Label)
			if s.Label == "steady-state" {
				fmt.Fprintf(&b, " (first %g ms excluded)", r.WarmupMS)
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

// journalStreamEvents reconstructs the device timeline from the per-record
// journal, for consumers that otherwise read the pprof projection.
//
// The projection is an aggregate: it carries no activity timestamps, so a
// reader built on it reports that the capture does not have them. On a capture
// whose journal is attached that statement is false about the file, and the
// recommendation it produced -- re-capture with timestamps -- sends someone
// back under GPU lock for work already on disk. Everything needed is in the
// member gaps reads by default.
//
// The second return says the replay envelopes are the only record of their
// graphs (see analyzeGaps), so a caller can carry that caveat forward instead
// of silently reporting envelope time as kernel time.
func journalStreamEvents(cap *cudacapture.Capture) (events []streamEvent, graphsOpaque bool, err error) {
	journals, _, err := requireJournals(cap)
	if err != nil {
		return nil, false, err
	}
	records, err := decodeJournals(journals, controllerMaxJournalRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return nil, false, err
	}
	acts := records.activities
	if len(records.graphs) != 0 && records.graphNodes == 0 {
		graphsOpaque = true
		acts = append(acts, records.graphs...)
	}
	for _, a := range acts {
		if a.end <= a.start {
			continue
		}
		subtype := a.subtype
		if subtype == "" {
			subtype = "kernel"
		}
		events = append(events, streamEvent{
			Subtype:     subtype,
			Name:        a.name,
			Stream:      a.stream,
			DurationNs:  uint64(a.end - a.start),
			Start:       uint64(a.start),
			End:         uint64(a.end),
			Correlation: a.correlation,
			GraphID:     a.graphID,
			Grid:        a.grid,
			Block:       a.block,
		})
	}
	return events, graphsOpaque, nil
}

// journalAPIRecords reads the host launch API records from the per-record
// journal, indexed by correlation id.
//
// Captures written through compactCaptureStatus omit the per-record arrays, so
// a reader that looks only in capture-status.json finds none and reports that
// the capture has no host launch timestamps. The journal that same capture
// carries held 124834 of them on the run that found this, all correlated.
func journalAPIRecords(cap *cudacapture.Capture) (map[uint32]apiCall, error) {
	journals, _, err := requireJournals(cap)
	if err != nil {
		return nil, err
	}
	records, err := decodeJournals(journals, controllerMaxJournalRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return nil, err
	}
	res := make(map[uint32]apiCall, len(records.apis))
	for corr, api := range records.apis {
		if corr == 0 || api.start <= 0 || api.end < api.start {
			continue
		}
		res[corr] = apiCall{Start: uint64(api.start), End: uint64(api.end)}
	}
	return res, nil
}
