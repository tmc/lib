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
	capturePath string
	warmupMS    float64
	maxRecords  int64
	json        bool
	limits      cudacapture.Limits
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

var errGapsHelp = errors.New("gaps: help requested")

func parseGapsArgs(rawArgs []string) (gapOptions, error) {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return gapOptions{}, err
	}
	opts := gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords, limits: limits}
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
		case strings.HasPrefix(arg, "--max-records=") || strings.HasPrefix(arg, "-max-records="):
			v, err := strconv.ParseInt(arg[strings.IndexByte(arg, '=')+1:], 10, 64)
			if err != nil || v <= 0 {
				return gapOptions{}, fmt.Errorf("invalid max records: %s", arg)
			}
			opts.maxRecords = v
		case strings.HasPrefix(arg, "-"):
			return gapOptions{}, fmt.Errorf("unknown flag: %s", arg)
		case opts.capturePath == "":
			opts.capturePath = arg
		default:
			return gapOptions{}, errors.New("multiple capture files specified")
		}
	}
	if opts.capturePath == "" {
		return gapOptions{}, errors.New("usage: cudaprof gaps [--warmup-ms=N] [--max-records=N] [--json] capture.pprof")
	}
	return opts, nil
}

func printGapsHelp() {
	fmt.Fprint(os.Stdout, `Usage: cudaprof gaps [options] capture.pprof

Reconstructs the device timeline from the capture journals attached to a
capture, merges kernel and memcpy intervals across streams per device, and
reports GPU duty cycle, the distribution of inter-activity idle gaps, and the
host concurrency state at each gap.

The analysis needs the per-record journal. pprof samples are aggregates and
carry no timestamps, so a capture whose journal was omitted by the artifact
budget, truncated, or lossy is refused instead of analyzed.

Options:
  --warmup-ms=N     warmup window excluded from the steady-state section (default 200)
  --max-records=N   maximum journal records to decode (default 1048576)
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
	ActivitySamples int                  `json:"activity_samples"`
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
		return nil, status, fmt.Errorf("capture omits the per-record journal, so the device timeline cannot be reconstructed: %s; capture-status.json reports run_completeness=%q and detailed_records=%q, and gap analysis needs the omitted journal itself, not the pprof projection",
			strings.Join(omitted, ", "), status.RunCompleteness, status.DetailedRecords)
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
		return nil, status, fmt.Errorf("capture dropped %d record(s), so the journal is missing activities the gaps would span: a duty cycle computed from the remainder would be wrong, not approximate", status.DroppedRecords)
	}
	return journals, status, nil
}

// gapRecords is everything gap analysis reads out of the journals.
type gapRecords struct {
	activities []gapActivity
	apis       map[uint32]gapAPI
	journals   []gapJournal
}

type gapActivity struct {
	device      uint32
	start, end  int64
	correlation uint32
}

type gapAPI struct{ start, end int64 }

// gapPayload is the subset of a typed journal payload gap analysis reads.
type gapPayload struct {
	Kind        string `json:"kind"`
	Subtype     string `json:"subtype"`
	State       string `json:"state"`
	Domain      string `json:"domain"`
	Valid       bool   `json:"valid"`
	Start       uint64 `json:"start"`
	End         uint64 `json:"end"`
	Device      uint32 `json:"device"`
	Correlation uint32 `json:"correlation"`
	Dropped     uint64 `json:"dropped"`
}

// decodeJournals decodes every journal member. A journal whose frames do not
// decode to completion, or that never finalized, is refused: both leave a
// timeline whose end is unknown, and a truncated prefix is indistinguishable
// from a short run once the records are counted.
func decodeJournals(journals map[string][]byte, maxRecords int64) (gapRecords, error) {
	out := gapRecords{apis: make(map[uint32]gapAPI)}
	names := make([]string, 0, len(journals))
	for name := range journals {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		data, compressed, err := decompressJournal(journals[name])
		if err != nil {
			return gapRecords{}, fmt.Errorf("read journal %s: %w", name, err)
		}
		if len(data) == 0 {
			return gapRecords{}, fmt.Errorf("journal %s is empty, so no device timeline was recorded", name)
		}
		summary := gapJournal{Name: name, Compressed: compressed, Bytes: len(data)}
		if err := decodeJournal(name, data, maxRecords, &out, &summary); err != nil {
			return gapRecords{}, err
		}
		out.journals = append(out.journals, summary)
	}
	if len(out.activities) == 0 {
		return gapRecords{}, errors.New("journals hold no valid activity records, so there is no device timeline to analyze")
	}
	return out, nil
}

// decompressJournal returns the journal frames, transparently decompressing a
// gzip-stored member. Detection is by content, so a journal is read the same
// way whether or not its name carries the .gz suffix.
func decompressJournal(data []byte) ([]byte, bool, error) {
	if len(data) < 2 || data[0] != 0x1f || data[1] != 0x8b {
		return data, false, nil
	}
	zr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, true, fmt.Errorf("open gzip journal: %w", err)
	}
	defer zr.Close()
	// The decompressed size is not bounded by the embedded size, so cap it at
	// the bound cudacapture applies to any single decoded input rather than
	// letting a malformed or hostile member expand without limit.
	plain, err := io.ReadAll(io.LimitReader(zr, cudacapture.MaxCaptureInputBytes+1))
	if err != nil {
		return nil, true, fmt.Errorf("decompress journal: %w", err)
	}
	if int64(len(plain)) > cudacapture.MaxCaptureInputBytes {
		return nil, true, fmt.Errorf("journal exceeds %d bytes decompressed", int64(cudacapture.MaxCaptureInputBytes))
	}
	return plain, true, nil
}

func decodeJournal(name string, data []byte, maxRecords int64, out *gapRecords, summary *gapJournal) error {
	limitsMaxRecords := uint32(math.MaxUint32)
	if maxRecords > 0 && maxRecords < math.MaxUint32 {
		limitsMaxRecords = uint32(maxRecords)
	}
	d := wire.NewDecoder(bytes.NewReader(data), wire.Limits{MaxRecords: limitsMaxRecords})
	first, finalized := true, false
	for {
		r, err := d.Next()
		if err != nil {
			switch {
			case errors.Is(err, io.EOF):
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
			out.activities = append(out.activities, gapActivity{
				device:      payload.Device,
				start:       int64(payload.Start),
				end:         int64(payload.End),
				correlation: payload.Correlation,
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
	records, err := decodeJournals(journals, opts.maxRecords)
	if err != nil {
		return nil, err
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
		Journals:   records.journals,
		Activities: len(records.activities),
		APIRecords: len(records.apis),
		Correlated: float64(matched) / float64(len(records.activities)) * 100,
		WarmupMS:   opts.warmupMS,
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
		entry := GapDevice{Device: d, Overall: computeGapStats(acts, records.apis, "overall")}
		t0 := acts[0].start
		var steady []gapActivity
		for _, a := range acts {
			if a.start-t0 > warmupNanos {
				steady = append(steady, a)
			}
		}
		if len(steady) != 0 && len(steady) != len(acts) {
			entry.SteadyState = computeGapStats(steady, records.apis, "steady-state")
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

func computeGapStats(acts []gapActivity, apis map[uint32]gapAPI, label string) *GapStats {
	if len(acts) == 0 {
		return nil
	}
	merged := mergeIntervals(acts)
	stats := &GapStats{Label: label, Activities: len(acts), MergedIntervals: len(merged)}
	stats.SpanNanos = acts[len(acts)-1].end - acts[0].start
	for _, m := range merged {
		stats.ActiveNanos += m.end - m.start
	}
	classGaps := map[string][]int64{}
	var gaps []int64
	for i := 0; i+1 < len(merged); i++ {
		prevEnd, nextStart := merged[i].end, merged[i+1].start
		gap := nextStart - prevEnd
		if gap <= 0 {
			continue
		}
		gaps = append(gaps, gap)
		api, ok := apis[merged[i+1].first.correlation]
		if !ok {
			classGaps[classUnmatched] = append(classGaps[classUnmatched], gap)
			stats.Attribution.HardwareNanos += gap
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
}

func millis(ns int64) string { return fmt.Sprintf("%.3f ms", float64(ns)/1e6) }
func micros(ns int64) string { return fmt.Sprintf("%.3f", float64(ns)/1e3) }

func pct(part, whole int64) float64 {
	if whole <= 0 {
		return 0
	}
	return float64(part) / float64(whole) * 100
}
