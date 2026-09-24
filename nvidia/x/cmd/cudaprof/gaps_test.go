package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// journalBuilder writes a synthetic CDW1 journal. Fixtures are built in full
// here so gap analysis is exercised without hardware or a real capture.
type journalBuilder struct {
	buf      bytes.Buffer
	sequence uint64
	session  [16]byte
}

func newJournalBuilder() *journalBuilder {
	b := &journalBuilder{}
	b.session[0] = 0x42
	b.append(recordBootstrap, map[string]any{"kind": "bootstrap"})
	b.append(recordCapability, map[string]any{"kind": "capability"})
	return b
}

func (b *journalBuilder) append(kind uint16, payload map[string]any) {
	data, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}
	b.sequence++
	rec := wire.Record{
		Kind:     kind,
		Identity: wire.Identity{Session: b.session, Process: 4242, Generation: 1},
		Sequence: b.sequence,
		Payload:  data,
	}
	if err := wire.Write(&b.buf, rec, wire.Limits{}); err != nil {
		panic(err)
	}
}

// kernel appends one kernel activity on device 0.
func (b *journalBuilder) kernel(start, end uint64, correlation uint32) *journalBuilder {
	return b.activity("kernel", 0, start, end, correlation)
}

func (b *journalBuilder) activity(subtype string, device uint32, start, end uint64, correlation uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": subtype, "valid": true,
		"start": start, "end": end, "device": device, "correlation": correlation,
	})
	return b
}

// kernelOn appends one kernel on an explicit device and context. The context
// matters only here: it is what a synchronization record is resolved through,
// and a fixture whose kernels name no context can never exercise resolution.
func (b *journalBuilder) kernelOn(device, context uint32, start, end uint64, correlation uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": "kernel", "valid": true,
		"start": start, "end": end, "device": device, "context": context, "correlation": correlation,
	})
	return b
}

// overhead appends one CUPTI overhead span. It carries no device, exactly as
// the producer writes it: CUPTI's own cost is charged to a host thread.
func (b *journalBuilder) overhead(start, end uint64, overheadKind uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": "overhead", "valid": true,
		"start": start, "end": end, "overhead_kind": overheadKind, "object_kind": 2,
		"process_id": 4242, "thread_id": 99,
	})
	return b
}

// graphNode appends one kernel that names the graph it ran as a node of,
// which is what makes a replay envelope's exclusion sound: the envelope is
// excluded because it spans records like this one.
func (b *journalBuilder) graphNode(device, context uint32, start, end uint64, correlation, graphID uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": "kernel", "valid": true,
		"start": start, "end": end, "device": device, "context": context,
		"correlation": correlation, "graph_id": graphID,
	})
	return b
}

// graphTrace appends one graph replay envelope, which does carry a device.
func (b *journalBuilder) graphTrace(device uint32, start, end uint64, graphID, correlation uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": "graph_trace", "valid": true,
		"start": start, "end": end, "device": device, "graph_id": graphID, "correlation": correlation,
	})
	return b
}

// sync appends one synchronization span. It names a context and no device,
// because CUpti_ActivitySynchronization has no device field to name.
func (b *journalBuilder) sync(context uint32, start, end uint64, syncType, correlation uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": "sync", "valid": true,
		"start": start, "end": end, "context": context, "sync_type": syncType,
		"cuda_event_id": 3, "correlation": correlation,
	})
	return b
}

// api appends one driver launch record correlated to an activity.
func (b *journalBuilder) api(start, end uint64, correlation uint32) *journalBuilder {
	b.append(recordAPI, map[string]any{
		"kind": "api", "domain": "driver", "start": start, "end": end, "correlation": correlation,
	})
	return b
}

func (b *journalBuilder) finalize() []byte {
	b.append(recordFinalized, map[string]any{"kind": "finalized", "state": "collected"})
	return append([]byte(nil), b.buf.Bytes()...)
}

// unfinalized returns the frames written so far without a finalized record.
func (b *journalBuilder) unfinalized() []byte { return append([]byte(nil), b.buf.Bytes()...) }

func gzipBytes(t *testing.T, data []byte) []byte {
	t.Helper()
	var out bytes.Buffer
	zw := gzip.NewWriter(&out)
	if _, err := zw.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}

// testCapture assembles a capture holding the given files plus a status.
func testCapture(files map[string][]byte, status map[string]any) *cudacapture.Capture {
	all := map[string][]byte{}
	for k, v := range files {
		all[k] = v
	}
	if status != nil {
		data, err := json.Marshal(map[string]any{"status": status})
		if err != nil {
			panic(err)
		}
		all["capture-status.json"] = data
	}
	return &cudacapture.Capture{
		Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}}},
		Files:   all,
		CodeUnavailable: map[string]string{
			"source": "source was not available in this capture",
			"ptx":    "PTX was not available in this capture",
			"cubin":  "CUBIN was not available in this capture",
			"sass":   "SASS was not available in this capture",
		},
	}
}

func completeStatus() map[string]any {
	return map[string]any{
		"capture_success":  true,
		"run_completeness": "complete",
		"dropped_records":  0,
		"detailed_records": "per-record activity and API arrays are not embedded in this status",
	}
}

// uniformJournal is a five-kernel run with uniform 1000 ns gaps. Every launch
// returns before the previous kernel ends, so every gap is Host Ahead.
func uniformJournal() []byte {
	b := newJournalBuilder()
	start := uint64(10_000)
	for i := uint32(1); i <= 5; i++ {
		b.kernel(start, start+4_000, i)
		b.api(start-3_000, start-2_000, i)
		start += 5_000
	}
	return b.finalize()
}

func TestGapsMergeIntervals(t *testing.T) {
	act := func(start, end int64) gapActivity { return gapActivity{start: start, end: end} }
	tests := []struct {
		name string
		in   []gapActivity
		want []mergedInterval
	}{
		{
			name: "disjoint",
			in:   []gapActivity{act(0, 10), act(20, 30)},
			want: []mergedInterval{{start: 0, end: 10}, {start: 20, end: 30}},
		},
		{
			name: "overlapping across streams",
			in:   []gapActivity{act(0, 30), act(10, 20)},
			want: []mergedInterval{{start: 0, end: 30}},
		},
		{
			name: "touching intervals merge",
			in:   []gapActivity{act(0, 10), act(10, 20)},
			want: []mergedInterval{{start: 0, end: 20}},
		},
		{
			name: "later end extends running interval",
			in:   []gapActivity{act(0, 10), act(5, 25), act(30, 40)},
			want: []mergedInterval{{start: 0, end: 25}, {start: 30, end: 40}},
		},
		{
			name: "empty",
			in:   nil,
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mergeIntervals(tt.in)
			if len(got) != len(tt.want) {
				t.Fatalf("intervals = %d, want %d (%+v)", len(got), len(tt.want), got)
			}
			for i := range got {
				if got[i].start != tt.want[i].start || got[i].end != tt.want[i].end {
					t.Errorf("interval %d = [%d,%d), want [%d,%d)", i, got[i].start, got[i].end, tt.want[i].start, tt.want[i].end)
				}
			}
		})
	}
}

func TestGapsPercentileNanos(t *testing.T) {
	sorted := []int64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	tests := []struct {
		name string
		p    float64
		want int64
	}{
		{"min", 0, 10},
		{"p25", 25, 30},
		{"median", 50, 50},
		{"p75", 75, 70},
		{"p90", 90, 90},
		{"p99", 99, 90},
		{"max", 100, 100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := percentileNanos(sorted, tt.p); got != tt.want {
				t.Errorf("percentileNanos(%g) = %d, want %d", tt.p, got, tt.want)
			}
		})
	}
	if got := percentileNanos(nil, 50); got != 0 {
		t.Errorf("percentileNanos(nil) = %d, want 0", got)
	}
	if got := percentileNanos([]int64{7}, 99); got != 7 {
		t.Errorf("single sample p99 = %d, want 7", got)
	}
}

func TestGapsDistribution(t *testing.T) {
	got := describeGaps([]int64{100, 1, 2, 3, 4})
	want := GapDistribution{Min: 1, P25: 2, Median: 3, P75: 4, P90: 4, P99: 4, Max: 100, Mean: 22}
	if got != want {
		t.Errorf("describeGaps = %+v, want %+v", got, want)
	}
	// The mean is 22 ns and the median is 3 ns: one long stall moves the mean
	// past every gap that actually occurred, and the median does not follow.
	if got.Mean <= float64(got.Max)/10 {
		t.Errorf("mean %g should be dragged by the outlier", got.Mean)
	}
}

func TestGapsHostClassification(t *testing.T) {
	// Two merged intervals: [0,100) and [200,300), so one 100 ns gap whose
	// class depends on where the second launch call sits relative to 100.
	acts := []gapActivity{{start: 0, end: 100, correlation: 1}, {start: 200, end: 300, correlation: 2}}
	tests := []struct {
		name         string
		api          *gapAPI
		wantClass    string
		wantIdle     int64
		wantBehind   int64
		wantDriver   int64
		wantHardware int64
		wantUnattrib int64
	}{
		{
			name:         "host ahead",
			api:          &gapAPI{start: 10, end: 50},
			wantClass:    classHostAhead,
			wantIdle:     100,
			wantHardware: 100,
		},
		{
			name:         "host during",
			api:          &gapAPI{start: 60, end: 140},
			wantClass:    classHostDuring,
			wantIdle:     100,
			wantDriver:   40,
			wantHardware: 60,
		},
		{
			name:         "host behind",
			api:          &gapAPI{start: 130, end: 170},
			wantClass:    classHostBehind,
			wantIdle:     100,
			wantBehind:   30,
			wantDriver:   40,
			wantHardware: 30,
		},
		{
			// Not hardware queue. A gap whose next activity correlated to
			// no API record has no known mechanism, and "hardware queue
			// latency" is a mechanism a reader acts on.
			name:         "unmatched correlation",
			api:          nil,
			wantClass:    classUnmatched,
			wantIdle:     100,
			wantUnattrib: 100,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			apis := map[uint32]gapAPI{}
			if tt.api != nil {
				apis[2] = *tt.api
			}
			s := computeGapStats(acts, apis, "overall", gapSpans{})
			if s.GapCount != 1 || s.IdleNanos != tt.wantIdle {
				t.Fatalf("gaps = %d idle = %d, want 1 gap of %d", s.GapCount, s.IdleNanos, tt.wantIdle)
			}
			for _, c := range s.Concurrency {
				wantCount, wantIdle := 0, int64(0)
				if c.Name == tt.wantClass {
					wantCount, wantIdle = 1, tt.wantIdle
				}
				if c.Count != wantCount || c.IdleNanos != wantIdle {
					t.Errorf("class %s = count %d idle %d, want count %d idle %d", c.Name, c.Count, c.IdleNanos, wantCount, wantIdle)
				}
			}
			a := s.Attribution
			if a.HostBehindNanos != tt.wantBehind || a.DriverSubmitNanos != tt.wantDriver || a.HardwareNanos != tt.wantHardware || a.UnattributedNanos != tt.wantUnattrib {
				t.Errorf("attribution = %+v, want behind %d driver %d hardware %d unattributed %d", a, tt.wantBehind, tt.wantDriver, tt.wantHardware, tt.wantUnattrib)
			}
			if sum := a.HostBehindNanos + a.DriverSubmitNanos + a.HardwareNanos + a.UnattributedNanos; sum != s.IdleNanos {
				t.Errorf("attribution arms sum to %d, want the idle they split, %d", sum, s.IdleNanos)
			}
		})
	}
}

func TestGapsDutyCycleAndDistribution(t *testing.T) {
	cap := testCapture(map[string][]byte{"capture.journal": uniformJournal()}, completeStatus())
	report, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Devices) != 1 || report.Devices[0].Device != 0 {
		t.Fatalf("devices = %+v, want one device 0", report.Devices)
	}
	s := report.Devices[0].Overall
	// Five 4000 ns kernels separated by four 1000 ns gaps.
	if s.Activities != 5 || s.MergedIntervals != 5 || s.GapCount != 4 {
		t.Fatalf("activities/intervals/gaps = %d/%d/%d, want 5/5/4", s.Activities, s.MergedIntervals, s.GapCount)
	}
	if s.SpanNanos != 24_000 || s.ActiveNanos != 20_000 || s.IdleNanos != 4_000 {
		t.Fatalf("span/active/idle = %d/%d/%d, want 24000/20000/4000", s.SpanNanos, s.ActiveNanos, s.IdleNanos)
	}
	if s.DutyCyclePct < 83.32 || s.DutyCyclePct > 83.34 {
		t.Errorf("duty cycle = %.4f, want 83.33", s.DutyCyclePct)
	}
	if s.Distribution.Median != 1000 || s.Distribution.Mean != 1000 || s.Distribution.Max != 1000 {
		t.Errorf("distribution = %+v, want uniform 1000 ns", s.Distribution)
	}
	ahead := classCount(s, classHostAhead)
	if ahead != 4 {
		t.Errorf("host ahead = %d, want 4", ahead)
	}
	if s.Attribution.HardwareNanos != 4_000 {
		t.Errorf("hardware queue = %d, want 4000", s.Attribution.HardwareNanos)
	}
	if report.Activities != 5 || report.APIRecords != 5 || report.Correlated != 100 {
		t.Errorf("report totals = %d activities, %d apis, %.1f%% correlated", report.Activities, report.APIRecords, report.Correlated)
	}
}

func classCount(s *GapStats, name string) int {
	for _, c := range s.Concurrency {
		if c.Name == name {
			return c.Count
		}
	}
	return -1
}

func TestGapsMergesStreamsPerDevice(t *testing.T) {
	// Device 0 runs two overlapping streams and device 1 runs one kernel; the
	// overlap must not count twice and the devices must not be merged.
	b := newJournalBuilder()
	b.activity("kernel", 0, 1_000, 3_000, 1)
	b.activity("kernel", 0, 2_000, 4_000, 2)
	b.activity("kernel", 0, 6_000, 7_000, 3)
	b.activity("memcpy", 1, 1_000, 2_000, 4)
	b.activity("kernel", 1, 5_000, 6_000, 5)
	data := b.finalize()

	cap := testCapture(map[string][]byte{"capture.journal": data}, completeStatus())
	report, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Devices) != 2 {
		t.Fatalf("devices = %d, want 2", len(report.Devices))
	}
	d0 := report.Devices[0].Overall
	if d0.MergedIntervals != 2 || d0.ActiveNanos != 4_000 || d0.IdleNanos != 2_000 {
		t.Errorf("device 0 = %d intervals, active %d, idle %d; want 2/4000/2000", d0.MergedIntervals, d0.ActiveNanos, d0.IdleNanos)
	}
	d1 := report.Devices[1].Overall
	if d1.MergedIntervals != 2 || d1.ActiveNanos != 2_000 || d1.IdleNanos != 3_000 {
		t.Errorf("device 1 = %d intervals, active %d, idle %d; want 2/2000/3000", d1.MergedIntervals, d1.ActiveNanos, d1.IdleNanos)
	}
}

func TestGapsCompressedJournalMatchesPlain(t *testing.T) {
	plain := uniformJournal()
	tests := []struct {
		name string
		file string
		data []byte
	}{
		{"plain", "capture.journal", plain},
		{"gzip same name", "capture.journal", gzipBytes(t, plain)},
		{"gzip dot gz name", "capture.journal.gz", gzipBytes(t, plain)},
	}
	var want string
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := testCapture(map[string][]byte{tt.file: tt.data}, completeStatus())
			report, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Journals) != 1 {
				t.Fatalf("journals = %+v", report.Journals)
			}
			if got := report.Journals[0].Compressed; got != (tt.name != "plain") {
				t.Errorf("compressed = %v for %s", got, tt.name)
			}
			if report.Journals[0].Bytes != len(plain) {
				t.Errorf("decoded bytes = %d, want %d", report.Journals[0].Bytes, len(plain))
			}
			body := reportBody(report)
			if want == "" {
				want = body
			} else if body != want {
				t.Errorf("report differs from the plain journal:\n%s\nwant\n%s", body, want)
			}
		})
	}
}

// reportBody renders everything but the journal identity, so a plain and a
// compressed journal can be compared directly.
func reportBody(r *GapReport) string {
	data, err := json.Marshal(r.Devices)
	if err != nil {
		panic(err)
	}
	return string(data)
}

func TestGapsRefusesIncompleteDetail(t *testing.T) {
	plain := uniformJournal()
	truncated := plain[:len(plain)-20]
	corrupt := append([]byte(nil), plain...)
	corrupt[wire.HeaderSize+4] ^= 0xff // flip a payload byte past the first header

	tests := []struct {
		name  string
		files map[string][]byte
		// status is the capture-status.json body; nil writes none at all.
		status map[string]any
		want   []string
	}{
		{
			name:  "journal omitted by the artifact budget",
			files: map[string][]byte{},
			status: map[string]any{
				"run_completeness": "complete",
				"detailed_records": "per-record detail is not in this artifact",
				"omitted_files": []map[string]any{{
					"name": "capture.journal", "bytes": 812345678,
					"sha256": "abc123", "reason": "exceeds capture file budget of 268435456 bytes",
				}},
			},
			want: []string{"omits the per-record journal", "capture.journal", "812345678", "abc123", "complete"},
		},
		{
			name:   "no journal member at all",
			files:  map[string][]byte{"kernels.ptx": []byte("// PTX\n")},
			status: completeStatus(),
			want:   []string{"contains no capture.journal member", "pprof samples are aggregates"},
		},
		{
			name:   "no journal and no status",
			files:  map[string][]byte{},
			status: nil,
			want:   []string{"contains no capture.journal member", "capture-status.json is absent"},
		},
		{
			name:   "empty journal member",
			files:  map[string][]byte{"capture.journal": {}},
			status: completeStatus(),
			want:   []string{"is empty"},
		},
		{
			name:   "truncated journal",
			files:  map[string][]byte{"capture.journal": truncated},
			status: completeStatus(),
			want:   []string{"truncated", "final frame is incomplete"},
		},
		{
			name:   "truncated compressed journal",
			files:  map[string][]byte{"capture.journal": gzipBytes(t, truncated)},
			status: completeStatus(),
			want:   []string{"truncated", "final frame is incomplete"},
		},
		{
			name:   "journal never finalized",
			files:  map[string][]byte{"capture.journal": newJournalBuilder().kernel(10, 20, 1).unfinalized()},
			status: completeStatus(),
			want:   []string{"never finalized", "no known end"},
		},
		{
			name:   "corrupt frame",
			files:  map[string][]byte{"capture.journal": corrupt},
			status: completeStatus(),
			want:   []string{"decode journal"},
		},
		{
			name:  "capture dropped records",
			files: map[string][]byte{"capture.journal": plain},
			status: map[string]any{
				"run_completeness": "partial", "dropped_records": 17,
				"detailed_records": "per-record detail is attached",
			},
			want: []string{"dropped 17 record(s)", "would be wrong, not approximate"},
		},
		{
			name:   "no valid activity records",
			files:  map[string][]byte{"capture.journal": newJournalBuilder().finalize()},
			status: completeStatus(),
			want:   []string{"no valid activity records"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := testCapture(tt.files, tt.status)
			report, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
			if err == nil {
				t.Fatalf("analysis succeeded on an unanalyzable capture: %+v", report)
			}
			for _, want := range tt.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not name %q", err, want)
				}
			}
		})
	}
}

func TestGapsParseArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    gapOptions
		wantErr string
	}{
		{name: "path only", args: []string{"c.pprof"}, want: gapOptions{capturePath: "c.pprof", warmupMS: 200, maxRecords: controllerMaxJournalRecords}},
		{name: "json", args: []string{"--json", "c.pprof"}, want: gapOptions{capturePath: "c.pprof", warmupMS: 200, maxRecords: controllerMaxJournalRecords, json: true}},
		{name: "warmup", args: []string{"--warmup-ms=12.5", "c.pprof"}, want: gapOptions{capturePath: "c.pprof", warmupMS: 12.5, maxRecords: controllerMaxJournalRecords}},
		{name: "max records", args: []string{"--max-records=7", "c.pprof"}, want: gapOptions{capturePath: "c.pprof", warmupMS: 200, maxRecords: 7}},
		{name: "journal bytes", args: []string{"--journal-bytes=7", "c.pprof"}, want: gapOptions{capturePath: "c.pprof", warmupMS: 200, maxRecords: controllerMaxJournalRecords, journalBytes: 7}},
		{name: "bad journal bytes", args: []string{"--journal-bytes=0", "c.pprof"}, wantErr: "invalid journal bytes"},
		{name: "unknown flag", args: []string{"--nope", "c.pprof"}, wantErr: "unknown flag"},
		{name: "bad warmup", args: []string{"--warmup-ms=x", "c.pprof"}, wantErr: "invalid warmup"},
		{name: "warmup until", args: []string{"--warmup-until=^hc_", "c.pprof"}, want: gapOptions{capturePath: "c.pprof", warmupMS: 200, warmupUntil: "^hc_", maxRecords: controllerMaxJournalRecords}},
		{name: "bad warmup until", args: []string{"--warmup-until=(", "c.pprof"}, wantErr: "invalid warmup kernel pattern"},
		{name: "two captures", args: []string{"a.pprof", "b.pprof"}, wantErr: "multiple capture files"},
		{name: "no capture", args: []string{"--json"}, wantErr: "usage: cudaprof gaps"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGapsArgs(tt.args)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.capturePath != tt.want.capturePath || got.warmupMS != tt.want.warmupMS || got.warmupUntil != tt.want.warmupUntil ||
				got.maxRecords != tt.want.maxRecords || got.json != tt.want.json ||
				(tt.want.journalBytes != 0 && got.journalBytes != tt.want.journalBytes) {
				t.Errorf("options = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestDecompressJournalLimit(t *testing.T) {
	raw := []byte("12345678")
	var compressed bytes.Buffer
	zw := gzip.NewWriter(&compressed)
	if _, err := zw.Write(raw); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name  string
		data  []byte
		limit int64
		want  string
	}{
		{"plain exact", raw, int64(len(raw)), string(raw)},
		{"gzip exact", compressed.Bytes(), int64(len(raw)), string(raw)},
		{"plain over", raw, int64(len(raw) - 1), "journal exceeds"},
		{"gzip over", compressed.Bytes(), int64(len(raw) - 1), "journal exceeds"},
		{"invalid", raw, 0, "invalid journal byte limit"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, _, err := decompressJournal(tt.data, tt.limit)
			var got []byte
			if err == nil {
				got, err = io.ReadAll(r)
				_ = r.Close()
			}
			if strings.Contains(tt.want, "journal exceeds") || strings.Contains(tt.want, "invalid") {
				if err == nil || !strings.Contains(err.Error(), tt.want) {
					t.Fatalf("error = %v, want %q", err, tt.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.want {
				t.Errorf("journal = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGapsCLI(t *testing.T) {
	dir := t.TempDir()
	capPath := filepath.Join(dir, "gaps.pprof")
	writeTestCaptureFile(t, capPath, testCapture(map[string][]byte{"capture.journal": uniformJournal()}, completeStatus()))

	text := captureStdout(t, func() error { return runGapsCommand([]string{capPath}) })
	for _, want := range []string{"duty cycle", "gap distribution", "host concurrency", classHostAhead, "idle attribution"} {
		if !strings.Contains(text, want) {
			t.Errorf("text report missing %q:\n%s", want, text)
		}
	}

	out := captureStdout(t, func() error { return runGapsCommand([]string{"--json", capPath}) })
	var report GapReport
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatalf("json report: %v\n%s", err, out)
	}
	if len(report.Devices) != 1 || report.Devices[0].Overall.GapCount != 4 {
		t.Errorf("json report = %+v", report.Devices)
	}
	if report.Cudaprof != currentBuild() {
		t.Errorf("json report cudaprof = %+v, want %+v", report.Cudaprof, currentBuild())
	}

	help := captureStdout(t, func() error { return runGapsCommand(nil) })
	if !strings.Contains(help, "Usage: cudaprof gaps") {
		t.Errorf("help text = %q", help)
	}

	// A capture whose journal is absent must fail the command, not print zeros.
	bare := filepath.Join(dir, "bare.pprof")
	writeTestCaptureFile(t, bare, testCapture(nil, completeStatus()))
	if err := runGapsCommand([]string{bare}); err == nil {
		t.Fatal("command succeeded on a capture with no journal")
	}
}

func writeTestCaptureFile(t *testing.T, path string, cap *cudacapture.Capture) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
}

func captureStdout(t *testing.T, fn func() error) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	// Drain the pipe while fn runs. Output larger than the pipe buffer would
	// otherwise block fn's write forever, because a read that starts only
	// after fn returns never empties the buffer fn is waiting on.
	done := make(chan string, 1)
	go func() {
		data, _ := io.ReadAll(r)
		done <- string(data)
	}()
	os.Stdout = w
	runErr := fn()
	os.Stdout = orig
	w.Close()
	out := <-done
	r.Close()
	if runErr != nil {
		t.Fatalf("command: %v", runErr)
	}
	return out
}

// TestGapsIgnoresNVTXMarkers covers the tier boundary. NVTX markers share the
// activity record kind with kernels and memcpys but are host annotations, not
// device work: they carry no device time, they can be stamped anywhere on the
// host timeline, and a range's two halves land at its ends rather than across
// it.
//
// Counting one as device activity does three separate wrong things -- it
// inflates Activities, it stretches SpanNanos to the outermost marker, and it
// splits a real device gap in two at the marker's timestamp -- so the control
// is that the same fixture with and without markers reports identically.
func TestGapsIgnoresNVTXMarkers(t *testing.T) {
	kernels := func(b *journalBuilder) {
		b.activity("kernel", 0, 1_000, 3_000, 1)
		b.activity("kernel", 0, 6_000, 7_000, 2)
	}

	plain := newJournalBuilder()
	kernels(plain)

	marked := newJournalBuilder()
	kernels(marked)
	// Well outside the device window on both sides, and squarely inside the
	// 3000..6000 gap: each placement corrupts a different statistic.
	for _, m := range []struct {
		flags uint32
		ts    uint64
	}{
		{markerFlagStart, 100},
		{markerFlagEnd, 9_000},
		{markerFlagInstantaneous, 4_500},
	} {
		marked.append(recordActivity, map[string]any{
			"kind": "activity", "subtype": "marker", "valid": true, "name": "prefill",
			"marker_domain": "ds4", "marker_id": 1, "marker_flags": m.flags,
			"process_id": 100, "thread_id": 200,
			"start": m.ts, "end": m.ts,
		})
	}

	opts := gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords}
	want, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": plain.finalize()}, completeStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	got, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": marked.finalize()}, completeStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(want.Devices) != 1 || len(got.Devices) != 1 {
		t.Fatalf("devices = %d with markers, %d without; want 1 each: a marker created a device of its own", len(got.Devices), len(want.Devices))
	}
	w, g := want.Devices[0].Overall, got.Devices[0].Overall
	if g.Activities != w.Activities {
		t.Errorf("activities = %d with markers, %d without", g.Activities, w.Activities)
	}
	if g.SpanNanos != w.SpanNanos {
		t.Errorf("span = %d with markers, %d without: a host marker moved an end of the device timeline", g.SpanNanos, w.SpanNanos)
	}
	if g.MergedIntervals != w.MergedIntervals {
		t.Errorf("merged intervals = %d with markers, %d without: a marker split a device gap", g.MergedIntervals, w.MergedIntervals)
	}
	if g.ActiveNanos != w.ActiveNanos || g.IdleNanos != w.IdleNanos {
		t.Errorf("active/idle = %d/%d with markers, %d/%d without", g.ActiveNanos, g.IdleNanos, w.ActiveNanos, w.IdleNanos)
	}
}

// TestGapsIgnoresCaptureRangeEdges guards the capture-range half of the same
// filter. A range edge says when the producer started and stopped collecting;
// it is not device work, and counting one would inflate Activities and move
// the ends of the device timeline.
//
// The fixture marks the edges valid and gives them a span on purpose. The
// producer does not emit them that way today, and a fixture that copied it
// would be caught by the validity check above and would pass with the subtype
// filter deleted -- a test of nothing. Recording a range's extent is the
// obvious next thing to put in this record, and the filter has to be what
// holds when that happens, not the absence of a field.
func TestGapsIgnoresCaptureRangeEdges(t *testing.T) {
	kernels := func(b *journalBuilder) {
		b.activity("kernel", 0, 1_000, 3_000, 1)
		b.activity("kernel", 0, 6_000, 7_000, 2)
	}

	plain := newJournalBuilder()
	kernels(plain)

	ranged := newJournalBuilder()
	kernels(ranged)
	// Before the device window, inside the 3000..6000 gap, and after it:
	// each placement corrupts a different statistic.
	for _, e := range []struct {
		edge  string
		start uint64
		end   uint64
	}{
		{"start", 100, 200},
		{"stop", 4_400, 4_600},
		{"start", 9_000, 9_500},
	} {
		ranged.append(recordActivity, map[string]any{
			"kind": "activity", "subtype": "capture_range", "valid": true,
			"edge": e.edge, "api": "cudaProfilerStart", "index": 0,
			"toggle_failures": 0, "start": e.start, "end": e.end,
		})
	}

	opts := gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords}
	want, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": plain.finalize()}, completeStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	got, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": ranged.finalize()}, completeStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(want.Devices) != 1 || len(got.Devices) != 1 {
		t.Fatalf("devices = %d with range edges, %d without; want 1 each", len(got.Devices), len(want.Devices))
	}
	w, g := want.Devices[0].Overall, got.Devices[0].Overall
	if g.Activities != w.Activities {
		t.Errorf("activities = %d with range edges, %d without", g.Activities, w.Activities)
	}
	if g.SpanNanos != w.SpanNanos {
		t.Errorf("span = %d with range edges, %d without: an edge moved an end of the device timeline", g.SpanNanos, w.SpanNanos)
	}
	if g.MergedIntervals != w.MergedIntervals {
		t.Errorf("merged intervals = %d with range edges, %d without: an edge split a device gap", g.MergedIntervals, w.MergedIntervals)
	}
	if g.ActiveNanos != w.ActiveNanos || g.IdleNanos != w.IdleNanos {
		t.Errorf("active/idle = %d/%d with range edges, %d/%d without", g.ActiveNanos, g.IdleNanos, w.ActiveNanos, w.IdleNanos)
	}
}

// TestGapsSpanCoversContainedActivities pins the device span against the case
// that breaks the obvious implementation: one activity wholly inside another.
// Activities are sorted by (start, end), so the container sorts first and the
// contained one is last -- taking the last element's end yields a span shorter
// than the active time measured within it, and the report then prints a
// percentage above 100.
//
// A non-overlapping fixture cannot see this, which is why the existing span
// assertions did not: they compare two fixtures that both sort cleanly.
func TestGapsSpanCoversContainedActivities(t *testing.T) {
	b := newJournalBuilder()
	b.activity("kernel", 0, 1_000, 5_000, 1)
	b.activity("kernel", 0, 1_200, 1_400, 2)

	opts := gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords}
	got, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	o := got.Devices[0].Overall
	if want := int64(4_000); o.SpanNanos != want {
		t.Errorf("span = %d, want %d: the span must reach the greatest end, not the end of the last activity by start order", o.SpanNanos, want)
	}
	if o.ActiveNanos > o.SpanNanos {
		t.Errorf("active %d exceeds span %d, so the report would print more than 100%% of span", o.ActiveNanos, o.SpanNanos)
	}
	if o.ActiveNanos+o.IdleNanos != o.SpanNanos {
		t.Errorf("active+idle = %d, span = %d: the two must agree", o.ActiveNanos+o.IdleNanos, o.SpanNanos)
	}
}

// TestGapsCountsMemsetsAsBusy pins the consequence the MEMSET tier exists for.
//
// The subtype filter in decodeJournals is a denylist -- markers and capture
// range edges are dropped and everything else is device work -- so this passes
// the moment memsets reach the journal at all. That is exactly why it is here:
// the whole defect was upstream of any filter. CUPTI was never asked for the
// kind, the stream therefore contained no memsets, and the time one occupied
// on the device was charged to idle by arithmetic that was working correctly
// on the records it had.
//
// The numbers are asserted absolutely, not just compared to the memset-free
// run, because "different from the baseline" would also be satisfied by a
// memset that landed on the wrong device or with the wrong span.
func TestGapsCountsMemsetsAsBusy(t *testing.T) {
	// Two kernels with one 3000 ns hole between them.
	kernels := func(b *journalBuilder) {
		b.activity("kernel", 0, 1_000, 3_000, 1)
		b.activity("kernel", 0, 6_000, 7_000, 2)
	}

	plain := newJournalBuilder()
	kernels(plain)

	withMemset := newJournalBuilder()
	kernels(withMemset)
	// Squarely inside the hole, touching neither kernel, so it splits the one
	// gap into two and removes exactly its own duration from idle.
	withMemset.activity("memset", 0, 4_000, 5_000, 3)

	opts := gapOptions{warmupMS: 0, maxRecords: controllerMaxJournalRecords}
	before, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": plain.finalize()}, completeStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	after, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": withMemset.finalize()}, completeStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(before.Devices) != 1 || len(after.Devices) != 1 {
		t.Fatalf("devices = %d without the memset, %d with; want 1 each", len(before.Devices), len(after.Devices))
	}
	b, a := before.Devices[0].Overall, after.Devices[0].Overall
	if b.IdleNanos != 3_000 {
		t.Fatalf("baseline idle = %d ns, want 3000: the fixture no longer has the hole this test is about", b.IdleNanos)
	}
	if b.ActiveNanos != 3_000 {
		t.Fatalf("baseline active = %d ns, want 3000 (2000+1000)", b.ActiveNanos)
	}
	// The memset is 1000 ns of device work that used to be reported as idle.
	if a.IdleNanos != 2_000 {
		t.Errorf("idle = %d ns with the memset, want 2000: %d ns of memset time is still charged to idle", a.IdleNanos, a.IdleNanos-2_000)
	}
	if a.ActiveNanos != 4_000 {
		t.Errorf("active = %d ns with the memset, want 4000", a.ActiveNanos)
	}
	if a.Activities != b.Activities+1 {
		t.Errorf("activities = %d with the memset, %d without: the record never reached the timeline", a.Activities, b.Activities)
	}
	// One hole became two, which is what makes the memset visible in the gap
	// report rather than merely subtracted from a total.
	if b.GapCount != 1 || a.GapCount != 2 {
		t.Errorf("gaps = %d without the memset, %d with; want 1 and 2", b.GapCount, a.GapCount)
	}
	if a.SpanNanos != b.SpanNanos {
		t.Errorf("span = %d with the memset, %d without: a contained record moved an end of the timeline", a.SpanNanos, b.SpanNanos)
	}
}

// twoKernelHole is two kernels on device 0 in context 1 with a single 3000 ns
// hole between them. Every test below adds one idle-explaining span inside
// that hole and asserts the hole is still there.
func twoKernelHole() *journalBuilder {
	b := newJournalBuilder()
	b.kernelOn(0, 1, 1_000, 3_000, 1)
	b.kernelOn(0, 1, 6_000, 7_000, 2)
	return b
}

func analyzeFixture(t *testing.T, data []byte) *GapReport {
	t.Helper()
	report, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": data}, completeStatus()), gapOptions{warmupMS: 0, maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	return report
}

// TestGapsKeepsOverheadOutOfDeviceBusy is the assertion the OVERHEAD tier
// stands or falls on, and it is an assertion that nothing changed.
//
// The subtype filter in decodeJournal is a denylist: markers and capture-range
// edges are dropped and everything else is device work. Enabling a new kind
// upstream therefore silently enrols it as device busy. For overhead that is
// not a small error, it is an inversion -- CUPTI's own cost would be counted
// as the program keeping the device busy, and the idle the tool itself caused
// would vanish from the number the tool exists to report. Nothing would fail:
// active would rise, idle would fall, and both would still sum to the span.
//
// So the test pins idle, active and the gap count against a run with no
// overhead at all, and then asserts the overlay picked the span up. Exclusion
// without attribution is half the value: a gap the tool reports should be
// labellable "this is CUPTI flushing, not your program".
func TestGapsKeepsOverheadOutOfDeviceBusy(t *testing.T) {
	before := analyzeFixture(t, twoKernelHole().finalize())
	with := twoKernelHole()
	// Squarely inside the hole, touching neither kernel.
	with.overhead(4_000, 5_000, 1<<16)
	after := analyzeFixture(t, with.finalize())

	if len(before.Devices) != 1 || len(after.Devices) != 1 {
		t.Fatalf("devices = %d without the overhead, %d with; want 1 each: an overhead carries no device and must not open a timeline", len(before.Devices), len(after.Devices))
	}
	b, a := before.Devices[0].Overall, after.Devices[0].Overall
	if b.IdleNanos != 3_000 || b.ActiveNanos != 3_000 || b.GapCount != 1 {
		t.Fatalf("baseline idle/active/gaps = %d/%d/%d, want 3000/3000/1: the fixture no longer has the hole this test is about", b.IdleNanos, b.ActiveNanos, b.GapCount)
	}
	if a.IdleNanos != b.IdleNanos {
		t.Errorf("idle = %d ns with the overhead, %d without: %d ns of CUPTI's own cost was counted as device work and subtracted from the idle it caused", a.IdleNanos, b.IdleNanos, b.IdleNanos-a.IdleNanos)
	}
	if a.ActiveNanos != b.ActiveNanos {
		t.Errorf("active = %d ns with the overhead, %d without: an overhead span entered the busy timeline", a.ActiveNanos, b.ActiveNanos)
	}
	if a.GapCount != b.GapCount {
		t.Errorf("gaps = %d with the overhead, %d without: the span split a real device gap in two", a.GapCount, b.GapCount)
	}
	if a.Activities != b.Activities {
		t.Errorf("activities = %d with the overhead, %d without: the count of device work admitted a record that is not device work", a.Activities, b.Activities)
	}
	// Excluded and attributed. Without this the gap is honest and unexplained,
	// which is where the tool already was.
	if a.Overlay.CuptiOverheadNanos != 1_000 {
		t.Errorf("overlay cupti_overhead = %d ns, want 1000: the gap cannot be labelled as the tool's own cost", a.Overlay.CuptiOverheadNanos)
	}
	if a.Overlay.CuptiOverheadGaps != 1 {
		t.Errorf("overlay cupti_overhead gap count = %d, want 1", a.Overlay.CuptiOverheadGaps)
	}
	if b.Overlay.CuptiOverheadNanos != 0 || b.Overlay.CuptiOverheadGaps != 0 {
		t.Errorf("baseline overlay = %d ns over %d gap(s), want zero: the overlay reports overhead in a run that has none", b.Overlay.CuptiOverheadNanos, b.Overlay.CuptiOverheadGaps)
	}
}

// TestGapsKeepsSyncOutOfDeviceBusy is the same denylist trap, in its sharpest
// form. A synchronization span is the host waiting, which is *why* the device
// was idle; counting it as busy would erase precisely the idle it was added to
// explain, and the erasure is exact rather than approximate. A reader would
// see a duty cycle improve as more of the run was spent blocked.
//
// It also covers the device resolution. The record carries a context and no
// device, and the only reason this sync lands on device 0 is that a kernel
// named context 1 on device 0.
func TestGapsKeepsSyncOutOfDeviceBusy(t *testing.T) {
	before := analyzeFixture(t, twoKernelHole().finalize())
	with := twoKernelHole()
	with.sync(1, 4_000, 5_500, 3, 9)
	after := analyzeFixture(t, with.finalize())

	b, a := before.Devices[0].Overall, after.Devices[0].Overall
	if b.IdleNanos != 3_000 {
		t.Fatalf("baseline idle = %d ns, want 3000", b.IdleNanos)
	}
	if a.IdleNanos != b.IdleNanos {
		t.Errorf("idle = %d ns with the sync, %d without: the host's wait was counted as device work, erasing %d ns of the idle it explains", a.IdleNanos, b.IdleNanos, b.IdleNanos-a.IdleNanos)
	}
	if a.ActiveNanos != b.ActiveNanos {
		t.Errorf("active = %d ns with the sync, %d without", a.ActiveNanos, b.ActiveNanos)
	}
	if a.GapCount != b.GapCount {
		t.Errorf("gaps = %d with the sync, %d without", a.GapCount, b.GapCount)
	}
	if a.Overlay.SyncNanos != 1_500 {
		t.Errorf("overlay synchronization = %d ns, want 1500: the idle is reported without the reason for it", a.Overlay.SyncNanos)
	}
	if a.Overlay.SyncGaps != 1 {
		t.Errorf("overlay synchronization gap count = %d, want 1", a.Overlay.SyncGaps)
	}
	if after.UnresolvedSyncs != 0 {
		t.Errorf("unresolved syncs = %d, want 0: a kernel named context 1 on device 0, so this sync is resolvable", after.UnresolvedSyncs)
	}
}

// TestGapsReportsSyncWithUnknownContextAsUnresolved covers the other branch of
// the resolution, which is the one a default would hide.
//
// CUpti_ActivitySynchronization has no device field. Filling one in from
// nothing means writing zero, and zero is a valid device id: on a single-GPU
// run the wrong answer and the right answer are the same number, so the defect
// never surfaces, and on a multi-GPU run one device's idle is explained by
// another device's wait. The fixture therefore runs entirely on device 1, so a
// fabricated device 0 has nowhere to hide.
func TestGapsReportsSyncWithUnknownContextAsUnresolved(t *testing.T) {
	b := newJournalBuilder()
	b.kernelOn(1, 4, 1_000, 3_000, 1)
	b.kernelOn(1, 4, 6_000, 7_000, 2)
	// Context 9 belongs to no record that carries a device.
	b.sync(9, 4_000, 5_000, 3, 9)
	report := analyzeFixture(t, b.finalize())

	if len(report.Devices) != 1 || report.Devices[0].Device != 1 {
		var got []uint32
		for _, d := range report.Devices {
			got = append(got, d.Device)
		}
		t.Fatalf("devices = %v, want [1] only: an unresolvable sync was placed on a device timeline, and device 0 is the timeline it invents", got)
	}
	if report.UnresolvedSyncs != 1 {
		t.Errorf("unresolved syncs = %d, want 1: a sync whose context nothing named was silently accounted for somewhere", report.UnresolvedSyncs)
	}
	if report.UnresolvedSyncNanos != 1_000 {
		t.Errorf("unresolved sync nanos = %d, want 1000", report.UnresolvedSyncNanos)
	}
	s := report.Devices[0].Overall
	if s.IdleNanos != 3_000 {
		t.Errorf("idle = %d ns, want 3000: the unresolvable sync still reached a device timeline", s.IdleNanos)
	}
	if s.Overlay.SyncNanos != 0 {
		t.Errorf("overlay synchronization = %d ns on device 1, want 0: a sync that could not be attributed was attributed anyway", s.Overlay.SyncNanos)
	}
	// The count is not enough on its own: a reader who is not told will read
	// the remaining overlay as complete.
	if !strings.Contains(report.text(), "could not be placed on a device") {
		t.Errorf("report text does not say a sync went unplaced:\n%s", report.text())
	}
}

// TestGapsKeepsGraphTraceOutOfDeviceBusy covers the third denylist entry.
//
// A graph replay envelope spans the kernel records of the nodes replayed
// inside it, so admitting it to the busy timeline does two things at once: it
// counts that device time twice, and -- because the envelope is contiguous
// across the scheduling gaps between nodes -- it merges over exactly the idle
// a reader would want to see. A graph would appear to run at a 100% duty
// cycle by definition.
//
// The envelope is kept as an overlay instead, where the difference between the
// span and the node time inside it is graph launch and scheduling cost that no
// per-node record holds.
func TestGapsKeepsGraphTraceOutOfDeviceBusy(t *testing.T) {
	// Both kernels name the graph they are nodes of. That is the condition
	// the exclusion rests on, so the fixture has to state it: a capture whose
	// kernels name no graph is the opaque case below, not this one.
	nodes := func() *journalBuilder {
		b := newJournalBuilder()
		b.graphNode(0, 1, 1_000, 3_000, 1, 5)
		b.graphNode(0, 1, 6_000, 7_000, 2, 5)
		return b
	}
	before := analyzeFixture(t, nodes().finalize())
	with := nodes()
	// The envelope covers both kernels and the hole between them, which is
	// what a real replay of two nodes looks like.
	with.graphTrace(0, 900, 7_100, 5, 3)
	after := analyzeFixture(t, with.finalize())

	b, a := before.Devices[0].Overall, after.Devices[0].Overall
	if a.IdleNanos != b.IdleNanos {
		t.Errorf("idle = %d ns with the graph trace, %d without: the replay envelope merged over the scheduling gap between its own nodes", a.IdleNanos, b.IdleNanos)
	}
	if a.ActiveNanos != b.ActiveNanos {
		t.Errorf("active = %d ns with the graph trace, %d without: the envelope's device time was counted on top of the nodes inside it", a.ActiveNanos, b.ActiveNanos)
	}
	if a.SpanNanos != b.SpanNanos {
		t.Errorf("span = %d with the graph trace, %d without: the envelope moved an end of the timeline", a.SpanNanos, b.SpanNanos)
	}
	if a.GapCount != b.GapCount {
		t.Errorf("gaps = %d with the graph trace, %d without", a.GapCount, b.GapCount)
	}
	if a.Overlay.GraphReplayNanos != 3_000 {
		t.Errorf("overlay graph_replay = %d ns, want 3000: the idle inside the replay is not attributable to graph scheduling", a.Overlay.GraphReplayNanos)
	}
	if a.Overlay.GraphReplayGaps != 1 {
		t.Errorf("overlay graph_replay gap count = %d, want 1", a.Overlay.GraphReplayGaps)
	}
}

// TestGapsOverlayDoesNotDoubleCountOverlappingSpans pins the one arithmetic
// property an overlay of this shape can get wrong. Two spans covering the same
// idle -- a CUPTI flush that happens during a synchronize is the ordinary case
// -- must contribute their union, not their sum. Summing them produces a
// figure that can exceed the idle it is a part of, which is how a reader
// learns to distrust every number beside it.
func TestGapsOverlayDoesNotDoubleCountOverlappingSpans(t *testing.T) {
	b := twoKernelHole()
	b.overhead(4_000, 4_800, 1<<16)
	b.overhead(4_400, 5_000, 1<<16)
	report := analyzeFixture(t, b.finalize())
	s := report.Devices[0].Overall
	if s.Overlay.CuptiOverheadNanos != 1_000 {
		t.Errorf("overlay cupti_overhead = %d ns, want 1000 (the union of 4000..4800 and 4400..5000): overlapping spans were summed", s.Overlay.CuptiOverheadNanos)
	}
	if s.Overlay.CuptiOverheadGaps != 1 {
		t.Errorf("overlay gap count = %d, want 1: two spans in one gap counted the gap twice", s.Overlay.CuptiOverheadGaps)
	}
	if s.Overlay.CuptiOverheadNanos > s.IdleNanos {
		t.Errorf("overlay cupti_overhead %d ns exceeds idle %d ns", s.Overlay.CuptiOverheadNanos, s.IdleNanos)
	}
}

// TestUnattributedIdleNamesNoMechanism pins the split that a whole campaign
// was misread through.
//
// Idle before an activity that correlated to no API record used to be added,
// whole, to the hardware-queue arm, whose stated meaning is "idle is hardware
// queue latency". That turns "we could not find the launch" into a specific
// mechanism, and a mechanism is what a reader goes and works on. The two arms
// have to be separable in the output, not only in the class table -- a reader
// who sums the attribution lines must not find the unknown inside the known.
func TestUnattributedIdleNamesNoMechanism(t *testing.T) {
	// Three activities, 100 us apart. The middle one correlates; the last
	// does not, so one gap is attributable and one is not.
	acts := []gapActivity{
		{start: 0, end: 100_000, correlation: 1},
		{start: 200_000, end: 300_000, correlation: 2},
		{start: 400_000, end: 500_000, correlation: 3},
	}
	apis := map[uint32]gapAPI{2: {start: 10_000, end: 50_000}} // host ahead
	s := computeGapStats(acts, apis, "overall", gapSpans{})
	if s.GapCount != 2 || s.IdleNanos != 200_000 {
		t.Fatalf("gaps/idle = %d/%d, want 2/200000", s.GapCount, s.IdleNanos)
	}
	a := s.Attribution
	if a.HardwareNanos != 100_000 || a.UnattributedNanos != 100_000 {
		t.Errorf("attribution = %+v, want hardware 100000 and unattributed 100000 kept apart", a)
	}
	if got := a.HostBehindNanos + a.DriverSubmitNanos + a.HardwareNanos + a.UnattributedNanos; got != s.IdleNanos {
		t.Errorf("arms sum to %d, want %d: the split must partition idle", got, s.IdleNanos)
	}

	// Asserted on the rendered text, not the struct, because the struct is
	// not what anyone read: the count of unmatched gaps was already printed
	// directly above the hardware-queue line, and it was still read past.
	var b strings.Builder
	writeGapStats(&b, s)
	out := b.String()
	hw := findGapLine(t, out, "hardware queue")
	un := findGapLine(t, out, "unattributed")
	if !strings.Contains(hw, "0.100 ms") || !strings.Contains(hw, "50.00%") {
		t.Errorf("hardware queue line absorbed the uncorrelated gap: %q", hw)
	}
	if !strings.Contains(un, "0.100 ms") || !strings.Contains(un, "50.00%") {
		t.Errorf("unattributed line does not carry the uncorrelated gap: %q", un)
	}
}

func findGapLine(t *testing.T, out, prefix string) string {
	t.Helper()
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), prefix) {
			return line
		}
	}
	t.Fatalf("no %q line in:\n%s", prefix, out)
	return ""
}

// TestOpaqueGraphReplayIsNotIdle covers the condition that makes the exclusion
// above unsound, which is the one a real capture hit.
//
// The envelope is excluded because it spans the node records. When CUPTI emits
// the envelope and nothing inside it -- no device record names a graph id --
// it spans nothing, and excluding it deletes the entire replay from the
// timeline and republishes its duration as device idle. On the capture that
// found this, 2942.915 ms of graph work read as idle and the duty cycle read
// 14.54% where the work was about 71%. An A/B that disabled graphs moved wall
// clock by 33 ms against a 2904 ms prediction: the work was always there.
//
// Both arms of the discriminator are asserted. A capture whose kernels do name
// their graph must keep the exclusion, or this fix trades one 5x error for
// another.
func TestOpaqueGraphReplayIsNotIdle(t *testing.T) {
	opaque := newJournalBuilder()
	opaque.kernelOn(0, 1, 1_000, 2_000, 1)  // not a graph node
	opaque.kernelOn(0, 1, 9_000, 10_000, 2) // not a graph node
	opaque.graphTrace(0, 3_000, 8_000, 5, 3)
	report := analyzeFixture(t, opaque.finalize())
	if !report.OpaqueGraphReplays {
		t.Fatal("a capture with replay envelopes and no graph-tagged record was not reported as opaque")
	}
	s := report.Devices[0].Overall
	// 1000 + 5000 + 1000 of work in a 9000 ns span: the replay is on the
	// timeline. Excluding it gives 2000 active and 7000 idle.
	if s.ActiveNanos != 7_000 || s.IdleNanos != 2_000 {
		t.Errorf("active/idle = %d/%d, want 7000/2000: the replay was dropped from the timeline and reported as idle", s.ActiveNanos, s.IdleNanos)
	}
	if s.DutyCyclePct < 77.7 || s.DutyCyclePct > 77.8 {
		t.Errorf("duty cycle = %.2f, want 77.78", s.DutyCyclePct)
	}
	// Counted as activity, it is no longer an idle overlay. Reporting it in
	// both places would let a reader add it to a total it is already inside.
	if s.Overlay.GraphReplayNanos != 0 {
		t.Errorf("graph replay overlay = %d ns, want 0: the envelope is on the timeline now", s.Overlay.GraphReplayNanos)
	}
	// The error this cannot avoid is stated rather than hidden: idle between
	// the nodes of a replay is invisible here and is counted as busy.
	if out := report.text(); !strings.Contains(out, "graph replays are opaque") {
		t.Errorf("report does not say the replays are opaque:\n%s", out)
	}

	// The other arm: one graph-tagged kernel is enough to make the envelope
	// the span of something, and the exclusion must survive.
	tagged := newJournalBuilder()
	tagged.kernelOn(0, 1, 1_000, 2_000, 1)
	tagged.graphNode(0, 1, 3_000, 4_000, 4, 5)
	tagged.kernelOn(0, 1, 9_000, 10_000, 2)
	tagged.graphTrace(0, 3_000, 8_000, 5, 3)
	report = analyzeFixture(t, tagged.finalize())
	if report.OpaqueGraphReplays {
		t.Error("a capture whose kernels name their graph was reported as opaque")
	}
	if s := report.Devices[0].Overall; s.ActiveNanos != 3_000 {
		t.Errorf("active = %d ns, want 3000: the envelope was counted on top of the node inside it", s.ActiveNanos)
	}
	if out := report.text(); strings.Contains(out, "graph replays are opaque") {
		t.Errorf("report claims opaque replays where a node record names its graph:\n%s", out)
	}
}

// namedKernel appends one kernel activity carrying its name, which is what the
// per-kernel census counts by.
func (b *journalBuilder) namedKernel(name string, start, end uint64, correlation uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": "kernel", "valid": true,
		"name": name, "start": start, "end": end, "device": 0, "correlation": correlation,
	})
	return b
}

// namedActivity appends one named activity of any subtype.
func (b *journalBuilder) namedActivity(subtype, name string, start, end uint64, correlation uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": subtype, "valid": true,
		"name": name, "start": start, "end": end, "device": 0, "correlation": correlation,
	})
	return b
}

// censusReport builds a one-capture report from a run of named kernels.
func censusReport(t *testing.T, kernels []struct {
	name     string
	duration uint64
	count    int
}) *GapReport {
	t.Helper()
	b := newJournalBuilder()
	var at uint64 = 1_000
	var corr uint32
	for _, k := range kernels {
		for i := 0; i < k.count; i++ {
			corr++
			b.namedKernel(k.name, at, at+k.duration, corr)
			b.api(at-500, at-100, corr)
			at += k.duration + 1_000
		}
	}
	cap := testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus())
	report, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	return report
}

// TestKernelCensusSeparatesLaunchChangesFromPressure is the distinction the
// census exists to make. A kernel whose launch count moved took a different
// path: that is the change. A kernel with the identical launch count in both
// arms was not touched, so device time that moved there came from elsewhere on
// the device -- and ranking the two together by delta alone puts the untouched
// kernel first.
func TestKernelCensusSeparatesLaunchChangesFromPressure(t *testing.T) {
	type row = struct {
		name     string
		duration uint64
		count    int
	}
	base := censusReport(t, []row{
		{"repack", 1_000, 10}, // removed by the change
		{"dense", 1_000, 5},   // untouched, and slower afterwards
		{"gone", 500, 2},      // absent from the target entirely
	})
	target := censusReport(t, []row{
		{"repack", 1_000, 4},
		{"dense", 3_000, 5},
		{"added", 400, 3},
	})
	d := diffKernelCensus(base, target)
	if d == nil {
		t.Fatal("no census diff")
	}
	changed := map[string]KernelDelta{}
	for _, r := range d.Changed {
		changed[r.Name] = r
	}
	unchanged := map[string]KernelDelta{}
	for _, r := range d.Unchanged {
		unchanged[r.Name] = r
	}
	for _, name := range []string{"repack", "gone", "added"} {
		if _, ok := changed[name]; !ok {
			t.Errorf("%s is not in the launch-count-changed half: %+v", name, d.Changed)
		}
		if _, ok := unchanged[name]; ok {
			t.Errorf("%s is reported as untouched, but its launch count moved", name)
		}
	}
	// A kernel present in one arm only is a launch count change from or to
	// zero, never an unchanged one: "the launch count did not move" would be
	// a claim about a kernel that did not run.
	if got := changed["gone"]; got.BaseLaunches != 2 || got.TargetLaunches != 0 {
		t.Errorf("gone = %+v, want 2 -> 0 launches", got)
	}
	if got := changed["added"]; got.BaseLaunches != 0 || got.TargetLaunches != 3 {
		t.Errorf("added = %+v, want 0 -> 3 launches", got)
	}
	// dense keeps its 5 launches in both arms, but removing 6 repack launches
	// ahead of it moves where it sits in the run. That is the third class:
	// the count says the kernel was left alone and the position says a window
	// containing it does not hold the same work in both arms.
	relocated := map[string]KernelDelta{}
	for _, r := range d.Relocated {
		relocated[r.Name] = r
	}
	dense, ok := relocated["dense"]
	if !ok {
		t.Fatalf("dense is in neither the untouched nor the moved half: unchanged %+v relocated %+v", d.Unchanged, d.Relocated)
	}
	if _, ok := unchanged["dense"]; ok {
		t.Error("dense is reported untouched, but its launches moved to a different part of the run")
	}
	if dense.PositionShift == 0 {
		t.Errorf("dense = %+v, want a nonzero median-position shift", dense)
	}
	if dense.BaseLaunches != 5 || dense.TargetLaunches != 5 {
		t.Errorf("dense launches = %d -> %d, want 5 -> 5", dense.BaseLaunches, dense.TargetLaunches)
	}
	if dense.DeltaNanos != 10_000 {
		t.Errorf("dense delta = %d ns, want +10000 (5 launches, 2000 ns slower each)", dense.DeltaNanos)
	}
	// The totals are the reading the halves exist to give: work removed on
	// one side, and on the other a cost that is only comparable if the window
	// it was measured over holds the same work in both arms.
	if d.RelocatedNanos != 10_000 {
		t.Errorf("moved total = %d ns, want 10000", d.RelocatedNanos)
	}
	if d.UnchangedNanos != 0 {
		t.Errorf("untouched total = %d ns, want 0: nothing here held its position", d.UnchangedNanos)
	}
	if want := int64(-6_000 - 1_000 + 1_200); d.ChangedNanos != want {
		t.Errorf("changed total = %d ns, want %d", d.ChangedNanos, want)
	}
	// Largest magnitude first, in each half independently: a -6000 ns row
	// must not be outranked by a +1200 ns one because it is negative.
	if d.Changed[0].Name != "repack" {
		t.Errorf("changed rows are not ranked by magnitude: %+v", d.Changed)
	}
}

// TestCensusCaveatFiresOnlyOnMixedOpacity guards the caveat in both
// directions. Printed on a matched pair it is noise a reader learns to skip;
// omitted on a mismatched one it lets a kernel whose records the capture did
// not carry read as a launch the change removed.
func TestCensusCaveatFiresOnlyOnMixedOpacity(t *testing.T) {
	opaque := &GapReport{OpaqueGraphReplays: true, Kernels: []KernelCensus{{Name: "k", Launches: 1}}}
	plain := &GapReport{Kernels: []KernelCensus{{Name: "k", Launches: 1}}}
	if got := diffKernelCensus(plain, opaque); got.Caveat == "" {
		t.Error("no caveat where only the target's replays are opaque")
	} else if !strings.Contains(got.Caveat, "target capture's graph replays are opaque") {
		t.Errorf("caveat names the wrong arm: %s", got.Caveat)
	}
	if got := diffKernelCensus(opaque, plain); !strings.Contains(got.Caveat, "base capture's graph replays are opaque") {
		t.Errorf("caveat names the wrong arm: %s", got.Caveat)
	}
	for _, pair := range [][2]*GapReport{{plain, plain}, {opaque, opaque}} {
		if got := diffKernelCensus(pair[0], pair[1]); got.Caveat != "" {
			t.Errorf("caveat on a matched pair: %s", got.Caveat)
		}
	}
}

// TestKernelCensusCountsOnlyKernels keeps memcpy, sync and overhead spans out
// of a per-kernel launch count: a census that counted them would report a
// launch count that moved where no kernel launch did.
func TestKernelCensusCountsOnlyKernels(t *testing.T) {
	b := newJournalBuilder()
	b.namedKernel("k", 1_000, 2_000, 1)
	// Named on purpose: an unnamed memcpy is excluded by the name check
	// rather than by the subtype check, so it cannot tell whether the subtype
	// check does anything.
	b.namedActivity("memcpy", "copyHtoD", 3_000, 4_000, 2)
	b.overhead(5_000, 6_000, 1)
	cap := testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus())
	report, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Kernels) != 1 || report.Kernels[0].Name != "k" || report.Kernels[0].Launches != 1 {
		t.Errorf("census = %+v, want one kernel k with one launch", report.Kernels)
	}
	if report.Kernels[0].DeviceNanos != 1_000 {
		t.Errorf("device time = %d ns, want 1000", report.Kernels[0].DeviceNanos)
	}
}

// TestUnnamedKernelRecordsAreCounted keeps a census total honest about what it
// does not cover. A kernel record without a name cannot be a row, and dropping
// it silently makes the rows look like the whole population.
func TestUnnamedKernelRecordsAreCounted(t *testing.T) {
	b := newJournalBuilder()
	b.namedKernel("k", 1_000, 2_000, 1)
	b.kernel(3_000, 4_000, 2)
	cap := testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus())
	report, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	if report.UnnamedKernelRecords != 1 {
		t.Errorf("unnamed kernel records = %d, want 1", report.UnnamedKernelRecords)
	}
	if len(report.Kernels) != 1 {
		t.Errorf("census = %+v, want only the named kernel", report.Kernels)
	}
}

// TestUnfinalizedNoCUDAJournal checks that a journal the controller admits as
// a process that ran no CUDA -- a shell in front of the program, say -- does
// not make gaps, compare or export refuse the capture, and that every other
// unfinalized journal still does. The evidence must be the producer's: it
// says it writes the device_active note, and the note is absent.
func TestUnfinalizedNoCUDAJournal(t *testing.T) {
	worker := func(capability map[string]any, body func(*journalBuilder)) []byte {
		b := &journalBuilder{}
		b.session[0] = 0x42
		b.append(recordBootstrap, map[string]any{"kind": "bootstrap"})
		b.append(recordCapability, capability)
		if body != nil {
			body(b)
		}
		return b.unfinalized()
	}
	noted := map[string]any{"kind": "capability", "device_active_note": true}
	tests := []struct {
		name    string
		journal []byte
		wantErr string // "" admits the journal as no-CUDA
	}{
		{name: "bootstrap and capability only", journal: worker(noted, nil)},
		{name: "host api only", journal: worker(noted, func(b *journalBuilder) { b.api(1, 2, 99) })},
		{name: "capture range edge only", journal: worker(noted, func(b *journalBuilder) {
			b.append(recordActivity, map[string]any{"kind": "activity", "subtype": "capture_range", "valid": true, "start": 1, "end": 1})
		})},
		{name: "producer without the note", journal: worker(map[string]any{"kind": "capability"}, nil), wantErr: "never finalized"},
		{name: "device active", journal: worker(noted, func(b *journalBuilder) {
			b.append(recordActivity, map[string]any{"kind": "activity", "subtype": "device_active"})
		}), wantErr: "never finalized"},
		{name: "device work", journal: worker(noted, func(b *journalBuilder) { b.kernel(10, 20, 1) }), wantErr: "never finalized"},
		{name: "producer failure", journal: worker(noted, func(b *journalBuilder) {
			b.append(recordFailure, map[string]any{"kind": "failure", "state": "producer_failure"})
		}), wantErr: "never finalized"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			journals := map[string][]byte{
				"worker-a-capture.journal": tt.journal,
				"worker-b-capture.journal": newJournalBuilder().kernel(100, 200, 1).api(90, 95, 1).finalize(),
			}
			got, err := decodeJournals(journals, controllerMaxJournalRecords, 1<<20)
			_, _, _, exportErr := decodeTraceJournals(journals, controllerMaxJournalRecords, 1<<20)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("gaps error = %v, want %q", err, tt.wantErr)
				}
				if exportErr == nil || !strings.Contains(exportErr.Error(), tt.wantErr) {
					t.Errorf("export error = %v, want %q", exportErr, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("gaps refused a no-CUDA journal: %v", err)
			}
			if exportErr != nil {
				t.Fatalf("export refused a no-CUDA journal: %v", exportErr)
			}
			if len(got.journals) != 2 || !got.journals[0].NoCUDA || got.journals[1].NoCUDA {
				t.Errorf("journals = %+v, want worker-a marked no-CUDA and worker-b not", got.journals)
			}
			if len(got.activities) != 1 {
				t.Errorf("activities = %d, want 1: the no-CUDA journal contributes none", len(got.activities))
			}
			if _, ok := got.apis[99]; ok || len(got.apis) != 1 {
				t.Errorf("apis = %v, want only worker-b's: a set-aside journal leaves nothing behind", got.apis)
			}
		})
	}
}
