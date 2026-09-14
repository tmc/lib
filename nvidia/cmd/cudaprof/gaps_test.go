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
			name:         "unmatched correlation",
			api:          nil,
			wantClass:    classUnmatched,
			wantIdle:     100,
			wantHardware: 100,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			apis := map[uint32]gapAPI{}
			if tt.api != nil {
				apis[2] = *tt.api
			}
			s := computeGapStats(acts, apis, "overall")
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
			if a.HostBehindNanos != tt.wantBehind || a.DriverSubmitNanos != tt.wantDriver || a.HardwareNanos != tt.wantHardware {
				t.Errorf("attribution = %+v, want behind %d driver %d hardware %d", a, tt.wantBehind, tt.wantDriver, tt.wantHardware)
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
		{name: "unknown flag", args: []string{"--nope", "c.pprof"}, wantErr: "unknown flag"},
		{name: "bad warmup", args: []string{"--warmup-ms=x", "c.pprof"}, wantErr: "invalid warmup"},
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
			if got.capturePath != tt.want.capturePath || got.warmupMS != tt.want.warmupMS ||
				got.maxRecords != tt.want.maxRecords || got.json != tt.want.json {
				t.Errorf("options = %+v, want %+v", got, tt.want)
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
	os.Stdout = w
	runErr := fn()
	w.Close()
	os.Stdout = orig
	data, _ := io.ReadAll(r)
	r.Close()
	if runErr != nil {
		t.Fatalf("command: %v", runErr)
	}
	return string(data)
}
