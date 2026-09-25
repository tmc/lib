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
	"github.com/tmc/lib/nvidia/x/cudaprof/gaps"
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
	b.append(wire.KindBootstrap, map[string]any{"kind": "bootstrap"})
	b.append(wire.KindCapability, map[string]any{"kind": "capability"})
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
	b.append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": subtype, "valid": true,
		"start": start, "end": end, "device": device, "correlation": correlation,
	})
	return b
}

// kernelOn appends one kernel on an explicit device and context. The context
// matters only here: it is what a synchronization record is resolved through,
// and a fixture whose kernels name no context can never exercise resolution.
func (b *journalBuilder) kernelOn(device, context uint32, start, end uint64, correlation uint32) *journalBuilder {
	b.append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "kernel", "valid": true,
		"start": start, "end": end, "device": device, "context": context, "correlation": correlation,
	})
	return b
}

// graphNode appends one kernel that names the graph it ran as a node of,
// which is what makes a replay envelope's exclusion sound: the envelope is
// excluded because it spans records like this one.
func (b *journalBuilder) graphNode(device, context uint32, start, end uint64, correlation, graphID uint32) *journalBuilder {
	b.append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "kernel", "valid": true,
		"start": start, "end": end, "device": device, "context": context,
		"correlation": correlation, "graph_id": graphID,
	})
	return b
}

// graphTrace appends one graph replay envelope, which does carry a device.
func (b *journalBuilder) graphTrace(device uint32, start, end uint64, graphID, correlation uint32) *journalBuilder {
	b.append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "graph_trace", "valid": true,
		"start": start, "end": end, "device": device, "graph_id": graphID, "correlation": correlation,
	})
	return b
}

// sync appends one synchronization span. It names a context and no device,
// because CUpti_ActivitySynchronization has no device field to name.
func (b *journalBuilder) sync(context uint32, start, end uint64, syncType, correlation uint32) *journalBuilder {
	b.append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "sync", "valid": true,
		"start": start, "end": end, "context": context, "sync_type": syncType,
		"cuda_event_id": 3, "correlation": correlation,
	})
	return b
}

// api appends one driver launch record correlated to an activity.
func (b *journalBuilder) api(start, end uint64, correlation uint32) *journalBuilder {
	b.append(wire.KindAPI, map[string]any{
		"kind": "api", "domain": "driver", "start": start, "end": end, "correlation": correlation,
	})
	return b
}

func (b *journalBuilder) finalize() []byte {
	b.append(wire.KindFinalized, map[string]any{"kind": "finalized", "state": "collected"})
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

func TestGapsCLI(t *testing.T) {
	dir := t.TempDir()
	capPath := filepath.Join(dir, "gaps.pprof")
	writeTestCaptureFile(t, capPath, testCapture(map[string][]byte{"capture.journal": uniformJournal()}, completeStatus()))

	text := captureStdout(t, func() error { return runGapsCommand([]string{capPath}) })
	for _, want := range []string{"duty cycle", "gap distribution", "host concurrency", gaps.ClassHostAhead, "idle attribution"} {
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

func analyzeFixture(t *testing.T, data []byte) *GapReport {
	t.Helper()
	report, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": data}, completeStatus()), gapOptions{warmupMS: 0, maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	return report
}

// TestGapsReportsSyncWithUnknownContextAsUnresolved checks that the text says
// a synchronization record went unplaced. The count is not enough on its own:
// a reader who is not told will read the remaining overlay as complete.
func TestGapsReportsSyncWithUnknownContextAsUnresolved(t *testing.T) {
	b := newJournalBuilder()
	b.kernelOn(1, 4, 1_000, 3_000, 1)
	b.kernelOn(1, 4, 6_000, 7_000, 2)
	// Context 9 belongs to no record that carries a device.
	b.sync(9, 4_000, 5_000, 3, 9)
	report := analyzeFixture(t, b.finalize())
	if report.UnresolvedSyncs != 1 {
		t.Fatalf("unresolved syncs = %d, want 1", report.UnresolvedSyncs)
	}
	if !strings.Contains(report.text(), "could not be placed on a device") {
		t.Errorf("report text does not say a sync went unplaced:\n%s", report.text())
	}
}

// TestUnattributedIdleNamesNoMechanism checks that the hardware-queue and
// unattributed arms are separable in the output, not only in the class table:
// a reader who sums the attribution lines must not find the unknown inside the
// known. It is asserted on the rendered text, not the struct, because the
// struct is not what anyone read: the count of unmatched gaps was already
// printed directly above the hardware-queue line, and it was still read past.
func TestUnattributedIdleNamesNoMechanism(t *testing.T) {
	// Three kernels, 100 us apart. The middle one correlates; the last does
	// not, so one gap is attributable and one is not.
	b := newJournalBuilder()
	b.kernel(1_000, 101_000, 1)
	b.kernel(201_000, 301_000, 2)
	b.kernel(401_000, 501_000, 3)
	b.api(11_000, 51_000, 2) // host ahead
	s := analyzeFixture(t, b.finalize()).Devices[0].Overall
	if s.GapCount != 2 || s.IdleNanos != 200_000 {
		t.Fatalf("gaps/idle = %d/%d, want 2/200000", s.GapCount, s.IdleNanos)
	}
	var out strings.Builder
	writeGapStats(&out, s)
	hw := findGapLine(t, out.String(), "hardware queue")
	un := findGapLine(t, out.String(), "unattributed")
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

// TestOpaqueGraphReplayIsNotIdle checks that the text states the error opaque
// replays cannot avoid -- idle between the nodes of a replay is invisible and
// counted as busy -- and does not claim it where a node record names its graph.
func TestOpaqueGraphReplayIsNotIdle(t *testing.T) {
	opaque := newJournalBuilder()
	opaque.kernelOn(0, 1, 1_000, 2_000, 1)  // not a graph node
	opaque.kernelOn(0, 1, 9_000, 10_000, 2) // not a graph node
	opaque.graphTrace(0, 3_000, 8_000, 5, 3)
	report := analyzeFixture(t, opaque.finalize())
	if out := report.text(); !strings.Contains(out, "graph replays are opaque") {
		t.Errorf("report does not say the replays are opaque:\n%s", out)
	}

	tagged := newJournalBuilder()
	tagged.kernelOn(0, 1, 1_000, 2_000, 1)
	tagged.graphNode(0, 1, 3_000, 4_000, 4, 5)
	tagged.kernelOn(0, 1, 9_000, 10_000, 2)
	tagged.graphTrace(0, 3_000, 8_000, 5, 3)
	report = analyzeFixture(t, tagged.finalize())
	if out := report.text(); strings.Contains(out, "graph replays are opaque") {
		t.Errorf("report claims opaque replays where a node record names its graph:\n%s", out)
	}
}

// namedKernel appends one kernel activity carrying its name, which is what the
// per-kernel census counts by.
func (b *journalBuilder) namedKernel(name string, start, end uint64, correlation uint32) *journalBuilder {
	b.append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "kernel", "valid": true,
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
	opaque := &GapReport{OpaqueGraphReplays: true, Kernels: []gaps.Kernel{{Name: "k", Launches: 1}}}
	plain := &GapReport{Kernels: []gaps.Kernel{{Name: "k", Launches: 1}}}
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

// TestUnfinalizedNoCUDAJournal checks that a journal the controller admits as
// a process that ran no CUDA -- a shell in front of the program, say -- does
// not make gaps, compare or export refuse the capture, and that every other
// unfinalized journal still does. The evidence must be the producer's: it
// says it writes the device_active note, and the note is absent.
func TestUnfinalizedNoCUDAJournal(t *testing.T) {
	worker := func(capability map[string]any, body func(*journalBuilder)) []byte {
		b := &journalBuilder{}
		b.session[0] = 0x42
		b.append(wire.KindBootstrap, map[string]any{"kind": "bootstrap"})
		b.append(wire.KindCapability, capability)
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
			b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "capture_range", "valid": true, "start": 1, "end": 1})
		})},
		{name: "producer without the note", journal: worker(map[string]any{"kind": "capability"}, nil), wantErr: "never finalized"},
		{name: "device active", journal: worker(noted, func(b *journalBuilder) {
			b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "device_active"})
		}), wantErr: "never finalized"},
		{name: "device work", journal: worker(noted, func(b *journalBuilder) { b.kernel(10, 20, 1) }), wantErr: "never finalized"},
		{name: "producer failure", journal: worker(noted, func(b *journalBuilder) {
			b.append(wire.KindFailure, map[string]any{"kind": "failure", "state": "producer_failure"})
		}), wantErr: "never finalized"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			journals := map[string][]byte{
				"worker-a-capture.journal": tt.journal,
				"worker-b-capture.journal": newJournalBuilder().kernel(100, 200, 1).api(90, 95, 1).finalize(),
			}
			_, _, _, exportErr := decodeTraceJournals(journals, controllerMaxJournalRecords, 1<<20)
			if tt.wantErr != "" {
				if exportErr == nil || !strings.Contains(exportErr.Error(), tt.wantErr) {
					t.Errorf("export error = %v, want %q", exportErr, tt.wantErr)
				}
				return
			}
			if exportErr != nil {
				t.Fatalf("export refused a no-CUDA journal: %v", exportErr)
			}
		})
	}
}
