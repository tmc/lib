package ncclinspector_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// Actual output of the CUDA-free production C++ header controls, not an
// authentic NCCL runtime capture. Source/command custody is in the task packet.
const progressCPPDigest = "91446f9f4155883c135872170d32d32615db1b7caa6aab22b1d041a13fa590a5"
const progressCPPJSONL = `{"schema":"nccl.progress-monitor.diagnostic.v1","source_commit":"12df1a11afad322be5a204a2db890161cbf8131d","build_id":"constructed-build","host":"constructed-host","boot_epoch":"constructed-boot","process_epoch":"constructed-process","capture_id":"constructed-capture","pid":123,"device":0,"worker":7,"sequence":1,"event":"registered","observed_ns":10,"host_clock":"std.chrono.steady_clock","gpu_clock":"globaltimer","time_unit":"nanoseconds","counter_unit":"owned_completed_work","snapshot_atomic":false,"upstream_loss":"unknown","poll_ms":1000,"stale_ms":5000,"max_records":4096,"max_bytes":16777216,"terminal":false,"exporter_retained_before":0,"exporter_dropped_before":0,"untracked_registrations":0,"code":0,"comm_hash":"ffffffffffffffff","rank":2,"registration":1,"generation":0,"launch_start_ns":0,"launch_end_ns":0,"launch_code":0,"copy_pending":false,"event_recorded":false,"copy_stale":false,"counter_reset":"unknown","mirror_present":false}
{"schema":"nccl.progress-monitor.diagnostic.v1","source_commit":"12df1a11afad322be5a204a2db890161cbf8131d","build_id":"constructed-build","host":"constructed-host","boot_epoch":"constructed-boot","process_epoch":"constructed-process","capture_id":"constructed-capture","pid":123,"device":0,"worker":7,"sequence":2,"event":"copy_launched","observed_ns":21,"host_clock":"std.chrono.steady_clock","gpu_clock":"globaltimer","time_unit":"nanoseconds","counter_unit":"owned_completed_work","snapshot_atomic":false,"upstream_loss":"unknown","poll_ms":1000,"stale_ms":5000,"max_records":4096,"max_bytes":16777216,"terminal":false,"exporter_retained_before":1,"exporter_dropped_before":0,"untracked_registrations":0,"code":0,"comm_hash":"ffffffffffffffff","rank":2,"registration":1,"generation":1,"launch_start_ns":20,"launch_end_ns":21,"launch_code":0,"copy_pending":true,"event_recorded":false,"copy_stale":false,"counter_reset":"unknown","mirror_present":false}
{"schema":"nccl.progress-monitor.diagnostic.v1","source_commit":"12df1a11afad322be5a204a2db890161cbf8131d","build_id":"constructed-build","host":"constructed-host","boot_epoch":"constructed-boot","process_epoch":"constructed-process","capture_id":"constructed-capture","pid":123,"device":0,"worker":7,"sequence":3,"event":"copy_complete","observed_ns":30,"host_clock":"std.chrono.steady_clock","gpu_clock":"globaltimer","time_unit":"nanoseconds","counter_unit":"owned_completed_work","snapshot_atomic":false,"upstream_loss":"unknown","poll_ms":1000,"stale_ms":5000,"max_records":4096,"max_bytes":16777216,"terminal":false,"exporter_retained_before":2,"exporter_dropped_before":0,"untracked_registrations":0,"code":0,"comm_hash":"ffffffffffffffff","rank":2,"registration":1,"generation":1,"launch_start_ns":20,"launch_end_ns":21,"launch_code":0,"copy_pending":false,"event_recorded":true,"copy_stale":false,"counter_reset":"unknown","mirror_present":true,"copy_success_min_ns":20,"copy_success_max_ns":30,"completed_work_count":[9007199254740993,9007199254740994,9007199254740995,9007199254740996,9007199254740997,9007199254740998,9007199254740999,9007199254741000,9007199254741001,9007199254741002,9007199254741003,9007199254741004,9007199254741005,9007199254741006,9007199254741007,9007199254741008],"completed_time_ns":[18446744073709551615,18446744073709551614,18446744073709551613,18446744073709551612,18446744073709551611,18446744073709551610,18446744073709551609,18446744073709551608,18446744073709551607,18446744073709551606,18446744073709551605,18446744073709551604,18446744073709551603,18446744073709551602,18446744073709551601,18446744073709551600],"channel_active_slot_mask":[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]}
{"schema":"nccl.progress-monitor.diagnostic.v1","source_commit":"12df1a11afad322be5a204a2db890161cbf8131d","build_id":"constructed-build","host":"constructed-host","boot_epoch":"constructed-boot","process_epoch":"constructed-process","capture_id":"constructed-capture","pid":123,"device":0,"worker":7,"sequence":4,"event":"worker_closed","observed_ns":40,"host_clock":"std.chrono.steady_clock","gpu_clock":"globaltimer","time_unit":"nanoseconds","counter_unit":"owned_completed_work","snapshot_atomic":false,"upstream_loss":"unknown","poll_ms":1000,"stale_ms":5000,"max_records":4096,"max_bytes":16777216,"terminal":true,"exporter_retained_before":3,"exporter_dropped_before":0,"untracked_registrations":0,"code":0,"mirror_present":false}
`

func progressCapture(raw []byte) ncclinspector.ProgressCapture {
	return ncclinspector.ProgressCapture{SHA256: fmt.Sprintf("%x", sha256.Sum256(raw)), BuildID: "constructed-build", Host: "constructed-host", BootEpoch: "constructed-boot", ProcessEpoch: "constructed-process", CaptureID: "constructed-capture", PID: 123, Worker: 7, Device: 0}
}
func progressWireRows(t *testing.T, raw []byte) []map[string]json.RawMessage {
	t.Helper()
	var rows []map[string]json.RawMessage
	for _, line := range bytes.Split(bytes.TrimSuffix(raw, []byte("\n")), []byte("\n")) {
		var row map[string]json.RawMessage
		if err := json.Unmarshal(line, &row); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
	}
	return rows
}
func progressWire(t *testing.T, rows []map[string]json.RawMessage) []byte {
	t.Helper()
	var out []byte
	for _, row := range rows {
		b, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, b...)
		out = append(out, '\n')
	}
	return out
}
func progressSet(row map[string]json.RawMessage, key string, value any) {
	b, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	row[key] = b
}
func progressClone(row map[string]json.RawMessage) map[string]json.RawMessage {
	out := make(map[string]json.RawMessage, len(row))
	for k, v := range row {
		out[k] = bytes.Clone(v)
	}
	return out
}
func progressPair(t *testing.T, mode string) []byte {
	t.Helper()
	rows := progressWireRows(t, []byte(progressCPPJSONL))
	launch := progressClone(rows[1])
	done := progressClone(rows[2])
	last := progressClone(rows[3])
	for _, row := range []map[string]json.RawMessage{launch, done} {
		progressSet(row, "generation", uint64(2))
		progressSet(row, "launch_start_ns", uint64(31))
		progressSet(row, "launch_end_ns", uint64(32))
	}
	progressSet(launch, "sequence", uint64(4))
	progressSet(launch, "exporter_retained_before", uint64(3))
	progressSet(launch, "observed_ns", uint64(32))
	progressSet(done, "sequence", uint64(5))
	progressSet(done, "exporter_retained_before", uint64(4))
	progressSet(done, "observed_ns", uint64(35))
	progressSet(done, "copy_success_min_ns", uint64(31))
	progressSet(done, "copy_success_max_ns", uint64(35))
	var counts []uint64
	if err := json.Unmarshal(done["completed_work_count"], &counts); err != nil {
		t.Fatal(err)
	}
	if mode == "increase" {
		counts[0] += 3
	}
	if mode == "decrease" {
		counts[0]--
	}
	progressSet(done, "completed_work_count", counts)
	progressSet(last, "sequence", uint64(6))
	progressSet(last, "exporter_retained_before", uint64(5))
	progressSet(last, "observed_ns", uint64(40))
	return progressWire(t, append(rows[:3], launch, done, last))
}
func progressRead(t *testing.T, raw []byte) *ncclinspector.ProgressTrace {
	t.Helper()
	trace, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), progressCapture(raw), ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	return trace
}

func TestProgressSnapshotCPPFormat(t *testing.T) {
	raw := []byte(progressCPPJSONL)
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != progressCPPDigest {
		t.Fatal("changed C++ output", got)
	}
	trace := progressRead(t, raw)
	rows := trace.Records()
	if len(rows) != 4 || rows[2].Mirror.Counts[0] != uint64(1<<53)+1 || rows[2].Mirror.GPUTimes[0] != ^uint64(0) || !rows[3].Terminal {
		t.Fatalf("lost exact source facts: %+v", rows)
	}
	rows[2].Mirror.Counts[0] = 0
	rows[0].Raw[0] = '!'
	if trace.Records()[2].Mirror.Counts[0] != uint64(1<<53)+1 || trace.Records()[0].Raw[0] != '{' {
		t.Fatal("mutable public alias")
	}
}

func TestProgressSnapshotRequiredPresence(t *testing.T) {
	rows := progressWireRows(t, []byte(progressCPPJSONL))
	for i, row := range rows {
		for key := range row {
			key := key
			i := i
			t.Run(fmt.Sprintf("%d_%s", i, key), func(t *testing.T) {
				cells := progressWireRows(t, []byte(progressCPPJSONL))
				delete(cells[i], key)
				raw := progressWire(t, cells)
				if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), progressCapture(raw), ncclinspector.Limits{}); err == nil {
					t.Fatal("accepted omitted field", key)
				}
			})
		}
	}
}

func TestProgressSnapshotWireRefusals(t *testing.T) {
	cases := []struct {
		name  string
		index int
		key   string
		value string
	}{
		{"unknown", 0, "invented", "1"}, {"schema", 0, "schema", `"stock-nccl"`}, {"unit", 0, "time_unit", `"microseconds"`}, {"host_clock", 0, "host_clock", `"wall"`},
		{"atomic", 0, "snapshot_atomic", "true"}, {"loss", 0, "upstream_loss", `"zero"`}, {"null", 0, "terminal", "null"}, {"pid_float", 0, "pid", "123.0"}, {"pid_exponent", 0, "pid", "123e0"}, {"pid_string", 0, "pid", `"123"`},
		{"count_overflow", 0, "exporter_dropped_before", "18446744073709551615"}, {"negative", 0, "sequence", "-1"}, {"retained_overflow", 0, "exporter_retained_before", "18446744073709551615"},
		{"config", 0, "max_bytes", "1"}, {"time_reverse", 1, "observed_ns", "0"}, {"hash", 0, "comm_hash", `"f"`}, {"rank", 0, "rank", "-1"},
		{"rank_high", 0, "rank", "2147483648"}, {"unknown_event", 0, "event", `"watchdog"`},
		{"missing_event", 2, "event_recorded", "false"}, {"stale", 2, "copy_stale", "true"}, {"generation", 2, "generation", "2"}, {"success_bound", 2, "copy_success_max_ns", "31"}, {"copy_code", 2, "code", "7"},
		{"array_short", 2, "completed_work_count", "[1]"}, {"array_null", 2, "completed_time_ns", "null"}, {"array_fraction", 2, "channel_active_slot_mask", "[1.5]"}, {"array_overflow", 2, "completed_work_count", "[18446744073709551616]"},
		{"terminal", 3, "terminal", "false"}, {"terminal_identity", 3, "rank", "2"}, {"counter_reset", 2, "counter_reset", `"known"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rows := progressWireRows(t, []byte(progressCPPJSONL))
			rows[tc.index][tc.key] = json.RawMessage(tc.value)
			raw := progressWire(t, rows)
			if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), progressCapture(raw), ncclinspector.Limits{}); err == nil {
				t.Fatal("accepted invalid wire")
			}
		})
	}
	for _, raw := range []string{strings.Replace(progressCPPJSONL, `"schema":`, `"schema":"duplicate","schema":`, 1), strings.Replace(progressCPPJSONL, `"schema":`, `"Schema":`, 1), strings.TrimSuffix(progressCPPJSONL, "\n"), progressCPPJSONL + strings.Split(progressCPPJSONL, "\n")[0] + "\n"} {
		if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), strings.NewReader(raw), progressCapture([]byte(raw)), ncclinspector.Limits{}); err == nil {
			t.Fatal("accepted duplicate, incomplete, aliased or post-terminal row")
		}
	}
}

func TestProgressSnapshotCopyAdmission(t *testing.T) {
	for _, mode := range []string{"failed_launch", "missing_launch", "stale_duration", "wrong_generation"} {
		t.Run(mode, func(t *testing.T) {
			rows := progressWireRows(t, []byte(progressCPPJSONL))
			switch mode {
			case "failed_launch":
				progressSet(rows[1], "event", "copy_launch_failed")
				progressSet(rows[1], "code", 7)
				progressSet(rows[1], "launch_code", 7)
				progressSet(rows[1], "copy_pending", false)
			case "missing_launch":
				rows = append(rows[:1], rows[2:]...)
				for i, row := range rows {
					progressSet(row, "sequence", i+1)
					progressSet(row, "exporter_retained_before", i)
				}
			case "stale_duration":
				progressSet(rows[2], "observed_ns", uint64(6000000000))
				progressSet(rows[2], "copy_success_max_ns", uint64(6000000000))
				progressSet(rows[3], "observed_ns", uint64(6000000001))
			case "wrong_generation":
				progressSet(rows[2], "generation", 2)
			}
			raw := progressWire(t, rows)
			if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), progressCapture(raw), ncclinspector.Limits{}); err == nil {
				t.Fatal("admitted invalid copy chain")
			}
		})
	}
}

type progressBrokenReader struct {
	calls int
	err   error
}

func (r *progressBrokenReader) Read([]byte) (int, error) { r.calls++; return 0, r.err }
func TestProgressSnapshotContextAndLimits(t *testing.T) {
	raw := []byte(progressCPPJSONL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	reader := &progressBrokenReader{err: io.EOF}
	_, err := ncclinspector.ReadProgressSnapshots(ctx, reader, progressCapture(raw), ncclinspector.Limits{})
	if !errors.Is(err, context.Canceled) || reader.calls != 0 {
		t.Fatal(err, reader.calls)
	}
	injected := errors.New("reader failed")
	_, err = ncclinspector.ReadProgressSnapshots(context.Background(), &progressBrokenReader{err: injected}, progressCapture(raw), ncclinspector.Limits{})
	if !errors.Is(err, injected) {
		t.Fatal(err)
	}
	for _, limits := range []ncclinspector.Limits{{Bytes: len(raw) - 1}, {Line: 10}, {Records: 3}, {Bytes: -1}} {
		if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), progressCapture(raw), limits); err == nil {
			t.Fatal("accepted reader limit", limits)
		}
	}
	c := progressCapture(raw)
	c.SHA256 = strings.Repeat("0", 64)
	if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), c, ncclinspector.Limits{}); err == nil {
		t.Fatal("digest ignored")
	}
}

func TestProgressSnapshotRetirementAndStickyStale(t *testing.T) {
	for _, mode := range []string{"retired", "stale"} {
		t.Run(mode, func(t *testing.T) {
			rows := progressWireRows(t, []byte(progressCPPJSONL))
			state := progressClone(rows[1])
			progressSet(state, "event_recorded", true)
			progressSet(state, "observed_ns", uint64(24))
			if mode == "retired" {
				progressSet(state, "event", "unregister_requested")
				drained := progressClone(state)
				progressSet(drained, "event", "unregistered_drained")
				progressSet(drained, "observed_ns", uint64(25))
				rows = append(rows[:2], state, drained, rows[2], rows[3])
			} else {
				progressSet(state, "event", "copy_pending_stale")
				progressSet(state, "copy_stale", true)
				rows = append(rows[:2], state, rows[2], rows[3])
			}
			for i, row := range rows {
				progressSet(row, "sequence", i+1)
				progressSet(row, "exporter_retained_before", i)
			}
			raw := progressWire(t, rows)
			if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), progressCapture(raw), ncclinspector.Limits{}); err == nil {
				t.Fatal("retired/stale registration became fresh", mode)
			}
		})
	}
}

func TestProgressSnapshotArrayElementPresence(t *testing.T) {
	for _, key := range []string{"completed_work_count", "completed_time_ns", "channel_active_slot_mask"} {
		t.Run(key, func(t *testing.T) {
			rows := progressWireRows(t, []byte(progressCPPJSONL))
			var cells []json.RawMessage
			if err := json.Unmarshal(rows[2][key], &cells); err != nil {
				t.Fatal(err)
			}
			cells[len(cells)/2] = json.RawMessage("null")
			value, err := json.Marshal(cells)
			if err != nil {
				t.Fatal(err)
			}
			rows[2][key] = value
			raw := progressWire(t, rows)
			if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), progressCapture(raw), ncclinspector.Limits{}); err == nil {
				t.Fatal("absent mirror integer became zero", key)
			}
		})
	}
}
