package ncclinspector_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// Independent constructed Inspector control, read by the existing public reader.
// No operation/sequence is derived from diagnostic counter slot0.
const progressInspectorJSONL = `{"header":{"id":"0xffffffffffffffff","rank":2,"n_ranks":3,"nnodes":1},"metadata":{"inspector_output_format_version":"v4.1","git_rev":"constructed-inspector","rec_mechanism":"constructed","dump_timestamp_us":1,"hostname":"constructed-host","pid":123},"coll_perf":{"coll":"AllReduce","coll_sn":71,"coll_msg_size_bytes":8,"coll_exec_time_us":2,"coll_algobw_gbs":0,"coll_busbw_gbs":0}}
`

func progressInspector(t *testing.T) ncclinspector.Record {
	t.Helper()
	rows, err := ncclinspector.Read(strings.NewReader(progressInspectorJSONL), ncclinspector.Limits{})
	if err != nil || len(rows) != 1 {
		t.Fatal(rows, err)
	}
	return rows[0]
}
func progressBinding(raw []byte) ncclinspector.ProgressBinding {
	return ncclinspector.ProgressBinding{Capture: progressCapture(raw), Communicator: "0xffffffffffffffff", Rank: 2, Registration: 1, Evidence: "constructed-independent-association"}
}
func progressSummary(t *testing.T, raw []byte) ncclinspector.ProgressSummary {
	t.Helper()
	out, err := ncclinspector.SummarizeProgress(progressRead(t, raw), progressBinding(raw), progressInspector(t))
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestProgressSummaryPublicChain(t *testing.T) {
	for _, mode := range []string{"increase", "unchanged", "decrease"} {
		t.Run(mode, func(t *testing.T) {
			raw := progressPair(t, mode)
			out := progressSummary(t, raw)
			if !out.ExporterComplete || out.UpstreamLoss != "unknown" || out.ResetCoverage != "unknown" || !slices.Contains(out.Withheld, "stall") || !slices.Contains(out.Withheld, "GPU completion") || !slices.Contains(out.Withheld, "population zero loss") {
				t.Fatalf("promoted diagnostic evidence: %+v", out)
			}
			if out.Inspector.Collective.Sequence != 71 || out.Inspector.Metadata.GitRevision != "constructed-inspector" || out.Asserted != progressBinding(raw) {
				t.Fatal("lost actual Inspector facts or asserted provenance")
			}
			if mode == "decrease" {
				if len(out.Comparisons) != 0 || !slices.Contains(out.Withheld, "decreasing diagnostic counter pair") {
					t.Fatal("counter decrease admitted", out)
				}
				return
			}
			if len(out.Comparisons) != 1 || len(out.Comparisons[0].Slots) != 16 {
				t.Fatal("public counter comparison missing", out)
			}
			slot := out.Comparisons[0].Slots[0]
			want := uint64(0)
			if mode == "increase" {
				want = 3
			}
			if slot.Before != uint64(1<<53)+1 || slot.After-slot.Before != want || slot.Delta != want || slot.Status != map[bool]string{true: "increased", false: "unchanged"}[want != 0] {
				t.Fatal("wrong observed arithmetic", slot)
			}
			if out.Comparisons[0].Slots[15].Status != "unchanged" {
				t.Fatal("synthetic P2P slot became an operation sequence")
			}
		})
	}
}

func TestProgressSummaryIncompleteWithholds(t *testing.T) {
	for _, mode := range []string{"missing_terminal", "reported_drop", "retained_gap", "untracked", "initialization_failed"} {
		t.Run(mode, func(t *testing.T) {
			rows := progressWireRows(t, progressPair(t, "increase"))
			switch mode {
			case "missing_terminal":
				rows = rows[:len(rows)-1]
			case "reported_drop":
				for i := 1; i < len(rows); i++ {
					progressSet(rows[i], "exporter_dropped_before", 1)
					progressSet(rows[i], "sequence", i+2)
				}
			case "retained_gap":
				rows = append(rows[:1], rows[2:]...)
			case "untracked":
				for _, row := range rows {
					progressSet(row, "untracked_registrations", 1)
				}
			case "initialization_failed":
				progressSet(rows[len(rows)-1], "event", "initialization_failed")
			}
			raw := progressWire(t, rows)
			out := progressSummary(t, raw)
			if out.ExporterComplete || len(out.Comparisons) != 0 || !slices.Contains(out.Withheld, "incomplete or lossy exporter artifact") {
				t.Fatal("missing/lost output became complete", out)
			}
		})
	}
}

func TestProgressSummaryFailureBreaksComparison(t *testing.T) {
	rows := progressWireRows(t, progressPair(t, "increase"))
	failed := progressClone(rows[3])
	progressSet(failed, "event", "copy_launch_failed")
	progressSet(failed, "code", 7)
	progressSet(failed, "launch_code", 7)
	progressSet(failed, "copy_pending", false)
	// A third successful copy follows the failed generation. It cannot bridge
	// generation1 to3 as observed unchanged/progress across an unobserved attempt.
	launch := progressClone(rows[3])
	done := progressClone(rows[4])
	for _, row := range []map[string]json.RawMessage{launch, done} {
		progressSet(row, "generation", 3)
	}
	rows = append(rows[:3], failed, launch, done, rows[len(rows)-1])
	for i, row := range rows {
		progressSet(row, "sequence", i+1)
		progressSet(row, "exporter_retained_before", i)
	}
	raw := progressWire(t, rows)
	out := progressSummary(t, raw)
	if !out.ExporterComplete || len(out.Comparisons) != 0 {
		t.Fatal("failed attempt was bridged", out)
	}
}

func TestProgressSummaryBindingRefusals(t *testing.T) {
	raw := progressPair(t, "increase")
	trace := progressRead(t, raw)
	for _, name := range []string{"hash", "rank", "boot", "process", "capture", "worker", "registration", "build", "edited_record", "absent_raw", "absent_trace"} {
		t.Run(name, func(t *testing.T) {
			b := progressBinding(raw)
			r := progressInspector(t)
			tr := trace
			switch name {
			case "hash":
				b.Communicator = "0xffffffff"
			case "rank":
				b.Rank = 1
			case "boot":
				b.Capture.BootEpoch = "different"
			case "process":
				b.Capture.ProcessEpoch = "different"
			case "capture":
				b.Capture.CaptureID = "different"
			case "worker":
				b.Capture.Worker++
			case "registration":
				b.Registration++
			case "build":
				b.Capture.BuildID = "different"
			case "edited_record":
				r.Metadata.PID++
			case "absent_raw":
				r.Raw = nil
			case "absent_trace":
				tr = &ncclinspector.ProgressTrace{}
			}
			if _, err := ncclinspector.SummarizeProgress(tr, b, r); err == nil {
				t.Fatal("mismatched provenance admitted")
			}
		})
	}
}

func TestProgressSummaryInspectorDropsDoNotCoverMonitor(t *testing.T) {
	rows := progressWireRows(t, progressPair(t, "increase"))
	rows = rows[:len(rows)-1]
	raw := progressWire(t, rows)
	var inspector map[string]json.RawMessage
	if err := json.Unmarshal([]byte(progressInspectorJSONL), &inspector); err != nil {
		t.Fatal(err)
	}
	delete(inspector, "coll_perf")
	inspector["dump_stats"] = json.RawMessage(`{"coll_records":0,"coll_dropped_total":0,"coll_dropped_since_last_dump":0,"p2p_records":0,"p2p_dropped_total":0,"p2p_dropped_since_last_dump":0}`)
	b, err := json.Marshal(inspector)
	if err != nil {
		t.Fatal(err)
	}
	b = append(b, '\n')
	actual, err := ncclinspector.Read(bytes.NewReader(b), ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	out, err := ncclinspector.SummarizeProgress(progressRead(t, raw), progressBinding(raw), actual[0])
	if err != nil {
		t.Fatal(err)
	}
	if out.Inspector.Stats == nil || out.ExporterComplete || len(out.Comparisons) != 0 || out.UpstreamLoss != "unknown" {
		t.Fatal("Inspector zero covered missing mirror terminal", out)
	}
}

func TestProgressSummaryReaderCaptureIdentity(t *testing.T) {
	raw := []byte(progressCPPJSONL)
	for _, name := range []string{"boot", "process", "capture", "host", "pid", "worker", "device", "build"} {
		t.Run(name, func(t *testing.T) {
			c := progressCapture(raw)
			switch name {
			case "boot":
				c.BootEpoch = "changed"
			case "process":
				c.ProcessEpoch = "changed"
			case "capture":
				c.CaptureID = "changed"
			case "host":
				c.Host = "changed"
			case "pid":
				c.PID++
			case "worker":
				c.Worker++
			case "device":
				c.Device++
			case "build":
				c.BuildID = "changed"
			}
			if _, err := ncclinspector.ReadProgressSnapshots(context.Background(), bytes.NewReader(raw), c, ncclinspector.Limits{}); err == nil {
				t.Fatal(fmt.Sprintf("reader ignored %s", name))
			}
		})
	}
}
