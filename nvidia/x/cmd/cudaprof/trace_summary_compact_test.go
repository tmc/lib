package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/traceanalysis"
)

func compactFixture(t *testing.T, counts []*int, ordinals []int) traceanalysis.Summary {
	t.Helper()
	var records timeline.Records
	add := func(event, boundary string, when int64, token, count *int) {
		name, err := marker.Encode(marker.Metadata{Version: 2, RequestID: "same", Role: "measured", Phase: "decode", Event: event, Boundary: boundary, ExecutionUnit: "logical_forward", Token: token, Executions: count})
		if err != nil {
			t.Fatal(err)
		}
		records.Markers = append(records.Markers, timeline.Marker{Journal: "journal", Process: 7, Time: when, Flags: timeline.MarkerInstantaneous, Name: name, Sequence: uint64(len(records.Markers) + 1)})
	}
	zero := 0
	add("request_start", "submission", 0, nil, &zero)
	for i, c := range counts {
		ordinal := i
		if ordinals != nil {
			ordinal = ordinals[i]
		}
		add("delivered", "consumer_delivery", int64(10+i*10), &ordinal, c)
	}
	add("request_end", "completion", int64(20+len(counts)*10), nil, &zero)
	// End counter must not decrease; inherit the last delivery when available.
	if len(counts) > 0 && counts[len(counts)-1] != nil {
		name, err := marker.Encode(marker.Metadata{Version: 2, RequestID: "same", Role: "measured", Phase: "cleanup", Event: "request_end", Boundary: "completion", ExecutionUnit: "logical_forward", Executions: counts[len(counts)-1]})
		if err != nil {
			t.Fatal(err)
		}
		records.Markers[len(records.Markers)-1].Name = name
	}
	session, err := traceanalysis.New(records, traceanalysis.Options{CollectorComplete: true})
	if err != nil {
		t.Fatal(err)
	}
	return session.Summary()
}
func intPointer(n int) *int { return &n }
func TestCompactRequestCadence(t *testing.T) {
	for _, tt := range []struct {
		name   string
		counts []int
		hist   map[int]int
		tail   int
	}{
		{"zero two zero", []int{1, 1, 3, 3}, map[int]int{0: 2, 2: 1}, 1},
		{"all zero", []int{0, 0, 0}, map[int]int{0: 2}, 2},
		{"delta two is one interval", []int{1, 3}, map[int]int{2: 1}, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var counts []*int
			for _, n := range tt.counts {
				counts = append(counts, intPointer(n))
			}
			q := compactTraceSummary(compactFixture(t, counts, nil), traceanalysis.KernelShapes{}, 10).Requests[0]
			if q.ValidIntervals == nil || *q.ValidIntervals != len(counts)-1 || q.TrailingZero == nil || *q.TrailingZero != tt.tail {
				t.Fatalf("%+v", q)
			}
			for _, d := range q.Deltas {
				if tt.hist[d.Delta] != d.Intervals || d.Spacing.Count != d.Intervals {
					t.Fatalf("%+v", d)
				}
			}
			if q.Wall == nil || q.FirstDelivery == nil || *q.FirstDelivery != 10 {
				t.Fatalf("missing lifecycle: %+v", q)
			}
		})
	}
	counts := make([]*int, 32)
	for i := range counts {
		counts[i] = intPointer(min(i+1, 30))
	}
	q := compactTraceSummary(compactFixture(t, counts, nil), traceanalysis.KernelShapes{}, 10).Requests[0]
	if q.Delivered != 32 || q.Endpoints != 34 || *q.ValidIntervals != 31 || *q.TrailingZero != 2 || q.Deltas[0].Intervals != 2 || q.Deltas[1].Intervals != 29 {
		t.Fatalf("real cadence: %+v", q)
	}
}
func TestCompactRequestValidation(t *testing.T) {
	for _, tt := range []struct {
		name     string
		counts   []*int
		ordinals []int
	}{
		{"none", nil, nil}, {"one", []*int{intPointer(1)}, nil},
		{"missing", []*int{intPointer(1), nil}, nil},
		{"decreasing", []*int{intPointer(2), intPointer(1)}, nil},
		{"duplicate", []*int{intPointer(1), intPointer(2)}, []int{0, 0}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			q := compactTraceSummary(compactFixture(t, tt.counts, tt.ordinals), traceanalysis.KernelShapes{}, 10).Requests[0]
			if len(q.Unavailable) == 0 || q.ValidIntervals != nil || q.Wall != nil || q.FirstDelivery != nil || q.AllSpacing != nil || q.TrailingZero != nil || len(q.Deltas) != 0 {
				t.Fatalf("reconstructed invalid metrics: %+v", q)
			}
		})
	}
}
func TestCompactBounds(t *testing.T) {
	label := strings.Repeat("界", 10000)
	s := traceanalysis.Summary{Requests: make([]traceanalysis.Request, 1000)}
	for i := range s.Requests {
		s.Requests[i] = traceanalysis.Request{Journal: label, ID: label, Endpoints: make([]traceanalysis.Endpoint, 1000), Unavailable: make([]string, 1000)}
		for j := range s.Requests[i].Unavailable {
			s.Requests[i].Unavailable[j] = label
		}
	}
	shapes := traceanalysis.KernelShapes{Distributions: make([]traceanalysis.LaunchDistribution, 1000)}
	for i := range shapes.Distributions {
		shapes.Distributions[i] = traceanalysis.LaunchDistribution{Journal: label, Symbol: label, Sources: make([]traceanalysis.Source, 1000)}
		for j := range shapes.Distributions[i].Sources {
			shapes.Distributions[i].Sources[j].Journal = label
		}
	}
	out := compactTraceSummary(s, shapes, 2)
	data, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) > 20000 || out.OmittedRequests != 998 || out.OmittedShapes != 998 || out.Requests[0].OmittedSources != 996 || out.Requests[0].OmittedUnavailable != 992 || out.Shapes[0].OmittedSources != 996 {
		t.Fatalf("unbounded output (%d bytes): %+v", len(data), out.Requests[0])
	}
	again, _ := json.Marshal(compactTraceSummary(s, shapes, 2))
	if string(data) != string(again) {
		t.Fatal("nondeterministic output")
	}
	if out.Requests[0].ID.SHA256 == "" || out.Requests[0].ID.OmittedBytes == 0 {
		t.Fatal("truncated identity lacks hash")
	}
}

func TestCompactEnvelope(t *testing.T) {
	path := t.TempDir() + "/partial.pprof"
	writeTestCaptureFile(t, path, testCapture(nil, map[string]any{"run_completeness": "partial"}))
	for _, tt := range []struct {
		view    string
		version int
		schema  string
	}{{"overview", 2, "cudaprof.trace-summary.v2"}, {"detailed", 1, "cudaprof.trace-summary.v1"}} {
		var out bytes.Buffer
		if err := runTraceSummary([]string{"-json", "-view", tt.view, path}, &out); err != nil {
			t.Fatal(err)
		}
		var result traceViewResult
		if err := json.Unmarshal(out.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if result.Version != tt.version || result.Schema != tt.schema || result.SHA256 != fmt.Sprintf("%x", sha256.Sum256(data)) || result.State != "unavailable" || result.Qualifications["isolation"] == "clear" {
			t.Fatalf("lost qualification/identity: %+v", result)
		}
		if result.Consumer != currentBuild() {
			t.Fatal("consumer stamp changed")
		}
	}
	for _, n := range []string{"0", "101"} {
		if err := runTraceSummary([]string{"-top", n, path}, new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted top=%s", n)
		}
	}
}

func TestCompactIdentityAndVersions(t *testing.T) {
	var records timeline.Records
	for _, identity := range []struct {
		pid  uint32
		role string
	}{{1, "measured"}, {2, "measured"}, {1, "warmup"}} {
		for i, event := range []string{"request_start", "delivered", "delivered", "request_end"} {
			boundary := "consumer_delivery"
			if i == 0 {
				boundary = "submission"
			}
			if i == 3 {
				boundary = "completion"
			}
			count := i
			ordinal := i - 1
			m := marker.Metadata{Version: 2, RequestID: "same", Role: identity.role, Phase: "decode", Event: event, Boundary: boundary, Executions: &count, ExecutionUnit: "logical_forward"}
			if event == "delivered" {
				m.Token = &ordinal
			}
			name, err := marker.Encode(m)
			if err != nil {
				t.Fatal(err)
			}
			records.Markers = append(records.Markers, timeline.Marker{Journal: "p", Process: identity.pid, Name: name, Time: int64(i + 1), Flags: timeline.MarkerInstantaneous, Sequence: uint64(len(records.Markers) + 1)})
		}
	}
	ordinal := 0
	legacy, err := marker.Encode(marker.Metadata{Version: 1, Phase: "decode", Event: "delivered", Token: &ordinal})
	if err != nil {
		t.Fatal(err)
	}
	records.Markers = append(records.Markers, timeline.Marker{Journal: "p", Process: 1, Name: legacy, Time: 2, Flags: timeline.MarkerInstantaneous})
	session, err := traceanalysis.New(records, traceanalysis.Options{CollectorComplete: true})
	if err != nil {
		t.Fatal(err)
	}
	out := compactTraceSummary(session.Summary(), traceanalysis.KernelShapes{}, 10)
	if len(out.Requests) != 3 || out.MarkerVersions[2] != 12 || out.MarkerVersions[1] != 1 {
		t.Fatalf("merged identities or versions: %+v", out)
	}
	for _, r := range out.Requests {
		if r.ValidIntervals == nil || *r.ValidIntervals != 1 {
			t.Fatalf("%+v", r)
		}
		for _, source := range r.Sources {
			if source.Index < 0 || source.Index >= len(records.Markers) || source.Sequence != records.Markers[source.Index].Sequence {
				t.Fatalf("unresolvable source: %+v", source)
			}
		}
	}
}
func TestCompactNativeCompletionUnknown(t *testing.T) {
	cap, binding := isolationBindFixture(t)
	binding.CompletionKnown = false
	raw, _ := json.Marshal(map[string]any{"status": map[string]any{"isolation_binding": binding}})
	cap.Files["capture-status.json"] = raw
	cap.Files[isolation.Attachment] = isolationBindReceipt(t, binding, false)
	path := t.TempDir() + "/completion-unknown.pprof"
	writeTestCaptureFile(t, path, cap)
	var out bytes.Buffer
	if err := runTraceSummary([]string{"-json", path}, &out); err != nil {
		t.Fatal(err)
	}
	var result traceViewResult
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(result.Qualifications["isolation"], "unchecked:") || !strings.Contains(result.Qualifications["isolation"], "completion boundary unknown") {
		t.Fatalf("promoted unknown completion: %+v", result.Qualifications)
	}
}

// An optional retained capture exercises the existing reader and validator
// without building a CLI or running a new GPU workload.
func TestCompactRetainedCapture(t *testing.T) {
	path := os.Getenv("CUDAPROF_COMPACT_INPUT")
	if path == "" {
		t.Skip("set CUDAPROF_COMPACT_INPUT to a retained native capture")
	}
	var out bytes.Buffer
	if err := runTraceSummary([]string{"-json", path}, &out); err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		Schema         string            `json:"schema"`
		SHA            string            `json:"sha256"`
		Qualifications map[string]string `json:"qualifications"`
		Evidence       compactSummary    `json:"evidence"`
	}
	if err := json.Unmarshal(out.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Schema != "cudaprof.trace-summary.v2" || len(envelope.SHA) != 64 || len(out.Bytes()) > 100000 {
		t.Fatalf("unbounded/unbound retained summary: %d bytes", len(out.Bytes()))
	}
	if output := os.Getenv("CUDAPROF_COMPACT_OUTPUT"); output != "" {
		if err := os.WriteFile(output, out.Bytes(), 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("bytes=%d requests=%d collector=%s isolation=%s", out.Len(), len(envelope.Evidence.Requests), envelope.Qualifications["collector"], envelope.Qualifications["isolation"])
	for _, r := range envelope.Evidence.Requests {
		t.Logf("request=%s endpoints=%d delivered=%d intervals=%v tail=%v deltas=%+v", r.ID.Text, r.Endpoints, r.Delivered, compactOptionalInt(r.ValidIntervals), compactOptionalInt(r.TrailingZero), r.Deltas)
	}
}

func compactOptionalInt(n *int) any {
	if n == nil {
		return nil
	}
	return *n
}

func TestCompactLifecycleCounts(t *testing.T) {
	for _, tt := range []struct {
		name         string
		starts, ends int
		negative     bool
	}{
		{"valid", 1, 1, false}, {"missing start", 0, 1, false}, {"missing end", 1, 0, false},
		{"duplicate start", 2, 1, false}, {"duplicate end", 1, 2, false}, {"negative start counter", 1, 1, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var records timeline.Records
			count, ordinal := 0, 0
			add := func(event, boundary string) {
				m := marker.Metadata{Version: 2, RequestID: "r", Role: "control", Phase: "decode", Event: event, Boundary: boundary, ExecutionUnit: "logical_forward", Executions: &count}
				if event == "delivered" {
					m.Token = &ordinal
				}
				name, err := marker.Encode(m)
				if err != nil {
					t.Fatal(err)
				}
				if tt.negative && event == "request_start" {
					name = strings.Replace(name, `"executions":0`, `"executions":-1`, 1)
				}
				records.Markers = append(records.Markers, timeline.Marker{Journal: "p", Process: 1, Name: name, Time: int64(len(records.Markers) + 1), Flags: timeline.MarkerInstantaneous})
			}
			for i := 0; i < tt.starts; i++ {
				add("request_start", "submission")
			}
			count = 1
			add("delivered", "consumer_delivery")
			count = 2
			ordinal = 1
			add("delivered", "consumer_delivery")
			for i := 0; i < tt.ends; i++ {
				add("request_end", "completion")
			}
			session, err := traceanalysis.New(records, traceanalysis.Options{CollectorComplete: true})
			if err != nil {
				t.Fatal(err)
			}
			row := compactTraceSummary(session.Summary(), traceanalysis.KernelShapes{}, 10).Requests[0]
			starts := tt.starts
			if tt.negative {
				starts = 0
			}
			if row.Starts != starts || row.Ends != tt.ends || row.Delivered != 2 {
				t.Fatalf("assumed lifecycle counts: %+v", row)
			}
			valid := tt.name == "valid"
			if valid != (row.ValidIntervals != nil) || valid != (row.Wall != nil) {
				t.Fatalf("manufactured lifecycle metrics: %+v", row)
			}
		})
	}
}
