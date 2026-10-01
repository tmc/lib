package main

import (
	"archive/tar"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"regexp"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func TestExportCompareArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{"normal", []string{"-iteration-marker", "marker", "-iteration-count", "2", "a", "b"}, false},
		{"inferred summary", []string{"-summary", "-summary-marker", "marker", "-iteration-marker", "marker", "a", "b"}, false},
		{"invalid inferred marker", []string{"-summary-marker", "[", "-iteration-marker", "marker", "a", "b"}, true},
		{"runtime summary", []string{"-summary", "-summary-domain", "runtime", "-iteration-marker", "marker", "a", "b"}, false},
		{"invalid summary domain", []string{"-summary-domain", "automatic", "-iteration-marker", "marker", "a", "b"}, true},
		{"missing marker", []string{"a", "b"}, true},
		{"negative skip", []string{"-iteration-marker", "x", "-skip-iterations", "-1", "a", "b"}, true},
		{"same labels", []string{"-iteration-marker", "x", "-base-label", "x", "-target-label", "x", "a", "b"}, true},
		{"negative override", []string{"-iteration-marker", "x", "-target-skip-iterations", "-2", "a", "b"}, true},
		{"unequal skips", []string{"-iteration-marker", "x", "-base-skip-iterations", "2", "-target-skip-iterations", "0", "-demangle", "a", "b"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseExportCompareArgs(tt.args, io.Discard)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

func TestExportCompareWindowAndGates(t *testing.T) {
	cap := compareCapture(iterationRecs(5), completeStatus())
	_, records, err := decodeCompleteCapture(cap, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	w := iterationWindow{marker: regexp.MustCompile("Embed"), skip: 1, count: 2}
	arm := alignmentArm{Expected: 5, MarkerVerdict: "unchecked"}
	if err := selectAlignmentWindow(&arm, records, w); err != nil {
		t.Fatal(err)
	}
	if arm.Start != 2000 || arm.End != 4000 || arm.Iterations != 2 || arm.Markers != 5 || arm.MarkerVerdict != "passed" {
		t.Fatalf("arm=%+v", arm)
	}
	if !arm.Stationarity.Pass || !arm.FullGate.Pass {
		t.Fatalf("gates=%+v %+v", arm.Stationarity, arm.FullGate)
	}
	arm.Expected = 6
	if err := selectAlignmentWindow(&arm, records, w); err == nil {
		t.Fatal("missing invariant failure")
	}
	arm.Expected = 0
	arm.MarkerVerdict = "unchecked"
	records.Activities = append(records.Activities, timeline.Activity{Subtype: "kernel", Name: "extra", Start: 2100, End: 2200})
	if err := selectAlignmentWindow(&arm, records, w); err != nil {
		t.Fatal(err)
	}
	if arm.Stationarity.Pass || arm.FullGate.Pass {
		t.Fatal("alignment hid stationarity failure")
	}
	if arm.MarkerVerdict != "unchecked" {
		t.Fatal("invented workload invariant")
	}
	w.count = 8
	if err := selectAlignmentWindow(&arm, records, w); err == nil {
		t.Fatal("window beyond markers accepted")
	}
}

func TestExportCompareCompletenessRefusal(t *testing.T) {
	status := completeStatus()
	status["run_completeness"] = "partial"
	_, _, err := decodeCompleteCapture(compareCapture(iterationRecs(5), status), 0, 0)
	if err == nil {
		t.Fatal("partial capture accepted")
	}
}

func TestExportCompareBundle(t *testing.T) {
	makeArm := func(label string, origin, start, end int64) alignmentArm {
		return alignmentArm{Label: label, Trace: label + ".pftrace", Path: label + ".pprof", SHA256: strings.Repeat("a", 64), Origin: origin, Start: start, End: end, Records: 1, Completeness: "passed", MarkerVerdict: "unchecked", FullGate: iteration.Gate{Failures: []string{"variable kernels"}}, records: []traceRecord{{rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "kernel", Start: uint64(origin), End: uint64(end + 100), Complete: true}}}}
	}
	r := alignmentReceipt{Version: 1, TimeScale: 1, Marker: "marker", Arms: [2]alignmentArm{makeArm("base", 100, 300, 500), makeArm("target", 1000, 1500, 1800)}}
	if err := alignArms(&r); err != nil {
		t.Fatal(err)
	}
	if r.Anchor != 500 || r.Arms[0].Offset != 300 || r.Arms[1].Offset != 0 {
		t.Fatalf("offsets=%+v", r)
	}
	var b bytes.Buffer
	if err := writeComparisonBundle(&b, r); err != nil {
		t.Fatal(err)
	}
	tr := tar.NewReader(&b)
	members := map[string][]byte{}
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			t.Fatal(err)
		}
		members[h.Name] = data
	}
	if len(members) != 5 {
		t.Fatalf("got %d members", len(members))
	}
	var bindings tracecontract.Bindings
	if err := json.Unmarshal(members[".cudaprof-bindings.json"], &bindings); err != nil {
		t.Fatal(err)
	}
	if err := bindings.Verify(members); err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Manifest struct {
			Files []struct {
				Path    string
				Machine struct{ Name string }
				Clocks  struct {
					Clock  string
					Offset int64 `json:"offset_ns"`
				}
			}
			Time struct{ Clock, File string } `json:"trace_time"`
		} `json:"perfetto_manifest"`
	}
	if err := json.Unmarshal(members["perfetto_manifest.json"], &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Manifest.Files[1].Clocks.Offset != -300 || manifest.Manifest.Files[1].Clocks.Clock != "BOOTTIME" || manifest.Manifest.Time.File != "base.pftrace" {
		t.Fatalf("manifest=%+v", manifest)
	}
	if manifest.Manifest.Files[0].Machine.Name == manifest.Manifest.Files[1].Machine.Name {
		t.Fatal("machine collision")
	}
	var receipt alignmentReceipt
	if err := json.Unmarshal(members[".alignment.json"], &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.TimeScale != 1 || receipt.Arms[0].FullGate.Pass || receipt.Arms[1].Offset != 0 {
		t.Fatalf("receipt=%+v", receipt)
	}
	for _, name := range []string{"base.pftrace", "target.pftrace"} {
		data := members[name]
		for _, want := range []string{"kernel", "before selected window", "after selected window", "stationarity selected=false full=false"} {
			if !bytes.Contains(data, []byte(want)) {
				t.Errorf("%s missing %q", name, want)
			}
		}
	}
	r.Arms[0].records[0].rec.End = math.MaxInt64
	if err := alignArms(&r); err == nil {
		t.Fatal("overflowing shift accepted")
	}
}

func TestExportDemanglePreservesIdentity(t *testing.T) {
	const raw = "_ZN3mlx4core2cu6gatherIfiLi1ELi2EiEEvv"
	records := []traceRecord{{rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: raw, Start: 100, End: 200, Complete: true}}, {rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "not_mangled", Start: 300, End: 400, Complete: true}}}
	demangleTraceRecords(records)
	if records[0].rec.Name != raw || !strings.Contains(records[0].rec.DisplayName, "gather<float, int, 1, 2, int>") {
		t.Fatalf("name=%+v", records[0].rec)
	}
	if records[1].rec.DisplayName != "not_mangled" {
		t.Fatal("fallback changed")
	}
	var b bytes.Buffer
	if _, err := writeTraceOrigin(&b, records, 100, 1<<32, "arm: "); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{raw, "raw_symbol", "gather<float, int, 1, 2, int>"} {
		if !bytes.Contains(b.Bytes(), []byte(want)) {
			t.Errorf("trace missing %q", want)
		}
	}
	opts, err := parseExportArgs([]string{"-demangle", "capture.pprof"})
	if err != nil || !opts.demangle {
		t.Fatalf("demangle parse=%+v %v", opts, err)
	}
	if err := checkComparisonOutput("input.pprof", [2]string{"input.pprof", "other.pprof"}); err == nil {
		t.Fatal("output may overwrite input")
	}
}

func TestExportCompareSummaryClock(t *testing.T) {
	args, err := parseExportCompareArgs([]string{"-summary", "-iteration-marker", "x", "a", "b"}, io.Discard)
	if err != nil || !args.summary {
		t.Fatalf("summary option: %+v, %v", args, err)
	}
	var r alignmentReceipt
	for i, times := range [][3]int64{{100, 300, 500}, {1000, 1500, 1800}} {
		report := overview.Report{Hierarchy: &overview.Node{ID: "fixture", Kind: "capture", Label: "observed", Provenance: "observed", Metrics: overview.Metrics{GPUUnion: map[uint32][]overview.Interval{0: {{Journal: "j", Start: times[1], End: times[2]}}}}}}
		r.Arms[i] = alignmentArm{SHA256: strings.Repeat("a", 64), Origin: times[0], Start: times[1], End: times[2], Trace: fmt.Sprintf("arm-%d.pftrace", i), overview: &report, records: []traceRecord{{rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "raw", Start: uint64(times[0]), End: uint64(times[2]), Complete: true}}}}
	}
	if err := alignArms(&r); err != nil {
		t.Fatal(err)
	}
	var bundle bytes.Buffer
	if err := writeComparisonBundle(&bundle, r); err != nil {
		t.Fatal(err)
	}
	tr := tar.NewReader(&bundle)
	checked := 0
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			t.Fatal(err)
		}
		if h.Name != "arm-0.pftrace" && h.Name != "arm-1.pftrace" {
			continue
		}
		arm := 0
		if h.Name == "arm-1.pftrace" {
			arm = 1
		}
		if !bytes.Contains(data, []byte("raw")) || !bytes.Contains(data, []byte("GPU 0 activity union")) {
			t.Fatal("raw or overview lost")
		}
		wantStart := uint64(500)
		if arm == 1 {
			wantStart = 800
		}
		found := false
		for _, packet := range comparisonProtoFields(t, data)[perfetto.FieldTracePacket].bytes {
			fields := comparisonProtoFields(t, packet)
			for _, event := range fields[perfetto.FieldPacketTrackEvent].bytes {
				f := comparisonProtoFields(t, event)
				if f[perfetto.FieldTrackEventTrackUUID].value < uint64(arm+1)<<48 {
					continue
				}
				if fields[perfetto.FieldPacketTimestampClockId].value != perfetto.ClockBoottime {
					t.Fatal("overview packet clock differs")
				}
				if f[perfetto.FieldTrackEventType].value == perfetto.TrackEventTypeSliceBegin && fields[perfetto.FieldPacketTimestamp].value == wantStart {
					found = true
				}
			}
		}
		if !found {
			t.Fatalf("arm %d overview did not use aligned start %d", arm, wantStart)
		}
		checked++
	}
	if checked != 2 {
		t.Fatalf("checked %d arms", checked)
	}
}

type comparisonProtoField struct {
	value uint64
	bytes [][]byte
}

func comparisonProtoFields(t *testing.T, data []byte) map[int]comparisonProtoField {
	t.Helper()
	fields := make(map[int]comparisonProtoField)
	for len(data) > 0 {
		tag, n := binary.Uvarint(data)
		if n <= 0 {
			t.Fatal("invalid protobuf tag")
		}
		data = data[n:]
		number := int(tag >> 3)
		field := fields[number]
		switch tag & 7 {
		case 0:
			v, n := binary.Uvarint(data)
			if n <= 0 {
				t.Fatal("invalid protobuf varint")
			}
			field.value = v
			data = data[n:]
		case 2:
			size, n := binary.Uvarint(data)
			if n <= 0 || size > uint64(len(data)-n) {
				t.Fatal("invalid protobuf length")
			}
			field.bytes = append(field.bytes, data[n:n+int(size)])
			data = data[n+int(size):]
		case 1:
			if len(data) < 8 {
				t.Fatal("short fixed64")
			}
			data = data[8:]
		case 5:
			if len(data) < 4 {
				t.Fatal("short fixed32")
			}
			data = data[4:]
		default:
			t.Fatal("invalid protobuf type")
		}
		fields[number] = field
	}
	return fields
}

func TestExportCompareSeparateSummaryMarker(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		args := []string{"-summary", "-iteration-marker", "alignment", "a", "b"}
		if explicit {
			args = append([]string{"-summary-marker", "inferred"}, args...)
		}
		o, err := parseExportCompareArgs(args, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		if (o.summaryMarker != nil) != explicit {
			t.Fatal("alignment marker became inferred summary marker")
		}
		if explicit && o.summaryMarker.String() != "inferred" {
			t.Fatalf("summary marker = %v", o.summaryMarker)
		}
	}
}
