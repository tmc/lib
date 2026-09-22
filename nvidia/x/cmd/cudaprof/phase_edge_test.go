package main

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

func phaseResult(edges []capturePhaseEdge) captureResult {
	return captureResult{
		RunCompleteness:    runComplete,
		TransportReady:     true,
		CollectorFinalized: true,
		Capability:         []byte(`{"capture_range":"cudaProfilerApi","clocks_unified":true}`),
		Activities: []captureActivity{
			{Subtype: "kernel", Valid: true, Start: 20, End: 30},
			{Subtype: "kernel", Valid: true, Start: 40, End: 60},
			{Subtype: "kernel", Valid: true, Start: 80, End: 90},
		},
		PhaseReceipt: nil,
	}
}

func TestBuildPhaseReceiptMeasuresQualifiedPair(t *testing.T) {
	r := buildPhaseReceipt(phaseResult(nil), []capturePhaseEdge{
		{Sequence: 3, Timestamp: 10, Edge: "start", API: "cudaProfilerStart", Index: 7},
		{Sequence: 8, Timestamp: 70, Edge: "stop", API: "cudaProfilerStop", Index: 7},
	})
	if r.Status != "MEASURED" || r.API != "cudaProfilerStart" || r.DurationNanos != 60 {
		t.Fatalf("receipt = %+v, want measured 10..70", r)
	}
	if r.KernelCount != 2 || r.KernelNanos != 30 {
		t.Fatalf("kernel attribution = %d/%d, want 2/30", r.KernelCount, r.KernelNanos)
	}
}

func TestBuildPhaseReceiptIgnoresDeclaredRedundantEdges(t *testing.T) {
	edges := []capturePhaseEdge{
		{Sequence: 1, Timestamp: 10, Edge: "start", API: "cuProfilerStart", Index: 0},
		{Sequence: 2, Timestamp: 11, Edge: "redundant_start", API: "cudaProfilerStart", Index: 0},
		{Sequence: 3, Timestamp: 70, Edge: "stop", API: "cudaProfilerStop", Index: 0},
		{Sequence: 4, Timestamp: 71, Edge: "redundant_stop", API: "cuProfilerStop", Index: 1},
	}
	r := buildPhaseReceipt(phaseResult(nil), edges)
	if r.Status != "MEASURED" || r.DurationNanos != 60 {
		t.Fatalf("receipt = %+v, want measured primary pair", r)
	}
	if len(r.Edges) != len(edges) || r.Edges[1].Edge != "redundant_start" || r.Edges[3].Index != 1 {
		t.Fatalf("raw edges were not preserved: %+v", r.Edges)
	}
}

func TestObserveCaptureSessionBuildsMeasuredPhaseReceipt(t *testing.T) {
	dir := t.TempDir()
	session := "00112233445566778899aabbccddeeff"
	var sid [16]byte
	raw, err := hex.DecodeString(session)
	if err != nil {
		t.Fatal(err)
	}
	copy(sid[:], raw)
	h := captureHandshake{Protocol: 1, Session: session, Process: 41, Generation: 2, TransportReady: true}
	handshake, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), handshake, 0600); err != nil {
		t.Fatal(err)
	}
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability","capture_range":"cudaProfilerApi","clocks_unified":true}`)},
		{Kind: recordActivity, Timestamp: 10, Payload: []byte(`{"kind":"activity","subtype":"capture_range","edge":"start","api":"cudaProfilerStart","index":0,"toggle_failures":0}`)},
		{Kind: recordActivity, Timestamp: 20, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":11,"end":19}`)},
		{Kind: recordActivity, Timestamp: 30, Payload: []byte(`{"kind":"activity","subtype":"capture_range","edge":"stop","api":"cudaProfilerStop","index":0,"toggle_failures":0}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":1,"offered":6,"retained":6}`)},
	})
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.PhaseReceipt == nil || got.PhaseReceipt.Status != "MEASURED" {
		t.Fatalf("phase receipt = %+v, result = %+v", got.PhaseReceipt, got)
	}
	if got.PhaseReceipt.KernelCount != 1 || got.PhaseReceipt.KernelNanos != 8 {
		t.Fatalf("phase attribution = %+v, want one kernel / 8 ns", got.PhaseReceipt)
	}
}

func TestBuildPhaseReceiptRejectsInvalidPairs(t *testing.T) {
	tests := []struct {
		name   string
		edges  []capturePhaseEdge
		reason string
	}{
		{"duplicate", []capturePhaseEdge{
			{Sequence: 1, Timestamp: 10, Edge: "start", API: "cudaProfilerStart", Index: 1},
			{Sequence: 2, Timestamp: 20, Edge: "start", API: "cudaProfilerStart", Index: 1},
			{Sequence: 3, Timestamp: 30, Edge: "stop", API: "cudaProfilerStop", Index: 1},
		}, "edge_pair_invalid"},
		{"reversed", []capturePhaseEdge{
			{Sequence: 2, Timestamp: 20, Edge: "stop", API: "cudaProfilerStop", Index: 1},
			{Sequence: 3, Timestamp: 30, Edge: "start", API: "cudaProfilerStart", Index: 1},
		}, "edge_pair_invalid"},
		{"missing-stop", []capturePhaseEdge{
			{Sequence: 2, Timestamp: 20, Edge: "start", API: "cudaProfilerStart", Index: 1},
		}, "edge_pair_invalid"},
		{"toggle-failure", []capturePhaseEdge{
			{Sequence: 2, Timestamp: 20, Edge: "start", API: "cudaProfilerStart", Index: 1, ToggleFailures: 1},
			{Sequence: 3, Timestamp: 30, Edge: "stop", API: "cudaProfilerStop", Index: 1},
		}, "edge_pair_invalid"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			r := buildPhaseReceipt(phaseResult(nil), test.edges)
			if r.Status != "UNREAD" || r.Reason != test.reason {
				t.Fatalf("receipt = %+v, want UNREAD/%s", r, test.reason)
			}
		})
	}
}

func TestBuildPhaseReceiptRejectsUnsupportedClockAndIncompleteCapture(t *testing.T) {
	r := phaseResult(nil)
	r.Capability = []byte(`{"capture_range":"cudaProfilerApi","clocks_unified":false}`)
	got := buildPhaseReceipt(r, []capturePhaseEdge{
		{Sequence: 1, Timestamp: 10, Edge: "start", API: "cudaProfilerStart", Index: 1},
		{Sequence: 2, Timestamp: 20, Edge: "stop", API: "cudaProfilerStop", Index: 1},
	})
	if got.Reason != "clock_unified_false" {
		t.Fatalf("reason = %q, want clock_unified_false", got.Reason)
	}
	r = phaseResult(nil)
	r.RunCompleteness = runPartial
	got = buildPhaseReceipt(r, []capturePhaseEdge{
		{Sequence: 1, Timestamp: 10, Edge: "start", API: "cudaProfilerStart", Index: 1},
		{Sequence: 2, Timestamp: 20, Edge: "stop", API: "cudaProfilerStop", Index: 1},
	})
	if got.Reason != "capture_incomplete" {
		t.Fatalf("reason = %q, want capture_incomplete", got.Reason)
	}
}

func TestBuildPhaseReceiptBoundsRawEdges(t *testing.T) {
	edges := make([]capturePhaseEdge, maxPhaseEdges+1)
	for i := range edges {
		edges[i] = capturePhaseEdge{Sequence: uint64(i + 1), Timestamp: uint64(i + 1), Edge: "start", API: "cudaProfilerStart", Index: 1}
	}
	r := buildPhaseReceipt(phaseResult(nil), edges)
	if r.Status != "UNREAD" || r.Reason != "edge_limit" || r.OmittedEdges != 1 || len(r.Edges) != maxPhaseEdges {
		t.Fatalf("bounded receipt = %+v, want explicit edge_limit", r)
	}
}

func TestPhaseEdgeJournalRejectsMalformedAndTruncatedInput(t *testing.T) {
	var sid [16]byte
	decoded, err := hex.DecodeString("00112233445566778899aabbccddeeff")
	if err != nil {
		t.Fatal(err)
	}
	copy(sid[:], decoded)

	malformed := filepath.Join(t.TempDir(), "malformed.journal")
	appendRecords(t, malformed, sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"capture_range","edge":`)},
	})
	if _, _, _, _, _, _, err := observeJournal(malformed, "00112233445566778899aabbccddeeff", 41, 0); err == nil {
		t.Fatal("malformed phase edge journal decoded successfully")
	}

	valid := filepath.Join(t.TempDir(), "truncated.journal")
	appendRecords(t, valid, sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"capture_range","edge":"start","api":"cudaProfilerStart","index":0}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected"}`)},
	})
	b, err := os.ReadFile(valid)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(valid, b[:len(b)-1], 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, _, _, err := observeJournal(valid, "00112233445566778899aabbccddeeff", 41, 0); err == nil {
		t.Fatal("truncated phase edge journal decoded successfully")
	}
}

// TestBuildPhaseReceiptMultiplePhases covers a target that opens the capture
// range once per phase, which is what ds4-min does with -cuda-profiler-range.
// The enclosing receipt must stay UNREAD, because several intervals have no one
// duration, while still carrying a qualified receipt per phase: refusing to
// summarize must not mean discarding the measurement.
func TestBuildPhaseReceiptMultiplePhases(t *testing.T) {
	result := phaseResult(nil)
	result.Activities = []captureActivity{
		{Subtype: "kernel", Valid: true, Start: 12, End: 18}, // phase 0
		{Subtype: "kernel", Valid: true, Start: 14, End: 16}, // phase 0
		{Subtype: "kernel", Valid: true, Start: 32, End: 39}, // phase 1
		{Subtype: "kernel", Valid: true, Start: 20, End: 28}, // between phases, in neither
	}
	r := buildPhaseReceipt(result, []capturePhaseEdge{
		{Sequence: 1, Timestamp: 10, Edge: "start", API: "cuProfilerStart", Index: 0},
		{Sequence: 2, Timestamp: 19, Edge: "stop", API: "cuProfilerStop", Index: 0},
		{Sequence: 3, Timestamp: 30, Edge: "start", API: "cuProfilerStart", Index: 1},
		{Sequence: 4, Timestamp: 40, Edge: "stop", API: "cuProfilerStop", Index: 1},
	})
	if r.Status != "UNREAD" || r.Reason != "multiple_phases" {
		t.Fatalf("status/reason = %q/%q, want UNREAD/multiple_phases", r.Status, r.Reason)
	}
	if r.DurationNanos != 0 || r.KernelCount != 0 {
		t.Fatalf("enclosing receipt reported a measurement: duration=%d kernels=%d", r.DurationNanos, r.KernelCount)
	}
	if len(r.Phases) != 2 {
		t.Fatalf("phases = %d, want 2", len(r.Phases))
	}
	for i, want := range []struct {
		index                uint32
		start, stop, kernels uint64
		nanos                uint64
	}{
		{0, 10, 19, 2, 8},
		{1, 30, 40, 1, 7},
	} {
		got := r.Phases[i]
		if got.Status != "MEASURED" {
			t.Errorf("phase %d status = %q, want MEASURED", i, got.Status)
		}
		if got.Index != want.index || got.StartTimestamp != want.start || got.StopTimestamp != want.stop {
			t.Errorf("phase %d = index %d [%d,%d), want index %d [%d,%d)", i,
				got.Index, got.StartTimestamp, got.StopTimestamp, want.index, want.start, want.stop)
		}
		if got.KernelCount != want.kernels || got.KernelNanos != want.nanos {
			t.Errorf("phase %d kernels = %d/%dns, want %d/%dns", i, got.KernelCount, got.KernelNanos, want.kernels, want.nanos)
		}
	}
}

// TestBuildPhaseReceiptRejectsUnpairedEdges checks that an odd or malformed
// edge run is still refused outright. Splitting edges into pairs must not make
// a half-open range readable.
func TestBuildPhaseReceiptRejectsUnpairedEdges(t *testing.T) {
	for _, tt := range []struct {
		name  string
		edges []capturePhaseEdge
	}{
		{"odd count", []capturePhaseEdge{
			{Sequence: 1, Timestamp: 10, Edge: "start", API: "cuProfilerStart", Index: 0},
			{Sequence: 2, Timestamp: 19, Edge: "stop", API: "cuProfilerStop", Index: 0},
			{Sequence: 3, Timestamp: 30, Edge: "start", API: "cuProfilerStart", Index: 1},
		}},
		{"second pair inverted", []capturePhaseEdge{
			{Sequence: 1, Timestamp: 10, Edge: "start", API: "cuProfilerStart", Index: 0},
			{Sequence: 2, Timestamp: 19, Edge: "stop", API: "cuProfilerStop", Index: 0},
			{Sequence: 3, Timestamp: 40, Edge: "stop", API: "cuProfilerStop", Index: 1},
			{Sequence: 4, Timestamp: 30, Edge: "start", API: "cuProfilerStart", Index: 1},
		}},
		{"second pair index mismatch", []capturePhaseEdge{
			{Sequence: 1, Timestamp: 10, Edge: "start", API: "cuProfilerStart", Index: 0},
			{Sequence: 2, Timestamp: 19, Edge: "stop", API: "cuProfilerStop", Index: 0},
			{Sequence: 3, Timestamp: 30, Edge: "start", API: "cuProfilerStart", Index: 1},
			{Sequence: 4, Timestamp: 40, Edge: "stop", API: "cuProfilerStop", Index: 2},
		}},
		{"toggle failure in the second pair", []capturePhaseEdge{
			{Sequence: 1, Timestamp: 10, Edge: "start", API: "cuProfilerStart", Index: 0},
			{Sequence: 2, Timestamp: 19, Edge: "stop", API: "cuProfilerStop", Index: 0},
			{Sequence: 3, Timestamp: 30, Edge: "start", API: "cuProfilerStart", Index: 1, ToggleFailures: 1},
			{Sequence: 4, Timestamp: 40, Edge: "stop", API: "cuProfilerStop", Index: 1},
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := buildPhaseReceipt(phaseResult(nil), tt.edges)
			if r.Status != "UNREAD" || r.Reason != "edge_pair_invalid" {
				t.Fatalf("status/reason = %q/%q, want UNREAD/edge_pair_invalid", r.Status, r.Reason)
			}
			if len(r.Phases) != 0 {
				t.Fatalf("refused receipt still carried %d phases", len(r.Phases))
			}
		})
	}
}

// TestMeasurePhaseSeparatesSumFromUnion pins the distinction that decides
// whether a phase was GPU-bound. KernelNanos is a SUM over records, so
// concurrent work on separate streams counts more than once and the sum can
// exceed the phase. DeviceBusyNanos is the UNION and cannot, so busy and idle
// must partition the phase exactly.
func TestMeasurePhaseSeparatesSumFromUnion(t *testing.T) {
	result := phaseResult(nil)
	result.Activities = []captureActivity{
		{Subtype: "kernel", Valid: true, Start: 10, End: 40}, // stream A
		{Subtype: "kernel", Valid: true, Start: 20, End: 50}, // stream B, overlaps A
		{Subtype: "memcpy", Valid: true, Start: 70, End: 80, Bytes: 4096},
		{Subtype: "memset", Valid: true, Start: 75, End: 85},  // overlaps the memcpy
		{Subtype: "kernel", Valid: true, Start: 95, End: 200}, // straddles the stop edge
		{Subtype: "kernel", Valid: false, Start: 60, End: 65}, // invalid, not counted
	}
	r := buildPhaseReceipt(result, []capturePhaseEdge{
		{Sequence: 1, Timestamp: 5, Edge: "start", API: "cuProfilerStart", Index: 0},
		{Sequence: 2, Timestamp: 105, Edge: "stop", API: "cuProfilerStop", Index: 0},
	})
	if r.Status != "MEASURED" {
		t.Fatalf("status = %q reason = %q, want MEASURED", r.Status, r.Reason)
	}
	// Kernels: [10,40) and [20,50) are contained; [95,200) straddles the stop
	// edge and the Valid:false record is skipped.
	if r.KernelCount != 2 || r.KernelNanos != 60 {
		t.Errorf("kernels = %d/%dns, want 2/60ns (a sum: 30+30)", r.KernelCount, r.KernelNanos)
	}
	if r.MemcpyCount != 1 || r.MemcpyNanos != 10 || r.MemcpyBytes != 4096 {
		t.Errorf("memcpy = %d/%dns/%dB, want 1/10ns/4096B", r.MemcpyCount, r.MemcpyNanos, r.MemcpyBytes)
	}
	if r.MemsetCount != 1 || r.MemsetNanos != 10 {
		t.Errorf("memset = %d/%dns, want 1/10ns", r.MemsetCount, r.MemsetNanos)
	}
	// Union: [10,50) is 40, [70,85) is 15. Total 55, below the 80ns of sums.
	if r.DeviceBusyNanos != 55 {
		t.Errorf("device_busy = %dns, want 55ns (union, not the 80ns sum)", r.DeviceBusyNanos)
	}
	if sum := r.KernelNanos + r.MemcpyNanos + r.MemsetNanos; sum <= r.DeviceBusyNanos {
		t.Errorf("sum %dns did not exceed union %dns; the fixture no longer overlaps", sum, r.DeviceBusyNanos)
	}
	if r.DeviceBusyNanos+r.DeviceIdleNanos != r.DurationNanos {
		t.Errorf("busy %d + idle %d != duration %d", r.DeviceBusyNanos, r.DeviceIdleNanos, r.DurationNanos)
	}
}

// TestMeasurePhaseBusyNeverExceedsDuration is the property the ds4 comparison
// rests on: a GPU-busy fraction above 1 would mean the union leaked work from
// outside the phase, and every ratio computed from it would be wrong.
func TestMeasurePhaseBusyNeverExceedsDuration(t *testing.T) {
	result := phaseResult(nil)
	for i := 0; i < 64; i++ {
		result.Activities = append(result.Activities, captureActivity{
			Subtype: "kernel", Valid: true,
			Start: uint64(10 + i%8), End: uint64(90 - i%8),
		})
	}
	r := buildPhaseReceipt(result, []capturePhaseEdge{
		{Sequence: 1, Timestamp: 5, Edge: "start", API: "cuProfilerStart", Index: 0},
		{Sequence: 2, Timestamp: 105, Edge: "stop", API: "cuProfilerStop", Index: 0},
	})
	if r.Status != "MEASURED" {
		t.Fatalf("status = %q", r.Status)
	}
	if r.KernelNanos <= r.DurationNanos {
		t.Fatalf("kernel sum %d did not exceed duration %d; fixture is not concurrent", r.KernelNanos, r.DurationNanos)
	}
	if r.DeviceBusyNanos > r.DurationNanos {
		t.Fatalf("device_busy %d exceeds duration %d", r.DeviceBusyNanos, r.DurationNanos)
	}
	if r.DeviceBusyNanos+r.DeviceIdleNanos != r.DurationNanos {
		t.Fatalf("busy %d + idle %d != duration %d", r.DeviceBusyNanos, r.DeviceIdleNanos, r.DurationNanos)
	}
}
