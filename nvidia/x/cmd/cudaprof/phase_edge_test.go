package main

import (
	"encoding/hex"
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
