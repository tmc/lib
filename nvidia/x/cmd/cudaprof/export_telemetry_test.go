package main

import (
	"bytes"
	"encoding/json"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"testing"
)

func TestExportTelemetryQualification(t *testing.T) {
	sample := newGPUEdgeSample(10, 12, 15, 20, []gpuReading{{UUID: "GPU-x"}})
	receipt := &capturePhaseReceipt{Edges: []capturePhaseEdge{{Sequence: 7, Timestamp: 10, Edge: "start", GPU: sample}}}
	cap := &cudacapture.Capture{Files: map[string][]byte{}}
	encode := func(unified bool) {
		capability, _ := json.Marshal(phaseCapability{ClocksUnified: unified})
		data, _ := json.Marshal(struct {
			Status captureStatus `json:"status"`
		}{captureStatus{captureResult: captureResult{Capability: capability, PhaseReceipt: receipt}}})
		cap.Files["capture-status.json"] = data
	}
	encode(false)
	if _, err := telemetryEdges(cap); err == nil {
		t.Fatal("accepted unknown clock mapping")
	}
	encode(true)
	edges, err := telemetryEdges(cap)
	if err != nil || len(edges) != 1 {
		t.Fatalf("edges = %v, %v", edges, err)
	}
	var b bytes.Buffer
	if err := writeTelemetryTrace(&b, edges, 10, 2<<48); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b.Bytes(), []byte("GPU-x")) || !bytes.Contains(b.Bytes(), []byte("no per-kernel interpolation")) {
		t.Fatal("lost query provenance")
	}
	if err := writeTelemetryTrace(&b, edges, 13, 2<<48); err == nil {
		t.Fatal("accepted negative timestamp")
	}
	sample.TimestampDomain = ""
	encode(true)
	if _, err := telemetryEdges(cap); err == nil {
		t.Fatal("accepted old query receipt")
	}
}
