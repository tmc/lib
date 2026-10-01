package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

func TestTraceComparisonRecomputesBoundShapes(t *testing.T) {
	dir := t.TempDir()
	var paths, hashes [2]string
	for i := range 2 {
		b := newJournalBuilder()
		text := `cudaprof:v1 {"phase":"decode","token":0,"operation":"gemv","dtype":"float32","cache_length":32}`
		b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "marker", "valid": true, "name": text, "marker_id": 1, "marker_flags": timeline.MarkerStart, "thread_id": 2, "process_id": 7, "start": 1, "end": 1})
		b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "marker", "valid": true, "marker_id": 1, "marker_flags": timeline.MarkerEnd, "thread_id": 2, "process_id": 7, "start": 4, "end": 4})
		b.append(wire.KindAPI, map[string]any{"kind": "api", "domain": "driver", "correlation": 1, "thread_id": 2, "process_id": 7, "start": 2, "end": 3})
		b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "kernel", "valid": true, "name": "gemv<float>", "correlation": 1, "start": 10, "end": 20 + 5*i, "grid_x": 1, "grid_y": 1, "grid_z": 1, "block_x": 32, "block_y": 1, "block_z": 1})
		paths[i] = filepath.Join(dir, []string{"base.pprof", "target.pprof"}[i])
		writeTestCaptureFile(t, paths[i], testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus()))
		source, err := loadTraceSession(paths[i], "driver")
		if err != nil {
			t.Fatal(err)
		}
		hashes[i] = source.SHA256
	}
	var shape bytes.Buffer
	if err := runShapeCompare([]string{paths[0], paths[1]}, &shape); err != nil {
		t.Fatal(err)
	}
	shapePath := filepath.Join(dir, "shape.json")
	if err := os.WriteFile(shapePath, shape.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	archive := traceArchiveFixture(t, func(members map[string][]byte) {
		var a alignmentReceipt
		if err := json.Unmarshal(members[".alignment.json"], &a); err != nil {
			t.Fatal(err)
		}
		for i := range 2 {
			a.Arms[i].SHA256 = hashes[i]
			a.Arms[i].Path = paths[i]
		}
		members[".alignment.json"], _ = json.Marshal(a)
		index := tracecontract.Bindings{Version: 1, Schema: "cudaprof.bindings.v1"}
		for name, data := range members {
			if name == ".cudaprof-bindings.json" {
				continue
			}
			kind, source := "metadata", ""
			for i, arm := range a.Arms {
				if name == arm.Trace {
					kind = "trace"
					source = hashes[i]
				}
			}
			member, err := tracecontract.Digest(name, kind, source, data)
			if err != nil {
				t.Fatal(err)
			}
			index.Members = append(index.Members, member)
		}
		members[".cudaprof-bindings.json"], _ = index.JSON()
	})
	archivePath := filepath.Join(dir, "comparison.tar")
	if err := os.WriteFile(archivePath, archive, 0600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	args := []string{"-json", "-view", "comparison", "-shape", shapePath, archivePath}
	if err := runTraceSummary(args, &out); err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		Evidence comparisonSummary `json:"evidence"`
	}
	if err := json.Unmarshal(out.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if !envelope.Evidence.ShapeQualified || len(envelope.Evidence.Operations) != 1 || envelope.Evidence.Operations[0].SumDelta != 5 {
		t.Fatalf("summary: %s", out.String())
	}
	if !strings.Contains(out.String(), "full-capture stationarity unavailable") {
		t.Fatal("missing independent stationarity state")
	}
	// An asserted pass bit cannot substitute for the recomputed shape result.
	var receipt comparisonShapeReceipt
	if err := json.Unmarshal(shape.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	receipt.Shape.Groups[0].TargetDurations.Sum++
	bad, _ := json.Marshal(receipt)
	if err := os.WriteFile(shapePath, bad, 0600); err != nil {
		t.Fatal(err)
	}
	if err := runTraceSummary(args, new(bytes.Buffer)); err == nil {
		t.Fatal("accepted forged numerical shape receipt")
	}
}

func TestTraceCampaignDoesNotTrustInterval(t *testing.T) {
	c := campaignReport{Version: 1, Domain: "driver", Marker: "kernel", ExpectedMarkers: [2]int{2, 2}, Blocks: []campaignBlock{{Number: 1, Excluded: []string{"failed"}}, {Number: 2, Excluded: []string{"failed"}}, {Number: 3, Excluded: []string{"failed"}}}}
	if err := validateTraceCampaign(&c, defaultTraceInputLimits()); err != nil {
		t.Fatal(err)
	}
	c.Interval = &campaignInterval{Blocks: 3, Median: 10, Lower: 9, Upper: 11}
	if err := validateTraceCampaign(&c, defaultTraceInputLimits()); err == nil {
		t.Fatal("accepted interval without any included block evidence")
	}
}
