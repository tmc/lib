package main

import (
	"bytes"
	"encoding/json"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestShapeCompareArgs(t *testing.T) {
	for _, tt := range []struct {
		name string
		args []string
		want string
	}{
		{"missing", nil, "usage:"},
		{"domain", []string{"-domain=other", "base", "target"}, "invalid API domain"},
		{"limit", []string{"-max-records=-1", "base", "target"}, "negative decode limit"},
		{"marker", []string{"-iteration-marker=[", "base", "target"}, "iteration marker"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			err := runShapeCompare(tt.args, &out)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestShapeCompareCaptureReceipts(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	scratch := filepath.Join(home, "tmp")
	if err := os.MkdirAll(scratch, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(scratch, "shape-compare-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	b := newJournalBuilder()
	text := `cudaprof:v1 {"phase":"decode","token":0,"operation":"gemv","dtype":"float32","cache_length":32}`
	for _, m := range []struct {
		name  string
		flags uint32
		at    int
	}{{text, timeline.MarkerStart, 1}, {"", timeline.MarkerEnd, 4}} {
		b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "marker", "valid": true, "name": m.name, "marker_id": 1, "marker_flags": m.flags, "thread_id": 2, "process_id": 7, "start": m.at, "end": m.at})
	}
	b.append(wire.KindAPI, map[string]any{"kind": "api", "domain": "driver", "correlation": 1, "thread_id": 2, "process_id": 7, "start": 2, "end": 3})
	b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "kernel", "valid": true, "name": "gemv<float>", "correlation": 1, "start": 10, "end": 20, "grid_x": 1, "grid_y": 1, "grid_z": 1, "block_x": 32, "block_y": 1, "block_z": 1})
	journal := b.finalize()
	write := func(path string, status map[string]any) {
		t.Helper()
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		err = cudacapture.Write(f, testCapture(map[string][]byte{"capture.journal": journal}, status))
		closeErr := f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
	}
	good := filepath.Join(dir, "good.pprof")
	write(good, completeStatus())
	var out bytes.Buffer
	if err := runShapeCompare([]string{good, good}, &out); err != nil {
		t.Fatal(err)
	}
	var report struct {
		Arms []struct {
			Completeness string
			SHA256       string
			Stationarity iteration.Gate
		} `json:"arms"`
		Shape shape.Report `json:"shape_qualification"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if !report.Shape.Pass || len(report.Shape.Groups) != 1 {
		t.Fatalf("shape report = %s", out.String())
	}
	if len(report.Arms) != 2 || report.Arms[0].Stationarity.Pass || len(report.Arms[0].Stationarity.Failures) == 0 || report.Arms[0].Completeness != "passed" || len(report.Arms[0].SHA256) != 64 {
		t.Fatalf("missing marker/completeness receipts = %s", out.String())
	}
	partial := completeStatus()
	partial["run_completeness"] = "partial"
	bad := filepath.Join(dir, "partial.pprof")
	write(bad, partial)
	out.Reset()
	if err := runShapeCompare([]string{good, bad}, &out); err == nil || !strings.Contains(err.Error(), "run_completeness") {
		t.Fatalf("partial capture error = %v", err)
	}
	if out.Len() != 0 {
		t.Fatalf("partial result emitted: %s", out.String())
	}
}

func TestShapeRequestPairs(t *testing.T) {
	for _, tt := range []struct {
		name, text string
		valid      bool
	}{
		{"valid", `[{"base_journal":"a","base_process":1,"base_id":"r","target_journal":"b","target_process":2,"target_id":"s","role":"measured"}]`, true},
		{"unknown", `[{"extra":true}]`, false},
		{"duplicate", `[{"base_id":"r","base_id":"s"}]`, false},
		{"trailing", `[] []`, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "pairs.json")
			if err := os.WriteFile(path, []byte(tt.text), 0600); err != nil {
				t.Fatal(err)
			}
			_, err := readShapeRequestPairs(path)
			if (err == nil) != tt.valid {
				t.Fatalf("readShapeRequestPairs=%v", err)
			}
		})
	}
}
