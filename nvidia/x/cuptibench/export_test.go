package cuptibench

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func exportFixture() *Result {
	return &Result{Operations: 2, Quality: cuptirange.CollectionQuality{PassesStarted: 3, PassesStopped: 3, AllPassesSubmitted: true, Decoded: true}, Metadata: map[string]string{"capture_id": "capture-1", "benchmark": "BenchmarkDecode/full", "run_id": "run-1", "mode": "regions", "preparation": "fixture; cache unspecified"}, Ranges: []Range{
		{ID: "root", Name: "operation", Kind: "operation", Inclusive: true, Metrics: []Metric{{Name: "counter", Unit: "bytes", Value: 100, Additive: true}}},
		{ID: "region", ParentID: "root", Name: "attention", Kind: "region", Inclusive: true, Metrics: []Metric{{Name: "counter", Unit: "bytes", Value: 80, Additive: true}}},
		{ID: "a", ParentID: "region", Name: "q_proj", Kind: "region", Inclusive: true, Metrics: []Metric{{Name: "counter", Unit: "bytes", Value: 20, Additive: true}, {Name: "rate", Unit: "percent", Value: 75}}},
		{ID: "b", ParentID: "region", Name: "out_proj", Kind: "region", Inclusive: true, Metrics: []Metric{{Name: "counter", Unit: "bytes", Value: 30, Additive: true}}},
	}}
}

func TestExportTerminalRegionsAndLabels(t *testing.T) {
	r := exportFixture()
	files, err := buildExport("capture.pprof", []*Result{r})
	if err != nil {
		t.Fatal(err)
	}
	p, err := profile.ParseData(files["capture.pprof"])
	if err != nil {
		t.Fatal(err)
	}
	if err := p.CheckValid(); err != nil {
		t.Fatal(err)
	}
	if comments := strings.Join(p.Comments, "\n"); !strings.Contains(comments, "capture capture-1: terminal-region projection may omit work outside leaves; its total is not the whole-operation total; inclusive root retained in JSON") {
		t.Fatalf("profile lacks terminal-region coverage comment: %v", p.Comments)
	}
	if len(p.SampleType) != 1 || p.SampleType[0].Type != "counter" {
		t.Fatalf("sample types = %+v", p.SampleType)
	}
	var sum int64
	for _, sample := range p.Sample {
		sum += sample.Value[0]
		if sample.Label["benchmark"][0] != "BenchmarkDecode/full" || sample.Label["profile-operations"][0] != "2" || sample.Label["capture_id"][0] != "capture-1" {
			t.Fatalf("labels = %v", sample.Label)
		}
		if _, ok := sample.Label["kernel"]; ok {
			t.Fatal("region mislabeled as kernel")
		}
	}
	if sum != 50 {
		t.Fatalf("sum = %d, want only leaf 20+30; no normalization by passes", sum)
	}
	var m exportManifest
	if err := json.Unmarshal(files["capture.pprof.json"], &m); err != nil {
		t.Fatal(err)
	}
	if len(m.Results[0].Ranges) != 4 || !slices.Equal(m.Coverage[0].Excluded, []string{"root", "region"}) {
		t.Fatalf("manifest = %+v", m)
	}
	if got := m.Results[0].Ranges[0].Metrics[0].Value; got != 100 {
		t.Fatalf("inclusive root = %g, want 100 retained independently of leaf sum %d", got, sum)
	}
	if got := m.Results[0].Ranges[2].Metrics[1].Value; got != 75 {
		t.Fatalf("lost ratio in JSON: %v", got)
	}
}

func TestExportPartitionsMissingMetrics(t *testing.T) {
	r := exportFixture()
	r.Ranges[3].Metrics = append(r.Ranges[3].Metrics, Metric{Name: "extra", Unit: "instructions", Value: 9, Additive: true})
	files, err := buildExport("capture", []*Result{r})
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 3 {
		t.Fatalf("files = %v", files)
	}
	for _, path := range []string{"capture", "capture.part2"} {
		p, err := profile.ParseData(files[path])
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Sample) != 1 {
			t.Fatalf("%s samples = %d", path, len(p.Sample))
		}
		for _, v := range p.Sample[0].Value {
			if v == 0 {
				t.Fatal("missing metric encoded as measured zero")
			}
		}
	}
}

func TestExportKernelNameAndRepeatedRuns(t *testing.T) {
	a := exportFixture()
	a.Metadata["mode"] = "kernels"
	a.Ranges = []Range{{ID: "k", Name: "actual_cuda_kernel", Kind: "kernel", Metrics: []Metric{{Name: "counter", Unit: "bytes", Value: 10, Additive: true}}}}
	b := exportFixture()
	b.Metadata["mode"] = "kernels"
	b.Metadata["capture_id"], b.Metadata["run_id"] = "capture-2", "run-2"
	b.Ranges = a.Ranges
	files, err := buildExport("capture", []*Result{a, b})
	if err != nil {
		t.Fatal(err)
	}
	p, err := profile.ParseData(files["capture"])
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Sample) != 2 {
		t.Fatal("runs merged")
	}
	for _, sample := range p.Sample {
		if sample.Label["kernel"][0] != "actual_cuda_kernel" || sample.Label["benchmark"][0] != "BenchmarkDecode/full" {
			t.Fatalf("labels = %v", sample.Label)
		}
	}
	b.Metadata["capture_id"] = "capture-1"
	if _, err := buildExport("capture", []*Result{a, b}); err == nil {
		t.Fatal("duplicate capture double-counted")
	}
}

func TestIntegerCounterPolicy(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value float64
		valid bool
	}{
		{"zero", 0, true}, {"integer", 4096, true}, {"largest", 1<<53 - 1, true},
		{"ambiguous", 1 << 53, false}, {"overflow", 1 << 63, false}, {"fraction", 0.5, false},
		{"negative", -1, false}, {"nan", math.NaN(), false}, {"infinity", math.Inf(1), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			n, err := integerCounter(tc.value)
			if (err == nil) != tc.valid {
				t.Fatalf("%g: n=%d err=%v", tc.value, n, err)
			}
		})
	}
}

func TestExportRejectsInvalidMeasurements(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Result)
	}{
		{"dropped", func(r *Result) { r.Quality.DroppedRanges = 1 }},
		{"incomplete", func(r *Result) { r.Quality.AllPassesSubmitted = false }},
		{"notdecoded", func(r *Result) { r.Quality.Decoded = false }},
		{"missingpass", func(r *Result) { r.Quality.PassesStopped = 2 }},
		{"metadata", func(r *Result) { delete(r.Metadata, "capture_id") }},
		{"cycle", func(r *Result) { r.Ranges[0].ParentID = "a" }},
		{"missingparent", func(r *Result) { r.Ranges[1].ParentID = "absent" }},
		{"duplicate", func(r *Result) { r.Ranges[1].ID = "root" }},
		{"nonfinite", func(r *Result) { r.Ranges[2].Metrics[0].Value = math.NaN() }},
		{"precision", func(r *Result) { r.Ranges[2].Metrics[0].Value = 1 << 53 }},
		{"noadditive", func(r *Result) {
			for i := range r.Ranges {
				for j := range r.Ranges[i].Metrics {
					r.Ranges[i].Metrics[j].Additive = false
				}
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := exportFixture()
			tc.mutate(r)
			if _, err := buildExport("capture", []*Result{r}); err == nil {
				t.Fatal("accepted invalid result")
			}
		})
	}
}

func exportDir(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(home, "tmp")
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(root, "cuptibench-export-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

func TestPublishExportExclusiveAndRollback(t *testing.T) {
	dir := exportDir(t)
	path := filepath.Join(dir, "profile")
	files := map[string][]byte{path: []byte("profile"), path + ".json": []byte("manifest")}
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for range 2 {
		wg.Add(1)
		go func() { defer wg.Done(); errs <- publishExport(files, path+".json") }()
	}
	wg.Wait()
	close(errs)
	successes := 0
	for err := range errs {
		if err == nil {
			successes++
		}
	}
	if successes != 1 {
		t.Fatalf("successful writers=%d", successes)
	}
	for p, want := range files {
		got, err := os.ReadFile(p)
		if err != nil || string(got) != string(want) {
			t.Fatalf("%s: %q %v", p, got, err)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("left temporary files: %v", entries)
	}
	// An existing manifest must prevent even the first profile destination.
	other := filepath.Join(dir, "other")
	if err := os.WriteFile(other+".json", []byte("keep"), 0600); err != nil {
		t.Fatal(err)
	}
	if publishExport(map[string][]byte{other: []byte("new"), other + ".json": []byte("replace")}, other+".json") == nil {
		t.Fatal("overwrote existing manifest")
	}
	if _, err := os.Stat(other); !os.IsNotExist(err) {
		t.Fatalf("partial profile after collision: %v", err)
	}
}

func TestPackagePathEncoding(t *testing.T) {
	a, err := expandProfilePath("out-{package}.pprof", "example.com/a/b")
	if err != nil {
		t.Fatal(err)
	}
	b, err := expandProfilePath("out-{package}.pprof", "example.com/a_b")
	if err != nil {
		t.Fatal(err)
	}
	if a == b || strings.Contains(a, "/") {
		t.Fatalf("unsafe encodings: %q %q", a, b)
	}
	if _, err := expandProfilePath("out-{package}", ""); err == nil {
		t.Fatal("accepted missing package")
	}
}

func TestExportAggregateIsJSONOnly(t *testing.T) {
	r := exportFixture()
	r.Metadata["mode"] = "kernels"
	r.Ranges = []Range{{ID: "k", Name: "actual", Kind: "kernel", Metrics: []Metric{{Name: "counter", Unit: "bytes", Value: 10, Additive: true}}}}
	aggregate := exportFixture()
	aggregate.Metadata["mode"] = "whole"
	aggregate.Metadata["capture_id"] = "aggregate-1"
	aggregate.Ranges = aggregate.Ranges[:1]
	r.Aggregate = aggregate
	r.Metadata["aggregate_capture_id"] = "aggregate-1"
	files, err := buildExport("capture", []*Result{r})
	if err != nil {
		t.Fatal(err)
	}
	p, err := profile.ParseData(files["capture"])
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Sample) != 1 || p.Sample[0].Value[0] != 10 {
		t.Fatal("aggregate was added to kernel samples")
	}
	var m exportManifest
	if err := json.Unmarshal(files["capture.json"], &m); err != nil {
		t.Fatal(err)
	}
	if m.Results[0].Aggregate == nil || m.Results[0].Aggregate.Ranges[0].Metrics[0].Value != 100 {
		t.Fatal("aggregate lost from JSON")
	}
	aggregate.Quality.DroppedRanges = 1
	if _, err := buildExport("capture", []*Result{r}); err == nil {
		t.Fatal("failed aggregate exported")
	}
	aggregate.Quality.DroppedRanges = 0
	aggregate.Aggregate = r
	if _, err := buildExport("capture", []*Result{r}); err == nil {
		t.Fatal("cyclic aggregate accepted")
	}
}
