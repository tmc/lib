package cuptibench

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFlagRegistration(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	var p processExport
	p.init(fs)
	p.init(fs)
	if p.err != nil {
		t.Fatal(p.err)
	}
	if err := fs.Parse([]string{"-cuptimetrics=instructions,l2-read"}); err != nil {
		t.Fatal(err)
	}
	if p.metrics != "instructions,l2-read" {
		t.Fatal(p.metrics)
	}
	conflict := flag.NewFlagSet("conflict", flag.ContinueOnError)
	conflict.String("cuptimetrics", "owned", "existing")
	var q processExport
	q.init(conflict)
	if q.err == nil || !strings.Contains(q.err.Error(), "cuptimetrics") {
		t.Fatalf("conflict=%v", q.err)
	}
	if conflict.Lookup("cuptiprofile") != nil {
		t.Fatal("partially registered flags after conflict")
	}
	if err := q.finish(); err == nil {
		t.Fatal("Finish lost registration error")
	}
}

func TestProcessExportSnapshotAndFinish(t *testing.T) {
	path := filepath.Join(exportDir(t), "capture")
	p := processExport{path: path}
	r := exportFixture()
	if err := p.record(r, nil); err != nil {
		t.Fatal(err)
	}
	r.Ranges[2].Metrics[0].Value = 999
	r.Metadata["benchmark"] = "mutated"
	if err := p.finish(); err != nil {
		t.Fatal(err)
	}
	if err := p.finish(); err != nil {
		t.Fatalf("repeat Finish: %v", err)
	}
	if p.record(exportFixture(), nil) == nil {
		t.Fatal("recorded after Finish")
	}
	data, err := os.ReadFile(path + ".json")
	if err != nil {
		t.Fatal(err)
	}
	var m exportManifest
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m.Results[0].Ranges[2].Metrics[0].Value != 20 || m.Results[0].Metadata["benchmark"] != "BenchmarkDecode/full" {
		t.Fatal("caller mutation changed export")
	}
}

func TestFinishNoResultsAndNoOutput(t *testing.T) {
	p := processExport{path: filepath.Join(exportDir(t), "capture")}
	first := p.finish()
	if first == nil {
		t.Fatal("empty profile accepted")
	}
	if p.finish() != first {
		t.Fatal("Finish did not preserve error")
	}
	var q processExport
	if err := q.record(nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := q.finish(); err != nil {
		t.Fatal(err)
	}
}
