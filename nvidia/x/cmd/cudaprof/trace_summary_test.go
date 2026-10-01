package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestTraceCatalogAndParsers(t *testing.T) {
	var out bytes.Buffer
	if err := runTraceQuery([]string{"-list", "-json"}, &out); err != nil {
		t.Fatal(err)
	}
	var envelope traceViewResult
	if err := json.Unmarshal(out.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Schema != "cudaprof.query-catalog.v1" || !strings.Contains(out.String(), "recorded-gaps") || !strings.Contains(out.String(), "required_evidence") {
		t.Fatal(out.String())
	}
	for _, args := range [][]string{{"-list", "extra"}, {"arbitrarySQL", "missing"}, {"inventory"}} {
		if err := runTraceQuery(args, &out); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	for _, args := range [][]string{{"-view=other", "missing"}, {"-iteration-marker=[", "missing"}, {}} {
		if err := runTraceSummary(args, &out); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
func TestTraceSummaryEscaping(t *testing.T) {
	path := t.TempDir() + "/summary.html"
	result := traceViewResult{State: "partial", View: "overview", Input: "<script>alert(1)</script>", Evidence: map[string]string{"raw_symbol": "<img src=x onerror=alert(1)>"}}
	if err := writeTraceSummaryHTML(path, result); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "<script>") || strings.Contains(string(data), "<img src=") || !strings.Contains(string(data), "partial") {
		t.Fatalf("unsafe HTML: %s", data)
	}
}
func TestTraceSummaryCollectorGate(t *testing.T) {
	path := t.TempDir() + "/partial.pprof"
	writeTestCaptureFile(t, path, testCapture(nil, map[string]any{"run_completeness": "partial"}))
	var out bytes.Buffer
	if err := runTraceSummary([]string{"-json", path}, &out); err != nil {
		t.Fatal(err)
	}
	var result traceViewResult
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.State != "unavailable" || !strings.Contains(out.String(), "unavailable") {
		t.Fatal(out.String())
	}
}
