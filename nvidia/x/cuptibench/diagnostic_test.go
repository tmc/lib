package cuptibench

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func TestCollectRetainsFailureDiagnostics(t *testing.T) {
	p, c, w := testPlan(1)
	failure := errors.New("native Stop failure")
	c.captureErr = failure
	image := []byte{5, 6}
	trace := &cuptirange.LaunchTrace{Launches: []cuptirange.ObservedLaunch{{API: "cuGraphLaunch", Symbol: "private-symbol"}}, CompletedPasses: 1}
	c.mutate = func(v *cuptirange.Capture) {
		v.DiagnosticImage = image
		v.DiagnosticTrace = trace
		v.Quality.PassesStopped = 0
	}
	result, err := p.Collect(w)
	if !errors.Is(err, failure) || result.Diagnostics == nil || len(result.Ranges) != 0 {
		t.Fatalf("result=%+v error=%v", result, err)
	}
	image[0] = 99
	trace.Launches[0].Symbol = "changed"
	d := result.Diagnostics
	if d.Image[0] != 5 || d.Trace.Launches[0].Symbol != "private-symbol" {
		t.Fatal("borrowed backend storage")
	}
	raw, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "Diagnostics") || strings.Contains(string(raw), "private-symbol") {
		t.Fatal("diagnostics serialized into ordinary result")
	}
	if err := validateExportResult(result); err == nil || !strings.Contains(err.Error(), "diagnostic") {
		t.Fatalf("diagnostics admitted to export: %v", err)
	}
	summary := diagnosticSummary(d)
	if !strings.Contains(summary, "bytes=2 sha256=") || !strings.Contains(summary, "observed launches=1") || !strings.Contains(summary, "may be partial") {
		t.Fatal(summary)
	}
}

func TestDiagnosticsAbsentOnSuccess(t *testing.T) {
	p, _, w := testPlan(1)
	result, err := p.Collect(w)
	if err != nil || result.Diagnostics != nil {
		t.Fatalf("result=%+v error=%v", result, err)
	}
}

func TestRejectedEvaluationRetainsImage(t *testing.T) {
	p, c, w := testPlan(1)
	c.values = nil
	result, err := p.Collect(w)
	if err == nil || result.Diagnostics == nil || len(result.Diagnostics.Image) != 1 || len(result.Ranges) != 0 {
		t.Fatalf("result=%+v error=%v", result, err)
	}
}

type aggregateFailureCollector struct {
	*benchmarkCollector
	failure error
}

func (c *aggregateFailureCollector) CaptureReplay(dev *cuda.Device, limits cuptirange.Limits, options cuptirange.CaptureOptions, replay cuptirange.Replay) (*cuptirange.Capture, error) {
	if options.Mode == cuptirange.Whole {
		return &cuptirange.Capture{DiagnosticImage: []byte{7}}, c.failure
	}
	return c.benchmarkCollector.CaptureReplay(dev, limits, options, replay)
}

func BenchmarkAggregateFailureDiagnostics(b *testing.B) {
	if os.Getenv("CUPTIBENCH_DIAGNOSTIC_CONTROL") != "1" {
		b.Skip("subprocess control")
	}
	p, w, backend, _, _ := benchmarkFixture()
	p.config.Mode = Kernels
	failure := errors.New("aggregate collection failed")
	p.backend = &aggregateFailureCollector{backend, failure}
	got, err := benchmark(b, p, w, func(*Plan, Workload) (benchmarkTimer, error) { return &syntheticTimer{}, nil })
	if !errors.Is(err, failure) || got == nil || got.Diagnostics == nil || len(got.Diagnostics.Image) != 1 || got.Diagnostics.Image[0] != 7 || len(got.Ranges) != 0 {
		b.Fatalf("lost aggregate evidence result=%+v error=%v", got, err)
	}
	if validateExportResult(got) == nil {
		b.Fatal("aggregate failure exported")
	}
	b.Log(diagnosticSummary(got.Diagnostics))
	b.Fatal(err)
}

func TestAggregateFailureReachesBenchmarkDiagnostics(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, "-test.run=^$", "-test.bench=^BenchmarkAggregateFailureDiagnostics$", "-test.benchtime=1x")
	cmd.Env = append(os.Environ(), "CUPTIBENCH_DIAGNOSTIC_CONTROL=1")
	output, err := cmd.CombinedOutput()
	text := string(output)
	if err == nil || !strings.Contains(text, "aggregate collection failed") || !strings.Contains(text, "diagnostic-only counter image bytes=1") {
		t.Fatalf("error=%v output=%s", err, text)
	}
	if strings.Contains(text, "gpu-ns/op") || strings.Contains(text, "lost aggregate evidence") {
		t.Fatalf("failed benchmark output=%s", text)
	}
}
