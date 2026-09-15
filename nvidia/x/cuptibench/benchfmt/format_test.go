package benchfmt

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func fixture() Run {
	value := float64(200000)
	return Run{Version: 1, Config: map[string]string{"pkg": "example.com/decode", "goos": "linux", "goarch": "arm64"}, Samples: []Sample{{Name: "BenchmarkDecode/full-8", ID: "synthetic-1", Source: "synthetic-c.json", Operations: 100, Scope: "full", Provenance: map[string]string{"evidence": "synthetic parser control; no hardware execution"}, Measurements: []Measurement{{Unit: "gpu-ns/op", Value: &value, Basis: "total", Kind: "device-elapsed"}}}}}
}

func Example() {
	value := float64(200000)
	run := Run{Version: 1, Samples: []Sample{{Name: "BenchmarkDecode/full-8", ID: "example-1", Source: "c-run.json", Operations: 100, Scope: "full", Measurements: []Measurement{{Unit: "gpu-ns/op", Value: &value, Basis: "total", Kind: "device-elapsed"}}}}}
	var out bytes.Buffer
	_ = Write(&out, run)
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	fmt.Println(lines[len(lines)-1])
	// Output: BenchmarkDecode/full-8 100 2000 gpu-ns/op
}

func ExampleDecode() {
	run, err := Decode(strings.NewReader(`{"version":1,"samples":[]}`))
	fmt.Println(len(run.Samples), err)
	// Output: 0 <nil>
}

func ExampleWrite() {
	var out bytes.Buffer
	fmt.Println(Write(&out, Run{Version: 1}))
	// Output: benchfmt: no samples
}

func ExampleRun() {
	r := Run{Version: 1, Config: map[string]string{"pkg": "example.com/decode"}}
	fmt.Println(r.Config["pkg"])
	// Output: example.com/decode
}

func ExampleSample() {
	s := Sample{Name: "BenchmarkDecode/full-8", ID: "run-1", Operations: 128}
	fmt.Println(s.Name, s.Operations)
	// Output: BenchmarkDecode/full-8 128
}

func ExampleMeasurement() {
	n := float64(4096)
	m := Measurement{Unit: "bytes/op", Value: &n, Basis: "total", Kind: "counter"}
	fmt.Println(m.Unit, m.Basis)
	// Output: bytes/op total
}

func TestWriteOneRowPerSourceSample(t *testing.T) {
	r := fixture()
	var out bytes.Buffer
	if err := Write(&out, r); err != nil {
		t.Fatal(err)
	}
	got := out.String()
	if strings.Count(got, "\nBenchmark") != 1 || !strings.Contains(got, "BenchmarkDecode/full-8 100 2000 gpu-ns/op\n") {
		t.Fatal(got)
	}
	if !strings.Contains(got, `"source":"synthetic-c.json"`) {
		t.Fatal("lost provenance")
	}
	if strings.Contains(got, "-C") {
		t.Fatal("renamed comparison benchmark")
	}
	// Increasing operation count changes the denominator, never the sample count.
	r.Samples[0].Operations = 1000
	out.Reset()
	if err := Write(&out, r); err != nil {
		t.Fatal(err)
	}
	if strings.Count(out.String(), "\nBenchmark") != 1 || !strings.Contains(out.String(), " 1000 200 gpu-ns/op") {
		t.Fatal(out.String())
	}
}

func TestWriteRejectsInvalidDataAtomically(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Run)
	}{
		{"zero-operations", func(r *Run) { r.Samples[0].Operations = 0 }},
		{"missing", func(r *Run) { r.Samples[0].Measurements[0].Value = nil }},
		{"zero-time", func(r *Run) { *r.Samples[0].Measurements[0].Value = 0 }},
		{"nan", func(r *Run) { *r.Samples[0].Measurements[0].Value = math.NaN() }},
		{"infinity", func(r *Run) { *r.Samples[0].Measurements[0].Value = math.Inf(1) }},
		{"negative", func(r *Run) { *r.Samples[0].Measurements[0].Value = -1 }},
		{"name-injection", func(r *Run) { r.Samples[0].Name = "BenchmarkDecode\nBenchmarkFake" }},
		{"unit-injection", func(r *Run) { r.Samples[0].Measurements[0].Unit = "ns/op\n" }},
		{"config-injection", func(r *Run) { r.Config["pkg"] = "x\ngoos: fake" }},
		{"source-injection", func(r *Run) { r.Samples[0].Source = "x\nBenchmarkFake" }},
		{"no-scope", func(r *Run) { r.Samples[0].Scope = "" }},
		{"no-source", func(r *Run) { r.Samples[0].Source = "" }},
		{"no-id", func(r *Run) { r.Samples[0].ID = "" }},
		{"kernel-sum-is-not-latency", func(r *Run) { r.Samples[0].Measurements[0].Kind = "kernel-duration-sum" }},
		{"counter-is-not-timing", func(r *Run) { r.Samples[0].Measurements[0].Kind = "counter" }},
		{"raw-is-not-normalized", func(r *Run) { r.Samples[0].Measurements[0].Basis = "raw" }},
		{"ratio-total", func(r *Run) {
			r.Samples[0].Measurements[0].Kind = "ratio"
			r.Samples[0].Measurements[0].Unit = "pct/op"
		}},
		{"rate-total", func(r *Run) { r.Samples[0].Measurements[0].Kind = "rate"; r.Samples[0].Measurements[0].Unit = "GB/s" }},
		{"duplicate-row", func(r *Run) { r.Samples = append(r.Samples, r.Samples[0]) }},
		{"duplicate-unit", func(r *Run) {
			r.Samples[0].Measurements = append(r.Samples[0].Measurements, r.Samples[0].Measurements[0])
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := fixture()
			tc.mutate(&r)
			var out bytes.Buffer
			if Write(&out, r) == nil {
				t.Fatal("accepted invalid input")
			}
			if out.Len() != 0 {
				t.Fatal("published partial invalid run")
			}
		})
	}
}

func TestDecodeMissingAndUnknown(t *testing.T) {
	for _, input := range []string{`{"samplez":[]}`, `{"version":1,"samples":[]} {}`, `{"version":1,"samples":[{"unexpected":1}]}`} {
		if _, err := Decode(strings.NewReader(input)); err == nil {
			t.Fatal("accepted", input)
		}
	}
	r, err := Decode(strings.NewReader(`{"version":1,"samples":[{"name":"BenchmarkDecode/full-8","sample_id":"1","source":"c","scope":"full","operations":1,"measurements":[{"unit":"bytes/op","kind":"counter","basis":"total"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if Write(&bytes.Buffer{}, r) == nil {
		t.Fatal("missing value became zero")
	}
}

func TestMeasuredCounterZeroAndRawRatio(t *testing.T) {
	r := fixture()
	zero, ratio := float64(0), float64(0.75)
	r.Samples[0].Measurements = []Measurement{{Unit: "bytes/op", Value: &zero, Basis: "total", Kind: "counter"}, {Unit: "hit-ratio", Value: &ratio, Basis: "raw", Kind: "ratio"}}
	var out bytes.Buffer
	if err := Write(&out, r); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), " 0 bytes/op 0.75 hit-ratio") {
		t.Fatal(out.String())
	}
}

func TestBenchstatComparison(t *testing.T) {
	tool, err := exec.LookPath("benchstat")
	if err != nil {
		t.Skip("benchstat not installed")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cuptibench-benchstat-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	var imported bytes.Buffer
	if err := Write(&imported, fixture()); err != nil {
		t.Fatal(err)
	}
	external, goFile := filepath.Join(dir, "c.txt"), filepath.Join(dir, "go.txt")
	if err := os.WriteFile(external, imported.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	// Both arms are explicitly synthetic parser controls, not performance evidence.
	goData := "goarch: arm64\ngoos: linux\npkg: example.com/decode\nBenchmarkDecode/full-8 100 3000 gpu-ns/op\n"
	if err := os.WriteFile(goFile, []byte(goData), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(tool, "-format=csv", external, goFile)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("benchstat: %v\n%s", err, output)
	}
	if strings.Count(string(output), "Decode/full-8") != 1 {
		t.Fatalf("comparison split into different rows:\n%s", output)
	}
	if !strings.Contains(string(output), "c.txt") || !strings.Contains(string(output), "go.txt") {
		t.Fatalf("lost comparison arm:\n%s", output)
	}
	t.Logf("benchstat parsed one sample per arm; significance is intentionally unavailable:\n%s", output)
}

func TestCounterDeclarationCannotNormalizeRecognizableRate(t *testing.T) {
	for _, unit := range []string{"GB/s/op", "sm__throughput.avg.pct_of_peak_sustained_elapsed/op", "hit-ratio/op", "%/op"} {
		r := fixture()
		r.Samples[0].Measurements[0].Kind = "counter"
		r.Samples[0].Measurements[0].Unit = unit
		if Write(&bytes.Buffer{}, r) == nil {
			t.Fatalf("accepted rate or ratio counter %q", unit)
		}
	}
}

func TestSchemaVersionRequired(t *testing.T) {
	for _, input := range []string{`{"samples":[]}`, `{"version":0,"samples":[]}`, `{"version":2,"samples":[]}`} {
		if _, err := Decode(strings.NewReader(input)); err == nil {
			t.Fatalf("accepted version: %s", input)
		}
	}
	r := fixture()
	r.Version = 2
	var out bytes.Buffer
	if err := Write(&out, r); err == nil || out.Len() != 0 {
		t.Fatal("unsupported in-memory version wrote output")
	}
}

type endlessSpaces struct{ read int }

func (r *endlessSpaces) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = ' '
	}
	r.read += len(p)
	return len(p), nil
}

func TestDecodeInputBound(t *testing.T) {
	const valid = `{"version":1,"samples":[]}`
	for _, tc := range []struct {
		name, input string
		limit       int64
		accepted    bool
	}{
		{"exact", valid, int64(len(valid)), true},
		{"oversized-document", valid, int64(len(valid) - 1), false},
		{"exact-with-whitespace", valid + " ", int64(len(valid) + 1), true},
		{"trailing-whitespace-beyond-limit", valid + "  ", int64(len(valid) + 1), false},
		{"document-after-limit", valid + strings.Repeat(" ", 20) + "{}", int64(len(valid) + 4), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := decodeLimit(strings.NewReader(tc.input), tc.limit)
			if (err == nil) != tc.accepted {
				t.Fatalf("accepted=%v err=%v", tc.accepted, err)
			}
			if !tc.accepted && !strings.Contains(err.Error(), "input exceeds") {
				t.Fatalf("missing bound diagnostic: %v", err)
			}
		})
	}
	r := &endlessSpaces{}
	if _, err := decodeLimit(r, 32); err == nil || !strings.Contains(err.Error(), "input exceeds") {
		t.Fatalf("unbounded reader: %v", err)
	}
	if r.read != 33 {
		t.Fatalf("read %d bytes, want bounded limit+1", r.read)
	}
}

func TestDecodeDefaultLimit(t *testing.T) {
	// Generate whitespace as requested rather than allocating a 64 MiB fixture.
	r := &endlessSpaces{}
	if _, err := Decode(r); err == nil || !strings.Contains(err.Error(), "input exceeds") {
		t.Fatalf("default limit: %v", err)
	}
	if r.read != maxInputBytes+1 {
		t.Fatalf("read=%d want=%d", r.read, maxInputBytes+1)
	}
}
