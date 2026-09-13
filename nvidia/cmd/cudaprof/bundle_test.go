package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func createTestFlightRecorderBundle(t *testing.T, asGzip bool) (string, *cudaflightrecorder.Bundle) {
	t.Helper()

	m := cudaflightrecorder.Manifest{
		Timestamp:              time.Now().UTC().Truncate(time.Second),
		Reason:                 "test diagnostic snapshot",
		WindowRequestedNs:      1_000_000_000,
		GoSpanNs:               950_000_000,
		GPUSpanNs:              960_000_000,
		OverlapSpanNs:          940_000_000,
		GPURecords:             42,
		GPUDropped:             0,
		GoTraceBytes:           1024,
		ClockUncertaintyNs:     15,
		ClockUncertaintySource: "tsc_ptp",
		OverlapReliable:        true,
	}

	b := cudaflightrecorder.NewBundle(m)

	fn := &profile.Function{ID: 1, Name: "vec_add", Filename: "kernels.cu"}
	loc0 := &profile.Location{ID: 1, Address: 0x0000, Line: []profile.Line{{Function: fn, Line: 1}}}
	loc1 := &profile.Location{ID: 2, Address: 0x0010, Line: []profile.Line{{Function: fn, Line: 2}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "stall_scoreboard", Unit: "count"},
			{Type: "stall_mem", Unit: "count"},
		},
		Sample: []*profile.Sample{
			{Location: []*profile.Location{loc0}, Value: []int64{70, 0}},
			{Location: []*profile.Location{loc1}, Value: []int64{0, 30}},
		},
		Location: []*profile.Location{loc0, loc1},
		Function: []*profile.Function{fn},
	}
	var pprofBuf bytes.Buffer
	if err := p.Write(&pprofBuf); err != nil {
		t.Fatalf("write profile: %v", err)
	}

	b.Files["gpu.pprof"] = pprofBuf.Bytes()
	b.Files["go.trace"] = []byte("\x00\x00\x00fake_go_trace_bytes")
	b.Files["kernels.cu"] = []byte("extern \"C\" __global__ void vec_add(float *a, float *b) {}")
	b.Files["kernels.ptx"] = []byte(".version 8.0\n.target sm_90\n.visible .entry vec_add() { ret; }")
	b.Files["kernels.sass"] = []byte("Function : vec_add\n/*0000*/ LDG.E R0, [R2];\n/*0010*/ STG.E [R4], R0;\n/*0020*/ RET;\n")
	b.Files["workload.go"] = []byte("package main\nfunc main() {}\n")

	var tarBuf bytes.Buffer
	if err := cudaflightrecorder.WriteBundle(&tarBuf, b); err != nil {
		t.Fatalf("WriteBundle: %v", err)
	}

	tmpDir := t.TempDir()
	var bundlePath string
	if asGzip {
		bundlePath = filepath.Join(tmpDir, "bundle.tar.gz")
		var gzBuf bytes.Buffer
		gw := gzip.NewWriter(&gzBuf)
		if _, err := gw.Write(tarBuf.Bytes()); err != nil {
			t.Fatalf("gzip write: %v", err)
		}
		if err := gw.Close(); err != nil {
			t.Fatalf("gzip close: %v", err)
		}
		if err := os.WriteFile(bundlePath, gzBuf.Bytes(), 0644); err != nil {
			t.Fatalf("write bundle.tar.gz: %v", err)
		}
	} else {
		bundlePath = filepath.Join(tmpDir, "bundle.tar")
		if err := os.WriteFile(bundlePath, tarBuf.Bytes(), 0644); err != nil {
			t.Fatalf("write bundle.tar: %v", err)
		}
	}

	return bundlePath, b
}

func TestReadFlightRecorderBundle_InMemory(t *testing.T) {
	for _, asGzip := range []bool{false, true} {
		name := "tar"
		if asGzip {
			name = "tar.gz"
		}
		t.Run(name, func(t *testing.T) {
			bundlePath, origBundle := createTestFlightRecorderBundle(t, asGzip)

			fb, err := ReadFlightRecorderBundlePath(bundlePath)
			if err != nil {
				t.Fatalf("ReadFlightRecorderBundlePath: %v", err)
			}

			// Verify in-memory extraction of gpu.pprof
			if len(fb.GPUPprof) == 0 {
				t.Error("extracted GPUPprof is empty")
			}
			if !bytes.Equal(fb.GPUPprof, origBundle.Files["gpu.pprof"]) {
				t.Error("extracted GPUPprof content mismatch")
			}

			// Verify in-memory extraction of go.trace
			if len(fb.GoTrace) == 0 {
				t.Error("extracted GoTrace is empty")
			}
			if !bytes.Equal(fb.GoTrace, origBundle.Files["go.trace"]) {
				t.Error("extracted GoTrace content mismatch")
			}

			// Verify manifest
			if fb.Manifest.Reason != origBundle.Manifest.Reason {
				t.Errorf("Manifest.Reason = %q, want %q", fb.Manifest.Reason, origBundle.Manifest.Reason)
			}
			if fb.Manifest.GPURecords != origBundle.Manifest.GPURecords {
				t.Errorf("Manifest.GPURecords = %d, want %d", fb.Manifest.GPURecords, origBundle.Manifest.GPURecords)
			}
			if len(fb.Files["manifest.json"]) == 0 {
				t.Error("manifest.json file missing from Files map")
			}

			// Verify ToCapture conversion
			cap, err := fb.ToCapture()
			if err != nil {
				t.Fatalf("ToCapture: %v", err)
			}
			if cap.Profile == nil || len(cap.Profile.Sample) != 2 {
				t.Errorf("cap.Profile samples = %v, want 2", cap.Profile)
			}
			if cap.Code.SASS != "kernels.sass" {
				t.Errorf("cap.Code.SASS = %q, want kernels.sass", cap.Code.SASS)
			}
			if cap.Code.PTX != "kernels.ptx" {
				t.Errorf("cap.Code.PTX = %q, want kernels.ptx", cap.Code.PTX)
			}
			if cap.Code.Source != "kernels.cu" {
				t.Errorf("cap.Code.Source = %q, want kernels.cu", cap.Code.Source)
			}
		})
	}
}

func TestFlightRecorderBundle_InspectCommand(t *testing.T) {
	bundlePath, _ := createTestFlightRecorderBundle(t, true)

	// Test: cudaprof inspect bundle.tar.gz (single arg, prints full report)
	origStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w

	err = runInspectCommand([]string{bundlePath})
	w.Close()
	os.Stdout = origStdout

	if err != nil {
		t.Fatalf("runInspectCommand(bundle.tar.gz): %v", err)
	}

	var buf bytes.Buffer
	io.Copy(&buf, r)
	out := buf.String()

	var report inspectReport
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatalf("unmarshal inspect report: %v, raw:\n%s", err, out)
	}

	// Verify artifact names contain extracted bundle files
	hasPprof := false
	hasTrace := false
	hasManifest := false
	for _, name := range report.ArtifactNames {
		switch name {
		case "gpu.pprof":
			hasPprof = true
		case "go.trace":
			hasTrace = true
		case "manifest.json":
			hasManifest = true
		}
	}
	if !hasPprof || !hasTrace || !hasManifest {
		t.Errorf("ArtifactNames = %v, want gpu.pprof, go.trace, manifest.json", report.ArtifactNames)
	}

	// Also test inspect --coverage bundle.tar.gz
	r2, w2, _ := os.Pipe()
	os.Stdout = w2
	err = runInspectCommand([]string{"--coverage", bundlePath})
	w2.Close()
	os.Stdout = origStdout

	if err != nil {
		t.Fatalf("runInspectCommand(--coverage bundle.tar.gz): %v", err)
	}
	var buf2 bytes.Buffer
	io.Copy(&buf2, r2)
	var covReport inspectReport
	if err := json.Unmarshal(buf2.Bytes(), &covReport); err != nil {
		t.Fatalf("unmarshal covReport: %v", err)
	}
	if covReport.Coverage == nil || covReport.Coverage.ActivityRecords != 42 {
		t.Errorf("covReport.Coverage = %+v, want ActivityRecords=42", covReport.Coverage)
	}
}

func TestFlightRecorderBundle_ViewCommand(t *testing.T) {
	bundlePath, _ := createTestFlightRecorderBundle(t, true)
	outHTML := filepath.Join(t.TempDir(), "bundle-report.html")

	// Test: cudaprof view -html=... bundle.tar.gz
	err := runViewCommand([]string{"view", "-html=" + outHTML, bundlePath})
	if err != nil {
		t.Fatalf("runViewCommand(bundle.tar.gz): %v", err)
	}

	htmlBytes, err := os.ReadFile(outHTML)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	html := string(htmlBytes)

	// Verify HTML contains tiers
	for _, tier := range []string{"Go", "CUDA C", "PTX", "SASS"} {
		if !strings.Contains(html, tier) {
			t.Errorf("rendered HTML missing tier %q", tier)
		}
	}

	// Verify stall heatmap rendered on SASS line
	// PC 0x0000: 70 stalls (70%) -> stall-high
	// PC 0x0010: 30 stalls (30%) -> stall-high
	if !strings.Contains(html, "stall-heat") {
		t.Error("rendered HTML missing stall-heat class")
	}
	if !strings.Contains(html, "70.0%") {
		t.Error("rendered HTML missing 70.0% stall percentage")
	}
	if !strings.Contains(html, "stall_scoreboard: 70.0%") {
		t.Error("rendered HTML missing stall_scoreboard in tooltip")
	}
}
