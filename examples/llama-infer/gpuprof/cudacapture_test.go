package gpuprof

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// cudaprofProfile builds a profile shaped like the one cudaprof writes: four
// sample types, a kernel record carrying a name, and a memcpy record that
// carries none.
func cudaprofProfile() *profile.Profile {
	return &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "kernel_time", Unit: "nanoseconds"},
			{Type: "copy_time", Unit: "nanoseconds"},
			{Type: "copy_bytes", Unit: "bytes"},
			{Type: "activity", Unit: "count"},
		},
		Sample: []*profile.Sample{
			{
				Value: []int64{2272, 0, 0, 1},
				Label: map[string][]string{"subtype": {"kernel"}, "name": {"RMSNorm"}, "device": {"0"}},
			},
			{
				Value: []int64{4096, 0, 0, 1},
				Label: map[string][]string{"subtype": {"kernel"}, "name": {"LinearGEMV"}, "device": {"1"}},
			},
			{
				Value: []int64{1500, 0, 0, 1},
				Label: map[string][]string{"subtype": {"kernel"}, "name": {"RMSNorm"}, "device": {"0"}},
			},
			{
				Value: []int64{0, 500, 65536, 1},
				Label: map[string][]string{"subtype": {"memcpy"}, "device": {"0"}},
			},
		},
	}
}

// unavailableCode marks every code role explicitly absent, which is what a
// capture that retained no source, PTX, CUBIN, or SASS must record.
func unavailableCode() map[string]string {
	return map[string]string{
		"source": "not captured",
		"ptx":    "not captured",
		"cubin":  "not captured",
		"sass":   "not captured",
	}
}

func testCapture(p *profile.Profile) *cudacapture.Capture {
	return &cudacapture.Capture{
		Profile:         p,
		Files:           map[string][]byte{"run.json": []byte(`{"ok":true}`)},
		CodeUnavailable: unavailableCode(),
	}
}

func writeTestCapture(t *testing.T, p *profile.Profile) string {
	t.Helper()
	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, testCapture(p)); err != nil {
		t.Fatalf("write capture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "capture.pprof")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}
	return path
}

func TestReadCaptureSummary(t *testing.T) {
	got, err := ReadCaptureSummary(writeTestCapture(t, cudaprofProfile()))
	if err != nil {
		t.Fatalf("ReadCaptureSummary: %v", err)
	}
	if want := int64(2272 + 4096 + 1500); got.KernelTimeNS != want {
		t.Errorf("KernelTimeNS = %d, want %d", got.KernelTimeNS, want)
	}
	if got.CopyTimeNS != 500 {
		t.Errorf("CopyTimeNS = %d, want 500", got.CopyTimeNS)
	}
	if got.CopyBytes != 65536 {
		t.Errorf("CopyBytes = %d, want 65536", got.CopyBytes)
	}
	if got.Activities != 4 {
		t.Errorf("Activities = %d, want 4", got.Activities)
	}
	if got.KernelLaunches != 3 {
		t.Errorf("KernelLaunches = %d, want 3", got.KernelLaunches)
	}
	if want := []string{"LinearGEMV", "RMSNorm"}; !equalStrings(got.KernelNames, want) {
		t.Errorf("KernelNames = %v, want %v", got.KernelNames, want)
	}
	if want := []string{"0", "1"}; !equalStrings(got.Devices, want) {
		t.Errorf("Devices = %v, want %v", got.Devices, want)
	}
}

// TestReadCaptureSummaryDirectory covers the directory form, which is what an
// extracted capture looks like on disk.
func TestReadCaptureSummaryDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "extract")
	if err := cudacapture.WriteDir(dir, testCapture(cudaprofProfile())); err != nil {
		t.Fatalf("WriteDir: %v", err)
	}
	got, err := ReadCaptureSummary(dir)
	if err != nil {
		t.Fatalf("ReadCaptureSummary: %v", err)
	}
	if got.KernelLaunches != 3 {
		t.Errorf("KernelLaunches = %d, want 3", got.KernelLaunches)
	}
}

// TestSummarizeCaptureRejectsForeignProfile is the control against reporting a
// fabricated zero: a profile with none of the cudaprof sample types must fail
// rather than summarize as an idle run.
func TestSummarizeCaptureRejectsForeignProfile(t *testing.T) {
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
		Sample:     []*profile.Sample{{Value: []int64{7}}},
	}
	got, err := summarizeCapture(testCapture(p))
	if err == nil {
		t.Fatalf("summarizeCapture succeeded with %+v, want error", got)
	}
	if !strings.Contains(err.Error(), "no cudaprof sample types") {
		t.Errorf("error = %v, want it to name the missing sample types", err)
	}
}

func TestSummarizeCaptureRejectsMissingProfile(t *testing.T) {
	if _, err := summarizeCapture(nil); err == nil {
		t.Error("summarizeCapture(nil) succeeded, want error")
	}
	if _, err := summarizeCapture(&cudacapture.Capture{}); err == nil {
		t.Error("summarizeCapture with no profile succeeded, want error")
	}
}

func TestReadCaptureSummaryMissingPath(t *testing.T) {
	if _, err := ReadCaptureSummary(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Error("ReadCaptureSummary on a missing path succeeded, want error")
	}
}

// TestSummarizePartialSampleTypes covers a capture that carries only some of
// the cudaprof sample types: the present ones are summed and the absent ones
// stay zero without an error, because the capture does describe itself.
func TestSummarizePartialSampleTypes(t *testing.T) {
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "activity", Unit: "count"}},
		Sample: []*profile.Sample{
			{Value: []int64{1}, Label: map[string][]string{"subtype": {"kernel"}, "name": {"SwiGLU"}}},
		},
	}
	got, err := summarizeCapture(testCapture(p))
	if err != nil {
		t.Fatalf("summarizeCapture: %v", err)
	}
	if got.Activities != 1 || got.KernelLaunches != 1 {
		t.Errorf("Activities = %d, KernelLaunches = %d, want 1 and 1", got.Activities, got.KernelLaunches)
	}
	if got.KernelTimeNS != 0 {
		t.Errorf("KernelTimeNS = %d, want 0 for a capture with no kernel_time type", got.KernelTimeNS)
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
