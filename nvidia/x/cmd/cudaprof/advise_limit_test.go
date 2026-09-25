package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

// manyFindingsCapture writes a capture whose PC samples stall in n distinct
// kernels, so the advisor produces more findings than any small -limit.
func manyFindingsCapture(t *testing.T, n int) string {
	t.Helper()
	prof := &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "stall_scoreboard", Unit: "count"},
			{Type: "stall_mem", Unit: "count"},
		},
	}
	for i := 0; i < n; i++ {
		addr := uint64(0x1000 + i*0x10)
		loc := &profile.Location{ID: uint64(i + 1), Address: addr}
		prof.Location = append(prof.Location, loc)
		prof.Sample = append(prof.Sample, &profile.Sample{
			Location: []*profile.Location{loc},
			Value:    []int64{int64(900 - i), int64(100 + i)},
			Label:    map[string][]string{"kernel": {fmt.Sprintf("stall_kernel_%02d", i)}},
		})
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "many.pprof")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	cap := &cudacapture.Capture{
		Profile: prof,
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte("void k() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_MOCK"),
			"kernels.sass":  []byte(".section .text.k\n/*0000*/ EXIT;\n"),
		},
	}
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return path
}

// TestAdviseJSONIgnoresLimit is the B2 control. The summary counts every
// finding, so some format must be able to reach every finding. JSON is that
// format; applying -limit to it made 98% of the counted findings unreachable
// by any documented flag combination.
func TestAdviseJSONIgnoresLimit(t *testing.T) {
	const kernels = 8
	path := manyFindingsCapture(t, kernels)

	all := captureStdout(t, func() error {
		return runAdviseCommand([]string{"--format=json", "-limit", "0", path})
	})
	var full advise.Report
	if err := json.Unmarshal([]byte(all), &full); err != nil {
		t.Fatalf("decode unlimited json: %v", err)
	}
	if len(full.Items) <= 3 {
		t.Fatalf("capture produced %d findings, need more than the limit under test; control cannot fire", len(full.Items))
	}

	limited := captureStdout(t, func() error {
		return runAdviseCommand([]string{"--format=json", "-limit", "3", path})
	})
	var got advise.Report
	if err := json.Unmarshal([]byte(limited), &got); err != nil {
		t.Fatalf("decode limited json: %v", err)
	}
	if len(got.Items) != len(full.Items) {
		t.Errorf("--format=json with -limit 3 emitted %d of %d findings; JSON must emit every finding the summary counts", len(got.Items), len(full.Items))
	}
}

// TestAdviseMarkdownDeclaresWhatItWithheld is the other half of B2: the
// markdown summary counts findings it does not print, and must say so next to
// the count rather than only in a footer.
func TestAdviseMarkdownDeclaresWhatItWithheld(t *testing.T) {
	path := manyFindingsCapture(t, 8)

	out := captureStdout(t, func() error {
		return runAdviseCommand([]string{"-limit", "3", path})
	})
	if !strings.Contains(out, "Showing 3 of ") {
		t.Errorf("markdown report does not state what it withheld; want a \"Showing 3 of N\" line, got:\n%s", out)
	}
	if !strings.Contains(out, "--format=json") {
		t.Errorf("markdown report does not name the format that reaches every finding:\n%s", out)
	}
}

// TestAdviseLimitZeroMeansNoLimit pins the documented meaning of -limit 0.
func TestAdviseLimitZeroMeansNoLimit(t *testing.T) {
	path := manyFindingsCapture(t, 8)

	zero := captureStdout(t, func() error {
		return runAdviseCommand([]string{"--format=json", "-limit", "0", path})
	})
	var got advise.Report
	if err := json.Unmarshal([]byte(zero), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.Items) <= 3 {
		t.Fatalf("expected more than 3 findings with -limit 0, got %d", len(got.Items))
	}

	md := captureStdout(t, func() error { return runAdviseCommand([]string{"-limit", "0", path}) })
	if strings.Contains(md, "additional findings omitted") || strings.Contains(md, "Showing ") {
		t.Errorf("-limit 0 withheld findings:\n%s", md)
	}
}
