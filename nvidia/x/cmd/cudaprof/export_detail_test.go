package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime/debug"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
)

func TestExportDetailMatrix(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
		fail bool
	}{
		{"default", nil, "raw", false},
		{"legacy", []string{"-summary"}, "operation", false},
		{"legacy explicit", []string{"-summary", "-detail=operation"}, "operation", false},
		{"conflict before", []string{"-summary", "-detail=raw"}, "", true},
		{"conflict after", []string{"-detail=raw", "-summary"}, "", true},
		{"compact", []string{"-detail=compact"}, "compact", false},
		{"token", []string{"-detail=token"}, "token", false},
		{"orphan domain", []string{"-summary-domain=runtime"}, "", true},
		{"link domain", []string{"-submission-links", "-summary-domain=runtime"}, "raw", false},
		{"raw marker", []string{"-detail=raw", "-summary-marker=k"}, "", true},
		{"invalid", []string{"-detail=other"}, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts, err := parseExportArgs(append(tc.args, "capture.pprof"))
			if (err != nil) != tc.fail {
				t.Fatalf("error=%v", err)
			}
			if err == nil && opts.detail != tc.want {
				t.Fatalf("detail=%q, want %q", opts.detail, tc.want)
			}
		})
	}
}

func TestAtomicExport(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "input")
	output := filepath.Join(dir, "output")
	os.WriteFile(input, []byte("source"), 0600)
	os.WriteFile(output, []byte("previous"), 0600)
	write := func(w io.Writer) error { _, err := w.Write([]byte("replacement")); return err }
	if err := atomicExport(output, []string{input}, 3, write); err == nil {
		t.Fatal("byte budget ignored")
	}
	got, _ := os.ReadFile(output)
	if !bytes.Equal(got, []byte("previous")) {
		t.Fatalf("failure replaced output: %q", got)
	}
	if err := atomicExport(input, []string{input}, 100, write); err == nil {
		t.Fatal("path collision accepted")
	}
	link := filepath.Join(dir, "hardlink")
	if err := os.Link(input, link); err != nil {
		t.Fatal(err)
	}
	if err := atomicExport(link, []string{input}, 100, write); err == nil {
		t.Fatal("inode collision accepted")
	}
	if err := atomicExport(output, []string{input}, 100, write); err != nil {
		t.Fatal(err)
	}
	got, _ = os.ReadFile(output)
	if string(got) != "replacement" {
		t.Fatalf("output=%q", got)
	}
	files, _ := filepath.Glob(filepath.Join(dir, ".cudaprof-export-*"))
	if len(files) != 0 {
		t.Fatalf("temporary files retained: %v", files)
	}
}

func TestDetailOverviewExcludesRawLeaves(t *testing.T) {
	leaf := &overview.Node{Kind: "activity", Label: "raw"}
	operation := &overview.Node{Kind: "operation", Children: []*overview.Node{leaf}}
	token := &overview.Node{Kind: "token", Children: []*overview.Node{operation}}
	original := overview.Report{Hierarchy: &overview.Node{Kind: "capture", Children: []*overview.Node{token}}, Tokens: []overview.Token{{Index: 1}}}
	compact := detailOverview(original, "compact")
	if len(compact.Tokens) != 0 || len(compact.Hierarchy.Children[0].Children) != 0 {
		t.Fatal("compact retains token details or operation descendants")
	}
	operationReport := detailOverview(original, "operation")
	if len(operationReport.Hierarchy.Children[0].Children[0].Children) != 0 {
		t.Fatal("operation detail repeats raw leaves")
	}
	if !reflect.DeepEqual(original.Hierarchy.Children[0].Children[0].Children, []*overview.Node{leaf}) {
		t.Fatal("detail modified original report")
	}
}

func TestExportMemoryBudgetPreservesCaller(t *testing.T) {
	original := debug.SetMemoryLimit(64 << 20)
	defer debug.SetMemoryLimit(original)
	restore := exportMemoryBudget(128 << 20)
	if got := debug.SetMemoryLimit(-1); got != 64<<20 {
		t.Fatalf("raised caller limit: %d", got)
	}
	restore()
	restore = exportMemoryBudget(32 << 20)
	if got := debug.SetMemoryLimit(-1); got != 32<<20 {
		t.Fatalf("budget not applied: %d", got)
	}
	restore()
	if got := debug.SetMemoryLimit(-1); got != 64<<20 {
		t.Fatalf("caller limit not restored: %d", got)
	}
}

func TestExportLegacySummaryMatchesOperation(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "input.pprof")
	writeTestCaptureFile(t, input, compareCapture(iterationRecs(3), completeStatus()))
	legacy, explicit := filepath.Join(dir, "legacy.pftrace"), filepath.Join(dir, "explicit.pftrace")
	captureStdout(t, func() error { return runExportCommand([]string{"-summary", "-o", legacy, input}) })
	captureStdout(t, func() error { return runExportCommand([]string{"-detail=operation", "-o", explicit, input}) })
	a, err := os.ReadFile(legacy)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(explicit)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Fatal("legacy summary and operation detail differ")
	}
}
