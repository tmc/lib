package main

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestAdviseCompiledCode runs advise on a kernel compiled for sm_103a with
// no device: its SASS and its ptxas -v report.
func TestAdviseCompiledCode(t *testing.T) {
	dir := filepath.Join("..", "..", "cudaprof", "advise", "testdata", "tensor")
	args := []string{"-limit=0", filepath.Join(dir, "gemm.sm_103a.sass"), filepath.Join(dir, "gemm.sm_103a.ptxas.txt")}
	out := captureStdout(t, func() error { return runAdviseCommand(args) })
	for _, want := range []string{"compiled code only", "where tcgen05.mma applies", "on sm_103"} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
}

func TestReadCodeCaptureRefusesMixtures(t *testing.T) {
	for _, paths := range [][]string{{"a.sass", "run.pprof"}, {"a.sass", "b.sass"}} {
		if _, err := readCodeCapture(paths); err == nil {
			t.Errorf("readCodeCapture(%q) succeeded", paths)
		}
	}
}
