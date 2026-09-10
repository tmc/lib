//go:build darwin && arm64

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The command-buffer accounting rests on a source-level invariant: every Metal
// command buffer the inference runtime creates is created by
// (*metalPinnedOps).commandBuffer and recorded by (*metalPinnedOps).record.
// Before centralization the package held 28 creation sites and 2 recorders, so
// cbufs/tok read 86.00 against a gputrace count of 87.0 -- a metric with zero
// variance and a missing term. A count is only worth its narrow error bars if
// nothing escapes it.
//
// This is a source test, not a runtime one, and deliberately so: it costs no
// GPU, no model load and no quiet host, so it can gate every edit. The runtime
// half of the gate is BenchmarkMetalSpanProbe agreeing with gputrace.
func TestCommandBufferCreationIsCentralized(t *testing.T) {
	const (
		create = "Queue().CommandBuffer()"
		helper = "metal_ops_darwin.go"
		// metalTinyBackend is a separate toy backend, not part of the
		// inference runtime whose buffers cbufs/tok counts.
		tiny = "metal_tiny_darwin.go"
	)

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var offenders []string
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") || f == helper || f == tiny {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(src), "\n") {
			if strings.Contains(line, create) {
				offenders = append(offenders, fileLine(f, i+1, line))
			}
		}
	}
	if len(offenders) > 0 {
		t.Errorf("command buffers created outside %s:\n\t%s\n"+
			"route them through (*metalPinnedOps).run so their spans are counted",
			helper, strings.Join(offenders, "\n\t"))
	}
}

// The recorder must stay single, for the same reason.
func TestRegionRecordingIsCentralized(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var sites []string
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(src), "\n") {
			if strings.Contains(line, "regions.add(") {
				sites = append(sites, fileLine(f, i+1, line))
			}
		}
	}
	if len(sites) != 1 {
		t.Errorf("got %d region recording sites, want exactly 1:\n\t%s",
			len(sites), strings.Join(sites, "\n\t"))
	}
}

func fileLine(file string, line int, text string) string {
	return file + ":" + itoa(line) + ": " + strings.TrimSpace(text)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
