package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
)

const defaultExportBytes int64 = 512 << 20
const defaultExportTracks = 100000
const defaultExportMemory int64 = 768 << 20

// exportMemoryBudget temporarily tightens the Go runtime soft memory limit.
// It does not raise a caller-provided lower limit or claim to bound RSS.
func exportMemoryBudget(limit int64) func() {
	previous := debug.SetMemoryLimit(-1)
	if limit > 0 && limit < previous {
		debug.SetMemoryLimit(limit)
	}
	return func() { debug.SetMemoryLimit(previous) }
}

// resolveExportDetail handles legacy flags independently of argument order.
func resolveExportDetail(detail, domain string, summary, markerSet, domainSet bool) (string, error) {
	if detail == "" {
		if summary {
			return "operation", nil
		}
		if markerSet || domainSet {
			return "", fmt.Errorf("summary options require -summary or -detail")
		}
		return "raw", nil
	}
	switch detail {
	case "raw", "compact", "token", "operation":
	default:
		return "", fmt.Errorf("invalid detail %q", detail)
	}
	if summary && detail != "operation" {
		return "", fmt.Errorf("-summary requires -detail=operation")
	}
	if detail == "raw" && (markerSet || domainSet) {
		return "", fmt.Errorf("raw detail cannot use summary options")
	}
	if domain != "driver" && domain != "runtime" {
		return "", fmt.Errorf("invalid operation domain %q", domain)
	}
	return detail, nil
}

// limitedExportWriter refuses output before writing bytes beyond the budget.
type limitedExportWriter struct {
	io.Writer
	remaining int64
}

func (w *limitedExportWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		return 0, fmt.Errorf("trace output byte limit exceeded")
	}
	n, err := w.Writer.Write(p)
	w.remaining -= int64(n)
	return n, err
}

func atomicExport(path string, inputs []string, limit int64, write func(io.Writer) error) error {
	for _, input := range inputs {
		if err := checkComparisonOutput(path, [2]string{input, input}); err != nil {
			return err
		}
	}
	if limit <= 0 {
		return fmt.Errorf("trace output byte limit must be positive")
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".cudaprof-export-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	err = write(&limitedExportWriter{Writer: f, remaining: limit})
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// detailOverview retains bounded aggregate levels, excluding raw activity
// leaves already represented exactly once on the raw tracks.
func detailOverview(report overview.Report, detail string) overview.Report {
	var copyNode func(*overview.Node) *overview.Node
	copyNode = func(n *overview.Node) *overview.Node {
		if n == nil {
			return nil
		}
		switch n.Kind {
		case "capture", "phase", "token", "layer", "operation":
		default:
			return nil
		}
		if detail == "compact" && n.Kind != "capture" && n.Kind != "phase" && n.Kind != "token" {
			return nil
		}
		if detail == "token" && (n.Kind == "layer" || n.Kind == "operation") {
			return nil
		}
		c := *n
		c.Children = nil
		for _, child := range n.Children {
			if v := copyNode(child); v != nil {
				c.Children = append(c.Children, v)
			}
		}
		return &c
	}
	report.Hierarchy = copyNode(report.Hierarchy)
	if detail == "compact" {
		report.Tokens = nil
		report.Inferred = nil
	}
	return report
}
