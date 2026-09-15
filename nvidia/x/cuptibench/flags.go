package cuptibench

import (
	"encoding/json"
	"flag"
	"fmt"
	"sync"
)

type processExport struct {
	mu       sync.Mutex
	flags    *flag.FlagSet
	path     string
	metrics  string
	err      error
	finished bool
	results  []*Result
	owners   map[*Plan]bool
}

const (
	exportOwnerClosed uint32 = 1 << iota
	exportOwnerFailed
)

var process processExport

// Init registers -cuptiprofile and -cuptimetrics on flag.CommandLine. Repeated
// calls are harmless. A registration conflict is reported by Benchmark and
// Finish. Call Init before m.Run in TestMain, and Finish before os.Exit.
func Init() { process.init(flag.CommandLine) }

func (p *processExport) init(fs *flag.FlagSet) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.flags != nil {
		return
	}
	p.flags = fs
	for _, name := range []string{"cuptiprofile", "cuptimetrics"} {
		if fs.Lookup(name) != nil {
			p.err = fmt.Errorf("cuptibench: flag -%s is already registered", name)
			return
		}
	}
	fs.StringVar(&p.path, "cuptiprofile", "", "write CUPTI profiles and JSON results; use {package} for multiple test packages")
	fs.StringVar(&p.metrics, "cuptimetrics", "", "replace benchmark metrics with a comma-separated list of metric set names or native metric names")
}

func benchmarkMetricOverride() (string, error) {
	process.mu.Lock()
	defer process.mu.Unlock()
	if process.err != nil {
		return "", process.err
	}
	if process.finished {
		return "", fmt.Errorf("cuptibench: benchmark after Finish")
	}
	return process.metrics, nil
}

func recordBenchmarkResult(r *Result, owner *Plan) error { return process.record(r, owner) }

func (p *processExport) record(r *Result, owner *Plan) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.err != nil {
		return p.err
	}
	if p.finished {
		return fmt.Errorf("cuptibench: record after Finish")
	}
	if p.path == "" {
		return nil
	}
	if err := validateExportResult(r); err != nil {
		return err
	}
	// Retain an immutable snapshot; Benchmark returns the caller its own result.
	data, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("cuptibench: encode result: %w", err)
	}
	var copy Result
	if err := json.Unmarshal(data, &copy); err != nil {
		return fmt.Errorf("cuptibench: copy result: %w", err)
	}
	p.results = append(p.results, &copy)
	if owner != nil {
		if p.owners == nil {
			p.owners = make(map[*Plan]bool)
		}
		p.owners[owner] = true
	}
	return nil
}

// Finish writes queued benchmark profiles and their JSON manifest. It never
// overwrites an existing output. Repeated calls return the first completion
// result. All contributing benchmark plans must have closed successfully before
// Finish. A failed native cleanup permanently invalidates their queued results,
// including after a successful retry. Finish never closes caller resources.
func Finish() error { return process.finish() }

func (p *processExport) finish() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.finished {
		return p.err
	}
	p.finished = true
	if p.err != nil || p.path == "" {
		return p.err
	}
	for owner := range p.owners {
		status := owner.exportStatus.Load()
		if status&exportOwnerFailed != 0 {
			p.err = fmt.Errorf("cuptibench: profile owner cleanup failed; queued measurements are invalid")
			break
		}
		if status&exportOwnerClosed == 0 {
			p.err = fmt.Errorf("cuptibench: contributing plan is not closed; close all plans before Finish")
			break
		}
	}
	if p.err == nil {
		p.err = writeExport(p.path, p.results)
	}
	p.results = nil
	p.owners = nil
	return p.err
}
