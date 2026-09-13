package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

type inspectReport struct {
	Version       int                            `json:"version"`
	Profiles      []inspectProfile               `json:"profiles,omitempty"`
	Code          cudacapture.CodeFiles          `json:"code"`
	Unavailable   map[string]string              `json:"code_unavailable,omitempty"`
	Modules       []cudacapture.Module           `json:"modules,omitempty"`
	Descriptors   []cudacapture.MetricDescriptor `json:"descriptors,omitempty"`
	Observations  int                            `json:"observations,omitempty"`
	ArtifactNames []string                       `json:"artifacts"`
	Coverage      *inspectCoverage               `json:"coverage,omitempty"`
	TileIR        *inspectTileIRSummary          `json:"tileir,omitempty"`
	Remarks       *inspectRemarksSummary         `json:"remarks,omitempty"`
}

type inspectTileIRSummary struct {
	File       string   `json:"file"`
	IsBytecode bool     `json:"is_bytecode"`
	Producer   string   `json:"producer,omitempty"`
	Functions  []string `json:"functions,omitempty"`
	TMAOps     int      `json:"tma_operations"`
	MMAOps     int      `json:"mma_operations"`
	TotalOps   int      `json:"total_operations"`
}

type inspectRemarksSummary struct {
	File               string            `json:"file,omitempty"`
	Total              int               `json:"total"`
	TMAFailures        int               `json:"tma_failures"`
	TensorCoreMMAs     int               `json:"tensor_core_mmas"`
	TensorCoreFailures int               `json:"tensor_core_failures"`
	Records            []tileiras.Remark `json:"records,omitempty"`
}

type inspectCoverage struct {
	Outcome            string `json:"outcome"`
	CaptureSuccess     bool   `json:"capture_success"`
	TargetExitCode     int    `json:"target_exit_code"`
	TargetSignal       string `json:"target_signal,omitempty"`
	TransportReady     bool   `json:"transport_ready"`
	BootstrapSeen      bool   `json:"bootstrap_seen"`
	CollectorFinalized bool   `json:"collector_finalized"`
	ActivityRecords    uint64 `json:"activity_records"`
	DroppedRecords     uint64 `json:"dropped_records"`
	CaptureError       string `json:"capture_error,omitempty"`
	// RunCompleteness is the whole-run value. CaptureSuccess above qualifies
	// the target generation only, so a reader must not take it for a statement
	// about every worker generation in the run.
	RunCompleteness        string            `json:"run_completeness,omitempty"`
	UnqualifiedGenerations uint64            `json:"unqualified_generations,omitempty"`
	Workers                []inspectWorker   `json:"worker_generations,omitempty"`
	ProcessIDs             []int             `json:"process_ids,omitempty"`
	Devices                []uint32          `json:"devices,omitempty"`
	ActivityKinds          map[string]uint64 `json:"activity_kinds,omitempty"`
}

// inspectWorker republishes the per-generation disposition the capture
// recorded. A zero-record generation is reported as recorded; nothing here
// reads a count as evidence that work was lost or unobserved.
type inspectWorker struct {
	Process         uint64 `json:"process"`
	Generation      uint64 `json:"generation"`
	Records         uint64 `json:"records"`
	ActivityRecords uint64 `json:"activity_records"`
	Finalized       bool   `json:"finalized"`
	Failure         string `json:"failure,omitempty"`
	Disposition     string `json:"disposition"`
}

type inspectProfile struct {
	Type    string `json:"type"`
	Unit    string `json:"unit"`
	Samples int    `json:"samples"`
}

func runInspectCommand(args []string) error {
	args, limits, err := parseReadLimits(args)
	if err != nil {
		return err
	}
	if len(args) >= 1 && (args[0] == "--cfg" || args[0] == "--bbcfg") {
		return runInspectCFG(args, limits)
	}
	if len(args) != 2 || (args[0] != "--coverage" && args[0] != "--profiles" && args[0] != "--remarks" && args[0] != "--tileir") {
		return errors.New("usage: cudaprof inspect [--capture-file-bytes=N] [--capture-total-bytes=N] --coverage|--profiles|--remarks|--tileir|--cfg|--bbcfg [kernel] capture.pprof")
	}
	path, err := filepath.Abs(args[1])
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	version := 2
	if len(cap.CodeUnavailable) > 0 || len(cap.Modules) > 0 {
		version = 3
	}
	report := inspectReport{Version: version, Code: cap.Code, Unavailable: cap.CodeUnavailable, Modules: cap.Modules, Descriptors: cap.Descriptors, Observations: len(cap.Observations)}
	for name := range cap.Files {
		report.ArtifactNames = append(report.ArtifactNames, name)
	}
	if status, ok := cap.Files["capture-status.json"]; ok {
		var envelope struct {
			Status inspectCoverage `json:"status"`
		}
		if json.Unmarshal(status, &envelope) == nil {
			addCoverageDetails(&envelope.Status, status)
			report.Coverage = &envelope.Status
		}
	}
	report.TileIR = inspectTileIR(cap)
	report.Remarks = inspectRemarks(cap)

	if args[0] == "--profiles" {
		for _, typ := range cap.Profile.SampleType {
			report.Profiles = append(report.Profiles, inspectProfile{Type: typ.Type, Unit: typ.Unit, Samples: len(cap.Profile.Sample)})
		}
	}
	if args[0] == "--remarks" {
		if report.Remarks == nil {
			report.Remarks = &inspectRemarksSummary{}
		}
		return json.NewEncoder(os.Stdout).Encode(report.Remarks)
	}
	if args[0] == "--tileir" {
		if report.TileIR == nil {
			report.TileIR = &inspectTileIRSummary{}
		}
		return json.NewEncoder(os.Stdout).Encode(report.TileIR)
	}
	return json.NewEncoder(os.Stdout).Encode(report)
}

// addCoverageDetails reads the bounded aggregates the capture recorded. They
// are published directly rather than recomputed from a per-record array, which
// is what kept the status file from scaling with the journal.
func addCoverageDetails(coverage *inspectCoverage, status []byte) {
	var envelope struct {
		Status struct {
			TargetPID     int               `json:"target_pid"`
			Devices       []uint32          `json:"devices"`
			ActivityKinds map[string]uint64 `json:"activity_kinds"`
		} `json:"status"`
	}
	if json.Unmarshal(status, &envelope) != nil {
		return
	}
	if envelope.Status.TargetPID != 0 {
		coverage.ProcessIDs = []int{envelope.Status.TargetPID}
	}
	coverage.Devices = append(coverage.Devices, envelope.Status.Devices...)
	sort.Slice(coverage.Devices, func(i, j int) bool { return coverage.Devices[i] < coverage.Devices[j] })
	if len(envelope.Status.ActivityKinds) != 0 {
		coverage.ActivityKinds = envelope.Status.ActivityKinds
	}
	// A capture written before whole-run completeness existed publishes worker
	// dispositions but no verdict. Deriving it here, by the same rule the
	// writer uses, keeps a target-qualified capture_success from standing
	// alone in this output for those artifacts.
	if coverage.RunCompleteness == "" && len(coverage.Workers) != 0 {
		coverage.UnqualifiedGenerations = unqualifiedGenerations(workerDispositions(coverage.Workers))
		coverage.RunCompleteness = runCompletenessOf(coverage.TransportReady, coverage.CollectorFinalized, coverage.UnqualifiedGenerations)
	}
}

func workerDispositions(workers []inspectWorker) []captureWorker {
	out := make([]captureWorker, len(workers))
	for i, w := range workers {
		out[i] = captureWorker{Failure: w.Failure, Disposition: w.Disposition}
	}
	return out
}

func readCapturePath(path string) (*cudacapture.Capture, error) {
	return readCapturePathLimits(path, cudacapture.Limits{})
}

// readCapturePathLimits reads a capture under explicit limits. A zero Limits
// is the package default, so an artifact written under a raised producer
// budget is rejected until the reader asks for the same bound by name.
func readCapturePathLimits(path string, limits cudacapture.Limits) (*cudacapture.Capture, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return cudacapture.ReadDirLimits(path, limits)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return cudacapture.ReadLimits(io.Reader(f), limits)
}

// parseReadLimits removes the explicit read-side bounds from args. They mirror
// the producer flags by name, exist for reading an artifact this machine
// produced under a raised budget, and never become a default: an unset flag
// leaves the package default in force.
//
// Raising TotalBytes also raises InputBytes to at least the same value, since
// a larger embedded total implies a larger serialized profile to read it from.
func parseReadLimits(args []string) ([]string, cudacapture.Limits, error) {
	var limits cudacapture.Limits
	rest := make([]string, 0, len(args))
	for _, arg := range args {
		name, value, hasValue := strings.Cut(arg, "=")
		if !hasValue || (name != "--capture-file-bytes" && name != "--capture-total-bytes") {
			rest = append(rest, arg)
			continue
		}
		n, err := strconv.ParseInt(value, 10, 64)
		if err != nil || n <= 0 {
			return nil, cudacapture.Limits{}, fmt.Errorf("invalid %s %q", name, value)
		}
		if name == "--capture-file-bytes" {
			limits.FileBytes = n
		} else {
			limits.TotalBytes = n
		}
	}
	if limits.TotalBytes > cudacapture.MaxCaptureInputBytes {
		limits.InputBytes = limits.TotalBytes
	}
	return rest, limits, nil
}

func inspectTileIR(cap *cudacapture.Capture) *inspectTileIRSummary {
	if cap == nil || cap.Code.TileIR == "" {
		return nil
	}
	data, ok := cap.Files[cap.Code.TileIR]
	if !ok || len(data) == 0 {
		return nil
	}
	isBytecode := tileir.IsTileIRBytecode(data)
	var mod *tileir.Module
	var err error
	if isBytecode {
		bm, decErr := tileir.DecodeBytecodeBytes(data)
		if decErr == nil {
			mod = bm.ToModule()
		}
	} else {
		mod, err = tileir.ParseBytes(data)
	}
	if mod == nil || err != nil {
		return nil
	}
	summary := &inspectTileIRSummary{
		File:       cap.Code.TileIR,
		IsBytecode: isBytecode,
		Producer:   mod.Producer,
	}
	for _, fn := range mod.Functions {
		summary.Functions = append(summary.Functions, fn.Name)
		for _, op := range fn.Operations {
			summary.TotalOps++
			if op.IsTMA() {
				summary.TMAOps++
			}
			if op.IsMMA() {
				summary.MMAOps++
			}
		}
	}
	return summary
}

func inspectRemarks(cap *cudacapture.Capture) *inspectRemarksSummary {
	rems, name, _ := findRemarks(cap)
	if len(rems) == 0 {
		return nil
	}
	summary := &inspectRemarksSummary{
		File:    name,
		Total:   len(rems),
		Records: rems,
	}
	for _, r := range rems {
		if r.IsTMAFailure() {
			summary.TMAFailures++
		}
		if r.IsTensorCoreMMA() {
			summary.TensorCoreMMAs++
		}
		if r.IsTensorCoreFailure() {
			summary.TensorCoreFailures++
		}
	}
	return summary
}

func findRemarks(cap *cudacapture.Capture) ([]tileiras.Remark, string, error) {
	if cap == nil || len(cap.Files) == 0 {
		return nil, "", nil
	}
	candidateNames := []string{"tileiras-remarks.yaml", "remarks.yaml", "remarks.yml"}
	for _, name := range candidateNames {
		if d, ok := cap.Files[name]; ok {
			rems, err := tileiras.ParseRemarksYAML(d)
			if err == nil && len(rems) > 0 {
				return rems, name, nil
			}
		}
	}
	for name, d := range cap.Files {
		if strings.HasSuffix(name, ".yaml") || strings.HasSuffix(name, ".yml") {
			if strings.Contains(string(d), "--- !") || strings.Contains(string(d), "Remark") {
				rems, err := tileiras.ParseRemarksYAML(d)
				if err == nil && len(rems) > 0 {
					return rems, name, nil
				}
			}
		}
	}
	return nil, "", nil
}

