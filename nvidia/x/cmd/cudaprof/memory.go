package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// runMemoryCommand summarizes retained evidence, never inferring residency
// from allocation API names or transfer activity.
func runMemoryCommand(args []string) error {
	args, limits, err := parseReadLimits(args)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("memory", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	fs.Usage = func() {
		fmt.Fprintln(os.Stdout, "usage: cudaprof memory [options] capture.pprof")
		fs.PrintDefaults()
	}
	maxRecords := fs.Int64("max-records", 0, "journal record limit")
	journalBytes := fs.Int64("journal-bytes", 0, "decompressed journal byte limit")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *maxRecords < 0 || *journalBytes < 0 {
		return fmt.Errorf("memory: journal limits must not be negative")
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: cudaprof memory capture.pprof")
	}
	cap, captureSHA, err := readAlignmentCapture(fs.Arg(0), limits)
	if err != nil {
		return err
	}
	status, records, err := decodeCompleteCapture(cap, *maxRecords, *journalBytes)
	if err != nil {
		return err
	}
	gapStatus, _ := readGapStatus(cap)
	report := summarizeMemory(records, gapStatus.Capability)
	report.Capture = fs.Arg(0)
	report.SHA256 = captureSHA
	report.Completeness = status.RunCompleteness
	if report.Completeness == "" {
		report.Completeness = runComplete
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

type memoryEvidence struct {
	State  string `json:"state"`
	Reason string `json:"reason"`
}

type memoryTransfer struct {
	Journal     string `json:"journal"`
	Device      uint32 `json:"device"`
	Stream      uint32 `json:"stream"`
	CopyKind    uint8  `json:"copy_kind"`
	Bytes       uint64 `json:"bytes"`
	Start       int64  `json:"start_ns"`
	End         int64  `json:"end_ns"`
	Correlation uint32 `json:"correlation"`
}

type memoryAPICount struct {
	Domain  string `json:"domain"`
	Name    string `json:"name"`
	Records int    `json:"records"`
}

type memoryReport struct {
	WorkloadEvidence   memoryEvidence     `json:"workload_evidence"`
	Version            int                `json:"version"`
	Capture            string             `json:"capture"`
	SHA256             string             `json:"capture_sha256"`
	Completeness       string             `json:"collector_completeness"`
	Capability         json.RawMessage    `json:"collector_capability,omitempty"`
	Journals           []timeline.Journal `json:"journals"`
	Transfers          []memoryTransfer   `json:"transfers"`
	TransferRecords    int                `json:"transfer_records"`
	AllocationAPIs     []memoryAPICount   `json:"allocation_api_records"`
	AllocationRecords  int                `json:"allocation_api_record_count"`
	TransferEvidence   memoryEvidence     `json:"transfer_evidence"`
	AllocationEvidence memoryEvidence     `json:"allocation_evidence"`
	Residency          memoryEvidence     `json:"residency"`
	Migration          memoryEvidence     `json:"migration"`
	PageFaults         memoryEvidence     `json:"page_faults"`
	DeviceSupport      memoryEvidence     `json:"device_support"`
	Limits             []string           `json:"limits"`
}

func summarizeMemory(records timeline.Records, capability json.RawMessage) memoryReport {
	r := memoryReport{
		WorkloadEvidence: memoryEvidence{"unverified", "collector completeness does not establish output equality, independent workload coverage, or stationarity"},
		Version:          1, Capability: capability, Journals: records.Journals,
		Transfers: []memoryTransfer{}, AllocationAPIs: []memoryAPICount{},
		TransferEvidence:   memoryEvidence{"unavailable", "no retained memcpy activity; absence does not prove no transfers or migrations"},
		AllocationEvidence: memoryEvidence{"unavailable", "no retained allocation API calls; API collection may be disabled or allocation may precede the window"},
		Residency:          memoryEvidence{"unsupported", "current collector and decoded schema retain no per-range residency observations"},
		Migration:          memoryEvidence{"unsupported", "current collector does not enable CUPTI unified-memory counter activity; memcpy records are not migration records"},
		PageFaults:         memoryEvidence{"unsupported", "current collector does not enable CUPTI unified-memory page-fault counters"},
		DeviceSupport:      memoryEvidence{"unprobed", "capture analysis does not probe device/backend unified-memory profiling support"},
		Limits:             []string{"API activity records contain no allocation pointer, byte count, result code, or lifetime; calls are not successful allocations", "runtime and driver calls may describe the same operation and are counted separately", "transfer timestamps retain the journal clock; no host clock conversion or operation-window correlation is inferred", "allocation or transfer evidence does not establish current residency"},
	}
	for _, a := range records.Activities {
		if a.Subtype != "memcpy" {
			continue
		}
		r.Transfers = append(r.Transfers, memoryTransfer{a.Journal, a.Device, a.Stream, a.CopyKind, a.Bytes, a.Start, a.End, a.Correlation})
	}
	r.TransferRecords = len(r.Transfers)
	if r.TransferRecords > 0 {
		r.TransferEvidence = memoryEvidence{"observed", "retained memcpy activities only; transfer records do not establish residency or managed-memory migration"}
	}
	counts := make(map[string]*memoryAPICount)
	for _, call := range records.Calls {
		name := call.Name()
		if !strings.HasPrefix(name, "cuMemAlloc") && !strings.HasPrefix(name, "cudaMalloc") && !strings.HasPrefix(name, "cuMemHostAlloc") {
			continue
		}
		key := call.Domain + ":" + name
		if counts[key] == nil {
			counts[key] = &memoryAPICount{Domain: call.Domain, Name: name}
		}
		counts[key].Records++
		r.AllocationRecords++
	}
	for _, c := range counts {
		r.AllocationAPIs = append(r.AllocationAPIs, *c)
	}
	sort.Slice(r.AllocationAPIs, func(i, j int) bool {
		a, b := r.AllocationAPIs[i], r.AllocationAPIs[j]
		if a.Domain != b.Domain {
			return a.Domain < b.Domain
		}
		return a.Name < b.Name
	})
	if r.AllocationRecords > 0 {
		r.AllocationEvidence = memoryEvidence{"observed", "allocation API call records only; arguments, success and residency are unavailable"}
	}
	return r
}
