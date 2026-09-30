package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
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
	UnifiedMemory      []memoryCounterObservation `json:"unified_memory"`
	MemoryPolicies     []timeline.MemoryPolicy    `json:"memory_policy_observations"`
	Placement          memoryEvidence             `json:"placement_policy_evidence"`
	WorkloadEvidence   memoryEvidence             `json:"workload_evidence"`
	Version            int                        `json:"version"`
	Capture            string                     `json:"capture"`
	SHA256             string                     `json:"capture_sha256"`
	Completeness       string                     `json:"collector_completeness"`
	Capability         json.RawMessage            `json:"collector_capability,omitempty"`
	Journals           []timeline.Journal         `json:"journals"`
	Transfers          []memoryTransfer           `json:"transfers"`
	TransferRecords    int                        `json:"transfer_records"`
	AllocationAPIs     []memoryAPICount           `json:"allocation_api_records"`
	AllocationRecords  int                        `json:"allocation_api_record_count"`
	TransferEvidence   memoryEvidence             `json:"transfer_evidence"`
	AllocationEvidence memoryEvidence             `json:"allocation_evidence"`
	Residency          memoryEvidence             `json:"residency"`
	Migration          memoryEvidence             `json:"migration"`
	PageFaults         memoryEvidence             `json:"page_faults"`
	DeviceSupport      memoryEvidence             `json:"device_support"`
	Limits             []string                   `json:"limits"`
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
	addNativeMemoryEvidence(&r, records, capability)
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

type memoryCounterObservation struct {
	Record           timeline.UnifiedMemory `json:"record"`
	ValueMeaning     string                 `json:"value_meaning"`
	OperationWindows []int                  `json:"overlapping_application_ranges"`
	Correlation      string                 `json:"operation_correlation"`
}

func memoryCounterMeaning(kind uint32) string {
	switch kind {
	case 1, 2, 8:
		return "transferred bytes"
	case 3:
		return "faulting instruction program counter; not a fault count"
	case 4:
		return "GPU page-fault groups for the page; not individual page count"
	case 5, 7:
		return "affected memory-region bytes"
	default:
		return "native value retained without a supported quantitative interpretation"
	}
}
func addNativeMemoryEvidence(report *memoryReport, records timeline.Records, capability json.RawMessage) {
	report.UnifiedMemory = []memoryCounterObservation{}
	report.MemoryPolicies = records.MemoryPolicies
	report.Placement = memoryEvidence{"unavailable", "no retained range-attribute/advice/prefetch observations; allocation API records are not placement"}
	if len(records.MemoryPolicies) > 0 {
		report.Placement = memoryEvidence{"observed", "native advice/prefetch requests and successful scalar query outputs; preferred/last-prefetch locations are not current residency"}
	}
	var native struct {
		UM *struct {
			State   string `json:"state"`
			Results []int  `json:"native_configuration_results"`
		} `json:"unified_memory"`
		ClocksUnified bool `json:"clocks_unified"`
	}
	if json.Unmarshal(capability, &native) == nil && native.UM != nil {
		report.DeviceSupport = memoryEvidence{"native_configuration_receipt", "process-wide UM configuration state: " + native.UM.State + "; native configuration results are retained in collector_capability; this is separate from device allocation attributes"}
		if native.UM.State == "collected" || native.UM.State == "collected_partial" {
			report.Migration = memoryEvidence{"unavailable", "collector enabled supported UM counters but no retained migration records; no zero-migration claim"}
			report.PageFaults = memoryEvidence{"unavailable", "collector enabled supported UM counters but no retained fault records; no zero-fault claim"}
		} else {
			report.Migration = memoryEvidence{"unsupported", "UM collector state: " + native.UM.State}
			report.PageFaults = report.Migration
		}
	}
	for _, event := range records.UnifiedMemory {
		observation := memoryCounterObservation{Record: event, ValueMeaning: memoryCounterMeaning(event.CounterKind), OperationWindows: []int{}, Correlation: "unsupported: native UM records have no submission correlation and their streamId is invalid"}
		if native.ClocksUnified && event.Valid {
			for i, r := range records.Ranges {
				metadata, err := marker.ParseRange(r.Name)
				if err != nil || metadata.Operation == "" || r.Start < 0 || r.End < r.Start {
					continue
				}
				overlap := r.Start < event.End && event.Start < r.End
				if !event.EndValid || event.End == event.Start {
					overlap = r.Start <= event.Start && event.Start < r.End
				}
				if r.Journal == event.Journal && r.Process != 0 && r.Process == event.Process && overlap {
					observation.OperationWindows = append(observation.OperationWindows, i)
				}
			}
			if len(observation.OperationWindows) > 0 {
				observation.Correlation = "timestamp overlap in one process/journal only; not submission identity or causation"
			}
		}
		report.UnifiedMemory = append(report.UnifiedMemory, observation)
		if event.Valid {
			switch event.CounterKind {
			case 1, 2, 8:
				report.Migration = memoryEvidence{"observed", "retained native UM transfer records; migration does not establish current residency"}
			case 3, 4:
				report.PageFaults = memoryEvidence{"observed", "retained native fault-service records; CPU values are instruction addresses and GPU values count fault groups"}
			}
		}
	}
	report.Limits = append(report.Limits, "UM fault/migration service intervals are excluded from GPU occupancy and kernel duration sums", "UM streamId is invalid; overlapping application ranges are context only and require a unified recorded clock", "range advice and last-prefetch queries report policies/history, not physical page residency")
}
