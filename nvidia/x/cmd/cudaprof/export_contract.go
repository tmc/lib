package main

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

type traceExportIdentity struct {
	sha256, path, arm string
	shift             uint64
	summaryReferences int64
}

func traceRecordKind(record traceRecord) string {
	switch record.rec.Kind {
	case cupti.ActivityKindKernel, cupti.ActivityKindConcurrentKernel:
		return "kernel"
	case cupti.ActivityKindMemcpy:
		return "memcpy"
	case cupti.ActivityKindMemset:
		return "memset"
	case cupti.ActivityKindDriver, cupti.ActivityKindRuntime:
		return "api"
	case cupti.ActivityKindMarker:
		if record.rec.Start == record.rec.End {
			return "nvtx_point"
		}
		return "nvtx_range"
	case cupti.ActivityKindSynchronization:
		return "sync"
	case cupti.ActivityKindGraphTrace:
		return "graph_trace"
	case cupti.ActivityKindOverhead:
		return "overhead"
	}
	return "unknown"
}

func traceRole(record traceRecord) string {
	if record.sample != nil {
		return "host_stack_sample"
	}
	switch record.kind {
	case trackDevice:
		return "raw_device"
	case trackNVTX:
		return "raw_marker"
	}
	if record.rec.Kind == cupti.ActivityKindRuntime || record.rec.Kind == cupti.ActivityKindDriver {
		return "raw_api"
	}
	return "raw_host"
}

func rawTrack(writer *perfetto.Writer, id uint64, name string, record traceRecord, identity traceExportIdentity) error {
	scope := fmt.Sprintf("%s:%d:%d:%d", record.journal, record.group[0], record.group[1], record.rec.ContextID)
	track := tracecontract.Track{UUID: id, Name: name, Role: traceRole(record), Clock: identity.sha256 + ":" + identity.arm + ":" + record.journal + ":export-relative", Arm: identity.arm, CaptureSHA256: identity.sha256, Journal: record.journal, Scope: scope, Provenance: "observed raw journal records", Aggregation: "none"}
	if record.kind == trackDevice {
		device := record.rec.DeviceID
		track.Device = &device
	}
	return track.Write(writer, identity.shift)
}

func exportReceipt(command string, identity traceExportIdentity, records []traceRecord, status captureResult) tracecontract.Receipt {
	r := tracecontract.Receipt{Version: 1, Command: command, Options: map[string]string{}, InputPath: identity.path, InputSHA256: identity.sha256, Kind: "capture", ConsumerVersion: fmt.Sprintf("%v", currentBuild()), Backends: []tracecontract.Backend{{Name: "native"}}, Gates: map[string]tracecontract.Gate{}, ProjectedRows: map[string]int64{}, RawDeviceRecords: map[string]int64{}, Partial: true}
	r.Gates["collector"] = tracecontract.Gate{Disposition: "unavailable", Reason: "retained journals passed decode gates; whole-run collector status is absent or unsupported", Provenance: "native journal decode; capture-status.json"}
	for _, name := range []string{"import", "workload", "stationarity", "shape", "correctness"} {
		r.Gates[name] = tracecontract.Gate{Disposition: "unchecked", Reason: "export does not execute this independent qualification", Provenance: "export options"}
	}
	for _, record := range records {
		kind := traceRecordKind(record)
		r.ProjectedRows[kind]++
		if kind == "kernel" || kind == "memcpy" || kind == "memset" {
			r.RawDeviceRecords[kind]++
		}
	}
	if status.Protocol > 0 {
		r.ProducerVersion = strconv.Itoa(status.Protocol)
	}
	r.ClockDomain = "monotonic"
	r.ClockProvenance = "cupti"
	r.ProbeCoverage = map[string]tracecontract.ProbeCoverage{
		"activity": {Present: true, Ratio: 1.0},
	}
	if status.RecordLoss != nil {
		dropped := status.RecordLoss.Offered - status.RecordLoss.Retained
		if status.RecordLoss.Offered < status.RecordLoss.Retained {
			dropped = 0
		}
		r.LostRecords = map[string]tracecontract.Loss{
			"activity": {
				Dropped: dropped,
				Total:   status.RecordLoss.Offered,
			},
		}
	}
	if status.Protocol == captureProtocolVersion && status.RunCompleteness != "" {
		count := status.ActivityRecords
		r.CollectorActivityRecords = &count
		complete := status.RunCompleteness == runComplete && status.DroppedRecords == 0 && status.UnqualifiedGenerations == 0 && targetFailure(&status) == nil
		if status.RecordLoss != nil && status.RecordLoss.Offered != status.RecordLoss.Retained {
			complete = false
		}
		if complete {
			r.Gates["collector"] = tracecontract.Gate{Disposition: "passed", Reason: "supported whole-run collector status and retained journal decode gates passed", Provenance: "capture-status.json and native journal decode"}
		} else {
			r.Gates["collector"] = tracecontract.Gate{Disposition: "failed", Reason: "source capture does not establish complete whole-run collection", Provenance: "capture-status.json"}
		}
	}
	return r
}

func captureExportStatus(capture *cudacapture.Capture) captureResult {
	var envelope struct {
		Status captureResult `json:"status"`
	}
	if json.Unmarshal(capture.Files["capture-status.json"], &envelope) != nil {
		return captureResult{}
	}
	return envelope.Status
}

func writeExportReceipt(writer *perfetto.Writer, receipt tracecontract.Receipt, id, timestamp uint64) error {
	track := tracecontract.Track{UUID: id, Name: "export qualification receipt", Role: "qualification", Clock: receipt.InputSHA256 + ":export-relative", CaptureSHA256: receipt.InputSHA256, Provenance: "independent evidence dispositions", Aggregation: "none"}
	if err := track.Write(writer, timestamp); err != nil {
		return err
	}
	return receipt.Write(writer, id, timestamp)
}

func writeExportFindings(writer *perfetto.Writer, findings []tracecontract.Finding, id, timestamp uint64) error {
	if len(findings) == 0 {
		return nil
	}
	track := tracecontract.Track{
		UUID:        id,
		Name:        "diagnostic findings",
		Role:        "findings",
		Clock:       "monotonic",
		Provenance:  "admitted diagnostic rules",
		Aggregation: "none",
	}
	if err := track.Write(writer, timestamp); err != nil {
		return err
	}
	return tracecontract.WriteFindings(writer, id, timestamp, findings)
}
