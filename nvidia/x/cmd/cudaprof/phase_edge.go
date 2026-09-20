package main

import (
	"encoding/json"
	"fmt"
)

const maxPhaseEdges = 64

// capturePhaseEdge is the producer's capture-range edge, kept apart from GPU
// activities. Sequence and Timestamp are wire facts, not inferred host time.
type capturePhaseEdge struct {
	Sequence       uint64 `json:"sequence"`
	Timestamp      uint64 `json:"timestamp"`
	Edge           string `json:"edge"`
	API            string `json:"api"`
	Index          uint32 `json:"index"`
	ToggleFailures uint32 `json:"toggle_failures"`
}

// capturePhaseReceipt is deliberately UNREAD unless every qualification gate
// passes. It is host provenance, not a device or operand observation.
type capturePhaseReceipt struct {
	Status         string             `json:"status"`
	Reason         string             `json:"reason,omitempty"`
	API            string             `json:"api,omitempty"`
	Index          uint32             `json:"index,omitempty"`
	StartTimestamp uint64             `json:"start_timestamp,omitempty"`
	StopTimestamp  uint64             `json:"stop_timestamp,omitempty"`
	DurationNanos  uint64             `json:"duration_nanos,omitempty"`
	KernelCount    uint64             `json:"kernel_count,omitempty"`
	KernelNanos    uint64             `json:"kernel_nanos,omitempty"`
	OmittedEdges   uint64             `json:"omitted_edges,omitempty"`
	Edges          []capturePhaseEdge `json:"edges,omitempty"`
}

type phaseCapability struct {
	CaptureRange  string `json:"capture_range"`
	ClocksUnified bool   `json:"clocks_unified"`
}

func buildPhaseReceipt(result captureResult, edges []capturePhaseEdge) *capturePhaseReceipt {
	if len(edges) == 0 {
		return nil
	}
	r := &capturePhaseReceipt{Status: "UNREAD"}
	if len(edges) > maxPhaseEdges {
		r.OmittedEdges = uint64(len(edges) - maxPhaseEdges)
		edges = edges[:maxPhaseEdges]
		r.Reason = "edge_limit"
		r.Edges = append([]capturePhaseEdge(nil), edges...)
		return r
	}
	r.Edges = append([]capturePhaseEdge(nil), edges...)
	completeness := result.RunCompleteness
	if completeness == "" {
		completeness = runCompletenessOf(result.TransportReady, result.CollectorFinalized, unqualifiedGenerations(result.WorkerGenerations), result.DroppedRecords)
	}
	if completeness != runComplete {
		r.Reason = "capture_incomplete"
		return r
	}
	if result.DroppedRecords != 0 || result.RecordLoss != nil && result.RecordLoss.Offered != result.RecordLoss.Retained {
		r.Reason = "record_loss"
		return r
	}
	var cap phaseCapability
	if len(result.Capability) == 0 || json.Unmarshal(result.Capability, &cap) != nil {
		r.Reason = "capability_unreadable"
		return r
	}
	if cap.CaptureRange != "cudaProfilerApi" {
		r.Reason = "capture_range_unqualified"
		return r
	}
	if !cap.ClocksUnified {
		r.Reason = "clock_unified_false"
		return r
	}
	if len(edges) != 2 {
		r.Reason = "edge_pair_invalid"
		return r
	}
	start, stop := edges[0], edges[1]
	if start.Edge != "start" || stop.Edge != "stop" || !matchingPhaseAPIs(start.API, stop.API) || start.Index != stop.Index || start.ToggleFailures != 0 || stop.ToggleFailures != 0 || start.Sequence >= stop.Sequence || start.Timestamp == 0 || stop.Timestamp <= start.Timestamp {
		r.Reason = "edge_pair_invalid"
		return r
	}
	r.Status, r.API, r.Index = "MEASURED", start.API, start.Index
	r.StartTimestamp, r.StopTimestamp = start.Timestamp, stop.Timestamp
	r.DurationNanos = stop.Timestamp - start.Timestamp
	for _, activity := range result.Activities {
		if activity.Subtype != "kernel" || !activity.Valid || activity.Start < start.Timestamp || activity.End > stop.Timestamp {
			continue
		}
		r.KernelCount++
		r.KernelNanos += activity.End - activity.Start
	}
	return r
}

func matchingPhaseAPIs(start, stop string) bool {
	switch {
	case start == "cudaProfilerStart" && stop == "cudaProfilerStop":
		return true
	case start == "cuProfilerStart" && stop == "cuProfilerStop":
		return true
	default:
		return false
	}
}

func (r *capturePhaseReceipt) String() string {
	if r == nil {
		return "UNREAD: no capture-range edges"
	}
	if r.Status != "MEASURED" {
		return fmt.Sprintf("UNREAD: %s", r.Reason)
	}
	return fmt.Sprintf("MEASURED: %s [%d,%d)", r.API, r.StartTimestamp, r.StopTimestamp)
}
