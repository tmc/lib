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

	// FlushNanos is how long the producer spent flushing activity buffers
	// after a stop edge. The edge is stamped before the flush, so this time
	// lies outside the range.
	FlushNanos uint64 `json:"flush_nanos,omitempty"`

	// GPU is the GPU state the controller read once it saw this edge. It is
	// not a wire fact: see gpuEdgeSample for how far after the edge it lies.
	GPU *gpuEdgeSample `json:"gpu,omitempty"`
}

// capturePhaseReceipt is deliberately UNREAD unless every qualification gate
// passes. It is host provenance, not a device or operand observation.
type capturePhaseReceipt struct {
	Status         string `json:"status"`
	Reason         string `json:"reason,omitempty"`
	API            string `json:"api,omitempty"`
	Index          uint32 `json:"index,omitempty"`
	StartTimestamp uint64 `json:"start_timestamp,omitempty"`
	StopTimestamp  uint64 `json:"stop_timestamp,omitempty"`
	DurationNanos  uint64 `json:"duration_nanos,omitempty"`
	KernelCount    uint64 `json:"kernel_count,omitempty"`
	KernelNanos    uint64 `json:"kernel_nanos,omitempty"`

	// The Nanos totals above and below are SUMS over records, so work that
	// overlaps on separate streams counts more than once and a sum may exceed
	// DurationNanos. DeviceBusyNanos is a UNION of the same intervals and
	// cannot: it is the time the device had at least one activity in flight.
	//
	// DeviceUnresolvedNanos is graph replay time that no resolved record
	// covers. A replay whose nodes produced no records is known only by its
	// envelope, and the device may have been busy or idle anywhere inside it,
	// so that time is neither busy nor idle. Counting it as idle is how an
	// opaque replay read as a mostly idle device. When it is nonzero
	// DeviceIdleStatus is UNREAD and DeviceIdleNanos is only a floor.
	// DeviceBusyNanos + DeviceIdleNanos + DeviceUnresolvedNanos ==
	// DurationNanos exactly.
	MemcpyCount           uint64             `json:"memcpy_count,omitempty"`
	MemcpyNanos           uint64             `json:"memcpy_nanos,omitempty"`
	MemcpyBytes           uint64             `json:"memcpy_bytes,omitempty"`
	MemsetCount           uint64             `json:"memset_count,omitempty"`
	MemsetNanos           uint64             `json:"memset_nanos,omitempty"`
	DeviceBusyNanos       uint64             `json:"device_busy_nanos,omitempty"`
	DeviceIdleNanos       uint64             `json:"device_idle_nanos,omitempty"`
	DeviceUnresolvedNanos uint64             `json:"device_unresolved_nanos,omitempty"`
	DeviceIdleStatus      string             `json:"device_idle_status,omitempty"`
	OmittedEdges          uint64             `json:"omitted_edges,omitempty"`
	Edges                 []capturePhaseEdge `json:"edges,omitempty"`

	// Phases carries one MEASURED receipt per qualified pair when the target
	// opened the capture range more than once. The enclosing receipt stays
	// UNREAD in that case: several intervals have no single duration, and a
	// consumer must name which phase it means rather than be handed one.
	Phases []capturePhaseReceipt `json:"phases,omitempty"`
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
		completeness = runCompletenessOf(result.TransportReady, result.CollectorFinalized, unqualifiedGenerations(result.WorkerGenerations), result.DroppedRecords, result.ActivityRecords)
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
	primary := make([]capturePhaseEdge, 0, len(edges))
	for _, edge := range edges {
		if edge.Edge == "redundant_start" || edge.Edge == "redundant_stop" {
			continue
		}
		primary = append(primary, edge)
	}
	if len(primary) == 0 || len(primary)%2 != 0 {
		r.Reason = "edge_pair_invalid"
		return r
	}
	measured := make([]capturePhaseReceipt, 0, len(primary)/2)
	for i := 0; i+1 < len(primary); i += 2 {
		m, ok := measurePhase(result, primary[i], primary[i+1])
		if !ok {
			r.Reason = "edge_pair_invalid"
			return r
		}
		measured = append(measured, m)
	}
	if len(measured) == 1 {
		edges, omitted := r.Edges, r.OmittedEdges
		*r = measured[0]
		r.Edges, r.OmittedEdges = edges, omitted
		return r
	}
	// The pairs are individually well formed, so the edges are not invalid;
	// there is simply no one interval this receipt could describe.
	r.Reason = "multiple_phases"
	r.Phases = measured
	return r
}

// measurePhase qualifies one start/stop pair and measures the kernels that fall
// entirely inside it. It reports false rather than a partial receipt, so a pair
// that fails any gate cannot be read as a measurement.
//
// Graph replay envelopes are clipped to the pair rather than required to fall
// inside it: an envelope that crosses an edge still hides device time within
// the phase.
func measurePhase(result captureResult, start, stop capturePhaseEdge) (capturePhaseReceipt, bool) {
	if start.Edge != "start" || stop.Edge != "stop" || !matchingPhaseAPIs(start.API, stop.API) || start.Index != stop.Index || start.ToggleFailures != 0 || stop.ToggleFailures != 0 || start.Sequence >= stop.Sequence || start.Timestamp == 0 || stop.Timestamp <= start.Timestamp {
		return capturePhaseReceipt{}, false
	}
	m := capturePhaseReceipt{
		Status:         "MEASURED",
		API:            start.API,
		Index:          start.Index,
		StartTimestamp: start.Timestamp,
		StopTimestamp:  stop.Timestamp,
		DurationNanos:  stop.Timestamp - start.Timestamp,
	}
	busy := make([]interval, 0, len(result.Activities))
	var replays []interval
	for _, activity := range result.Activities {
		if !activity.Valid || activity.End < activity.Start {
			continue
		}
		if activity.Subtype == "graph_trace" {
			if lo, hi := max(activity.Start, start.Timestamp), min(activity.End, stop.Timestamp); lo < hi {
				replays = append(replays, interval{lo, hi})
			}
			continue
		}
		if activity.Start < start.Timestamp || activity.End > stop.Timestamp {
			continue
		}
		switch activity.Subtype {
		case "kernel":
			m.KernelCount++
			m.KernelNanos += activity.End - activity.Start
		case "memcpy":
			m.MemcpyCount++
			m.MemcpyNanos += activity.End - activity.Start
			m.MemcpyBytes += activity.Bytes
		case "memset":
			m.MemsetCount++
			m.MemsetNanos += activity.End - activity.Start
		default:
			continue
		}
		busy = append(busy, interval{activity.Start, activity.End})
	}
	m.DeviceBusyNanos = unionNanos(busy)
	covered := unionNanos(append(busy, replays...))
	m.DeviceUnresolvedNanos = covered - m.DeviceBusyNanos
	m.DeviceIdleNanos = m.DurationNanos - covered
	m.DeviceIdleStatus = "MEASURED"
	if m.DeviceUnresolvedNanos > 0 {
		m.DeviceIdleStatus = "UNREAD"
		m.Reason = "graph_replay_unresolved"
	}
	return m, true
}

func matchingPhaseAPIs(start, stop string) bool {
	return (start == "cudaProfilerStart" || start == "cuProfilerStart") &&
		(stop == "cudaProfilerStop" || stop == "cuProfilerStop")
}

func (r *capturePhaseReceipt) String() string {
	if r == nil {
		return "UNREAD: no capture-range edges"
	}
	if r.Status != "MEASURED" {
		if len(r.Phases) > 0 {
			return fmt.Sprintf("UNREAD: %s (%d qualified phases; select one by index)", r.Reason, len(r.Phases))
		}
		return fmt.Sprintf("UNREAD: %s", r.Reason)
	}
	if r.DeviceIdleStatus == "UNREAD" {
		return fmt.Sprintf("MEASURED: %s [%d,%d); device idle UNREAD: %s", r.API, r.StartTimestamp, r.StopTimestamp, r.Reason)
	}
	return fmt.Sprintf("MEASURED: %s [%d,%d)", r.API, r.StartTimestamp, r.StopTimestamp)
}
