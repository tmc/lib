package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// telemetryEdges requires the retained unified-clock receipt. Old receipts
// cannot place NVML queries against device time merely because values exist.
func telemetryEdges(cap *cudacapture.Capture) ([]capturePhaseEdge, error) {
	var envelope struct {
		Status captureStatus `json:"status"`
	}
	if err := json.Unmarshal(cap.Files["capture-status.json"], &envelope); err != nil {
		return nil, fmt.Errorf("telemetry status: %w", err)
	}
	var capability phaseCapability
	if err := json.Unmarshal(envelope.Status.Capability, &capability); err != nil || !capability.ClocksUnified {
		return nil, errors.New("telemetry requires retained unified CLOCK_MONOTONIC capability")
	}
	receipt := envelope.Status.PhaseReceipt
	if receipt == nil {
		return nil, errors.New("telemetry query receipts unavailable")
	}
	edges := append([]capturePhaseEdge(nil), receipt.Edges...)
	// Multi-phase receipt keeps edges in the enclosing receipt. Only use child
	// receipts when that list is absent, to avoid counting observations twice.
	if len(edges) == 0 {
		for _, phase := range receipt.Phases {
			edges = append(edges, phase.Edges...)
		}
	}
	var out []capturePhaseEdge
	for _, edge := range edges {
		q := edge.GPU
		if q == nil || q.Unavailable != "" {
			continue
		}
		if q.TimestampDomain != "CLOCK_MONOTONIC" || q.SampledAt == 0 || q.SampledAt < edge.Timestamp || q.ReadNanos > math.MaxUint64-q.SampledAt {
			return nil, errors.New("telemetry has an invalid or unknown query clock interval")
		}
		out = append(out, edge)
	}
	if len(out) == 0 {
		return nil, errors.New("telemetry query receipts unavailable or unsupported")
	}
	return out, nil
}

func writeTelemetryTrace(out io.Writer, edges []capturePhaseEdge, origin, base uint64) error {
	return writeTelemetryTraceAt(out, edges, origin, base, 0, 0)
}

func writeTelemetryTraceAt(out io.Writer, edges []capturePhaseEdge, origin, base, shift, clockID uint64) error {
	w := perfetto.NewWriter(out)
	w.TimestampClockID = clockID
	for i, edge := range edges {
		q := edge.GPU
		if q == nil || q.SampledAt < origin || q.ReadNanos > math.MaxUint64-q.SampledAt || shift > math.MaxUint64-(q.SampledAt-origin+q.ReadNanos) {
			return errors.New("telemetry query precedes trace origin")
		}
		// Every observation gets a lane: overlapping host queries are not nested.
		id := base + uint64(i) + 1
		if err := w.WriteTrackDescriptor(id, fmt.Sprintf("NVML %s query %d", edge.Edge, edge.Sequence), 0); err != nil {
			return err
		}
		data, err := json.Marshal(q)
		if err != nil {
			return err
		}
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("query_receipt", string(data)), perfetto.NewStringAnnotation("meaning", "host query interval; sensor observation/refresh unavailable; no per-kernel interpolation"), perfetto.NewUintAnnotation("raw_edge_sequence", edge.Sequence)}
		if err := w.WriteSliceBeginWithAnnotations(id, q.SampledAt-origin+shift, "NVML host query", "cudaprof_telemetry", args, nil); err != nil {
			return err
		}
		if err := w.WriteSliceEnd(id, q.SampledAt-origin+q.ReadNanos+shift); err != nil {
			return err
		}
	}
	return nil
}
