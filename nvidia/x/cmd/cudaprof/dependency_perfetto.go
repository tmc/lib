package main

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/dependency"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

const dependencyTrackBase = uint64(1 << 63)

// writeDependencyTrace preserves raw records and draws only explicit edges.
// Overlapping inventory nodes use separate lanes; the drawing never adds an
// edge between adjacent slices or a submission and a device activity.
func writeDependencyTrace(out io.Writer, raw []traceRecord, result dependencyResult) error {
	report, err := dependency.Analyze(result.Dependency.Graph)
	if err != nil {
		return err
	}
	origin := traceOrigin(raw)
	for _, node := range report.Graph.Nodes {
		if uint64(node.Start) < origin {
			origin = uint64(node.Start)
		}
	}
	if _, err := writeTraceOrigin(out, raw, origin, 0, ""); err != nil {
		return err
	}
	pw := perfetto.NewWriter(out)
	statusTrack := dependencyTrackBase
	statusName := "Dependency qualification: " + report.State
	if err := pw.WriteTrackDescriptor(statusTrack, statusName, 0); err != nil {
		return err
	}
	annotations := []perfetto.DebugAnnotation{
		perfetto.NewStringAnnotation("capture_sha256", result.SHA256),
		perfetto.NewStringAnnotation("collector_completeness", result.Completeness),
		perfetto.NewStringAnnotation("dependency_state", report.State),
		perfetto.NewStringAnnotation("missing_evidence", strings.Join(report.Reasons, "; ")),
		perfetto.NewStringAnnotation("limits", result.Limits),
		perfetto.NewIntAnnotation("node_count", int64(len(report.Graph.Nodes))),
		perfetto.NewIntAnnotation("explicit_edge_count", int64(len(report.Graph.Edges))),
	}
	if err := pw.WriteFlowInstant(statusTrack, 0, statusName, "dependency_qualification", annotations, nil, nil); err != nil {
		return err
	}

	type lane struct{ uuid, end uint64 }
	lanes := map[string][]lane{}
	nodes := append([]dependency.Node(nil), report.Graph.Nodes...)
	sort.SliceStable(nodes, func(i, j int) bool {
		if nodes[i].Clock != nodes[j].Clock {
			return nodes[i].Clock < nodes[j].Clock
		}
		if nodes[i].Start != nodes[j].Start {
			return nodes[i].Start < nodes[j].Start
		}
		return nodes[i].ID < nodes[j].ID
	})
	type event struct {
		ts, track       uint64
		kind            int // slice end, endpoint instant, slice begin
		name, category  string
		annotations     []perfetto.DebugAnnotation
		flow, terminate []uint64
	}
	events := []event{}
	tracks := map[string]uint64{}
	byID := map[string]dependency.Node{}
	nextTrack := statusTrack + 1
	for _, node := range nodes {
		group := lanes[node.Clock]
		slot := -1
		for i, track := range group {
			if track.end <= uint64(node.Start) {
				slot = i
				break
			}
		}
		if slot < 0 {
			slot = len(group)
			group = append(group, lane{uuid: nextTrack})
			nextTrack++
			name := fmt.Sprintf("Dependency nodes (%s): %s lane %d", report.State, node.Clock, slot)
			if err := pw.WriteTrackDescriptor(group[slot].uuid, name, 0); err != nil {
				return err
			}
		}
		track := group[slot].uuid
		group[slot].end = uint64(node.End)
		lanes[node.Clock] = group
		tracks[node.ID], byID[node.ID] = track, node
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("dependency_node", node.ID), perfetto.NewStringAnnotation("raw_record", node.Record), perfetto.NewStringAnnotation("clock_identity", node.Clock), perfetto.NewStringAnnotation("dependency_state", report.State)}
		if node.Start == node.End {
			events = append(events, event{ts: uint64(node.Start) - origin, track: track, kind: 1, name: node.ID, category: "dependency_node", annotations: args})
		} else {
			events = append(events, event{ts: uint64(node.Start) - origin, track: track, kind: 2, name: node.ID, category: "dependency_node", annotations: args}, event{ts: uint64(node.End) - origin, track: track, kind: 0})
		}
	}
	for i, edge := range report.Graph.Edges {
		id := dependencyTrackBase + uint64(i) + 1
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("edge_kind", edge.Kind), perfetto.NewStringAnnotation("ordering_evidence", edge.Record), perfetto.NewStringAnnotation("from_node", edge.From), perfetto.NewStringAnnotation("to_node", edge.To), perfetto.NewStringAnnotation("dependency_state", report.State)}
		events = append(events,
			event{ts: uint64(byID[edge.From].End) - origin, track: tracks[edge.From], kind: 1, name: "Explicit " + edge.Kind + " source", category: "dependency_edge", annotations: args, flow: []uint64{id}},
			event{ts: uint64(byID[edge.To].Start) - origin, track: tracks[edge.To], kind: 1, name: "Explicit " + edge.Kind + " target", category: "dependency_edge", annotations: args, terminate: []uint64{id}},
		)
	}
	sort.SliceStable(events, func(i, j int) bool {
		if events[i].ts != events[j].ts {
			return events[i].ts < events[j].ts
		}
		return events[i].kind < events[j].kind
	})
	for _, event := range events {
		switch event.kind {
		case 0:
			err = pw.WriteSliceEnd(event.track, event.ts)
		case 1:
			err = pw.WriteFlowInstant(event.track, event.ts, event.name, event.category, event.annotations, event.flow, event.terminate)
		case 2:
			err = pw.WriteSliceBeginWithAnnotations(event.track, event.ts, event.name, event.category, event.annotations, nil)
		}
		if err != nil {
			return err
		}
	}
	return nil
}
