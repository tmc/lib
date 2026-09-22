package main

import (
	"encoding/json"
	"fmt"
)

// graphTraceMode is how the injected agent traced CUDA graph replays, as its
// capability record states it under the key graph_trace_mode.
//
// The two modes are exclusive. With CUPTI_ACTIVITY_KIND_GRAPH_TRACE enabled,
// CUPTI reports one envelope per replay "without giving visibility about the
// execution of its nodes" (cupti_activity.h, CUpti_ActivityGraphTrace2), so a
// capture holds either the kernels inside its replays or the replays as
// wholes, never both.
type graphTraceMode string

const (
	// graphTraceNode leaves the envelope kind off: every kernel node of a
	// replay is a kernel record carrying its graph id and graph node id.
	graphTraceNode graphTraceMode = "node"
	// graphTraceGraph enables the envelope kind: one graph_trace record per
	// replay, and no kernel records for the nodes inside it.
	graphTraceGraph graphTraceMode = "graph"
	// graphTraceUnavailable is a graph request the agent could not meet. The
	// agent collected in node mode instead.
	graphTraceUnavailable graphTraceMode = "requested_unavailable"
)

// parseGraphTraceMode parses a --graph-trace value. Only the two modes can be
// requested; requested_unavailable is an outcome, not a request.
func parseGraphTraceMode(s string) (graphTraceMode, error) {
	switch m := graphTraceMode(s); m {
	case graphTraceNode, graphTraceGraph:
		return m, nil
	}
	return "", fmt.Errorf("invalid graph trace mode %q (%s or %s)", s, graphTraceNode, graphTraceGraph)
}

// captureGraphTraceMode reads the graph trace mode from an agent capability
// record. It reports false when the record does not state one: an agent that
// predates the key enabled graph tracing whenever CUPTI allowed it, and which
// mode a given run of it ended up in is not recoverable from the record, so
// the mode is UNREAD rather than assumed.
func captureGraphTraceMode(capability json.RawMessage) (graphTraceMode, bool) {
	var c struct {
		GraphTraceMode *string `json:"graph_trace_mode"`
	}
	if len(capability) == 0 || json.Unmarshal(capability, &c) != nil || c.GraphTraceMode == nil {
		return "", false
	}
	switch m := graphTraceMode(*c.GraphTraceMode); m {
	case graphTraceNode, graphTraceGraph, graphTraceUnavailable:
		return m, true
	}
	return "", false
}
