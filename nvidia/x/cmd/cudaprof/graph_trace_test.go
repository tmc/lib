package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseCaptureArgsGraphTrace(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    graphTraceMode
		wantErr string
	}{
		{name: "unset", args: []string{"--", "true"}, want: ""},
		{name: "node joined", args: []string{"--graph-trace=node", "--", "true"}, want: graphTraceNode},
		{name: "graph separated", args: []string{"--graph-trace", "graph", "--", "true"}, want: graphTraceGraph},
		{name: "single dash", args: []string{"-graph-trace=graph", "--", "true"}, want: graphTraceGraph},
		// An outcome the agent reports is not a mode a caller can request.
		{name: "outcome refused", args: []string{"--graph-trace=requested_unavailable", "--", "true"}, wantErr: "invalid graph trace mode"},
		{name: "unknown refused", args: []string{"--graph-trace=envelope", "--", "true"}, wantErr: "invalid graph trace mode"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			o, err := parseCaptureArgs(tc.args)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("parseCaptureArgs(%v) error = %v, want one naming %q", tc.args, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseCaptureArgs(%v): %v", tc.args, err)
			}
			if o.GraphTrace != tc.want {
				t.Errorf("parseCaptureArgs(%v).GraphTrace = %q, want %q", tc.args, o.GraphTrace, tc.want)
			}
		})
	}
}

// TestCaptureEnvironmentGraphTrace: the mode reaches the agent only through
// CUDAPROF_GRAPH_TRACE, and an unset flag leaves the caller's environment
// alone, as the other collection-policy flags do.
func TestCaptureEnvironmentGraphTrace(t *testing.T) {
	tests := []struct {
		name    string
		mode    graphTraceMode
		callEnv []string
		want    string
	}{
		{name: "unset", want: ""},
		{name: "unset keeps caller", callEnv: []string{"CUDAPROF_GRAPH_TRACE=graph"}, want: "graph"},
		{name: "node", mode: graphTraceNode, want: "node"},
		{name: "node overrides caller", mode: graphTraceNode, callEnv: []string{"CUDAPROF_GRAPH_TRACE=graph"}, want: "node"},
		{name: "graph", mode: graphTraceGraph, want: "graph"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			env := tc.callEnv
			if env == nil {
				env = []string{}
			}
			got := lookupEnv(captureEnvironment(captureOptions{Env: env, GraphTrace: tc.mode}, "session", "dir"), "CUDAPROF_GRAPH_TRACE")
			if got != tc.want {
				t.Errorf("CUDAPROF_GRAPH_TRACE = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestCaptureGraphTraceMode(t *testing.T) {
	tests := []struct {
		name       string
		capability string
		want       graphTraceMode
		wantOK     bool
	}{
		{name: "node", capability: `{"graph_traces":false,"graph_trace_mode":"node"}`, want: graphTraceNode, wantOK: true},
		{name: "graph", capability: `{"graph_traces":true,"graph_trace_mode":"graph"}`, want: graphTraceGraph, wantOK: true},
		{name: "unavailable", capability: `{"graph_traces":false,"graph_trace_mode":"requested_unavailable"}`, want: graphTraceUnavailable, wantOK: true},
		// An agent that predates the key: graph_traces true says envelopes
		// were enabled, but the mode was never stated and is not inferred.
		{name: "older agent", capability: `{"graph_traces":true}`, wantOK: false},
		{name: "no capability", capability: ``, wantOK: false},
		{name: "unknown value", capability: `{"graph_trace_mode":"envelope"}`, wantOK: false},
		{name: "not json", capability: `{`, wantOK: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := captureGraphTraceMode(json.RawMessage(tc.capability))
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("captureGraphTraceMode(%s) = %q, %v, want %q, %v", tc.capability, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

// TestGapReportGraphTraceMode: the gaps report states the mode the agent ran
// in, and says UNREAD for a capture whose agent did not state one.
func TestGapReportGraphTraceMode(t *testing.T) {
	tests := []struct {
		name       string
		capability map[string]any
		want       graphTraceMode
		wantText   string
	}{
		{name: "node", capability: map[string]any{"graph_trace_mode": "node"}, want: graphTraceNode, wantText: "graph trace mode node"},
		{name: "graph", capability: map[string]any{"graph_trace_mode": "graph", "graph_traces": true}, want: graphTraceGraph, wantText: "graph trace mode graph"},
		{name: "requested unavailable", capability: map[string]any{"graph_trace_mode": "requested_unavailable"}, want: graphTraceUnavailable, wantText: "graph trace mode requested_unavailable"},
		{name: "older agent", capability: map[string]any{"graph_traces": true}, want: "", wantText: "graph trace mode UNREAD"},
		{name: "no capability", want: "", wantText: "graph trace mode UNREAD"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			status := completeStatus()
			if tc.capability != nil {
				status["capability"] = tc.capability
			}
			cap := testCapture(map[string][]byte{"capture.journal": pacedJournal(8_000, 1_000)}, status)
			r, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords})
			if err != nil {
				t.Fatal(err)
			}
			if r.GraphTraceMode != tc.want {
				t.Errorf("GraphTraceMode = %q, want %q", r.GraphTraceMode, tc.want)
			}
			if text := r.text(); !strings.Contains(text, tc.wantText) {
				t.Errorf("report does not say %q:\n%s", tc.wantText, text)
			}
			if r.OpaqueGraphReplays {
				t.Errorf("OpaqueGraphReplays = true for a journal with no graph envelopes")
			}
		})
	}
}
