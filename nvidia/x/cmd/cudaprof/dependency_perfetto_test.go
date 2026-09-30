package main

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/dependency"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func dependencyTraceFixture() []traceRecord {
	return []traceRecord{{journal: "j", sequence: 3, kind: trackDevice, rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "raw_kernel", Start: 100, End: 200, Complete: true}}}
}

func TestDependencyPerfettoExplicitFlows(t *testing.T) {
	graph := dependency.Graph{Nodes: []dependency.Node{{ID: "producer", Record: "j:record:3", Clock: "j", Start: 100, End: 150}, {ID: "consumer", Record: "j:record:4", Clock: "j", Start: 160, End: 200}}, Edges: []dependency.Edge{{From: "producer", To: "consumer", Kind: "event_wait", Record: "j:record:5"}}}
	report, err := dependency.Analyze(graph)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := writeDependencyTrace(&output, dependencyTraceFixture(), dependencyResult{Dependency: report}); err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"raw_kernel", "Dependency qualification: partial", "producer", "consumer", "event_wait", "j:record:5", "ordering_evidence"} {
		if !bytes.Contains(output.Bytes(), []byte(text)) {
			t.Errorf("missing %q", text)
		}
	}
	for _, field := range []uint64{47, 48} {
		want := binary.AppendUvarint(nil, field<<3|1)
		want = binary.LittleEndian.AppendUint64(want, dependencyTrackBase+1)
		if !bytes.Contains(output.Bytes(), want) {
			t.Fatalf("missing explicit flow field %d", field)
		}
	}
}

func TestDependencyPerfettoDoesNotInventEdges(t *testing.T) {
	graph := dependency.Graph{Nodes: []dependency.Node{{ID: "a", Record: "j:record:3", Clock: "j", Start: 100, End: 150}, {ID: "b", Record: "j:record:4", Clock: "j", Start: 150, End: 200}}}
	report, err := dependency.Analyze(graph)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := writeDependencyTrace(&output, dependencyTraceFixture(), dependencyResult{Dependency: report}); err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(output.Bytes(), []byte("dependency_edge")) {
		t.Fatal("adjacency became a dependency")
	}
	if !bytes.Contains(output.Bytes(), []byte("raw_record")) || !bytes.Contains(output.Bytes(), []byte("partial")) {
		t.Fatal("source or qualification omitted")
	}
}

func TestDependencyPerfettoRejectsUnsupportedEdge(t *testing.T) {
	graph := dependency.Graph{Nodes: []dependency.Node{{ID: "a", Record: "j:3", Clock: "j", Start: 100, End: 150}, {ID: "b", Record: "j:4", Clock: "j", Start: 160, End: 200}}, Edges: []dependency.Edge{{From: "a", To: "b", Kind: "correlated_submission", Record: "j:5"}}}
	var output bytes.Buffer
	err := writeDependencyTrace(&output, dependencyTraceFixture(), dependencyResult{Dependency: dependency.Report{Graph: graph}})
	if err == nil || !strings.Contains(err.Error(), "unsupported edge") || output.Len() != 0 {
		t.Fatalf("error=%v bytes=%d", err, output.Len())
	}
}
