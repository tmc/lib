package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/dependency"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestDependencyEndpoints(t *testing.T) {
	message := `cudaprof:v1 {"phase":"decode","token":0,"event":"delivered"}`
	for _, tt := range []struct {
		name       string
		markers    []timeline.Marker
		wantStatus string
		missing    bool
	}{
		{"absent", nil, "", true},
		{"delivered", []timeline.Marker{{Journal: "j", Process: 1, Thread: 2, Time: 10, Name: message, Flags: timeline.MarkerInstantaneous}}, "observed", false},
		{"unknown thread", []timeline.Marker{{Journal: "j", Process: 1, Time: 10, Name: message, Flags: timeline.MarkerInstantaneous}}, "unsupported", true},
		{"range not event", []timeline.Marker{{Journal: "j", Process: 1, Thread: 2, Time: 10, Name: message, Flags: timeline.MarkerStart}}, "unsupported", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			graph, endpoints := dependencyInventory(timeline.Records{Markers: tt.markers})
			if tt.wantStatus != "" && (len(endpoints) != 1 || endpoints[0].Status != tt.wantStatus) {
				t.Fatalf("endpoints=%+v", endpoints)
			}
			missing := strings.Contains(strings.Join(graph.Missing, " "), "no scoped instantaneous")
			if missing != tt.missing {
				t.Fatalf("missing=%v", graph.Missing)
			}
			r, err := dependency.Analyze(graph)
			if err != nil || r.State != "partial" || len(r.CriticalPath) != 0 {
				t.Fatalf("report=%+v error=%v", r, err)
			}
		})
	}
}

func TestDependencyHTMLEscapesPartial(t *testing.T) {
	result := dependencyResult{Capture: `<script>alert(1)</script>`, Dependency: dependency.Report{State: "partial", Reasons: []string{"<missing>"}, Graph: dependency.Graph{Nodes: []dependency.Node{{ID: "activity:0", Record: "<raw>", Clock: "j", Start: 1, End: 2}}}}}
	var output bytes.Buffer
	if err := writeDependencyHTML(&output, result); err != nil {
		t.Fatal(err)
	}
	html := output.String()
	for _, want := range []string{"&lt;script&gt;alert(1)&lt;/script&gt;", "&lt;missing&gt;", "&lt;raw&gt;", "activity:0", "No complete critical path is established", "No explicit dependency edges are retained"} {
		if !strings.Contains(html, want) {
			t.Errorf("missing %q", want)
		}
	}
	if strings.Contains(html, "<script>") {
		t.Fatal("unescaped script")
	}
}
