package traceanalysis

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestRequestLifecycle(t *testing.T) {
	var records timeline.Records
	add := func(id, role, event, boundary string, when int64, token, executions *int) {
		phase := "decode"
		if event == "request_end" {
			phase = "cleanup"
		}
		name, err := marker.Encode(marker.Metadata{Version: 2, RequestID: id, Role: role, Phase: phase, Event: event, Boundary: boundary, Token: token, ExecutionUnit: "logical_forward", Executions: executions})
		if err != nil {
			t.Fatal(err)
		}
		records.Markers = append(records.Markers, timeline.Marker{Journal: "p", Process: 1, Thread: 1, Name: name, Time: when, Flags: timeline.MarkerInstantaneous})
	}
	zero, one, two, three := 0, 1, 2, 3
	for i, id := range []string{"a", "b"} {
		offset := int64(i * 100)
		add(id, "measured", "request_start", "submission", offset, &zero, &zero)
		add(id, "measured", "delivered", "consumer_delivery", offset+10, &zero, &one)
		add(id, "measured", "delivered", "consumer_delivery", offset+20, &one, &three)
		add(id, "measured", "request_end", "completion", offset+30, nil, &three)
	}
	s, err := New(records, Options{CollectorComplete: true})
	if err != nil {
		t.Fatal(err)
	}
	r := s.Summary()
	if len(r.Requests) != 2 || len(r.Overview.Tokens) != 4 {
		t.Fatalf("requests=%+v tokens=%+v", r.Requests, r.Overview.Tokens)
	}
	for _, q := range r.Requests {
		if len(q.Unavailable) != 0 || len(q.Spacing) != 1 || q.Spacing[0] != 10 || len(q.ExecutionDeltas) != 1 || q.ExecutionDeltas[0] != two || len(q.Diagnostics) != 1 {
			t.Fatalf("request=%+v", q)
		}
	}
}

func TestRequestMissingAndDecreasingCounters(t *testing.T) {
	zero, one, two := 0, 1, 2
	es := []Endpoint{
		{Source: Source{Journal: "p"}, Process: 1, Time: 0, Metadata: marker.Metadata{Version: 2, RequestID: "r", Role: "control", Event: "request_start", Boundary: "submission", ExecutionUnit: "logical_forward", Executions: &two}},
		{Source: Source{Journal: "p"}, Process: 1, Time: 1, Metadata: marker.Metadata{Version: 2, RequestID: "r", Role: "control", Event: "delivered", Boundary: "completion", Token: &zero, ExecutionUnit: "logical_forward", Executions: &one}},
		{Source: Source{Journal: "p"}, Process: 1, Time: 2, Metadata: marker.Metadata{Version: 2, RequestID: "r", Role: "control", Event: "request_end"}},
	}
	r := requests(es)
	if len(r) != 1 || r[0].Spacing != nil {
		t.Fatal(r)
	}
	text := strings.Join(r[0].Unavailable, ";")
	for _, want := range []string{"counter decreased", "counter missing", "boundary missing", "not consumer_delivery"} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q: %s", want, text)
		}
	}
}

func ExampleRequest() {
	name, _ := marker.Encode(marker.Metadata{Version: 2, RequestID: "r1", Role: "measured", Phase: "decode", Event: "request_start", Boundary: "submission"})
	s, _ := New(timeline.Records{Markers: []timeline.Marker{{Journal: "p", Process: 1, Thread: 1, Time: 1, Flags: timeline.MarkerInstantaneous, Name: name}}}, Options{CollectorComplete: true})
	r := s.Summary().Requests[0]
	fmt.Println(r.ID, r.Role)
	fmt.Println(len(r.Unavailable) > 0)
	// Output:
	// r1 measured
	// true
}

func TestRequestOrdinalGates(t *testing.T) {
	for _, tt := range []struct {
		name    string
		ordinal int
		when    int64
	}{
		{"duplicate", 0, 2}, {"backward", -1, 2}, {"skipped", 2, 2}, {"same time", 1, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			zero, one, two := 0, 1, 2
			base := marker.Metadata{Version: 2, RequestID: "r", Role: "measured", Boundary: "consumer_delivery", ExecutionUnit: "logical_forward"}
			endpoint := func(time int64, event string, token, count *int, boundary string) Endpoint {
				m := base
				m.Event = event
				m.Token = token
				m.Executions = count
				m.Boundary = boundary
				return Endpoint{Source: Source{Journal: "p"}, Process: 1, Time: time, Metadata: m}
			}
			r := requests([]Endpoint{endpoint(0, "request_start", nil, &zero, "submission"), endpoint(1, "delivered", &zero, &one, "consumer_delivery"), endpoint(tt.when, "delivered", &tt.ordinal, &two, "consumer_delivery"), endpoint(3, "request_end", nil, &two, "completion")})
			if len(r) != 1 || len(r[0].Unavailable) == 0 || r[0].Spacing != nil || r[0].ExecutionDeltas != nil {
				t.Fatalf("unqualified derived metrics: %+v", r)
			}
		})
	}
}
