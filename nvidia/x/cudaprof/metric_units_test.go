package cudaprof_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestGuidanceRequiresProducerUnits(t *testing.T) {
	for _, kind := range []string{"aggregate", "pc"} {
		t.Run(kind, func(t *testing.T) {
			s := validProfileSnapshot()
			if kind == "aggregate" {
				s.MetricUnits = nil
			} else {
				s.PCStalls[16]["unknown"] = 100
			}
			stored, err := cudaprof.NewProfileSnapshot(s)
			if err != nil {
				t.Fatal(err)
			}
			g, err := stored.Guidance(profileTarget(stored, "work", "", "device"))
			if !errors.Is(err, cudaprof.ErrGuidanceUnavailable) || g == nil || g.Available || g.UnavailableReason == "" {
				t.Fatalf("unitless evidence admitted: %+v %v", g, err)
			}
		})
	}
}

func TestMetricUnitsRoundTripAndIsolation(t *testing.T) {
	s := validProfileSnapshot()
	stored, err := cudaprof.NewProfileSnapshot(s)
	if err != nil {
		t.Fatal(err)
	}
	s.MetricUnits["stall_mem"] = "changed"
	data, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	var decoded cudaprof.ProfileSnapshot
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	g, err := decoded.Guidance(profileTarget(&decoded, "work", "", "device"))
	if err != nil {
		t.Fatal(err)
	}
	if g.MetricUnits["stall_mem"] != "samples" {
		t.Fatalf("unit=%q", g.MetricUnits["stall_mem"])
	}
	g.MetricUnits["stall_mem"] = "mutated"
	if decoded.MetricUnits["stall_mem"] != "samples" {
		t.Fatal("guidance units alias input")
	}
	for _, units := range []map[string]string{{"": "samples"}, {"stall_mem": ""}} {
		s.MetricUnits = units
		if _, err := cudaprof.NewProfileSnapshot(s); err == nil {
			t.Fatal("accepted malformed unit declaration")
		}
	}
}
