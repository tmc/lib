package tracecontract

import (
	"math"
	"testing"
)

func TestAdmissionRejectsContradictions(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*EvidenceContext)
		req    Requirement
	}{
		{"negative uncertainty", func(e *EvidenceContext) { v := int64(-1); e.Receipt.ClockUncertaintyNS = &v }, Requirement{Name: "clock", MaxUncertaintyNS: 10}},
		{"nan coverage", func(e *EvidenceContext) {
			e.Receipt.ProbeCoverage["driver_api"] = ProbeCoverage{Present: true, Ratio: math.NaN()}
		}, Requirement{Name: "probe", Probe: "driver_api", MinProbeCoverage: 1}},
		{"unknown backend version", func(e *EvidenceContext) { e.Receipt.Backends[0].Version = "" }, Requirement{Name: "backend", Backend: "native", MinVersion: "12.0"}},
		{"numeric backend version", func(e *EvidenceContext) { e.Receipt.Backends[0].Version = "9.10" }, Requirement{Name: "backend", Backend: "native", MinVersion: "12.0"}},
		{"empty correlation", func(e *EvidenceContext) { e.Receipt.Correlation = &CorrelationOutcome{} }, Requirement{Name: "correlation", RequireCorrelation: true}},
		{"inverted interval", func(e *EvidenceContext) { e.Intervals = []Interval{{Start: 3000, End: 2000}} }, Requirement{Name: "duration", RequireReturn: true}},
		{"interval clock mismatch", func(e *EvidenceContext) { e.Intervals = []Interval{{Start: 2000, End: 3000, Clock: "wall"}} }, Requirement{Name: "clock", ClockDomain: "monotonic"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e := EvidenceContext{Receipt: validReceiptFixture(), HasReturn: true}
			tt.mutate(&e)
			if a := EvaluateRequirements([]Requirement{tt.req}, e); a.Admitted {
				t.Fatal("contradiction admitted")
			}
		})
	}
}

func TestConfirmedFindingRequiresControl(t *testing.T) {
	f := Admit("id", "rule", FindingExperimentallyConfirmed, "", nil, "", nil, EvidenceContext{})
	if f.Admission.Admitted || f.Class != FindingUnavailable {
		t.Fatal("confirmation admitted without control")
	}
}
