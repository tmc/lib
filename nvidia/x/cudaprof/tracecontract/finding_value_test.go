package tracecontract

import (
	"math"
	"testing"
)

func TestFindingRejectsInvalidMeasurementAndClass(t *testing.T) {
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		f := Admit("id", "rule", FindingObserved, "measurement", &v, "ns", nil, EvidenceContext{})
		if _, err := f.JSON(); err != nil {
			t.Fatalf("withheld finding not serializable: %v", err)
		}
		if f.Admission.Admitted || f.Class != FindingUnavailable || f.CheckClaim(FindingObserved) == nil {
			t.Fatal("nonfinite finding admitted")
		}
	}
	f := Admit("id", "rule", FindingClass("invented"), "invalid", nil, "", nil, EvidenceContext{})
	if f.Admission.Admitted || f.CheckClaim(FindingClass("invented")) == nil {
		t.Fatal("invalid evidence class admitted")
	}
}

func TestRuleSpecificReceiptGate(t *testing.T) {
	ev := EvidenceContext{Receipt: Receipt{Gates: map[string]Gate{"shape": {Disposition: "unchecked"}, "import": {Disposition: "passed"}}}}
	if EvaluateRequirements([]Requirement{{Name: "shape", ReceiptGate: "shape"}}, ev).Admitted {
		t.Fatal("unchecked shape passed")
	}
	if !EvaluateRequirements([]Requirement{{Name: "import", ReceiptGate: "import"}}, ev).Admitted {
		t.Fatal("unrelated shape blocked import")
	}
}
