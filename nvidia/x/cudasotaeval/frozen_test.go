package cudasotaeval

import (
	"context"
	"math"
	"testing"
)

func TestFrozenProtocolRejectsMutation(t *testing.T) {
	p := DefaultProtocol(PhaseQualification)
	if err := p.Freeze(); err != nil {
		t.Fatal(err)
	}
	p.Workloads[0].Concurrency++
	if err := p.Validate(); err == nil {
		t.Fatal("accepted mutated frozen workload")
	}
}

func TestProtocolRejectsNonFiniteBudget(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), -1} {
		p := DefaultProtocol(PhaseQualification)
		p.Budgets.MaxRSSGrowthMB = value
		if err := p.Validate(); err == nil {
			t.Fatalf("accepted budget %v", value)
		}
	}
}

func TestPortableFixtureReceiptJSON(t *testing.T) {
	result, err := NewGroundTruthFixture().Execute(context.Background(), []GroundTruthRequest{{ID: "fixture-receipt", WorkloadName: "fixture", Data: []float64{1}, Fault: FaultNone, Resilience: ResilienceNormal}})
	if err != nil {
		t.Fatal(err)
	}
	receipt := result.DiagnosticInputs["fixture-receipt"].Receipt
	if _, err := receipt.JSON(); err != nil {
		t.Fatal(err)
	}
	if receipt.Kind != "capture" || receipt.Backends[0].Name != "portable_synthetic_fixture" {
		t.Fatalf("misclassified fixture %+v", receipt)
	}
}
