package cudadiag

import (
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

func TestIntervention_VerificationAndOracle(t *testing.T) {
	// 1. Control run with host submission delay
	ctrlFix := BuildSyntheticFixture(InjectionHostDelay)
	ctrlRes := Diagnose(ctrlFix.Workload, ctrlFix.OpID, ctrlFix.Input)

	// 2. Intervention run: CPU affinity pinned, reducing launch delay from 70us to 15us
	invFix := BuildSyntheticFixture(InjectionHostDelay)
	// Mutate intervention input to simulate resolved delay
	invFix.Input.HostCalls[0].Span = tracecontract.Interval{Start: 2000, End: 17000, Clock: "monotonic"}
	invFix.Input.Sched = BuildSyntheticFixture(InjectionNone).Input.Sched
	invRes := Diagnose(invFix.Workload, invFix.OpID, invFix.Input)

	plan := InterventionPlan{
		HypothesisID:      string(MechanismHostSubmissionDelay),
		TargetWorkload:    ctrlFix.Workload,
		CorrectnessOracle: "gemm_oracle_ref",
		InterventionType:  "cpu_affinity_pinning",
		Ordering:          "A/B paired randomized block",
		StoppingRules: StoppingRules{
			MaxTrials:       10,
			MinEffectSize:   0.25, // 25% reduction required
			OracleTolerance: 1e-5,
		},
	}

	ctrlOut := []float64{1.000000, 2.000000, 3.000000}
	invOut := []float64{1.000001, 2.000000, 3.000002} // within tolerance 1e-5

	plan.ReferenceOutput = []float64{1, 2, 3}
	receipt, err := EvaluateIntervention(plan, ctrlRes, ctrlOut, invRes, invOut)
	if err != nil {
		t.Fatalf("expected successful intervention evaluation, got err: %v; control=%+v intervention=%+v", err, ctrlRes.PrimaryFinding.Admission, invRes.PrimaryFinding.Admission)
	}

	if !receipt.Verified {
		t.Fatalf("expected receipt.Verified to be true, got false")
	}
	if receipt.ControlSHA256 == "" || receipt.InterventionSHA256 == "" {
		t.Fatalf("missing SHA256 hashes in receipt")
	}

	// Now diagnose the bottleneck run with verified intervention attached -> finding promoted to FindingExperimentallyConfirmed
	ctrlFix.Input.Intervention = &receipt
	confirmedRes := Diagnose(ctrlFix.Workload, ctrlFix.OpID, ctrlFix.Input)

	if confirmedRes.PrimaryFinding.Class != tracecontract.FindingExperimentallyConfirmed {
		t.Fatalf("expected class FindingExperimentallyConfirmed, got %s", confirmedRes.PrimaryFinding.Class)
	}
	if err := confirmedRes.PrimaryFinding.CheckClaim(tracecontract.FindingExperimentallyConfirmed); err != nil {
		t.Fatalf("check claim for confirmed finding failed: %v", err)
	}
}

func TestIntervention_NumericalCorrectness(t *testing.T) {
	plan := InterventionPlan{
		HypothesisID:      string(MechanismHostSubmissionDelay),
		CorrectnessOracle: "gemm_oracle_ref",
		InterventionType:  "cpu_affinity_pinning",
		StoppingRules: StoppingRules{
			OracleTolerance: 1e-4,
			MinEffectSize:   0.20,
		},
	}

	ctrlFix := BuildSyntheticFixture(InjectionHostDelay)
	ctrlRes := Diagnose(ctrlFix.Workload, ctrlFix.OpID, ctrlFix.Input)

	invFix := BuildSyntheticFixture(InjectionNone)
	invRes := Diagnose(invFix.Workload, invFix.OpID, invFix.Input)

	plan.ReferenceOutput = []float64{1, 2, 3}
	// Case 1: Exceeding tolerance
	badOut := []float64{1.0, 2.5, 3.0} // diff of 0.5 exceeds 1e-4
	ctrlOut := []float64{1.0, 2.0, 3.0}

	_, err := EvaluateIntervention(plan, ctrlRes, ctrlOut, invRes, badOut)
	if err == nil {
		t.Fatalf("expected error on oracle tolerance violation, got nil")
	}

	// Case 2: NaN in output
	nanOut := []float64{1.0, math.NaN(), 3.0}
	_, err = EvaluateIntervention(plan, ctrlRes, ctrlOut, invRes, nanOut)
	if err == nil {
		t.Fatalf("expected error on NaN in output, got nil")
	}

	// Case 3: Inf in output
	infOut := []float64{1.0, math.Inf(1), 3.0}
	_, err = EvaluateIntervention(plan, ctrlRes, ctrlOut, invRes, infOut)
	if err == nil {
		t.Fatalf("expected error on Inf in output, got nil")
	}
}

func TestIntervention_InsufficientEffect(t *testing.T) {
	plan := InterventionPlan{
		HypothesisID:      string(MechanismHostSubmissionDelay),
		CorrectnessOracle: "gemm_oracle_ref",
		InterventionType:  "cpu_affinity_pinning",
		StoppingRules: StoppingRules{
			OracleTolerance: 1e-4,
			MinEffectSize:   0.50, // requires 50% improvement
		},
	}

	ctrlFix := BuildSyntheticFixture(InjectionHostDelay)
	ctrlRes := Diagnose(ctrlFix.Workload, ctrlFix.OpID, ctrlFix.Input)

	// Intervention with only minor improvement (e.g. 70us -> 65us = ~7% improvement)
	invFix := BuildSyntheticFixture(InjectionHostDelay)
	invFix.Input.HostCalls[0].Span.End = invFix.Input.HostCalls[0].Span.Start + (65 * time.Microsecond).Nanoseconds()
	invRes := Diagnose(invFix.Workload, invFix.OpID, invFix.Input)

	ctrlOut := []float64{1.0, 2.0}
	plan.ReferenceOutput = []float64{1, 2}
	invOut := []float64{1.0, 2.0}

	_, err := EvaluateIntervention(plan, ctrlRes, ctrlOut, invRes, invOut)
	if err == nil {
		t.Fatalf("expected error on insufficient effect size, got nil")
	}
}
