package cudaregress_test

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func Example() {
	ctx := context.Background()

	// 1. Ingest an admitted incident from cudadiag / tracecontract.
	incident := cudaregress.Incident{
		ID:       "inc-gemm-slowdown-101",
		Admitted: true,
		Workload: cudaworkload.WorkloadIdentity{
			WorkloadName: "llm_prefill_gemm",
			RunID:        "run-42",
		},
		Diagnosis: cudadiag.DiagnosisResult{
			Admitted:  true,
			Mechanism: cudadiag.MechanismDeviceBottleneck,
		},
		PrimaryFinding: tracecontract.Finding{
			ID:    "finding-dev-bottleneck",
			Class: tracecontract.FindingObserved,
			Admission: tracecontract.Admission{
				Admitted: true,
				Status:   "admitted",
			},
		},
		ObservedEnvironment: cudaregress.Environment{
			DeviceIndex:       0,
			Model:             "NVIDIA GB10",
			ComputeCapability: "sm_121a",
			Driver:            "550.54.14",
			CUDA:              "12.8",
		},
		CreatedAt: time.Now(),
	}

	// 2. Generate a bounded, replayable ExperimentPlan.
	plan, planRec, err := cudaregress.NewPlanFromIncident(incident, cudaregress.PlanOptions{
		Oracle:     cudaregress.CorrectnessOracle{Method: "numerical-tolerance"},
		WarmupRuns: 1,
		Rounds:     3,
		ExecutionScope: cudaregress.TimingScope{
			Channel: cudatrial.TimingSummedKernel,
		},
		StoppingRules: cudaregress.StoppingRules{
			RelativeThreshold: 0.10, // 10% regression threshold
			MinSamples:        3,
		},
	})
	if err != nil {
		fmt.Printf("plan generation failed: %v\n", err)
		return
	}

	fmt.Printf("Plan created: %s (closure valid: %t)\n", plan.IncidentID, planRec.ArtifactClosureValid)

	// 3. Replay with runner executing baseline and candidate arms.
	runner := cudaregress.ReplayFunc(func(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
		dur := 100 * time.Microsecond
		if armName == "candidate" {
			dur = 125 * time.Microsecond // 25% regression
		}
		r := cudatrial.Receipt{
			Name:         armName,
			Round:        round,
			Complete:     true,
			Digest:       "digest-ok",
			SummedKernel: dur,
			DeviceNs:     dur.Nanoseconds(),
			TimingKind:   cudatrial.TimingSummedKernel,
		}
		_ = r.RecordTiming(cudatrial.TimingSummedKernel, dur)
		return r, []float64{1.0, 2.0}, nil
	})

	execRec, compRec, err := cudaregress.Replay(ctx, plan, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: incident.ObservedEnvironment,
		ReferenceValues:    []float64{1, 2},
	})
	if err != nil {
		fmt.Printf("replay failed: %v\n", err)
		return
	}

	fmt.Printf("Rounds completed: %d\n", execRec.RoundsCompleted)
	fmt.Printf("Verdict: %s (regression detected: %t)\n", compRec.Verdict, compRec.RegressionDetected)

	// Output:
	// Plan created: inc-gemm-slowdown-101 (closure valid: true)
	// Rounds completed: 3
	// Verdict: fail (regression detected: true)
}

func ExampleNewPlanFromIncident() {
	incident := cudaregress.Incident{
		ID:       "inc-norm-01",
		Admitted: true,
		Workload: cudaworkload.WorkloadIdentity{WorkloadName: "layernorm"},
		ObservedEnvironment: cudaregress.Environment{
			Model:             "NVIDIA H100",
			ComputeCapability: "sm_90",
		},
	}

	plan, receipt, err := cudaregress.NewPlanFromIncident(incident, cudaregress.PlanOptions{
		WarmupRuns: 2,
		Rounds:     4,
	})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	fmt.Printf("Plan ID: %s, Rounds: %d, Verified: %d\n",
		plan.IncidentID, plan.Rounds, receipt.ArtifactsVerified)

	// Output:
	// Plan ID: inc-norm-01, Rounds: 4, Verified: 0
}
