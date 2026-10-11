package cudadiag

import (
	"math"
	"testing"
)

func TestDiagnoseRejectsInvalidScheduler(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*DiagnosticInput)
	}{
		{"unknown loss", func(in *DiagnosticInput) { in.Sched.UnknownLoss = true }},
		{"dropped records", func(in *DiagnosticInput) { in.Sched.Loss = 1 }},
		{"reversed interval", func(in *DiagnosticInput) { in.Sched.Intervals[0].Span.End = in.Sched.Intervals[0].Span.Start - 1 }},
		{"invalid state", func(in *DiagnosticInput) { in.Sched.Intervals[0].State = "invented" }},
		{"host clock mismatch", func(in *DiagnosticInput) { in.HostCalls[0].Span.Clock = "other" }},
		{"kernel clock mismatch", func(in *DiagnosticInput) { in.Kernels[0].Span.Clock = "other" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f := BuildSyntheticFixture(InjectionHostDelay)
			tt.mutate(&f.Input)
			if r := Diagnose(f.Workload, f.OpID, f.Input); r.Admitted {
				t.Fatal("contradiction admitted")
			}
		})
	}
}

func TestOracleRejectsInvalidTolerance(t *testing.T) {
	for _, v := range []float64{-1, math.NaN(), math.Inf(1)} {
		if err := CheckNumericalCorrectness([]float64{1}, []float64{2}, v); err == nil {
			t.Fatalf("invalid tolerance %v accepted", v)
		}
	}
}
