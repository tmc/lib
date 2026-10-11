package nvtelemetry_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestBenchmarkCorrelationJoulesPerToken(t *testing.T) {
	energy := []nvtelemetry.EnergySummary{
		{
			DeviceUUID:     "GPU-1",
			WindowName:     "measurement",
			StartMonoNanos: 0, EndMonoNanos: 10_000_000_000, TotalDurationNanos: 10_000_000_000, CoveredDurationNanos: 10_000_000_000,
			EnergyConsumedJoules: 100.0,
			CoverageRatio:        1.0,
			Valid:                true,
		},
	}

	requests := []nvtelemetry.BenchmarkRecord{
		{
			RequestID:      "req-1",
			StartTimeNanos: 1_000_000_000,
			EndTimeNanos:   2_000_000_000,
			OutputTokens:   200,
			Success:        true,
		},
		{
			RequestID:      "req-2",
			StartTimeNanos: 2_000_000_000,
			EndTimeNanos:   3_000_000_000,
			OutputTokens:   300,
			Success:        true,
		},
	}

	corr, err := nvtelemetry.CorrelateBenchmark("measurement", 0, 10_000_000_000, energy, requests, nvtelemetry.BenchmarkOptions{ClockProvenance: "local"})
	if err != nil {
		t.Fatalf("CorrelateBenchmark failed: %v", err)
	}

	if !corr.TokensAvailable || !corr.EnergyAvailable {
		t.Fatalf("expected tokens and energy available: %+v", corr)
	}

	if corr.StartMonoNanos != 0 || corr.EndMonoNanos != 10_000_000_000 {
		t.Fatalf("missing joined interval: %+v", corr)
	}
	if corr.TotalOutputTokens != 500 {
		t.Errorf("TotalOutputTokens = %d, want 500", corr.TotalOutputTokens)
	}
	if corr.TotalEnergyJoules != 100.0 {
		t.Errorf("TotalEnergyJoules = %f, want 100.0", corr.TotalEnergyJoules)
	}

	// 100 Joules / 500 tokens = 0.2 Joules/token
	if math.Abs(corr.JoulesPerOutputToken-0.2) > 0.001 {
		t.Errorf("JoulesPerOutputToken = %f, want 0.2", corr.JoulesPerOutputToken)
	}
	// 500 tokens / 100 Joules = 5.0 tokens/Joule
	if math.Abs(corr.OutputTokensPerJoule-5.0) > 0.001 {
		t.Errorf("OutputTokensPerJoule = %f, want 5.0", corr.OutputTokensPerJoule)
	}
}

func TestBenchmarkCorrelationZeroTokens(t *testing.T) {
	energy := []nvtelemetry.EnergySummary{
		{
			DeviceUUID:     "GPU-1",
			WindowName:     "measurement",
			StartMonoNanos: 0, EndMonoNanos: 10_000_000_000, TotalDurationNanos: 10_000_000_000, CoveredDurationNanos: 10_000_000_000,
			EnergyConsumedJoules: 50.0,
			CoverageRatio:        1.0,
			Valid:                true,
		},
	}

	requests := []nvtelemetry.BenchmarkRecord{
		{
			RequestID:      "req-empty",
			StartTimeNanos: 1_000_000_000,
			EndTimeNanos:   2_000_000_000,
			OutputTokens:   0,
			Success:        true,
		},
	}

	corr, err := nvtelemetry.CorrelateBenchmark("measurement", 0, 10_000_000_000, energy, requests, nvtelemetry.BenchmarkOptions{ClockProvenance: "local"})
	if err != nil {
		t.Fatalf("CorrelateBenchmark failed: %v", err)
	}

	if corr.TokensAvailable {
		t.Errorf("TokensAvailable = true, want false for zero tokens")
	}
	if corr.RefusalReason != nvtelemetry.ReasonZeroTokens {
		t.Errorf("RefusalReason = %s, want %s", corr.RefusalReason, nvtelemetry.ReasonZeroTokens)
	}
	if math.IsInf(corr.JoulesPerOutputToken, 0) || math.IsNaN(corr.JoulesPerOutputToken) {
		t.Errorf("JoulesPerOutputToken produced Inf or NaN: %f", corr.JoulesPerOutputToken)
	}
}

func TestBenchmarkCorrelationMisalignedClock(t *testing.T) {
	energy := []nvtelemetry.EnergySummary{
		{
			DeviceUUID:     "GPU-1",
			WindowName:     "measurement",
			StartMonoNanos: 0, EndMonoNanos: 10_000_000_000, TotalDurationNanos: 10_000_000_000, CoveredDurationNanos: 10_000_000_000,
			EnergyConsumedJoules: 50.0,
			CoverageRatio:        1.0,
			Valid:                true,
		},
	}

	// Negative duration request (clock misaligned or step)
	requests := []nvtelemetry.BenchmarkRecord{
		{
			RequestID:      "req-bad-clock",
			StartTimeNanos: 5_000_000_000,
			EndTimeNanos:   4_000_000_000, // end before start!
			OutputTokens:   10,
			Success:        true,
		},
	}

	corr, err := nvtelemetry.CorrelateBenchmark("measurement", 0, 10_000_000_000, energy, requests, nvtelemetry.BenchmarkOptions{ClockProvenance: "local"})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}

	if corr.ClockAligned {
		t.Errorf("ClockAligned = true, want false for negative duration")
	}
	if !strings.Contains(corr.RefusalReason, nvtelemetry.ReasonMisalignedClock) {
		t.Errorf("RefusalReason = %s, want containing %s", corr.RefusalReason, nvtelemetry.ReasonMisalignedClock)
	}
}

func TestBenchmarkCorrelationRefusesUnknownClockProvenance(t *testing.T) {
	energy := []nvtelemetry.EnergySummary{
		{
			DeviceUUID:     "GPU-1",
			WindowName:     "measurement",
			StartMonoNanos: 0, EndMonoNanos: 10_000_000_000, TotalDurationNanos: 10_000_000_000, CoveredDurationNanos: 10_000_000_000,
			EnergyConsumedJoules: 50.0,
			CoverageRatio:        1.0,
			Valid:                true,
		},
	}
	requests := []nvtelemetry.BenchmarkRecord{
		{
			RequestID:      "req-1",
			StartTimeNanos: 1_000_000_000,
			EndTimeNanos:   2_000_000_000,
			OutputTokens:   10,
			Success:        true,
		},
	}

	corr, err := nvtelemetry.CorrelateBenchmark("measurement", 0, 10_000_000_000, energy, requests, nvtelemetry.BenchmarkOptions{
		ClockProvenance: "unknown",
	})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if corr.ClockAligned {
		t.Fatalf("ClockAligned = true, want false for unknown clock provenance")
	}
	if !strings.Contains(corr.RefusalReason, "unknown_clock_provenance") {
		t.Errorf("RefusalReason = %s, want containing unknown_clock_provenance", corr.RefusalReason)
	}
}

func TestBenchmarkCorrelationRefusesExcessiveClockSkew(t *testing.T) {
	energy := []nvtelemetry.EnergySummary{
		{
			DeviceUUID:     "GPU-1",
			WindowName:     "measurement",
			StartMonoNanos: 0, EndMonoNanos: 10_000_000_000, TotalDurationNanos: 10_000_000_000, CoveredDurationNanos: 10_000_000_000,
			EnergyConsumedJoules: 50.0,
			CoverageRatio:        1.0,
			Valid:                true,
		},
	}
	requests := []nvtelemetry.BenchmarkRecord{
		{
			RequestID:      "req-1",
			StartTimeNanos: 1_000_000_000,
			EndTimeNanos:   2_000_000_000,
			OutputTokens:   10,
			Success:        true,
		},
	}

	corr, err := nvtelemetry.CorrelateBenchmark("measurement", 0, 10_000_000_000, energy, requests, nvtelemetry.BenchmarkOptions{
		ClockProvenance:         "ntp",
		MaxClockSkewNanos:       1_000_000, // 1ms
		EstimatedClockSkewNanos: 5_000_000, // 5ms > 1ms
	})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if corr.ClockAligned {
		t.Fatalf("ClockAligned = true, want false for excessive skew")
	}
	if !strings.Contains(corr.RefusalReason, nvtelemetry.ReasonMisalignedClock) {
		t.Errorf("RefusalReason = %s, want containing %s", corr.RefusalReason, nvtelemetry.ReasonMisalignedClock)
	}
}

func TestBenchmarkCorrelationRefusesPartialEnergyCoverage(t *testing.T) {
	// Device only covered 85% of the window
	energy := []nvtelemetry.EnergySummary{
		{
			DeviceUUID:     "GPU-1",
			WindowName:     "measurement",
			StartMonoNanos: 0, EndMonoNanos: 10_000_000_000, TotalDurationNanos: 10_000_000_000, CoveredDurationNanos: 8_500_000_000,
			EnergyConsumedJoules: 50.0,
			CoverageRatio:        0.85,
			Valid:                true,
		},
	}
	requests := []nvtelemetry.BenchmarkRecord{
		{
			RequestID:      "req-1",
			StartTimeNanos: 1_000_000_000,
			EndTimeNanos:   2_000_000_000,
			OutputTokens:   10,
			Success:        true,
		},
	}

	// Default requires 1.0 (100%) coverage
	corr, err := nvtelemetry.CorrelateBenchmark("measurement", 0, 10_000_000_000, energy, requests, nvtelemetry.BenchmarkOptions{ClockProvenance: "local"})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if corr.EnergyAvailable {
		t.Fatalf("EnergyAvailable = true, want false for partial coverage")
	}
	if !strings.Contains(corr.RefusalReason, "partial energy coverage") {
		t.Errorf("RefusalReason = %s, want containing partial energy coverage", corr.RefusalReason)
	}
}

func TestBenchmarkCorrelationRefusesMalformedJoins(t *testing.T) {
	validEnergy := nvtelemetry.EnergySummary{DeviceUUID: "GPU-1", WindowName: "measurement", EnergyConsumedJoules: 100, CoverageRatio: 1, Valid: true, StartMonoNanos: 0, EndMonoNanos: 10, TotalDurationNanos: 10, CoveredDurationNanos: 10}
	validRequest := nvtelemetry.BenchmarkRecord{RequestID: "request", StartTimeNanos: 1, EndTimeNanos: 2, OutputTokens: 10, Success: true}
	tests := []struct {
		name                              string
		opts                              nvtelemetry.BenchmarkOptions
		editEnergy                        func(*nvtelemetry.EnergySummary)
		editRequest                       func(*nvtelemetry.BenchmarkRecord)
		duplicateEnergy, duplicateRequest bool
	}{
		{name: "default unknown clock"},
		{name: "whitespace clock", opts: nvtelemetry.BenchmarkOptions{ClockProvenance: " "}},
		{name: "negative skew", opts: nvtelemetry.BenchmarkOptions{ClockProvenance: "local", EstimatedClockSkewNanos: -100, MaxClockSkewNanos: 1}},
		{name: "minimum int skew", opts: nvtelemetry.BenchmarkOptions{ClockProvenance: "local", EstimatedClockSkewNanos: math.MinInt64, MaxClockSkewNanos: math.MaxInt64}},
		{name: "wrong same name interval", editEnergy: func(e *nvtelemetry.EnergySummary) { e.StartMonoNanos = 20; e.EndMonoNanos = 30 }},
		{name: "duration mismatch", editEnergy: func(e *nvtelemetry.EnergySummary) { e.TotalDurationNanos = 100 }},
		{name: "ratio duration mismatch", editEnergy: func(e *nvtelemetry.EnergySummary) { e.CoveredDurationNanos = 5 }},
		{name: "zero coverage", editEnergy: func(e *nvtelemetry.EnergySummary) { e.CoverageRatio = 0 }},
		{name: "negative coverage", editEnergy: func(e *nvtelemetry.EnergySummary) { e.CoverageRatio = -1 }},
		{name: "excess coverage", editEnergy: func(e *nvtelemetry.EnergySummary) { e.CoverageRatio = 2 }},
		{name: "nan coverage", editEnergy: func(e *nvtelemetry.EnergySummary) { e.CoverageRatio = math.NaN() }},
		{name: "infinite energy", editEnergy: func(e *nvtelemetry.EnergySummary) { e.EnergyConsumedJoules = math.Inf(1) }},
		{name: "nan energy", editEnergy: func(e *nvtelemetry.EnergySummary) { e.EnergyConsumedJoules = math.NaN() }},
		{name: "missing device", editEnergy: func(e *nvtelemetry.EnergySummary) { e.DeviceUUID = "" }},
		{name: "duplicate device", duplicateEnergy: true},
		{name: "missing request", editRequest: func(r *nvtelemetry.BenchmarkRecord) { r.RequestID = "" }},
		{name: "negative tokens", editRequest: func(r *nvtelemetry.BenchmarkRecord) { r.OutputTokens = -1 }},
		{name: "duplicate request", duplicateRequest: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, r := validEnergy, validRequest
			if tt.editEnergy != nil {
				tt.editEnergy(&e)
			}
			if tt.editRequest != nil {
				tt.editRequest(&r)
			}
			opts := tt.opts
			if tt.editEnergy != nil || tt.editRequest != nil || tt.duplicateEnergy || tt.duplicateRequest {
				opts.ClockProvenance = "local"
			}
			energy, requests := []nvtelemetry.EnergySummary{e}, []nvtelemetry.BenchmarkRecord{r}
			if tt.duplicateEnergy {
				energy = append(energy, e)
			}
			if tt.duplicateRequest {
				requests = append(requests, r)
			}
			got, err := nvtelemetry.CorrelateBenchmark("measurement", 0, 10, energy, requests, opts)
			if err != nil {
				t.Fatal(err)
			}
			if got.RefusalReason == "" || got.JoulesPerOutputToken != 0 || got.OutputTokensPerJoule != 0 {
				t.Fatalf("unsafe join accepted: %+v", got)
			}
		})
	}
}
