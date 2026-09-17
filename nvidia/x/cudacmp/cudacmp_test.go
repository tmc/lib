package cudacmp

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func TestNewTrial(t *testing.T) {
	tests := []struct {
		name    string
		cfg     TrialConfig
		wantErr bool
	}{
		{
			name: "valid config",
			cfg: TrialConfig{
				WorkloadTokensPath: "test.json",
				ContextLength:      4096,
				PrefillTokens:      2048,
				DecodeTokens:       16,
				InterleavedRuns:    5,
				WarmupRuns:         1,
				COraclePath:        "/bin/echo",
				Residency:          cudaroofline.DeviceResident,
			},
			wantErr: false,
		},
		{
			name: "negative interleaved runs",
			cfg: TrialConfig{
				InterleavedRuns: -1,
			},
			wantErr: true,
		},
		{
			name: "negative warmup runs",
			cfg: TrialConfig{
				WarmupRuns: -1,
			},
			wantErr: true,
		},
		{
			name: "negative prefill tokens",
			cfg: TrialConfig{
				PrefillTokens: -1,
			},
			wantErr: true,
		},
		{
			name: "negative decode tokens",
			cfg: TrialConfig{
				DecodeTokens: -1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trial, err := NewTrial(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewTrial() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && trial == nil {
				t.Fatal("expected non-nil trial on success")
			}
			if !tt.wantErr && trial.Config().ContextLength != tt.cfg.ContextLength {
				t.Errorf("Config().ContextLength = %d, want %d", trial.Config().ContextLength, tt.cfg.ContextLength)
			}
		})
	}
}

func TestRender(t *testing.T) {
	reports := []DifferentialReport{
		{
			Phase:              PhasePrefill,
			Family:             FamilyExperts,
			Availability:       AvailableBoth,
			Available:          true,
			CObserved:          true,
			GoObserved:         true,
			CDurationNs:        12_500_000,
			GoDurationNs:       10_000_000,
			DeltaNs:            -2_500_000,
			Speedup:            1.25,
			NullDriftNs:        50_000,
			ExceedsDrift:       true,
			StatisticallyValid: true,
		},
		{
			Phase:              PhaseDecode,
			Family:             FamilyDenseQ8,
			Availability:       AvailableBoth,
			Available:          true,
			CObserved:          true,
			GoObserved:         true,
			CDurationNs:        8_000_000,
			GoDurationNs:       8_020_000,
			DeltaNs:            20_000,
			Speedup:            0.9975,
			NullDriftNs:        50_000,
			ExceedsDrift:       false,
			StatisticallyValid: false,
		},
		{
			Phase:        PhaseDecode,
			Family:       FamilyResidual,
			Availability: AvailableCOnly,
			Available:    false,
			CObserved:    true,
			GoObserved:   false,
			CDurationNs:  1_000_000,
			NullDriftNs:  50_000,
		},
	}

	var buf bytes.Buffer
	if err := Render(&buf, reports); err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "PHASE") || !strings.Contains(out, "FAMILY") {
		t.Errorf("missing headers in output: %s", out)
	}
	if !strings.Contains(out, "experts") || !strings.Contains(out, "dense_q8") {
		t.Errorf("missing family names in output: %s", out)
	}
	if !strings.Contains(out, "YES") || !strings.Contains(out, "NOISE") {
		t.Errorf("missing validity flags in output: %s", out)
	}
	if !strings.Contains(out, "UNAVAILABLE") {
		t.Errorf("expected UNAVAILABLE in output, got: %s", out)
	}
}

func TestHelperMethods(t *testing.T) {
	km := KernelMetric{
		Timing: LatencyDecomposition{DeviceNs: 5_000_000},
	}
	if got := km.Duration(); got != 5*time.Millisecond {
		t.Errorf("km.Duration() = %v, want 5ms", got)
	}

	pc := PhaseCensus{
		SummedKernelNs: 10_000_000,
		GPUBusyNs:      8_000_000,
	}
	if got := pc.SummedKernelDuration(); got != 10*time.Millisecond {
		t.Errorf("pc.SummedKernelDuration() = %v, want 10ms", got)
	}
	if got := pc.GPUBusyDuration(); got != 8*time.Millisecond {
		t.Errorf("pc.GPUBusyDuration() = %v, want 8ms", got)
	}
	if got := pc.Duration(); got != 10*time.Millisecond {
		t.Errorf("pc.Duration() = %v, want 10ms", got)
	}

	dr := DifferentialReport{
		CDurationNs:  15_000_000,
		GoDurationNs: 10_000_000,
		DeltaNs:      -5_000_000,
	}
	if got := dr.CDuration(); got != 15*time.Millisecond {
		t.Errorf("CDuration() = %v, want 15ms", got)
	}
	if got := dr.GoDuration(); got != 10*time.Millisecond {
		t.Errorf("GoDuration() = %v, want 10ms", got)
	}
	if got := dr.Delta(); got != -5*time.Millisecond {
		t.Errorf("Delta() = %v, want -5ms", got)
	}
}
