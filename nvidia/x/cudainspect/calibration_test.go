package cudainspect

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func observedAddkReport(t testing.TB, sass string) *Report {
	t.Helper()
	i, err := NewInspector(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	r, err := i.Report(context.Background(), roundtripCUBIN(t), Request{
		Symbol:   "addk",
		Required: []string{"precision_attribution", "target"},
		Optional: []string{"feature_suffix"},
		SASS:     sass,
	})
	if err != nil {
		t.Fatalf("Report: %v", err)
	}
	return r
}

func validAddkObservation(r *Report, fp32Active float64) WorkloadObservation {
	return WorkloadObservation{
		ArtifactSHA256: r.Artifact.SHA256,
		Entry:          "addk",
		Complete:       true,
		Identity: cudacensus.Identity{
			DeviceName:    "NVIDIA GB10",
			Architecture:  "sm_121",
			ComputeMajor:  12,
			ComputeMinor:  1,
			DriverVersion: "580.65.06",
			NVMLVersion:   "13.580.65.06",
		},
		Residency: cudaroofline.DeviceResident,
		Timing: DeviceTiming{
			Duration: 100 * time.Millisecond,
			Clock:    "cuda_event",
			Wall:     140 * time.Millisecond,
		},
		SampleInterval: 10 * time.Millisecond,
		Readings: []cudacensus.Reading{
			{Counter: cudacensus.FP32Active, Value: fp32Active, Available: true},
			{Counter: cudacensus.DRAMBandwidthUtil, Value: 0.05, Available: true},
			{Counter: cudacensus.HMMATensorActive, Available: false, Reason: "GPM metric not supported"},
		},
		RanAt: time.Date(2026, 10, 18, 12, 0, 0, 0, time.UTC),
	}
}

func TestCalibrateWorkloadPositiveScalarFP32(t *testing.T) {
	r := observedAddkReport(t, roundtripSASS(t))
	m, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatal(err)
	}

	var invoked int
	res, err := CalibrateWorkload(context.Background(), CalibrationRequest{
		Workload:  "gemm-10240-fp32",
		Report:    r,
		Model:     m,
		Shape:     GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 1},
		Separates: "fp32 from fp16_tensor",
		FailsWhen: "a mixed kernel interleaves HMMA and FFMA in comparable shares",
		Observe: func(context.Context) (WorkloadObservation, error) {
			invoked++
			return validAddkObservation(r, 0.5), nil
		},
	})
	if err != nil {
		t.Fatalf("CalibrateWorkload: %v", err)
	}
	if invoked != 1 {
		t.Fatalf("Observe invoked %d times, want 1", invoked)
	}
	if res.ComparisonError != nil {
		t.Fatalf("unexpected ComparisonError: %v", res.ComparisonError)
	}
	if res.EvidenceState != StateObserved || res.Entry != "addk" || res.Artifact != r.Artifact {
		t.Fatalf("unexpected artifact/entry association: %+v", res)
	}
	if res.Attribution.Precision != cudaroofline.FP32 || res.Attribution.Method != cudacensus.FromInstructionMix || res.Attribution.Share != 1 {
		t.Fatalf("Attribution = %+v", res.Attribution)
	}
	if want := uint64(2 * 10240 * 10240 * 10240); res.Calibration.AnalyticFLOPs != want {
		t.Fatalf("AnalyticFLOPs = %d, want %d", res.Calibration.AnalyticFLOPs, want)
	}
	if want := "2*m*n*k, m=10240 n=10240 k=10240"; res.Calibration.Derivation != want {
		t.Fatalf("Derivation = %q, want %q", res.Calibration.Derivation, want)
	}
	// Device duration (100ms) must be used, never host wall time (140ms).
	if res.Calibration.Duration != 100*time.Millisecond {
		t.Fatalf("Duration = %v, want 100ms", res.Calibration.Duration)
	}
	if !res.Calibration.Estimate.Available || res.Calibration.Estimate.AchievedTFLOPS != 20.0 {
		t.Fatalf("Estimate = %+v", res.Calibration.Estimate)
	}
	if len(res.Calibration.Estimate.Derivation.Readings) != 3 {
		t.Fatalf("Readings len = %d, want 3 (preserving absent reading)", len(res.Calibration.Estimate.Derivation.Readings))
	}

	wantAnalytic := 21.47483648
	if got := res.Calibration.AnalyticTFLOPS(); math.Abs(got-wantAnalytic) > 1e-6 {
		t.Fatalf("AnalyticTFLOPS = %v, want %v", got, wantAnalytic)
	}
	wantRel := (20.0 - wantAnalytic) / wantAnalytic
	if math.Abs(res.RelativeError-wantRel) > 1e-12 || res.RelativeError >= 0 {
		t.Fatalf("RelativeError = %v, want negative %v", res.RelativeError, wantRel)
	}

	if res.Receipt == nil {
		t.Fatal("Receipt is nil")
	}
	if res.Receipt.Samples != 1 || math.Abs(res.Receipt.Error-math.Abs(wantRel)) > 1e-12 {
		t.Fatalf("Receipt = %+v", *res.Receipt)
	}
	if _, err := cudacensus.NewThreshold(0.8, *res.Receipt); err != nil {
		t.Fatalf("NewThreshold from calibration receipt: %v", err)
	}
}

func TestCalibrateWorkloadNegativeRefusals(t *testing.T) {
	r := observedAddkReport(t, roundtripSASS(t))
	m, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatal(err)
	}
	baseReq := func() CalibrationRequest {
		return CalibrationRequest{
			Workload: "gemm-10240-fp32",
			Report:   r,
			Model:    m,
			Shape:    GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 1},
			Observe: func(context.Context) (WorkloadObservation, error) {
				return validAddkObservation(r, 0.5), nil
			},
		}
	}

	for _, tt := range []struct {
		name    string
		mutate  func(*CalibrationRequest)
		wantSub string
	}{
		{
			name: "host_wall_only_zero_device_duration",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validAddkObservation(r, 0.5)
					obs.Timing = DeviceTiming{Duration: 0, Clock: "cuda_event", Wall: 100 * time.Millisecond}
					return obs, nil
				}
			},
			wantSub: "host wall time",
		},
		{
			name: "host_wall_clock_origin_rejected",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validAddkObservation(r, 0.5)
					obs.Timing = DeviceTiming{Duration: 100 * time.Millisecond, Clock: "wall_elapsed", Wall: 100 * time.Millisecond}
					return obs, nil
				}
			},
			wantSub: "not a device clock",
		},
		{
			name: "missing_clock_origin_rejected",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validAddkObservation(r, 0.5)
					obs.Timing.Clock = ""
					return obs, nil
				}
			},
			wantSub: "clock origin is required",
		},
		{
			name: "wrong_artifact_digest",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validAddkObservation(r, 0.5)
					obs.ArtifactSHA256 = strings.Repeat("a", 64)
					return obs, nil
				}
			},
			wantSub: "observed artifact digest",
		},
		{
			name: "wrong_entry_symbol",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validAddkObservation(r, 0.5)
					obs.Entry = "intmix"
					return obs, nil
				}
			},
			wantSub: "observed entry",
		},
		{
			name: "callback_returns_error",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					return WorkloadObservation{}, errors.New("driver launch failure")
				}
			},
			wantSub: "driver launch failure",
		},
		{
			name: "observation_err_set",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validAddkObservation(r, 0.5)
					obs.Err = errors.New("device trap")
					return obs, nil
				}
			},
			wantSub: "device trap",
		},
		{
			name: "incomplete_execution",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validAddkObservation(r, 0.5)
					obs.Complete = false
					return obs, nil
				}
			},
			wantSub: "incomplete",
		},
		{
			name: "shape_zero_dimension",
			mutate: func(req *CalibrationRequest) {
				req.Shape = GEMMShape{M: 1024, N: 0, K: 1024, Repetitions: 1}
			},
			wantSub: "must be positive",
		},
		{
			name: "shape_overflow_dimensions",
			mutate: func(req *CalibrationRequest) {
				req.Shape = GEMMShape{M: 1 << 32, N: 1 << 32, K: 1, Repetitions: 1}
			},
			wantSub: "overflows uint64",
		},
		{
			name: "shape_overflow_repetitions",
			mutate: func(req *CalibrationRequest) {
				req.Shape = GEMMShape{M: 1024, N: 1024, K: 1024, Repetitions: math.MaxUint64}
			},
			wantSub: "overflows uint64",
		},
		{
			name: "model_architecture_mismatch",
			mutate: func(req *CalibrationRequest) {
				other, err := cudaroofline.NewModel("sm_90", cudaroofline.DeviceResident)
				if err != nil {
					t.Fatal(err)
				}
				req.Model = other
			},
			wantSub: "does not match observation architecture",
		},
		{
			name: "model_residency_mismatch",
			mutate: func(req *CalibrationRequest) {
				other, err := cudaroofline.NewModel("sm_121", cudaroofline.HostPageTableResident)
				if err != nil {
					t.Fatal(err)
				}
				req.Model = other
			},
			wantSub: "does not match model residency",
		},
		{
			name: "inferred_unbound_sass_rejected",
			mutate: func(req *CalibrationRequest) {
				unbound := strings.Replace(roundtripSASS(t), "/* 0x0000000502097221 */", "", 1)
				req.Report = observedAddkReport(t, unbound)
			},
			wantSub: "want observed bound to .text.addk bytes",
		},
		{
			name: "missing_fails_when_for_receipt",
			mutate: func(req *CalibrationRequest) {
				req.Separates = "fp32 from fp16_tensor"
				req.FailsWhen = ""
			},
			wantSub: "requires both separates and failsWhen",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req := baseReq()
			tt.mutate(&req)
			res, err := CalibrateWorkload(context.Background(), req)
			if !errors.Is(err, ErrCalibration) || res != nil || !strings.Contains(err.Error(), tt.wantSub) {
				t.Fatalf("CalibrateWorkload = %+v, %v; want nil, ErrCalibration containing %q", res, err, tt.wantSub)
			}
		})
	}
}

func TestCalibrateWorkloadUnavailableAndRejectedOutcomesVisible(t *testing.T) {
	m, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("absent_counters", func(t *testing.T) {
		r := observedAddkReport(t, roundtripSASS(t))
		res, err := CalibrateWorkload(context.Background(), CalibrationRequest{
			Workload: "gemm-10240-fp32-silent",
			Report:   r,
			Model:    m,
			Shape:    GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 1},
			Observe: func(context.Context) (WorkloadObservation, error) {
				obs := validAddkObservation(r, 0)
				obs.Readings = []cudacensus.Reading{
					{Counter: cudacensus.FP32Active, Available: false, Reason: "GPM metric not supported"},
					{Counter: cudacensus.DRAMBandwidthUtil, Available: false, Reason: "GPM metric not supported"},
				}
				return obs, nil
			},
		})
		if err != nil {
			t.Fatalf("CalibrateWorkload: %v", err)
		}
		if res.ComparisonError == nil || !math.IsNaN(res.RelativeError) {
			t.Fatalf("ComparisonError = %v, RelativeError = %v; want non-nil error and NaN", res.ComparisonError, res.RelativeError)
		}
		if res.Calibration.Estimate.Available || res.Calibration.Estimate.Confidence != cudacensus.ConfidenceUnavailable {
			t.Fatalf("Estimate = %+v, want unavailable", res.Calibration.Estimate)
		}
		if len(res.Calibration.Estimate.Derivation.Readings) != 2 {
			t.Fatalf("absent readings not preserved: %+v", res.Calibration.Estimate.Derivation.Readings)
		}
		if _, err := res.ThresholdReceipt("fp32", "absent counters"); err == nil {
			t.Fatal("ThresholdReceipt succeeded for unavailable estimate")
		}
	})

	t.Run("non_fp32_absent_peak", func(t *testing.T) {
		// Replace the FADD mnemonic with HMMA.16816.F32.F16 while keeping the
		// 16-byte instruction encodings intact so the fixture exercises an
		// observed non-FP32 attribution against the FP32-only peak table.
		hmmaSASS := strings.Replace(roundtripSASS(t), "FADD R9, R2, R5", "HMMA.16816.F32.F16 R9, R2, R5, R9", 1)
		r := observedAddkReport(t, hmmaSASS)
		res, err := CalibrateWorkload(context.Background(), CalibrationRequest{
			Workload: "gemm-10240-fp16-tensor",
			Report:   r,
			Model:    m,
			Shape:    GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 1},
			Observe: func(context.Context) (WorkloadObservation, error) {
				obs := validAddkObservation(r, 0)
				obs.Readings = []cudacensus.Reading{
					{Counter: cudacensus.HMMATensorActive, Value: 0.8, Available: true},
					{Counter: cudacensus.DRAMBandwidthUtil, Value: 0.2, Available: true},
				}
				return obs, nil
			},
		})
		if err != nil {
			t.Fatalf("CalibrateWorkload: %v", err)
		}
		if res.Attribution.Precision != cudaroofline.FP16Tensor || res.Attribution.Method != cudacensus.FromInstructionMix {
			t.Fatalf("Attribution = %+v, want fp16_tensor from instruction_mix", res.Attribution)
		}
		if res.ComparisonError == nil || !strings.Contains(res.ComparisonError.Error(), "no fp16_tensor peak") || !math.IsNaN(res.RelativeError) {
			t.Fatalf("ComparisonError = %v, RelativeError = %v; want missing fp16_tensor peak error and NaN", res.ComparisonError, res.RelativeError)
		}
		if res.Calibration.Estimate.Available {
			t.Fatalf("Estimate = %+v, want unavailable", res.Calibration.Estimate)
		}
	})

	t.Run("above_ceiling_rejection", func(t *testing.T) {
		r := observedAddkReport(t, roundtripSASS(t))
		res, err := CalibrateWorkload(context.Background(), CalibrationRequest{
			Workload: "gemm-10240-fp32-misscaled",
			Report:   r,
			Model:    m,
			Shape:    GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 1},
			Observe: func(context.Context) (WorkloadObservation, error) {
				return validAddkObservation(r, 85.0), nil
			},
		})
		if err != nil {
			t.Fatalf("CalibrateWorkload: %v", err)
		}
		if !res.Calibration.Estimate.Rejected || res.Calibration.Estimate.Available {
			t.Fatalf("Estimate = %+v, want rejected and unavailable", res.Calibration.Estimate)
		}
		if res.ComparisonError == nil || !strings.Contains(res.ComparisonError.Error(), "rejected above the ceiling") || !math.IsNaN(res.RelativeError) {
			t.Fatalf("ComparisonError = %v, RelativeError = %v; want above-ceiling error and NaN", res.ComparisonError, res.RelativeError)
		}
		if _, err := res.ThresholdReceipt("fp32", "misscaled percentage"); err == nil {
			t.Fatal("ThresholdReceipt succeeded for rejected estimate")
		}
	})
}
