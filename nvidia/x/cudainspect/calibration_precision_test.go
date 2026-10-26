package cudainspect

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func constructedCalibrationPrecisionModel(t testing.TB) *cudaroofline.Model {
	t.Helper()
	c := cudaroofline.Ceiling{
		Architecture:      "sm_121",
		Residency:         cudaroofline.DeviceResident,
		DeviceName:        "NVIDIA GB10",
		MaxSMClockMHz:     2400,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    239.6,
		ComputeProvenance: cudaroofline.Unverified,
		MemoryProvenance:  cudaroofline.Measured,
		PrecisionPeaks: map[cudaroofline.Precision]cudaroofline.PrecisionPeak{
			cudaroofline.TF32Tensor: {
				PeakTFLOPS: 80.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense TF32 tensor assertion",
			},
			cudaroofline.FP16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense FP16 tensor assertion",
			},
			cudaroofline.BF16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense BF16 tensor assertion",
			},
			cudaroofline.FP8Tensor: {
				PeakTFLOPS: 320.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense FP8 tensor assertion",
			},
		},
	}
	if err := cudaroofline.ValidateCeiling(c); err != nil {
		t.Fatalf("ValidateCeiling: %v", err)
	}
	return cudaroofline.NewModelWithCeiling(c)
}

func validTensorObservation(r *Report, hmmaActive float64) WorkloadObservation {
	obs := validAddkObservation(r, 0)
	obs.Identity.MaxSMClockMHz = 2400
	obs.Readings = []cudacensus.Reading{
		{Counter: cudacensus.HMMATensorActive, Value: hmmaActive, Available: true},
		{Counter: cudacensus.DRAMBandwidthUtil, Value: 0.05, Available: true},
	}
	return obs
}

func TestCalibrateWorkloadFloatingPrecisionWithConstructedProfile(t *testing.T) {
	m := constructedCalibrationPrecisionModel(t)

	for _, tc := range []struct {
		name      string
		mnemonic  string
		precision cudaroofline.Precision
		wantPeak  float64
		activity  float64
	}{
		{
			name:      "fp16_tensor",
			mnemonic:  "HMMA.16816.F32.F16 R9, R2, R5, R9",
			precision: cudaroofline.FP16Tensor,
			wantPeak:  160.0,
			activity:  0.5, // 80.0 TFLOPS achieved (exceeds 40.0 FP32 peak, bounded by 160.0 FP16Tensor peak)
		},
		{
			name:      "tf32_tensor",
			mnemonic:  "HMMA.1688.F32.TF32 R9, R2, R5, R9",
			precision: cudaroofline.TF32Tensor,
			wantPeak:  80.0,
			activity:  0.75, // 60.0 TFLOPS achieved (exceeds 40.0 FP32 peak, bounded by 80.0 TF32Tensor peak)
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sass := strings.Replace(roundtripSASS(t), "FADD R9, R2, R5", tc.mnemonic, 1)
			r := observedAddkReport(t, sass)

			res, err := CalibrateWorkload(context.Background(), CalibrationRequest{
				Workload:  "gemm-10240-" + string(tc.precision),
				Report:    r,
				Model:     m,
				Shape:     GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 4},
				Separates: string(tc.precision) + " from fp32",
				FailsWhen: "a kernel mixes scalar FP32 and tensor HMMA instructions",
				Observe: func(context.Context) (WorkloadObservation, error) {
					return validTensorObservation(r, tc.activity), nil
				},
			})
			if err != nil {
				t.Fatalf("CalibrateWorkload: %v", err)
			}
			if res.ComparisonError != nil || math.IsNaN(res.RelativeError) {
				t.Fatalf("ComparisonError = %v, RelativeError = %v; want nil error and finite relative error",
					res.ComparisonError, res.RelativeError)
			}
			if res.Attribution.Precision != tc.precision || res.Attribution.Method != cudacensus.FromInstructionMix {
				t.Fatalf("Attribution = %+v, want %s/%s", res.Attribution, tc.precision, cudacensus.FromInstructionMix)
			}
			wantAchieved := tc.activity * tc.wantPeak
			if !res.Calibration.Estimate.Available || res.Calibration.Estimate.Rejected || res.Calibration.Estimate.AchievedTFLOPS != wantAchieved {
				t.Fatalf("Estimate = %+v, want Available=true AchievedTFLOPS=%v", res.Calibration.Estimate, wantAchieved)
			}
			if res.Calibration.Estimate.Derivation.PeakTFLOPS != tc.wantPeak {
				t.Errorf("Derivation.PeakTFLOPS = %v, want %v", res.Calibration.Estimate.Derivation.PeakTFLOPS, tc.wantPeak)
			}
			if res.Calibration.Estimate.Derivation.DeviceName != "NVIDIA GB10" || res.Calibration.Estimate.Derivation.MaxSMClockMHz != 2400 {
				t.Errorf("Derivation identity = (%q, %d), want (NVIDIA GB10, 2400)",
					res.Calibration.Estimate.Derivation.DeviceName, res.Calibration.Estimate.Derivation.MaxSMClockMHz)
			}
			if res.Receipt == nil {
				t.Fatal("Receipt is nil")
			}
		})
	}
}

func TestCalibrateWorkloadRefusesINT8TensorBeforeObserve(t *testing.T) {
	immaSASS := strings.Replace(roundtripSASS(t), "FADD R9, R2, R5", "IMMA.16832.S32.S8.S8 R9, R2, R5, R9", 1)
	r := observedAddkReport(t, immaSASS)
	m := constructedCalibrationPrecisionModel(t)

	var invoked int
	res, err := CalibrateWorkload(context.Background(), CalibrationRequest{
		Workload: "gemm-10240-int8-tensor",
		Report:   r,
		Model:    m,
		Shape:    GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 1},
		Observe: func(context.Context) (WorkloadObservation, error) {
			invoked++
			return validTensorObservation(r, 0.5), nil
		},
	})
	if !errors.Is(err, ErrCalibration) || res != nil {
		t.Fatalf("CalibrateWorkload(INT8Tensor) = (%+v, %v), want (nil, ErrCalibration)", res, err)
	}
	if !strings.Contains(err.Error(), "integer TOPS/OPs") {
		t.Errorf("error = %q, want integer TOPS/OPs refusal", err)
	}
	if invoked != 0 {
		t.Fatalf("Observe invoked %d times for unsupported integer workload, want 0", invoked)
	}
}

func TestCalibrateWorkloadIdentityAndProfileConsistencyRefusals(t *testing.T) {
	hmmaSASS := strings.Replace(roundtripSASS(t), "FADD R9, R2, R5", "HMMA.16816.F32.F16 R9, R2, R5, R9", 1)
	r := observedAddkReport(t, hmmaSASS)

	for _, tc := range []struct {
		name    string
		mutate  func(*CalibrationRequest)
		wantSub string
	}{
		{
			name: "model_missing_device_name",
			mutate: func(req *CalibrationRequest) {
				req.Model.Ceiling.DeviceName = ""
			},
			wantSub: "DeviceName is required",
		},
		{
			name: "model_sparse_peak_rejected",
			mutate: func(req *CalibrationRequest) {
				pk := req.Model.Ceiling.PrecisionPeaks[cudaroofline.FP16Tensor]
				pk.Sparse = true
				req.Model.Ceiling.PrecisionPeaks[cudaroofline.FP16Tensor] = pk
			},
			wantSub: "sparse (2:4) peaks are unsupported",
		},
		{
			name: "observation_device_name_mismatch",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validTensorObservation(r, 0.5)
					obs.Identity.DeviceName = "NVIDIA H100"
					return obs, nil
				}
			},
			wantSub: "observation device name",
		},
		{
			name: "observation_missing_max_sm_clock",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validTensorObservation(r, 0.5)
					obs.Identity.MaxSMClockMHz = 0
					return obs, nil
				}
			},
			wantSub: "must state max_sm_clock_mhz",
		},
		{
			name: "observation_max_sm_clock_mismatch",
			mutate: func(req *CalibrationRequest) {
				req.Observe = func(context.Context) (WorkloadObservation, error) {
					obs := validTensorObservation(r, 0.5)
					obs.Identity.MaxSMClockMHz = 1980
					return obs, nil
				}
			},
			wantSub: "does not match model ceiling max_sm_clock_mhz",
		},
		{
			name: "implicit_architecture_suffix_alias_rejected",
			mutate: func(req *CalibrationRequest) {
				req.Model.Ceiling.Architecture = "sm_121a"
			},
			wantSub: "does not exactly match observation architecture",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := CalibrationRequest{
				Workload: "gemm-10240-fp16-tensor",
				Report:   r,
				Model:    constructedCalibrationPrecisionModel(t),
				Shape:    GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 4},
				Observe: func(context.Context) (WorkloadObservation, error) {
					return validTensorObservation(r, 0.5), nil
				},
			}
			tc.mutate(&req)
			res, err := CalibrateWorkload(context.Background(), req)
			if !errors.Is(err, ErrCalibration) || res != nil || !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("CalibrateWorkload = (%+v, %v), want (nil, ErrCalibration containing %q)", res, err, tc.wantSub)
			}
		})
	}
}

func ExampleCalibrateWorkload_precisionCeiling() {
	m := cudaroofline.NewModelWithCeiling(cudaroofline.Ceiling{
		Architecture:      "sm_121a",
		Residency:         cudaroofline.DeviceResident,
		DeviceName:        "NVIDIA GB10 (constructed example profile)",
		MaxSMClockMHz:     2400,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    239.6,
		ComputeProvenance: cudaroofline.Unverified,
		MemoryProvenance:  cudaroofline.Measured,
		PrecisionPeaks: map[cudaroofline.Precision]cudaroofline.PrecisionPeak{
			cudaroofline.FP16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed example profile: dense FP16 tensor assertion",
			},
		},
	})
	err := cudaroofline.ValidateCeiling(m.Ceiling)
	fmt.Printf("valid=%v device=%q clock=%d\n", err == nil, m.Ceiling.DeviceName, m.Ceiling.MaxSMClockMHz)
	// Output:
	// valid=true device="NVIDIA GB10 (constructed example profile)" clock=2400
}
