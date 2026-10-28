package cudainspect

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// ErrCalibration indicates that a calibration request or workload observation
// failed validation or could not be associated with the inspected artifact.
var ErrCalibration = errors.New("cudainspect: invalid workload calibration")

// GEMMShape describes the dimensions of a dense matrix multiply of (M, K) by
// (K, N) executed Repetitions times.
type GEMMShape struct {
	M           uint64
	N           uint64
	K           uint64
	Repetitions uint64
}

// AnalyticWork computes the checked floating-point operation count
// 2*M*N*K*Repetitions and returns a human-checkable derivation string. It
// refuses zero dimensions or uint64 overflow.
func (s GEMMShape) AnalyticWork() (flops uint64, derivation string, err error) {
	if s.M == 0 || s.N == 0 || s.K == 0 || s.Repetitions == 0 {
		return 0, "", fmt.Errorf("%w: GEMM dimensions and repetitions must be positive (m=%d n=%d k=%d repetitions=%d)",
			ErrCalibration, s.M, s.N, s.K, s.Repetitions)
	}
	total := uint64(2)
	for _, factor := range []uint64{s.M, s.N, s.K, s.Repetitions} {
		prod, ok := checkedMulUint64(total, factor)
		if !ok {
			return 0, "", fmt.Errorf("%w: analytic GEMM work 2*%d*%d*%d*%d overflows uint64",
				ErrCalibration, s.M, s.N, s.K, s.Repetitions)
		}
		total = prod
	}
	if s.Repetitions == 1 {
		return total, fmt.Sprintf("2*m*n*k, m=%d n=%d k=%d", s.M, s.N, s.K), nil
	}
	return total, fmt.Sprintf("2*m*n*k*repetitions, m=%d n=%d k=%d repetitions=%d", s.M, s.N, s.K, s.Repetitions), nil
}

func checkedMulUint64(a, b uint64) (uint64, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > math.MaxUint64/b {
		return 0, false
	}
	return a * b, true
}

// DeviceTiming records a kernel execution duration measured on the device
// clock. Wall is an optional host wall-clock observation kept strictly
// separate; it is never substituted when Duration is zero or negative.
//
// Callers using a trial harness must supply the explicit summed kernel device
// duration and name the device clock origin (such as "cuda_event" or
// "cupti_activity") rather than a host wall timer.
type DeviceTiming struct {
	Duration time.Duration // kernel execution span on the device clock
	Clock    string        // device timing source, e.g. "cuda_event" or "cupti_activity"
	Wall     time.Duration // optional host wall time, never used as Duration
}

var hostWallClocks = map[string]bool{
	"wall":         true,
	"wall_elapsed": true,
	"host":         true,
	"host_wall":    true,
	"host_clock":   true,
	"api_call":     true,
	"time.since":   true,
}

func (t DeviceTiming) validate() error {
	if t.Duration <= 0 {
		return fmt.Errorf("%w: missing or nonpositive device duration (%v); host wall time (%v) is not a substitute",
			ErrCalibration, t.Duration, t.Wall)
	}
	clock := strings.ToLower(strings.TrimSpace(t.Clock))
	if clock == "" {
		return fmt.Errorf("%w: device timing clock origin is required", ErrCalibration)
	}
	if hostWallClocks[clock] {
		return fmt.Errorf("%w: timing clock %q is a host or API timer, not a device clock", ErrCalibration, t.Clock)
	}
	return nil
}

// WorkloadObservation is the evidence returned by a calibration callback for
// one execution and counter-sampling tick of a known-FLOP workload.
//
// Host validation checks consistency between the supplied report, model, and
// callback fields; it does not authenticate that the callback executed on a
// physical GPU or that unverified peaks are hardware measurements.
type WorkloadObservation struct {
	ArtifactSHA256 string
	Entry          string
	Complete       bool
	Err            error
	Identity       cudacensus.Identity
	Residency      cudaroofline.Residency
	Timing         DeviceTiming
	SampleInterval time.Duration
	Readings       []cudacensus.Reading
	RanAt          time.Time
}

// ObserveWorkloadFunc executes or replays one calibration workload tick and
// returns its observation.
type ObserveWorkloadFunc func(context.Context) (WorkloadObservation, error)

// CalibrationRequest configures a known-workload calibration of the counter
// estimator against an inspected kernel report.
type CalibrationRequest struct {
	Workload  string
	Report    *Report
	Model     *cudaroofline.Model
	Shape     GEMMShape
	Separates string
	FailsWhen string
	Observe   ObserveWorkloadFunc
}

// CalibrationResult retains the estimator calibration, its signed relative
// error when the estimate is available, an optional single-observation
// threshold receipt when Separates and FailsWhen were supplied, and the raw
// observation, identity, and static attribution evidence.
//
// When the estimator declines to produce a figure (because required counters or
// non-FP32 peaks are absent) or rejects an estimate above the roofline ceiling,
// ComparisonError holds the error from [cudacensus.Calibration.RelativeError]
// and RelativeError is NaN rather than zero.
type CalibrationResult struct {
	Calibration     cudacensus.Calibration
	RelativeError   float64
	ComparisonError error
	Receipt         *cudacensus.CalibrationReceipt

	Artifact      Digest
	SASS          Digest
	Entry         string
	EvidenceState string
	Mix           cudacensus.InstructionMix
	Attribution   cudacensus.Attribution
	Observation   WorkloadObservation
}

// ThresholdReceipt produces a single-observation threshold calibration receipt
// through [cudacensus.Calibration.Receipt].
func (r *CalibrationResult) ThresholdReceipt(separates, failsWhen string) (cudacensus.CalibrationReceipt, error) {
	if r == nil {
		return cudacensus.CalibrationReceipt{}, fmt.Errorf("%w: nil calibration result", ErrCalibration)
	}
	return r.Calibration.Receipt(separates, failsWhen)
}

// CalibrateWorkload executes req.Observe, derives static precision attribution
// from req.Report's observed instruction mix, calls [cudacensus.NewEstimate],
// and constructs a [cudacensus.Calibration] against the checked analytic GEMM
// work of req.Shape.
//
// It refuses missing or invalid reports, inferred SASS listings whose encodings
// are not bound to the artifact's .text section, overflowed or zero GEMM
// shapes, failed or incomplete callback runs, mismatched artifact digests or
// entry symbols, mismatched model/identity architectures or residencies, and
// missing, nonpositive, or host-wall timing. Unavailable or rejected estimates
// remain visible in CalibrationResult with ComparisonError set and
// RelativeError set to NaN.
func CalibrateWorkload(ctx context.Context, req CalibrationRequest) (*CalibrationResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	workload := strings.TrimSpace(req.Workload)
	if workload == "" {
		return nil, fmt.Errorf("%w: workload name is required", ErrCalibration)
	}
	if req.Report == nil {
		return nil, fmt.Errorf("%w: report is required", ErrCalibration)
	}
	if req.Model == nil {
		return nil, fmt.Errorf("%w: roofline model is required", ErrCalibration)
	}
	if err := cudaroofline.ValidateCeiling(req.Model.Ceiling); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCalibration, err)
	}
	flops, derivation, err := req.Shape.AnalyticWork()
	if err != nil {
		return nil, err
	}
	wantReceipt := strings.TrimSpace(req.Separates) != "" || strings.TrimSpace(req.FailsWhen) != ""
	if wantReceipt && (strings.TrimSpace(req.Separates) == "" || strings.TrimSpace(req.FailsWhen) == "") {
		return nil, fmt.Errorf("%w: threshold receipt requires both separates and failsWhen", ErrCalibration)
	}
	if req.Observe == nil {
		return nil, fmt.Errorf("%w: observation callback is required", ErrCalibration)
	}

	pa, err := req.Report.PrecisionAttribution()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCalibration, err)
	}
	if pa.State != StateObserved {
		return nil, fmt.Errorf("%w: precision_attribution is %s, want %s bound to .text.%s bytes",
			ErrCalibration, pa.State, StateObserved, req.Report.Request.Symbol)
	}
	mix, state, err := req.Report.InstructionMix()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCalibration, err)
	}
	attr, err := cudacensus.AttributeFromMix(mix)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCalibration, err)
	}

	obs, err := req.Observe(ctx)
	if err != nil {
		return nil, fmt.Errorf("%w: observation callback failed: %v", ErrCalibration, err)
	}
	if obs.Err != nil {
		return nil, fmt.Errorf("%w: observed workload execution failed: %v", ErrCalibration, obs.Err)
	}
	if !obs.Complete {
		return nil, fmt.Errorf("%w: observed workload execution is incomplete", ErrCalibration)
	}
	if strings.ToLower(strings.TrimSpace(obs.ArtifactSHA256)) != strings.ToLower(req.Report.Artifact.SHA256) {
		return nil, fmt.Errorf("%w: observed artifact digest %q does not match report artifact %q",
			ErrCalibration, obs.ArtifactSHA256, req.Report.Artifact.SHA256)
	}
	if obs.Entry != req.Report.Request.Symbol {
		return nil, fmt.Errorf("%w: observed entry %q does not match report symbol %q",
			ErrCalibration, obs.Entry, req.Report.Request.Symbol)
	}
	if err := validateObservationEnvironment(req.Report, req.Model, obs); err != nil {
		return nil, err
	}
	if err := obs.Timing.validate(); err != nil {
		return nil, err
	}
	if obs.SampleInterval < 0 {
		return nil, fmt.Errorf("%w: negative sample interval %v", ErrCalibration, obs.SampleInterval)
	}
	interval := obs.SampleInterval
	if interval == 0 {
		interval = obs.Timing.Duration
	}
	if obs.RanAt.IsZero() {
		return nil, fmt.Errorf("%w: observation timestamp is required", ErrCalibration)
	}
	if len(obs.Readings) == 0 {
		return nil, fmt.Errorf("%w: observation carries no counter readings", ErrCalibration)
	}

	readings := append([]cudacensus.Reading(nil), obs.Readings...)
	est := cudacensus.NewEstimate(req.Model, cudacensus.Inputs{
		Readings:    readings,
		Attribution: attr,
		Interval:    interval,
	})

	cal := cudacensus.Calibration{
		Workload:      workload,
		Device:        formatCalibrationDevice(obs.Identity),
		AnalyticFLOPs: flops,
		Derivation:    derivation,
		Duration:      obs.Timing.Duration,
		Estimate:      est,
		RanAt:         obs.RanAt,
	}
	if err := cal.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCalibration, err)
	}

	obsCopy := obs
	obsCopy.Readings = append([]cudacensus.Reading(nil), obs.Readings...)
	res := &CalibrationResult{
		Calibration:   cal,
		RelativeError: math.NaN(),
		Artifact:      req.Report.Artifact,
		SASS:          *req.Report.SASS,
		Entry:         req.Report.Request.Symbol,
		EvidenceState: state,
		Mix:           mix,
		Attribution:   attr,
		Observation:   obsCopy,
	}

	relErr, cmpErr := cal.RelativeError()
	if cmpErr != nil {
		res.ComparisonError = cmpErr
		return res, nil
	}
	res.RelativeError = relErr

	if wantReceipt {
		rec, err := cal.Receipt(req.Separates, req.FailsWhen)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrCalibration, err)
		}
		res.Receipt = &rec
	}
	return res, nil
}

func validateObservationEnvironment(r *Report, m *cudaroofline.Model, obs WorkloadObservation) error {
	if strings.TrimSpace(obs.Identity.DeviceName) == "" || strings.TrimSpace(obs.Identity.DriverVersion) == "" || strings.TrimSpace(obs.Identity.Architecture) == "" {
		return fmt.Errorf("%w: observation identity must state device name, architecture, and driver version", ErrCalibration)
	}
	if obs.Residency == "" || obs.Residency != m.Ceiling.Residency {
		return fmt.Errorf("%w: observation residency %q does not match model residency %q",
			ErrCalibration, obs.Residency, m.Ceiling.Residency)
	}
	obsMajor, obsMinor, obsSuffix, err := cudakernel.ParseArch(obs.Identity.Architecture)
	if err != nil {
		return fmt.Errorf("%w: invalid observation architecture %q: %v", ErrCalibration, obs.Identity.Architecture, err)
	}
	if obs.Identity.ComputeMajor != 0 || obs.Identity.ComputeMinor != 0 {
		if int(obs.Identity.ComputeMajor) != obsMajor || int(obs.Identity.ComputeMinor) != obsMinor {
			return fmt.Errorf("%w: observation compute capability %d.%d disagrees with architecture %q",
				ErrCalibration, obs.Identity.ComputeMajor, obs.Identity.ComputeMinor, obs.Identity.Architecture)
		}
	}

	target := r.Feature("target")
	if !target.Known() {
		return fmt.Errorf("%w: report target feature is unavailable", ErrCalibration)
	}
	var repTarget struct {
		Architecture string `json:"architecture"`
		Major        int    `json:"major"`
		Minor        int    `json:"minor"`
	}
	if err := decodeValue(target.Value, &repTarget); err != nil {
		return fmt.Errorf("%w: report target: %v", ErrCalibration, err)
	}
	if obsMajor != repTarget.Major || obsMinor != repTarget.Minor {
		return fmt.Errorf("%w: observation architecture %q does not match report target %q",
			ErrCalibration, obs.Identity.Architecture, repTarget.Architecture)
	}

	var repSuffix string
	if sf := r.Feature("feature_suffix"); sf.Known() {
		_ = decodeValue(sf.Value, &repSuffix)
	}
	if obsSuffix != "" && repSuffix != "" && obsSuffix != repSuffix {
		return fmt.Errorf("%w: observation suffix %q does not match report SASS suffix %q",
			ErrCalibration, obsSuffix, repSuffix)
	}

	modelMajor, modelMinor, modelSuffix, err := cudakernel.ParseArch(m.Ceiling.Architecture)
	if err != nil {
		return fmt.Errorf("%w: invalid model architecture %q: %v", ErrCalibration, m.Ceiling.Architecture, err)
	}
	if modelMajor != obsMajor || modelMinor != obsMinor {
		return fmt.Errorf("%w: model architecture %q does not match observation architecture %q",
			ErrCalibration, m.Ceiling.Architecture, obs.Identity.Architecture)
	}
	if modelSuffix != "" && modelSuffix != obsSuffix && modelSuffix != repSuffix {
		return fmt.Errorf("%w: model architecture %q claims suffix %q not established by observation %q or report SASS",
			ErrCalibration, m.Ceiling.Architecture, modelSuffix, obs.Identity.Architecture)
	}
	return nil
}

func formatCalibrationDevice(id cudacensus.Identity) string {
	return fmt.Sprintf("%s %s, driver %s", strings.TrimSpace(id.DeviceName), strings.TrimSpace(id.Architecture), strings.TrimSpace(id.DriverVersion))
}
