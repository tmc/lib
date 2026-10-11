package cudaregress

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// ReplayRunner executes one arm invocation within a trial round.
type ReplayRunner interface {
	RunArm(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error)
}

// ReplayFunc adapts a function to the ReplayRunner interface.
type ReplayFunc func(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error)

// RunArm calls the adapted ReplayFunc.
func (f ReplayFunc) RunArm(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
	return f(ctx, armName, round)
}

// ReplayOptions configures preflight validation and execution policies for replay.
type ReplayOptions struct {
	CurrentEnvironment Environment
	AllowDriftFork     bool
	DriftRationale     string
	BaselineFamilies   []string
	CandidateFamilies  []string
	BaselineResources  map[string]string
	CandidateResources map[string]string
	ReferenceValues    []float64
}

// CheckNumericalCorrectness compares candidate outputs against reference outputs under a tolerance bound.
func CheckNumericalCorrectness(reference, candidate []float64, tolerance float64) error {
	if math.IsNaN(tolerance) || math.IsInf(tolerance, 0) || tolerance < 0 {
		return errors.New("cudaregress: invalid numerical tolerance")
	}
	if len(reference) == 0 || len(candidate) == 0 {
		return errors.New("cudaregress: empty output vector for oracle evaluation")
	}
	if len(reference) != len(candidate) {
		return fmt.Errorf("cudaregress: output length mismatch: reference has %d, candidate has %d",
			len(reference), len(candidate))
	}
	for i, refVal := range reference {
		candVal := candidate[i]
		if math.IsNaN(refVal) || math.IsNaN(candVal) {
			return fmt.Errorf("cudaregress: NaN encountered at index %d", i)
		}
		if math.IsInf(refVal, 0) || math.IsInf(candVal, 0) {
			return fmt.Errorf("cudaregress: Inf encountered at index %d", i)
		}
		diff := math.Abs(refVal - candVal)
		if diff > tolerance {
			return fmt.Errorf("%w: at index %d diff %g > tolerance %g (ref=%v, cand=%v)",
				ErrCorrectnessOracleFailed, i, diff, tolerance, refVal, candVal)
		}
	}
	return nil
}

// SafeCommandArm constructs a verified trial arm that directly executes a declared binary.
// It verifies that binaryPath corresponds to a declared and digest-verified binary artifact,
// and executes directly without shell interpolation.
func SafeCommandArm(plan *ExperimentPlan, name, binaryPath string, args ...string) (cudatrial.Arm, error) {
	if plan == nil {
		return cudatrial.Arm{}, errors.New("cudaregress: nil plan")
	}
	if err := ValidatePlan(plan); err != nil {
		return cudatrial.Arm{}, err
	}

	cleanBinary := filepath.Clean(binaryPath)
	found := false
	for _, b := range plan.BinaryArtifacts {
		if filepath.Clean(b.Path) == cleanBinary {
			found = true
			break
		}
	}
	if !found {
		return cudatrial.Arm{}, fmt.Errorf("%w: binary %q not in declared binary artifacts",
			ErrUnsafeCommandRefused, cleanBinary)
	}

	for _, arg := range args {
		if strings.Contains(arg, ";") || strings.Contains(arg, "|") || strings.Contains(arg, "`") || strings.Contains(arg, "$(") {
			return cudatrial.Arm{}, fmt.Errorf("%w: shell meta-character in argument %q",
				ErrUnsafeCommandRefused, arg)
		}
	}

	return cudatrial.Arm{
		Name: name,
		Run: func(ctx context.Context) (cudatrial.Receipt, error) {
			if err := ValidatePlan(plan); err != nil {
				return cudatrial.Receipt{}, err
			}

			cmd := exec.CommandContext(ctx, cleanBinary, args...)
			start := time.Now()
			out, err := cmd.CombinedOutput()
			finish := time.Now()
			if err != nil {
				return cudatrial.Receipt{Complete: false, Err: err}, err
			}
			wall := finish.Sub(start)
			r := cudatrial.Receipt{
				Name:        name,
				Complete:    true,
				Digest:      fmt.Sprintf("%x", sha256.Sum256(out)),
				Started:     start,
				Finished:    finish,
				WallElapsed: wall,
				WallNs:      wall.Nanoseconds(),
				TimingKind:  cudatrial.TimingWallElapsed,
			}
			_ = r.RecordTiming(cudatrial.TimingWallElapsed, wall)
			return r, nil
		},
	}, nil
}

// Replay executes the experiment plan across warmup, null control, and interleaved rounds,
// returning separated execution and comparison receipts.
func Replay(ctx context.Context, plan *ExperimentPlan, runner ReplayRunner, opts ReplayOptions) (*ExecutionReceipt, *ComparisonReceipt, error) {
	if plan == nil {
		return nil, nil, errors.New("cudaregress: nil plan")
	}
	if runner == nil {
		return nil, nil, errors.New("cudaregress: nil replay runner")
	}

	// 1. Validate plan structure and artifact closure.
	if err := ValidatePlan(plan); err != nil {
		return nil, nil, err
	}

	if plan.CorrectnessOracle.Method == "numerical-tolerance" && len(opts.ReferenceValues) == 0 {
		return nil, nil, errors.New("cudaregress: numerical oracle requires reference values")
	}

	if plan.CorrectnessOracle.Method == "digest-exact" && plan.CorrectnessOracle.ExpectedDigest == "" {
		return nil, nil, errors.New("cudaregress: digest oracle requires expected digest")
	}

	// 2. Validate execution environment against plan requirements.
	effectivePlan := plan
	if err := ValidateEnvironment(plan, opts.CurrentEnvironment); err != nil {
		if opts.AllowDriftFork {
			forked, forkErr := DerivePlan(plan, opts.CurrentEnvironment, opts.DriftRationale)
			if forkErr != nil {
				return nil, nil, fmt.Errorf("cudaregress: fork failed: %w", forkErr)
			}
			effectivePlan = forked
		} else {
			return nil, nil, err
		}
	}

	// 3. Validate timing channel.
	timingKind := effectivePlan.TimingScope.Channel
	switch timingKind {
	case cudatrial.TimingWallElapsed, cudatrial.TimingSummedKernel, cudatrial.TimingGPUBusy, cudatrial.TimingAPICall:
	default:
		return nil, nil, fmt.Errorf("%w: %q", ErrUnsupportedTimingChannel, timingKind)
	}

	execReceipt := &ExecutionReceipt{
		ExecutionID:         fmt.Sprintf("exec-%s-%d", effectivePlan.PlanID, time.Now().UnixNano()),
		PlanID:              effectivePlan.PlanID,
		PlanDigest:          effectivePlan.PlanDigest,
		StartedAt:           time.Now(),
		EnvironmentObserved: opts.CurrentEnvironment,
		OutputDigests:       make(map[string]string),
	}

	checkReceipt := func(r cudatrial.Receipt, values []float64, err error) error {
		if err != nil {
			return err
		}
		if r.Err != nil {
			return r.Err
		}
		if !r.Complete {
			return errors.New("cudaregress: incomplete receipt")
		}
		switch effectivePlan.CorrectnessOracle.Method {
		case "numerical-tolerance":
			return CheckNumericalCorrectness(opts.ReferenceValues, values, effectivePlan.CorrectnessOracle.Tolerance)
		case "digest-exact":
			if r.Digest != effectivePlan.CorrectnessOracle.ExpectedDigest {
				return ErrCorrectnessOracleFailed
			}
		}
		return nil
	}
	// 4. Warmup passes.
	for w := 0; w < effectivePlan.WarmupRuns; w++ {
		if err := ctx.Err(); err != nil {
			execReceipt.FinishedAt = time.Now()
			execReceipt.Partial = true
			execReceipt.HaltReason = err.Error()
			return execReceipt, nil, fmt.Errorf("%w: warmup interrupted: %v", ErrExecutionHalted, err)
		}
		r, values, err := runner.RunArm(ctx, "baseline", -(w + 1))
		err = checkReceipt(r, values, err)
		if err != nil {
			execReceipt.FinishedAt = time.Now()
			execReceipt.Partial = true
			execReceipt.HaltReason = fmt.Sprintf("warmup failed: %v", err)
			return execReceipt, nil, fmt.Errorf("%w: warmup baseline failed: %v", ErrExecutionHalted, err)
		}
		r, values, err = runner.RunArm(ctx, "candidate", -(w + 1))
		err = checkReceipt(r, values, err)
		if err != nil {
			execReceipt.FinishedAt = time.Now()
			execReceipt.Partial = true
			execReceipt.HaltReason = fmt.Sprintf("warmup failed: %v", err)
			return execReceipt, nil, fmt.Errorf("%w: warmup candidate failed: %v", ErrExecutionHalted, err)
		}
		execReceipt.WarmupRunsCompleted++
	}

	// 5. Baseline-vs-Baseline Null Control to calibrate empirical noise floor.
	if effectivePlan.NullControls.Enabled && effectivePlan.NullControls.BasketRuns > 1 {
		var nullDurs []time.Duration
		for n := 1; n <= effectivePlan.NullControls.BasketRuns; n++ {
			if err := ctx.Err(); err != nil {
				execReceipt.FinishedAt = time.Now()
				execReceipt.Partial = true
				execReceipt.HaltReason = err.Error()
				return execReceipt, nil, fmt.Errorf("%w: null control interrupted: %v", ErrExecutionHalted, err)
			}
			nr, values, err := runner.RunArm(ctx, "baseline", n)
			err = checkReceipt(nr, values, err)
			if err != nil {
				execReceipt.FinishedAt = time.Now()
				execReceipt.Partial = true
				execReceipt.HaltReason = fmt.Sprintf("null control failed: %v", err)
				return execReceipt, nil, fmt.Errorf("%w: null control failed: %v", ErrExecutionHalted, err)
			}
			nr.Round = n
			execReceipt.NullControlReceipts = append(execReceipt.NullControlReceipts, nr)
			dur, timingErr := nr.DurationFor(timingKind)
			if timingErr != nil {
				return execReceipt, nil, timingErr
			}
			nullDurs = append(nullDurs, dur)
		}
		if len(nullDurs) >= 2 {
			sort.Slice(nullDurs, func(i, j int) bool { return nullDurs[i] < nullDurs[j] })
			execReceipt.NullDrift = nullDurs[len(nullDurs)-1] - nullDurs[0]
		}
	}

	// 6. Interleaved Arm Execution.
	arms := []cudatrial.Arm{{Name: "baseline"}, {Name: "candidate"}}

	consecutiveFailures := 0
	maxFailures := effectivePlan.StoppingRules.MaxConsecutiveFailures
	if maxFailures <= 0 {
		maxFailures = 3
	}

	for round := 1; round <= effectivePlan.Rounds; round++ {
		roundArms := arms
		if round%2 == 0 && effectivePlan.ExecutionOrder == "ABBA" {
			roundArms = []cudatrial.Arm{arms[1], arms[0]}
		}

		for _, arm := range roundArms {
			if err := ctx.Err(); err != nil {
				execReceipt.FinishedAt = time.Now()
				execReceipt.Partial = true
				execReceipt.HaltReason = err.Error()
				return execReceipt, nil, fmt.Errorf("%w: %v", ErrExecutionHalted, err)
			}

			r, candVals, err := runner.RunArm(ctx, arm.Name, round)
			if err == nil {
				err = r.Err
			}
			if err == nil && !r.Complete {
				err = errors.New("incomplete arm receipt")
			}
			r.Round = round
			r.Name = arm.Name
			if err != nil {
				r.Complete = false
				r.Err = err
				execReceipt.ArmReceipts = append(execReceipt.ArmReceipts, r)
				consecutiveFailures++
				execReceipt.Partial = true
				if consecutiveFailures >= maxFailures {
					execReceipt.FinishedAt = time.Now()
					execReceipt.Partial = true
					execReceipt.HaltReason = fmt.Sprintf("exceeded max consecutive failures (%d): %v", maxFailures, err)
					return execReceipt, nil, fmt.Errorf("%w: %s", ErrExecutionHalted, execReceipt.HaltReason)
				}
				continue
			}

			consecutiveFailures = 0

			if r.Digest != "" {
				execReceipt.OutputDigests[arm.Name] = r.Digest
			}

			// Validate correctness oracle
			{
				oracle := effectivePlan.CorrectnessOracle
				switch oracle.Method {
				case "digest-exact":
					if oracle.ExpectedDigest != "" && r.Digest != oracle.ExpectedDigest {
						r.Complete = false
						r.Err = fmt.Errorf("%w: digest %q != expected %q",
							ErrCorrectnessOracleFailed, r.Digest, oracle.ExpectedDigest)
						execReceipt.ArmReceipts = append(execReceipt.ArmReceipts, r)
						execReceipt.FinishedAt = time.Now()
						execReceipt.Partial = true
						execReceipt.HaltReason = r.Err.Error()
						return execReceipt, nil, r.Err
					}
				case "numerical-tolerance":
					if len(opts.ReferenceValues) > 0 {
						tol := oracle.Tolerance
						if oErr := CheckNumericalCorrectness(opts.ReferenceValues, candVals, tol); oErr != nil {
							r.Complete = false
							r.Err = oErr
							execReceipt.ArmReceipts = append(execReceipt.ArmReceipts, r)
							execReceipt.FinishedAt = time.Now()
							execReceipt.Partial = true
							execReceipt.HaltReason = oErr.Error()
							return execReceipt, nil, oErr
						}
					}
				}
			}

			execReceipt.ArmReceipts = append(execReceipt.ArmReceipts, r)
		}
		execReceipt.RoundsCompleted++
	}

	execReceipt.FinishedAt = time.Now()
	execReceipt.OutputsVerified = !execReceipt.Partial && (effectivePlan.CorrectnessOracle.Method == "numerical-tolerance" || effectivePlan.CorrectnessOracle.Method == "digest-exact")

	// 7. Produce ComparisonReceipt.
	compReceipt, err := CompareExecution(effectivePlan, execReceipt, opts)
	if err != nil {
		return execReceipt, nil, err
	}

	return execReceipt, compReceipt, nil
}

// CompareExecution analyzes completed trial receipts, evaluating differential performance,
// noise floor significance, missing architectural families, and resource divergence.
func CompareExecution(plan *ExperimentPlan, exec *ExecutionReceipt, opts ReplayOptions) (*ComparisonReceipt, error) {
	if plan == nil || exec == nil {
		return nil, errors.New("cudaregress: nil plan or execution receipt")
	}

	timingKind := plan.TimingScope.Channel
	var baseDurs, candDurs []time.Duration

	for _, r := range exec.ArmReceipts {
		if !r.Complete {
			continue
		}
		dur, err := r.DurationFor(timingKind)
		if err != nil {
			return nil, fmt.Errorf("cudaregress: missing requested timing channel: %w", err)
		}
		switch r.Name {
		case "baseline":
			baseDurs = append(baseDurs, dur)
		case "candidate":
			candDurs = append(candDurs, dur)
		}
	}

	comp := &ComparisonReceipt{
		ExecutionID:         exec.ExecutionID,
		PlanID:              plan.PlanID,
		EvaluatedAt:         time.Now(),
		BaselineArm:         "baseline",
		CandidateArm:        "candidate",
		TimingChannel:       timingKind,
		NullDriftNoiseFloor: exec.NullDrift,
		Verdict:             cudaprofilehistory.Unknown,
	}

	// Architectural families preservation: distinguish missing families from measured zero.
	baseFamMap := make(map[string]bool)
	candFamMap := make(map[string]bool)
	for _, f := range opts.BaselineFamilies {
		baseFamMap[f] = true
	}
	for _, f := range opts.CandidateFamilies {
		candFamMap[f] = true
	}

	for f := range baseFamMap {
		if !candFamMap[f] {
			comp.MissingFamilies = append(comp.MissingFamilies, f)
		}
	}
	for f := range candFamMap {
		if !baseFamMap[f] {
			comp.MissingFamilies = append(comp.MissingFamilies, f)
		}
	}
	sort.Strings(comp.MissingFamilies)

	// Resource changes preservation: keep hardware/resource differences strictly separate from performance.
	allResources := make(map[string]bool)
	for k := range opts.BaselineResources {
		allResources[k] = true
	}
	for k := range opts.CandidateResources {
		allResources[k] = true
	}
	resKeys := make([]string, 0, len(allResources))
	for k := range allResources {
		resKeys = append(resKeys, k)
	}
	sort.Strings(resKeys)

	for _, k := range resKeys {
		bv := opts.BaselineResources[k]
		cv := opts.CandidateResources[k]
		if bv != cv {
			comp.ResourceChanges = append(comp.ResourceChanges, ResourceChange{
				Resource:  k,
				Baseline:  bv,
				Candidate: cv,
			})
		}
	}

	minSamples := plan.StoppingRules.MinSamples
	if minSamples <= 0 {
		minSamples = 3
	}

	if len(baseDurs) < minSamples || len(candDurs) < minSamples {
		comp.Reason = fmt.Sprintf("insufficient samples: baseline has %d, candidate has %d (minimum required: %d)",
			len(baseDurs), len(candDurs), minSamples)
		return comp, nil
	}

	medianDuration := func(durs []time.Duration) time.Duration {
		sorted := make([]time.Duration, len(durs))
		copy(sorted, durs)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		return sorted[len(sorted)/2]
	}

	baseMed := medianDuration(baseDurs)
	candMed := medianDuration(candDurs)

	comp.BaselineDuration = baseMed
	comp.CandidateDuration = candMed
	comp.Delta = candMed - baseMed
	if baseMed > 0 {
		comp.Ratio = float64(candMed) / float64(baseMed)
	}

	comp.PerformanceChanges = append(comp.PerformanceChanges, PerformanceChange{
		Channel:   timingKind,
		Baseline:  baseMed,
		Candidate: candMed,
		Delta:     comp.Delta,
		Ratio:     comp.Ratio,
	})

	absDelta := comp.Delta
	if absDelta < 0 {
		absDelta = -absDelta
	}
	comp.ExceedsNoiseFloor = absDelta > exec.NullDrift

	relThresh := plan.StoppingRules.RelativeThreshold

	// Verdict evaluation
	if exec.Partial || !exec.OutputsVerified || len(comp.MissingFamilies) != 0 || len(comp.ResourceChanges) != 0 || baseMed <= 0 || candMed <= 0 {
		comp.Reason = "comparison withheld: incomplete, unverified, or changed evidence"
		return comp, nil
	}
	if !comp.ExceedsNoiseFloor && exec.NullDrift > 0 {
		comp.Verdict = cudaprofilehistory.Unknown
		comp.Reason = fmt.Sprintf("duration delta (%v) within null control noise floor (%v)", comp.Delta, exec.NullDrift)
		return comp, nil
	}

	if comp.Ratio > 1.0+relThresh {
		comp.RegressionDetected = true
		comp.Verdict = cudaprofilehistory.Fail
		comp.Reason = fmt.Sprintf("slowdown ratio %.2fx exceeds regression threshold %.2fx (delta %v)",
			comp.Ratio, 1.0+relThresh, comp.Delta)
	} else {
		comp.RegressionDetected = false
		comp.Verdict = cudaprofilehistory.Pass
		comp.Reason = fmt.Sprintf("performance ratio %.2fx within acceptable threshold %.2fx",
			comp.Ratio, 1.0+relThresh)
	}

	return comp, nil
}
