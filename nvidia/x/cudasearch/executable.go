package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// Executable is a compiled candidate over private workload buffers. Outputs
// waits for execution and readback. Run measures device execution through the
// requested timing channel; it must not substitute host elapsed time. Close
// drains work and retains failed cleanup resources for retry. Identity includes
// compiled artifacts and launch configuration, not floating-point output bytes.
type Executable interface {
	Identity() string
	Outputs(context.Context) (map[string][]float32, error)
	Run(context.Context) (cudatrial.Receipt, error)
	Close() error
}

// OutputPolicy declares the comparison abs(got-want) <= Absolute+Relative*abs(want).
// Zero tolerances require exact finite values; signed zeros differ unless
// IgnoreZeroSign is set. Infinities must match in sign. NaNs require NaNsEqual.
type OutputPolicy struct {
	Absolute, Relative        float64
	NaNsEqual, IgnoreZeroSign bool
}

// OutputComparison records the observed deviation, including rejected outputs.
type OutputComparison struct {
	Elements                 int
	MaxAbsolute, MaxRelative float64
}

// ExecutableEvaluator compiles extracted expressions, verifies both arms, and
// measures them in ABBA order. Baseline is borrowed; compiled candidates are
// closed after evaluation. Failed cleanup remains owned until Close succeeds.
// It is not safe for concurrent use or mutation during Evaluate. References
// must cover every externally observable output, including secondary consumers.
// Held-out winner confirmation and promotion remain separate steps.
type ExecutableEvaluator struct {
	Generation int // Execution-owner generation; zero permits evaluation but not promotion.
	Workload   WorkloadBindings
	CostModel  egraph.CostModel
	Compile    func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error)
	// CompileTimeout limits compilation through its context. Zero uses the
	// caller's deadline. Compile must honor cancellation; synchronous native
	// calls cannot be forcibly interrupted. Late results are closed and rejected.
	CompileTimeout time.Duration
	Baseline       Executable
	References     map[string][]float32
	Policy         OutputPolicy
	Rounds         int // even, in [2,1024]
	TimingKind     cudatrial.TimingKind
	// HardwareLockChecker must verify a live lease, not only an environment flag.
	HardwareLockChecker func() bool
	// Receipts and Comparison retain the last attempt, including failures.
	Receipts        []cudatrial.Receipt
	Comparison      map[string]OutputComparison
	Preparation     map[string][]PreparationPhase
	pending         []Executable
	attemptIdentity string
	comparisonArm   string
}

// Close retries destruction of candidates retained after cleanup failure.
// The borrowed baseline remains the caller's responsibility.
func (e *ExecutableEvaluator) Close() error {
	for len(e.pending) > 0 {
		if err := e.pending[0].Close(); err != nil {
			return err
		}
		e.pending = e.pending[1:]
	}
	return nil
}

// Evaluate implements Evaluator. A correctness failure prevents timing. Every
// run resets workload state; unavailable or mixed device timing is rejected.
func (e *ExecutableEvaluator) Evaluate(ctx context.Context, s *egraph.SearchState, root egraph.Id) (candidate *CandidatePlan, err error) {
	return e.evaluate(ctx, s, root, nil)
}

func (e *ExecutableEvaluator) evaluate(ctx context.Context, s *egraph.SearchState, root egraph.Id, selection *egraph.Expr) (candidate *CandidatePlan, err error) {
	e.Receipts = nil
	e.attemptIdentity = ""
	e.comparisonArm = ""
	e.Comparison = make(map[string]OutputComparison)
	e.Preparation = make(map[string][]PreparationPhase)
	if len(e.pending) != 0 {
		return nil, fmt.Errorf("cudasearch: candidate cleanup pending")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e.CompileTimeout < 0 {
		return nil, fmt.Errorf("cudasearch: negative compile timeout")
	}
	if e.Generation < 0 {
		return nil, ErrStaleGeneration
	}
	if (selection == nil && (s == nil || s.Graph == nil)) || e.Compile == nil || e.Baseline == nil || e.Baseline.Identity() == "" || len(e.References) == 0 {
		return nil, fmt.Errorf("cudasearch: incomplete executable evaluator")
	}
	if err := e.Workload.Validate(); err != nil {
		return nil, err
	}
	if e.Rounds < 2 || e.Rounds > 1024 || e.Rounds%2 != 0 {
		return nil, fmt.Errorf("cudasearch: ABBA rounds must be even and in [2,1024]")
	}
	if e.TimingKind != cudatrial.TimingSummedKernel && e.TimingKind != cudatrial.TimingGPUBusy {
		return nil, ErrTimingKindMismatch
	}
	if err := e.Policy.validate(); err != nil {
		return nil, err
	}
	if e.Workload.NumericalPolicy == NumericalPolicyExact && (e.Policy.Absolute != 0 || e.Policy.Relative != 0) {
		return nil, fmt.Errorf("cudasearch: nonzero tolerance conflicts with exact workload policy")
	}
	if objective := e.Workload.Objective; objective != "" && objective != ObjectiveMinLatency {
		return nil, fmt.Errorf("%w: unsupported executable objective %q", ErrMissingBinding, objective)
	}
	if e.HardwareLockChecker == nil || !e.HardwareLockChecker() {
		return nil, cudatrial.ErrHardwareLockRequired
	}
	model := e.CostModel
	if model == nil {
		model = egraph.AstSizeCost{}
	}
	expr := selection
	var derivation []egraph.Action
	if expr == nil {
		if err := chargeWork(ctx, false); err != nil {
			return nil, err
		}
		expr, _, err = extractSearch(ctx, s, root, model)
		if err != nil {
			return nil, err
		}
		derivation = s.Clone().Derivation
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := chargeWork(ctx, true); err != nil {
		return nil, err
	}
	compileCtx := ctx
	cancelCompile := func() {}
	if e.CompileTimeout > 0 {
		compileCtx, cancelCompile = context.WithTimeout(ctx, e.CompileTimeout)
	}
	executable, err := e.Compile(compileCtx, expr, e.Workload)
	// Inspect the deadline before canceling our child context. A compiler may
	// return an owner even when it reports failure or ignores cancellation.
	if contextErr := compileCtx.Err(); contextErr != nil {
		err = errors.Join(err, fmt.Errorf("cudasearch: compile: %w", contextErr))
	}
	cancelCompile()
	if executable != nil {
		e.attemptIdentity = executable.Identity()
		defer func() {
			if closeErr := executable.Close(); closeErr != nil {
				e.pending = append(e.pending, executable)
				candidate = nil
				err = errors.Join(err, fmt.Errorf("cudasearch: close candidate: %w", closeErr))
			}
		}()
	}
	if err != nil {
		return nil, err
	}
	if executable == nil || executable.Identity() == "" {
		return nil, fmt.Errorf("cudasearch: compiler returned no executable identity")
	}
	e.Preparation["baseline"] = preparationOf(e.Baseline)
	e.Preparation["candidate"] = preparationOf(executable)
	reset := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !e.HardwareLockChecker() {
			return cudatrial.ErrHardwareLockRequired
		}
		if e.Workload.ResetState != nil {
			return e.Workload.ResetState()
		}
		return nil
	}
	for i, arm := range []Executable{e.Baseline, executable} {
		if err := reset(); err != nil {
			return nil, err
		}
		e.comparisonArm = []string{"baseline", "candidate"}[i]
		e.Comparison = nil
		outputs, err := arm.Outputs(ctx)
		if err != nil {
			return nil, err
		}
		comparison, err := e.Policy.compare(outputs, e.References)
		e.Comparison = comparison
		if err != nil {
			return nil, err
		}
	}
	makeArm := func(name string, x Executable) cudatrial.Arm {
		return cudatrial.Arm{Name: name, Run: func(ctx context.Context) (cudatrial.Receipt, error) {
			if err := reset(); err != nil {
				return cudatrial.Receipt{}, err
			}
			r, err := x.Run(ctx)
			if err != nil {
				return r, err
			}
			if !r.Complete || r.Err != nil {
				return r, fmt.Errorf("cudasearch: incomplete execution receipt")
			}
			if r.TimingKind != e.TimingKind {
				return r, ErrTimingKindMismatch
			}
			d, err := r.DurationFor(e.TimingKind)
			if err != nil {
				return r, err
			}
			if d <= 0 {
				return r, fmt.Errorf("cudasearch: nonpositive measured duration")
			}
			r.Timings = maps.Clone(r.Timings)
			r.Provenance = maps.Clone(r.Provenance)
			if r.Provenance == nil {
				r.Provenance = make(map[string]string)
			}
			r.Provenance["executable_identity"] = x.Identity()
			return r, nil
		}}
	}
	e.Receipts, err = cudatrial.InterleaveWithOptions(ctx, []cudatrial.Arm{makeArm("baseline", e.Baseline), makeArm("candidate", executable)}, e.Rounds, cudatrial.Options{RequireHardwareLock: true, HardwareLockChecker: e.HardwareLockChecker})
	if err != nil {
		return nil, err
	}
	var durations []time.Duration
	var selected cudatrial.Receipt
	for _, r := range e.Receipts {
		if r.Name == "candidate" {
			d, _ := r.DurationFor(e.TimingKind)
			durations = append(durations, d)
		}
	}
	sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
	median := durations[len(durations)/2]
	for _, r := range e.Receipts {
		d, _ := r.DurationFor(e.TimingKind)
		if r.Name == "candidate" && d == median {
			selected = r
			break
		}
	}
	trials := cloneTrials(e.Receipts)
	selected.Timings = maps.Clone(selected.Timings)
	selected.Provenance = maps.Clone(selected.Provenance)
	return &CandidatePlan{Generation: e.Generation, WorkloadKey: e.Workload.Key(), InputKey: inputKey(e.Workload), VerificationPolicy: e.Policy, ID: executable.Identity(), Expr: expr, Derivation: derivation, Cost: float64(median.Nanoseconds()), CostSource: CostMeasured, Receipt: &selected, Trials: trials, Validated: true, Comparison: maps.Clone(e.Comparison), Preparation: preparationOf(executable)}, nil
}

func (p OutputPolicy) validate() error {
	if p.Absolute < 0 || p.Relative < 0 || math.IsNaN(p.Absolute) || math.IsNaN(p.Relative) || math.IsInf(p.Absolute, 0) || math.IsInf(p.Relative, 0) {
		return fmt.Errorf("cudasearch: invalid output tolerance")
	}
	return nil
}

func (p OutputPolicy) compare(got, want map[string][]float32) (map[string]OutputComparison, error) {
	stats := make(map[string]OutputComparison)
	if len(got) != len(want) {
		return stats, fmt.Errorf("%w: output set differs", ErrVerificationFailed)
	}
	keys := make([]string, 0, len(want))
	for key := range want {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var failure error
	for _, name := range keys {
		expected := want[name]
		actual, ok := got[name]
		if !ok || len(actual) != len(expected) {
			return stats, fmt.Errorf("%w: output %q extent differs", ErrVerificationFailed, name)
		}
		var s OutputComparison
		for i, w := range expected {
			g := actual[i]
			s.Elements++
			valid := true
			switch {
			case math.IsNaN(float64(g)) || math.IsNaN(float64(w)):
				valid = p.NaNsEqual && math.IsNaN(float64(g)) && math.IsNaN(float64(w))
			case math.IsInf(float64(g), 0) || math.IsInf(float64(w), 0):
				valid = g == w
			case g == 0 && w == 0:
				valid = p.IgnoreZeroSign || math.Float32bits(g) == math.Float32bits(w)
			default:
				d := math.Abs(float64(g) - float64(w))
				rel := 0.0
				if w != 0 {
					rel = d / math.Abs(float64(w))
				} else if d != 0 {
					rel = math.Inf(1)
				}
				s.MaxAbsolute = math.Max(s.MaxAbsolute, d)
				s.MaxRelative = math.Max(s.MaxRelative, rel)
				valid = d <= p.Absolute+p.Relative*math.Abs(float64(w))
			}
			if !valid && failure == nil {
				failure = fmt.Errorf("%w: output %q index %d got %g want %g", ErrVerificationFailed, name, i, g, w)
			}
		}
		stats[name] = s
	}
	return stats, failure
}
