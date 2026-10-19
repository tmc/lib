package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/egraphcore"
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
	// Compile borrows the expression through successful Close of its returned
	// executable, including an owner returned with an error. It must release
	// any retained expression references when Close succeeds. Failed cleanup
	// keeps the expression and its reservations alive until evaluator Close.
	Compile func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error)
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
	// FailureEvidence optionally exports actual candidate comparison failures
	// and checks caller-owned offline quarantine after compile-owner registration.
	FailureEvidence *FailureEvidence
	// Receipts, Comparison and Preparation retain the last admitted attempt,
	// including failures, until another attempt is admitted or Close succeeds.
	// With capped search accounting their retained metadata remains reserved;
	// callers must Close the evaluator independently of returned candidates.
	Receipts           []cudatrial.Receipt
	Comparison         map[string]OutputComparison
	Preparation        map[string][]PreparationPhase
	pending            []Executable
	pendingOutput      io.Closer
	attemptMemory      *egraphcore.MemoryScope
	pendingExtraction  *egraph.Extraction
	pendingMemory      *egraphcore.MemoryScope
	memoryReservations []*memoryReservation
	attemptIdentity    string
	comparisonArm      string
}

// Close retries destruction of candidates retained after cleanup failure.
// The borrowed baseline remains the caller's responsibility.
func (e *ExecutableEvaluator) Close() error {
	if e.pendingOutput != nil {
		if err := e.pendingOutput.Close(); err != nil {
			return err
		}
		e.pendingOutput = nil
	}
	for len(e.pending) > 0 {
		if err := e.pending[0].Close(); err != nil {
			return err
		}
		e.pending = e.pending[1:]
	}
	e.pending = nil
	e.pendingExtraction.Close()
	e.pendingExtraction = nil
	e.pendingMemory.Close()
	e.pendingMemory = nil
	for _, reservation := range e.memoryReservations {
		reservation.release()
	}
	e.memoryReservations = nil
	e.attemptMemory.Close()
	e.attemptMemory = nil
	e.Receipts = nil
	e.Comparison = nil
	e.Preparation = nil
	e.attemptIdentity = ""
	e.comparisonArm = ""
	return nil
}

// Evaluate implements Evaluator. A correctness failure prevents timing. Every
// run resets workload state; unavailable or mixed device timing is rejected.
func (e *ExecutableEvaluator) Evaluate(ctx context.Context, s *egraph.SearchState, root egraph.Id) (candidate *CandidatePlan, err error) {
	return e.evaluate(ctx, s, root, nil)
}

func (e *ExecutableEvaluator) evaluate(ctx context.Context, s *egraph.SearchState, root egraph.Id, selection *egraph.Expr) (candidate *CandidatePlan, err error) {
	if len(e.pending) != 0 || e.pendingOutput != nil {
		return nil, fmt.Errorf("cudasearch: candidate cleanup pending")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := e.beginAttempt(ctx); err != nil {
		return nil, err
	}
	resultMemory, err := mockMemoryScope(ctx)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil || candidate == nil {
			resultMemory.Close()
		}
	}()
	if e.CompileTimeout < 0 {
		return nil, fmt.Errorf("cudasearch: negative compile timeout")
	}
	if c := e.FailureEvidence; c != nil && ((c.Recorder == nil) != (c.Writer == nil)) {
		return nil, fmt.Errorf("cudasearch: incomplete failure evidence exporter")
	}
	if c := e.FailureEvidence; c != nil && c.Quarantine != nil && c.Quarantine.maxBytes == 0 {
		return nil, fmt.Errorf("cudasearch: uninitialized quarantine")
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
	baselineMemory, memoryErr := reserveBorrowedExecutable(ctx, e.Baseline)
	if memoryErr != nil {
		return nil, memoryErr
	}
	defer baselineMemory.release()
	model := e.CostModel
	if model == nil {
		model = egraph.AstSizeCost{}
	}
	expr := selection
	var extraction *egraph.Extraction
	var derivation []egraph.Action
	if expr == nil {
		if err := chargeWork(ctx, false); err != nil {
			return nil, err
		}
		extraction, err = extractSearch(ctx, s, root, model)
		if err != nil {
			return nil, err
		}
		expr = extraction.Expr
		defer func() {
			if candidate == nil || err != nil {
				extraction.Close()
			}
		}()
		derivationBytes, memoryErr := storageBytesWithMemory(s.Derivation, resultMemory)
		if memoryErr != nil && searchMemoryBudget(ctx) != nil && searchMemoryBudget(ctx).limit > 0 {
			return nil, memoryErr
		}
		memoryErr = resultMemory.Reserve(derivationBytes)
		if memoryErr != nil {
			return nil, memoryErr
		}
		derivation = make([]egraph.Action, len(s.Derivation))
		copy(derivation, s.Derivation)
		for i := range derivation {
			derivation[i].Subst = maps.Clone(derivation[i].Subst)
			derivation[i].Binding = maps.Clone(derivation[i].Binding)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cleanupSlot, cleanupMemory, err := reservePendingExecutable(ctx)
	if err != nil {
		return nil, err
	}
	cleanupTransferred := false
	defer func() {
		if !cleanupTransferred {
			cleanupMemory.Close()
		}
	}()
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
		defer func() {
			if closeErr := executable.Close(); closeErr != nil {
				if cleanupSlot == nil {
					e.pending = append(e.pending, executable)
				} else {
					cleanupSlot[0] = executable
					e.pending = cleanupSlot
				}
				e.pendingExtraction = extraction
				e.pendingMemory = cleanupMemory
				extraction = nil
				cleanupTransferred = true
				candidate = nil
				err = errors.Join(err, fmt.Errorf("cudasearch: close candidate: %w", closeErr))
			}
		}()
	}
	if executable != nil {
		id := executable.Identity()
		if memoryErr := e.attemptMemory.Reserve(int64(len(id))); memoryErr != nil {
			err = errors.Join(err, memoryErr)
		} else {
			e.attemptIdentity = strings.Clone(id)
		}
	}
	if err != nil {
		return nil, err
	}
	if executable == nil || executable.Identity() == "" {
		return nil, fmt.Errorf("cudasearch: compiler returned no executable identity")
	}
	association, associationMemory, err := e.candidateAssociation(ctx, executable.Identity())
	if err != nil {
		return nil, err
	}
	defer associationMemory.release()
	if e.FailureEvidence != nil {
		if err := e.FailureEvidence.Quarantine.check(association); err != nil {
			return nil, err
		}
	}
	for _, arm := range []struct {
		name string
		x    Executable
	}{{"baseline", e.Baseline}, {"candidate", executable}} {
		if _, ok := arm.x.(preparationReporter); ok {
			phases, copyErr := executablePreparation(e.attemptMemory, arm.x)
			if copyErr != nil {
				return nil, copyErr
			}
			e.Preparation[arm.name] = phases
		}
	}
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
		if err := e.compareExecutable(ctx, arm, i == 1, association); err != nil {
			return nil, err
		}
	}
	makeArm := func(name string, x Executable) cudatrial.Arm {
		return cudatrial.Arm{Name: name, Run: func(ctx context.Context) (cudatrial.Receipt, error) {
			if err := reset(); err != nil {
				return cudatrial.Receipt{}, err
			}
			r, err := x.Run(ctx)
			detached, copyErr := r.CloneWithMemory(e.attemptMemory.Reserve)
			if copyErr != nil {
				return r, errors.Join(err, r.Err, copyErr)
			}
			r = detached
			if err != nil {
				return r, errors.Join(err, r.Err)
			}
			if !r.Complete || r.Err != nil {
				return r, errors.Join(fmt.Errorf("cudasearch: incomplete execution receipt"), r.Err)
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
			id := x.Identity()
			if reserveErr := e.attemptMemory.Reserve(256 + int64(len(id))); reserveErr != nil {
				return r, reserveErr
			}
			r.Provenance["executable_identity"] = strings.Clone(id)
			return r, nil
		}}
	}
	e.Receipts, err = cudatrial.InterleaveWithOptions(ctx, []cudatrial.Arm{makeArm("baseline", e.Baseline), makeArm("candidate", executable)}, e.Rounds, cudatrial.Options{RequireHardwareLock: true, HardwareLockChecker: e.HardwareLockChecker, ReserveMemory: e.attemptMemory.Reserve})
	if err != nil {
		return nil, err
	}
	if err := bookkeepingSlots[time.Duration](e.attemptMemory, e.Rounds); err != nil {
		return nil, err
	}
	durations := make([]time.Duration, 0, e.Rounds)
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
	ledger := searchMemoryBudget(ctx)
	if extraction == nil {
		ledger = nil
	}
	if err := bookkeepingSlots[CandidatePlan](resultMemory, 1); err != nil {
		return nil, err
	}
	if err := bookkeepingSlots[cudatrial.Receipt](resultMemory, 1); err != nil {
		return nil, err
	}
	candidate = &CandidatePlan{Generation: e.Generation, WorkloadKey: e.Workload.Key(), InputKey: inputKey(e.Workload), VerificationPolicy: e.Policy, ID: executable.Identity(), Expr: expr, Derivation: derivation, Cost: float64(median.Nanoseconds()), CostSource: CostMeasured, Receipt: &selected, Trials: nil, Validated: true, Comparison: nil, Preparation: e.Preparation["candidate"], extraction: extraction, extractionLedger: ledger, metadata: resultMemory, metadataLedger: searchMemoryBudget(ctx)}
	if err := e.detachCandidate(resultMemory, candidate); err != nil {
		return nil, err
	}
	return candidate, nil
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
