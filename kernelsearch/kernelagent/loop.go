package kernelagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// Mode separates synthetic host adapters from native execution admission.
type Mode string

const (
	HostStub Mode = "host-stub"
	Native   Mode = "native"
)

// ErrDeviceFault marks a reported device fault. Its cause remains unknown;
// the loop stops the epoch and quarantines the attempt without banning a rule.
var ErrDeviceFault = errors.New("kernelagent: device fault; causal proposal unknown")

// DeviceState records per-measurement conditions. Nil means unavailable, not
// zero. Source must identify the observer, and synthetic host adapters must
// label it synthetic. ThermalThrottle must be known and false for admission.
type DeviceState struct {
	Observed                                    time.Time
	Source                                      string
	ClockMHz, TemperatureC, PowerW, PowerLimitW *float64
	ThermalThrottle                             *bool
}

// Identity is the declared P0 target/tool/lease record. HostStub values describe
// fixtures, not observed GPU identity. A native lease is rechecked before work.
type Identity struct {
	Mode                                       Mode
	GPU, SM, Driver, CUDA, NVRTC, Tools, Lease string
	Initial                                    DeviceState
}

// NumericalDeclaration binds one operation's numerical contract and oracle.
// Tolerances use cudasearch's existing OutputPolicy semantics. Text fields
// declare domain behavior, including cases which are not applicable.
type NumericalDeclaration struct {
	Operation, Oracle                          string
	Policy                                     cudasearch.NumericalPolicy
	Tolerance                                  cudasearch.OutputPolicy
	NaNs, Infinities, SignedZero, Empty, Masks string
}

// Evaluation binds retained input bits and reference outputs to the actual
// search workload. The trusted compiler must use these exact private inputs.
// References must be independently produced by the declared cudaref oracle.
type Evaluation struct {
	Workload           cudasearch.WorkloadBindings
	Inputs, References map[string][]float32
}

// Config supplies trusted owner adapters. A proposer only selects catalog
// entries; it cannot supply source, paths, commands or an executable. Compile,
// Observe and ResetState must honor cancellation and their declared budgets.
// Callbacks are trusted Go code, not a security sandbox or enforced peak cap.
// Budget is passed unchanged to the cudasearch strategy, including its
// extraction, compilation and memory limits.
type Config struct {
	Identity            Identity
	Numerical           NumericalDeclaration
	Screen, HeldOut     Evaluation
	Generation          int
	Baseline            cudasearch.Executable // borrowed; caller retains and closes it
	Compile             func(context.Context, *egraph.Expr, cudasearch.WorkloadBindings) (cudasearch.Executable, error)
	Observe             func(context.Context) (DeviceState, error)
	Model               *cudaroofline.Model
	Operation           cudaroofline.Operation
	CostModel           egraph.CostModel
	Catalog             map[string]*egraph.Rewrite
	Proposer            Proposer
	ExpectedModel       ModelIdentity
	Prompt, EvaluatorID string
	Budget              cudasearch.SearchBudget
	Strategy            string // beam or mcts; mechanics remain in cudasearch
	BeamWidth           int
	Seed                int64
	Rounds              int
	Timing              cudatrial.TimingKind
	CompileTimeout      time.Duration
	MaxEvidenceElements int // bound before copying input/reference FP32 bits
}

// Loop drives sequential search epochs. The zero value is unusable. Close
// retries candidate cleanup; the archive and baseline remain caller-owned.
// A Loop is not safe for concurrent use.
type Loop struct {
	cfg     Config
	archive *Archive
	pending []*cudasearch.ExecutableEvaluator
}

// NewLoop validates the operation, numerical policy, P0 identity and bounds.
// It retains the supplied trusted callbacks; do not mutate them during an epoch.
func NewLoop(a *Archive, cfg Config) (*Loop, error) {
	if a == nil || cfg.Proposer == nil || cfg.Compile == nil || cfg.Observe == nil || cfg.Baseline == nil || cfg.Baseline.Identity() == "" || cfg.Generation <= 0 || cfg.EvaluatorID == "" {
		return nil, fmt.Errorf("kernelagent: incomplete loop configuration")
	}
	if cfg.Identity.Mode != HostStub && cfg.Identity.Mode != Native {
		return nil, fmt.Errorf("kernelagent: invalid execution mode")
	}
	for _, value := range []string{cfg.Identity.GPU, cfg.Identity.SM, cfg.Identity.Driver, cfg.Identity.CUDA, cfg.Identity.NVRTC, cfg.Identity.Tools, cfg.Identity.Lease} {
		if value == "" || len(value) > 1024 {
			return nil, fmt.Errorf("kernelagent: missing or oversized P0 identity")
		}
	}
	if err := validState(cfg.Identity.Initial, cfg.Identity.Mode); err != nil {
		return nil, err
	}
	n := cfg.Numerical
	for _, value := range []string{n.Operation, n.Oracle, string(n.Policy), n.NaNs, n.Infinities, n.SignedZero, n.Empty, n.Masks} {
		if value == "" || len(value) > 1024 {
			return nil, fmt.Errorf("kernelagent: numerical policy must be declared per operation")
		}
	}
	if n.Tolerance.Absolute < 0 || n.Tolerance.Relative < 0 || math.IsNaN(n.Tolerance.Absolute) || math.IsNaN(n.Tolerance.Relative) || math.IsInf(n.Tolerance.Absolute, 0) || math.IsInf(n.Tolerance.Relative, 0) || (n.Policy == cudasearch.NumericalPolicyExact && (n.Tolerance.Absolute != 0 || n.Tolerance.Relative != 0)) {
		return nil, fmt.Errorf("kernelagent: invalid numerical tolerance")
	}
	for _, e := range []Evaluation{cfg.Screen, cfg.HeldOut} {
		if err := e.Workload.Validate(); err != nil {
			return nil, err
		}
		if e.Workload.NumericalPolicy != n.Policy || len(e.References) == 0 || len(e.Inputs) == 0 {
			return nil, fmt.Errorf("kernelagent: evaluation does not match numerical declaration")
		}
		for _, name := range e.Workload.InputNames {
			if _, ok := e.Inputs[name]; !ok {
				return nil, fmt.Errorf("kernelagent: missing retained input %q", name)
			}
		}
		remaining := cfg.MaxEvidenceElements
		if remaining <= 0 {
			return nil, fmt.Errorf("kernelagent: positive input evidence bound required")
		}
		for _, data := range []map[string][]float32{e.Inputs, e.References} {
			for name, values := range data {
				if name == "" || len(name) > 1024 || len(values) > remaining {
					return nil, fmt.Errorf("kernelagent: input evidence exceeds bound")
				}
				remaining -= len(values)
			}
		}
	}
	if cfg.Model == nil || cudaroofline.ValidateCeiling(cfg.Model.Ceiling) != nil || cfg.Model.Ceiling.Architecture != cfg.Identity.SM || cfg.Operation.Name != n.Operation || cfg.Model.Floor(cfg.Operation).Basis == cudaroofline.FloorAbsent {
		return nil, fmt.Errorf("kernelagent: missing or mismatched roofline authority")
	}
	if cfg.Strategy != "beam" && cfg.Strategy != "mcts" {
		return nil, fmt.Errorf("kernelagent: unsupported search strategy")
	}
	if cfg.Rounds < 2 || cfg.Rounds > 1024 || cfg.Rounds%2 != 0 || cfg.CompileTimeout <= 0 || cfg.Budget.Timeout <= 0 || cfg.Budget.MaxEvaluations <= 0 {
		return nil, fmt.Errorf("kernelagent: finite evaluation, compile and epoch bounds required")
	}
	if cfg.Timing != cudatrial.TimingSummedKernel && cfg.Timing != cudatrial.TimingGPUBusy {
		return nil, cudasearch.ErrTimingKindMismatch
	}
	if len(cfg.Prompt) > maxProposalBytes || cfg.Prompt == "" || cfg.ExpectedModel.Name == "" || len(cfg.ExpectedModel.Name) > 1024 || cfg.ExpectedModel.Version == "" || len(cfg.ExpectedModel.Version) > 1024 || (cfg.ExpectedModel.Kind != "fixed" && cfg.ExpectedModel.Kind != "model") {
		return nil, fmt.Errorf("kernelagent: incomplete proposer identity or prompt")
	}
	if _, err := allowedRules(cfg.Catalog); err != nil {
		return nil, err
	}
	return &Loop{cfg: cfg, archive: a}, nil
}

// Result owns one epoch's search result and held-out confirmation. The search
// winner's expression stays valid until Close, so a Result can be passed to
// [Loop.Remeasure]. Promotion is detached evidence and remains valid after
// Close; it contains no live executable and does not mutate an executing plan.
type Result struct {
	Search       *cudasearch.SearchResult
	Confirmation *cudasearch.CandidatePlan // nil unless the winner was confirmed
	Promotion    *cudasearch.PromotionProposal
	Reason       string
}

// Close releases the search result and confirmation candidate and clears
// both fields. It does not close the loop's evaluators, the archive or the
// borrowed baseline. Close is idempotent.
func (r *Result) Close() {
	if r == nil {
		return
	}
	r.Confirmation.Close()
	r.Confirmation = nil
	r.Search.Close()
	r.Search = nil
}

// RunEpoch records P0, validates a data-only proposal, then invokes the real
// beam or MCTS engine with a concrete [cudasearch.ExecutableEvaluator], whose
// Observer writes each attempt and outcome to the archive. Correctness, ABBA
// interleaving and work budgets are enforced by that evaluator. A slower result
// is retained but cannot be promoted.
//
// On success the caller owns the Result and must Close it. On error RunEpoch
// has already closed every owner it created and returns a nil Result.
func (l *Loop) RunEpoch(ctx context.Context, epoch string, initial *egraph.SearchState, diagnostics *DiagnosticInput, counters []string) (*Result, error) {
	result := new(Result)
	if err := l.runEpoch(ctx, epoch, initial, diagnostics, counters, result); err != nil {
		return nil, err
	}
	return result, nil
}

// runEpoch fills result. On error it has closed result's owners.
func (l *Loop) runEpoch(ctx context.Context, epoch string, initial *egraph.SearchState, diagnostics *DiagnosticInput, counters []string, result *Result) (err error) {
	defer func() {
		if err != nil {
			result.Close()
		}
	}()
	if err := l.Close(); err != nil {
		return err
	}
	if epoch == "" || len(epoch) > 128 {
		return fmt.Errorf("kernelagent: invalid epoch")
	}
	for _, r := range l.archive.Records() {
		if r.Epoch == epoch {
			return fmt.Errorf("kernelagent: epoch already recorded")
		}
	}
	c := l.cfg
	if _, err := l.archive.append(Record{Epoch: epoch, Kind: "epoch", Generation: c.Generation, Identity: &c.Identity, Numerical: &c.Numerical, Roofline: &c.Model.Ceiling, Operation: &c.Operation}); err != nil {
		return err
	}
	if c.Identity.Mode == Native && !nativeLease() {
		_, archiveErr := l.archive.append(Record{Epoch: epoch, Kind: "admission", Outcome: "rejected", Reason: cudatrial.ErrHardwareLockRequired.Error()})
		return errors.Join(cudatrial.ErrHardwareLockRequired, archiveErr)
	}
	ctx, cancel := context.WithTimeout(ctx, c.Budget.Timeout)
	defer cancel()
	summary, err := Summarize(diagnostics, c.Baseline.Identity(), c.Screen.Workload.Key(), c.Identity.SM, counters)
	if err != nil {
		return err
	}
	allowed, _ := allowedRules(c.Catalog)
	reply, proposalErr := c.Proposer.Propose(ctx, Request{Epoch: epoch, Operation: c.Numerical.Operation, Prompt: c.Prompt, Allowed: allowed, Diagnostics: summary})
	if ctx.Err() != nil {
		proposalErr = ctx.Err()
	}
	proposal := ProposalEvidence{Model: c.ExpectedModel, Prompt: c.Prompt, ReturnedBytes: len(reply.Edit)}
	if len(reply.Model.Name) <= 1024 && len(reply.Model.Version) <= 1024 && len(reply.Model.Kind) <= 64 {
		reported := reply.Model
		proposal.ReportedModel = &reported
	}
	if len(reply.Edit) <= maxProposalBytes {
		proposal.ReturnedEdit = string(reply.Edit)
	}
	var rules []*egraph.Rewrite
	if proposalErr == nil && reply.Model != c.ExpectedModel {
		proposalErr = fmt.Errorf("kernelagent: proposer identity mismatch")
	}
	if proposalErr == nil {
		rules, proposal.Rules, proposalErr = admitProposal(reply.Edit, c.Catalog)
	}
	r := Record{Epoch: epoch, Kind: "proposal", Proposal: &proposal, Diagnostics: &summary, Outcome: "accepted"}
	if proposalErr != nil {
		r.Outcome, r.Reason = "rejected", proposalErr.Error()
	}
	if _, err := l.archive.append(r); err != nil {
		return err
	}
	if proposalErr != nil {
		return proposalErr
	}
	defer func() {
		if closeErr := l.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		}
	}()
	screen := l.evaluator(epoch, "screen", c.Screen, &proposal, c.Generation, cancel)
	var search *cudasearch.SearchResult
	if c.Strategy == "beam" {
		search, err = cudasearch.NewBeamSearch(cudasearch.BeamConfig{EvaluatorID: c.EvaluatorID, BeamWidth: c.BeamWidth, Budget: c.Budget}).Search(ctx, initial, rules, screen)
	} else {
		search, err = cudasearch.NewMCTSSearch(cudasearch.MCTSConfig{EvaluatorID: c.EvaluatorID, Seed: c.Seed, Budget: c.Budget}).Search(ctx, initial, rules, screen)
	}
	result.Search = search
	if search != nil {
		r := Record{Epoch: epoch, Kind: "search", Generation: c.Generation, Policy: &search.PolicyID, Reason: errorText(err)}
		if search.Winner != nil {
			r.CandidateID = search.Winner.ID
		}
		_, archiveErr := l.archive.append(r)
		err = errors.Join(err, archiveErr)
	}
	if err != nil {
		return err
	}
	if search == nil || search.Winner == nil {
		return cudasearch.ErrNoAdmittedPlan
	}
	// The screening evaluator's compiled owners are no longer needed. The
	// winner's expression is owned by search, not by the evaluator, and stays
	// valid through confirmation.
	if err := l.Close(); err != nil {
		return err
	}
	if !improves(search.Winner) {
		result.Reason = "no interleaved improvement"
		_, err := l.archive.append(Record{Epoch: epoch, Kind: "decision", CandidateID: search.Winner.ID, Outcome: "not-promoted", Reason: result.Reason})
		return err
	}
	held := l.evaluator(epoch, "held-out", c.HeldOut, &proposal, c.Generation, cancel)
	result.Confirmation, err = held.Confirm(ctx, search.Winner)
	if err != nil {
		return err
	}
	if !improves(result.Confirmation) {
		result.Reason = "held-out regression"
		_, err := l.archive.append(Record{Epoch: epoch, Kind: "decision", CandidateID: search.Winner.ID, Outcome: "not-promoted", Reason: result.Reason})
		return err
	}
	promotion, err := search.ProposePromotion(result.Confirmation, c.Generation)
	if err != nil {
		return err
	}
	if _, err := l.archive.append(Record{Epoch: epoch, Kind: "promotion", CandidateID: promotion.CandidateID, Promotion: promotion}); err != nil {
		return err
	}
	result.Promotion = promotion
	return nil
}

// Close retries cleanup of compiled candidates retained by the owner evaluator.
// A failed Close retains that evaluator for another retry.
func (l *Loop) Close() error {
	for len(l.pending) > 0 {
		if err := l.pending[0].Close(); err != nil {
			return err
		}
		l.pending = l.pending[1:]
	}
	return nil
}

func nativeLease() bool { _, err := cudalock.Inherited(os.Getenv); return err == nil }

func validState(s DeviceState, mode Mode) error {
	if s.Observed.IsZero() || s.Source == "" || (mode == HostStub && s.Source != "synthetic") || s.ThermalThrottle == nil || *s.ThermalThrottle {
		return fmt.Errorf("kernelagent: unavailable or thermally throttled conditions")
	}
	for _, v := range []*float64{s.ClockMHz, s.TemperatureC, s.PowerW, s.PowerLimitW} {
		if v == nil || math.IsNaN(*v) || math.IsInf(*v, 0) {
			return fmt.Errorf("kernelagent: missing or invalid per-measurement conditions")
		}
	}
	if *s.ClockMHz <= 0 || *s.PowerLimitW <= 0 || *s.PowerW < 0 {
		return fmt.Errorf("kernelagent: invalid clock or power limit")
	}
	return nil
}

func bitEvidence(values map[string][]float32) map[string][]uint32 {
	out := make(map[string][]uint32, len(values))
	for name, values := range values {
		bits := make([]uint32, len(values))
		for i, v := range values {
			bits[i] = math.Float32bits(v)
		}
		out[name] = bits
	}
	return out
}

// archiveObserver is the archive's only attempt write path. It is installed
// as the concrete evaluator's Observer, so the evaluator's own boundary
// decides when an attempt starts and finishes; this package never wraps or
// replaces the evaluator itself.
type archiveObserver struct {
	loop         *Loop
	epoch, phase string
	binding      Evaluation
	proposal     *ProposalEvidence
	current      *Record
	cancel       context.CancelFunc
}

// evaluator returns a concrete owner evaluator for one phase. The loop retains
// it for Close; callers must not use it after the next Close succeeds.
func (l *Loop) evaluator(epoch, phase string, binding Evaluation, proposal *ProposalEvidence, generation int, cancel context.CancelFunc) *cudasearch.ExecutableEvaluator {
	c := l.cfg
	o := &archiveObserver{loop: l, epoch: epoch, phase: phase, binding: binding, proposal: proposal, cancel: cancel}
	check := func() bool { return c.Identity.Mode == HostStub || nativeLease() }
	e := &cudasearch.ExecutableEvaluator{Generation: generation, Workload: binding.Workload, References: binding.References, Policy: c.Numerical.Tolerance, CostModel: c.CostModel, Rounds: c.Rounds, TimingKind: c.Timing, CompileTimeout: c.CompileTimeout, HardwareLockChecker: check, Observer: o}
	e.Compile = func(ctx context.Context, expr *egraph.Expr, b cudasearch.WorkloadBindings) (cudasearch.Executable, error) {
		r := o.current
		if r == nil {
			return nil, fmt.Errorf("kernelagent: compile outside a recorded attempt")
		}
		r.StructuralID = expr.Fingerprint()
		x, err := c.Compile(ctx, expr, b)
		r.Compile = "failed"
		if err == nil && x != nil {
			r.Compile = "succeeded"
		}
		if x == nil {
			return nil, err
		}
		return x, err
	}
	e.Baseline = c.Baseline
	e.Run = func(ctx context.Context, arm string, x cudasearch.Executable) (cudatrial.Receipt, error) {
		return o.run(ctx, arm, x)
	}
	l.pending = append(l.pending, e)
	return e
}

// StartEvaluation durably appends the attempt record. A failed append
// refuses the attempt, so no extraction or compilation follows, and cancels
// the epoch.
func (o *archiveObserver) StartEvaluation(_ context.Context, s cudasearch.EvaluationStart) error {
	if o.current != nil {
		return fmt.Errorf("kernelagent: overlapping attempts")
	}
	c := o.loop.cfg
	b := o.binding.Workload
	r := Record{Epoch: o.epoch, Kind: "attempt", Generation: s.Generation, Phase: o.phase, Evaluation: string(s.Kind), OwnerAttempt: s.Attempt,
		Identity: &c.Identity, Numerical: &c.Numerical, Proposal: o.proposal, WorkloadKey: s.WorkloadKey, BaselineID: s.BaselineID, ConfirmationOf: s.ConfirmationOf,
		Inputs: bitEvidence(o.binding.Inputs), References: bitEvidence(o.binding.References), Derivation: s.Derivation, Verification: "unverified", Unmeasured: true,
		Roofline: &c.Model.Ceiling, Operation: &c.Operation}
	r.Workload = &WorkloadEvidence{Name: b.Name, Shape: b.Shape.Dims(), DType: b.DType.String(), Layout: b.Layout, InputNames: b.InputNames, InputIdentities: b.InputIdentities, WeightIdentities: b.WeightIdentities, NumericalPolicy: string(b.NumericalPolicy), Objective: string(b.Objective)}
	id, err := o.loop.archive.append(r)
	if err != nil {
		o.cancel()
		return err
	}
	r.Attempt, r.Kind = id, "outcome"
	o.current = &r
	return nil
}

// FinishEvaluation durably appends the outcome from the owner's detached
// evidence. An admitted candidate whose trials do not qualify, or whose
// outcome cannot be appended, is refused, which the evaluator reports by
// withholding the candidate.
func (o *archiveObserver) FinishEvaluation(_ context.Context, f cudasearch.EvaluationFinish) error {
	r := o.current
	o.current = nil
	if r == nil {
		return fmt.Errorf("kernelagent: outcome without a recorded attempt")
	}
	c := o.loop.cfg
	r.CandidateID = f.Evidence.ExecutableID
	r.MeasurementReused = f.MeasurementReused
	r.VerificationArm = f.Evidence.ComparisonArm
	switch r.VerificationArm {
	case "baseline":
		r.VerificationIdentity = r.BaselineID
	case "candidate":
		r.VerificationIdentity = r.CandidateID
	}
	switch {
	case errors.Is(f.Err, cudasearch.ErrVerificationFailed):
		r.Verification = "failed"
	case len(f.Evidence.Trials) > 0 || f.Candidate != nil:
		// The owner times neither arm until both arms have been compared.
		r.Verification = "passed"
	}
	r.Outcome = "quarantined"
	var refused error
	if f.Candidate != nil && f.Err == nil {
		if err := validTrials(f.Evidence.Trials, f.Candidate.ID, c.Timing); err != nil {
			refused = err
		} else {
			r.Outcome = "qualified"
			cost := f.Candidate.Cost
			r.CostNs = &cost
		}
	}
	for _, trial := range f.Evidence.Trials {
		t := Trial{Name: trial.Name, Round: trial.Round, Identity: trial.Provenance["executable_identity"], Complete: trial.Complete, Kind: string(trial.TimingKind), Timings: make(map[string]int64)}
		for kind, d := range trial.Timings {
			t.Timings[string(kind)] = int64(d)
		}
		// Some real adapters populate the owner's specific duration fields rather
		// than Timings. Preserve their owner-defined availability as well.
		for _, kind := range []cudatrial.TimingKind{cudatrial.TimingSummedKernel, cudatrial.TimingGPUBusy, cudatrial.TimingWallElapsed, cudatrial.TimingAPICall} {
			if trial.TimingKind == kind || trial.HasTiming(kind) {
				if d, err := trial.DurationFor(kind); err == nil {
					t.Timings[string(kind)] = int64(d)
				}
			}
		}
		if trial.Err != nil {
			t.Error = trial.Err.Error()
		}
		r.Trials = append(r.Trials, t)
	}
	r.Comparison = make(map[string]Comparison)
	for name, comparison := range f.Evidence.Comparison {
		r.Comparison[name] = Comparison{comparison.Elements, strconv.FormatFloat(comparison.MaxAbsolute, 'g', -1, 64), strconv.FormatFloat(comparison.MaxRelative, 'g', -1, 64)}
	}
	r.Reason = errorText(errors.Join(f.Err, refused))
	if errors.Is(f.Err, ErrDeviceFault) {
		r.FaultCauseUnknown = true
		o.cancel()
	}
	if _, err := o.loop.archive.append(*r); err != nil {
		o.cancel()
		return errors.Join(refused, err)
	}
	return refused
}

// run records conditions around the original owner's Run. The concrete
// evaluator retains the original executable and its optional capabilities.
func (a *archiveObserver) run(ctx context.Context, arm string, x cudasearch.Executable) (receipt cudatrial.Receipt, err error) {
	c := a.loop.cfg
	if a.current == nil {
		return receipt, fmt.Errorf("kernelagent: measurement outside a recorded attempt")
	}
	s := Sample{Identity: x.Identity(), Status: "rejected", Floor: c.Model.Floor(c.Operation)}
	defer func() {
		if err != nil {
			s.Error = err.Error()
		}
		a.current.Samples = append(a.current.Samples, s)
	}()
	if c.Identity.Mode == Native && !nativeLease() {
		return receipt, cudatrial.ErrHardwareLockRequired
	}
	s.Before, err = c.Observe(ctx)
	if err != nil {
		return receipt, err
	}
	if err := validState(s.Before, c.Identity.Mode); err != nil {
		return receipt, err
	}
	receipt, err = x.Run(ctx)
	if err != nil {
		return receipt, err
	}
	s.After, err = c.Observe(ctx)
	if err != nil {
		return receipt, err
	}
	if err := validState(s.After, c.Identity.Mode); err != nil {
		return receipt, err
	}
	if receipt.TimingKind != c.Timing || !receipt.HasTiming(c.Timing) {
		s.Status = "unmeasured"
		return receipt, cudatrial.ErrTimingUnavailable
	}
	d, err := receipt.DurationFor(c.Timing)
	if err != nil {
		s.Status = "unmeasured"
		return receipt, cudatrial.ErrTimingUnavailable
	}
	if arm == "candidate" {
		a.current.Unmeasured = false
	}
	if d < 0 {
		return receipt, fmt.Errorf("kernelagent: invalid negative device duration")
	}
	s.Floor = c.Model.Judge(c.Operation, uint64(d.Nanoseconds()))
	if s.Floor.Verdict != cudaroofline.VerdictAccepted {
		return receipt, fmt.Errorf("kernelagent: measurement artifact: %s", s.Floor.Reason)
	}
	s.Status = "accepted"
	receipt.Provenance = maps.Clone(receipt.Provenance)
	if receipt.Provenance == nil {
		receipt.Provenance = make(map[string]string)
	}
	before, _ := json.Marshal(s.Before)
	after, _ := json.Marshal(s.After)
	receipt.Provenance["kernelagent_conditions_before"], receipt.Provenance["kernelagent_conditions_after"] = string(before), string(after)
	receipt.Provenance["kernelagent_mode"] = string(c.Identity.Mode)
	return receipt, nil
}

func validTrials(trials []cudatrial.Receipt, id string, kind cudatrial.TimingKind) error {
	if len(trials) < 4 || len(trials)%4 != 0 {
		return fmt.Errorf("kernelagent: uninterleaved comparison")
	}
	for i, t := range trials {
		name := "baseline"
		if i%4 == 1 || i%4 == 2 {
			name = "candidate"
		}
		if t.Name != name || t.Round != i/2+1 || !t.Complete || t.Err != nil || t.TimingKind != kind || (name == "candidate" && t.Provenance["executable_identity"] != id) {
			return fmt.Errorf("kernelagent: uninterleaved or invalid comparison")
		}
		if d, err := t.DurationFor(kind); err != nil || d <= 0 {
			return cudatrial.ErrTimingUnavailable
		}
		if t.Provenance["kernelagent_conditions_before"] == "" || t.Provenance["kernelagent_conditions_after"] == "" {
			return fmt.Errorf("kernelagent: missing per-measurement conditions")
		}
	}
	return nil
}

func improves(c *cudasearch.CandidatePlan) bool {
	if c == nil || c.Receipt == nil {
		return false
	}
	var baseline []time.Duration
	for _, r := range c.Trials {
		if r.Name == "baseline" {
			d, err := r.DurationFor(r.TimingKind)
			if err != nil {
				return false
			}
			baseline = append(baseline, d)
		}
	}
	if len(baseline) == 0 {
		return false
	}
	sort.Slice(baseline, func(i, j int) bool { return baseline[i] < baseline[j] })
	return c.Cost < float64(baseline[len(baseline)/2].Nanoseconds())
}
