package kernelagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type fixtureCost struct{}

func (fixtureCost) NodeCost(n egraph.ENode, children []float64) float64 {
	c := map[string]float64{"baseline": 10, "fast": 1, "wrong": 0.5, "slow": 2}[n.Op]
	for _, child := range children {
		c += child
	}
	return c
}

type stubExecutable struct {
	id, op   string
	values   map[string][]float32
	duration time.Duration
	kind     cudatrial.TimingKind
	fixture  *fixture
	closed   bool
}

func (x *stubExecutable) Identity() string { return x.id }
func (x *stubExecutable) Outputs(context.Context) (map[string][]float32, error) {
	x.fixture.events = append(x.fixture.events, "verify:"+x.op)
	return x.values, nil
}
func (x *stubExecutable) Run(context.Context) (cudatrial.Receipt, error) {
	x.fixture.events = append(x.fixture.events, "run:"+x.op)
	x.fixture.runs[x.op]++
	if x.fixture.fault && x.op == "fast" {
		return cudatrial.Receipt{}, ErrDeviceFault
	}
	r := cudatrial.Receipt{Complete: true, TimingKind: x.kind}
	if x.duration >= 0 {
		if x.fixture.legacyTiming {
			r.SummedKernel = x.duration
		} else {
			_ = r.RecordTiming(x.kind, x.duration)
		}
	}
	return r, nil
}
func (x *stubExecutable) Close() error { x.closed = true; return nil }

type fixture struct {
	cfg                               Config
	state                             *egraph.SearchState
	archive                           *Archive
	events                            []string
	runs                              map[string]int
	fastDuration                      time.Duration
	thermal, missingConditions, fault bool
	fastKind                          cudatrial.TimingKind
	legacyTiming                      bool
	heldOutWrong                      bool
	compiled                          []*stubExecutable // every owner returned by Compile
}

func syntheticState() DeviceState {
	clock, temp, power, limit, throttle := 1200.0, 42.0, 10.0, 100.0, false
	return DeviceState{Observed: time.Unix(1, 0), Source: "synthetic", ClockMHz: &clock, TemperatureC: &temp, PowerW: &power, PowerLimitW: &limit, ThermalThrottle: &throttle}
}

func newFixture(dir string) (*fixture, error) {
	f := &fixture{runs: make(map[string]int), fastDuration: 50 * time.Nanosecond, fastKind: cudatrial.TimingSummedKernel}
	a, err := OpenArchive(dir, 1<<20, 1000)
	if err != nil {
		return nil, err
	}
	f.archive = a
	makeBinding := func(name string, x, y []float32) (Evaluation, error) {
		shape, err := cudashape.New(len(x))
		if err != nil {
			return Evaluation{}, err
		}
		out, err := cudaref.ElementwiseBinary(x, y, "add")
		return Evaluation{Workload: cudasearch.WorkloadBindings{Name: "add", Shape: shape, DType: cudair.Float32, Layout: "row_major", InputNames: []string{"x", "y"}, InputIdentities: map[string]string{"x": name + "-x", "y": name + "-y"}, NumericalPolicy: cudasearch.NumericalPolicyFP32, Objective: cudasearch.ObjectiveMinLatency}, Inputs: map[string][]float32{"x": x, "y": y}, References: map[string][]float32{"out": out}}, err
	}
	screen, err := makeBinding("screen", []float32{1, 2}, []float32{3, 4})
	if err != nil {
		return nil, err
	}
	held, err := makeBinding("held", []float32{4, -2}, []float32{-1, 5})
	if err != nil {
		return nil, err
	}
	base := &stubExecutable{id: "host-stub:baseline:tool-fixture-v1", op: "baseline", values: screen.References, duration: 100 * time.Nanosecond, kind: cudatrial.TimingSummedKernel, fixture: f}
	// Outputs follow private bindings for each evaluation, just as the real
	// compiler/executable adapters must bind/reset their workload buffers.
	var active Evaluation
	screen.Workload.ResetState = func() error { active = screen; base.values = active.References; return nil }
	held.Workload.ResetState = func() error { active = held; base.values = active.References; return nil }
	active = screen
	model := cudaroofline.NewModelWithCeiling(cudaroofline.Ceiling{Architecture: "sm_fixture", Residency: cudaroofline.DeviceResident, PeakTFLOPS: 1, PeakMemoryGBps: 1, ComputeProvenance: cudaroofline.Unverified, MemoryProvenance: cudaroofline.Unverified})
	f.cfg = Config{Identity: Identity{Mode: HostStub, GPU: "synthetic-gpu", SM: "sm_fixture", Driver: "synthetic", CUDA: "synthetic", NVRTC: "stub-no-compiler", Tools: "fixture-v1", Lease: "not-applicable-host-stub", Initial: syntheticState()}, Numerical: NumericalDeclaration{Operation: "add", Oracle: "cudaref.ElementwiseBinary/add", Policy: cudasearch.NumericalPolicyFP32, Tolerance: cudasearch.OutputPolicy{Absolute: 1e-6, Relative: 1e-6}, NaNs: "reject unless both NaNs and NaNsEqual", Infinities: "match sign", SignedZero: "preserve sign", Empty: "no empty outputs in this operation fixture", Masks: "not applicable"}, Screen: screen, HeldOut: held, Generation: 1, Baseline: base, Model: model, Operation: cudaroofline.Operation{Name: "add", FLOPs: 2, Bytes: 24, Precision: cudaroofline.FP32}, CostModel: fixtureCost{}, ExpectedModel: ModelIdentity{"fixed", "enumerator", "1"}, Prompt: "select registered schedule edits; return only rules JSON", EvaluatorID: "host-stub-compiler-observer-v1", Strategy: "beam", BeamWidth: 4, Seed: 7, Rounds: 2, Timing: cudatrial.TimingSummedKernel, CompileTimeout: time.Second, Budget: cudasearch.SearchBudget{MaxStates: 16, MaxNodesPerState: 64, MaxClassesPerState: 32, MaxDepth: 2, MaxActions: 8, MaxTotalActions: 16, MaxEvaluations: 16, Timeout: 30 * time.Second}} // every record is fsynced; loaded hosts are slow
	f.cfg.Observe = func(context.Context) (DeviceState, error) {
		s := syntheticState()
		if f.thermal {
			*s.ThermalThrottle = true
		}
		if f.missingConditions {
			s.TemperatureC = nil
		}
		return s, nil
	}
	f.cfg.Compile = func(_ context.Context, expr *egraph.Expr, b cudasearch.WorkloadBindings) (cudasearch.Executable, error) {
		op := expr.Op
		binding := screen
		if b.Key() == held.Workload.Key() {
			binding = held
		}
		values := make([]float32, len(binding.Inputs["x"]))
		for i, x := range binding.Inputs["x"] {
			values[i] = x + binding.Inputs["y"][i]
			if op == "wrong" || (f.heldOutWrong && op == "fast" && binding.Workload.Key() == held.Workload.Key()) {
				values[i]++
			}
		}
		duration := map[string]time.Duration{"baseline": 100 * time.Nanosecond, "slow": 150 * time.Nanosecond, "wrong": time.Nanosecond, "fast": f.fastDuration}[op]
		if _, ok := map[string]bool{"baseline": true, "slow": true, "wrong": true, "fast": true}[op]; !ok {
			return nil, fmt.Errorf("unsupported fixture schedule %q", op)
		}
		kind := cudatrial.TimingSummedKernel
		if op == "fast" {
			kind = f.fastKind
		}
		x := &stubExecutable{id: "host-stub:" + op + ":tool-fixture-v1", op: op, values: map[string][]float32{"out": values}, duration: duration, kind: kind, fixture: f}
		f.compiled = append(f.compiled, x)
		return x, nil
	}
	f.cfg.Catalog = make(map[string]*egraph.Rewrite)
	for _, name := range []string{"fast", "wrong", "slow"} {
		r, err := egraph.NewRewrite("schedule-"+name, "(baseline ?x)", "("+name+" ?x)")
		if err != nil {
			return nil, err
		}
		f.cfg.Catalog[name] = r
	}
	f.cfg.Proposer = &FixedEnumerator{Edits: []json.RawMessage{json.RawMessage(`{"rules":["fast","wrong","slow"]}`)}}
	g := egraph.New()
	x := g.Add(egraph.ENode{Op: "x"})
	root := g.Add(egraph.ENode{Op: "baseline", Children: []egraph.Id{x}})
	f.state = egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	meta := egraph.TypedMetadata{Shape: screen.Workload.Shape, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, LoopInvariant: true, NumericalPolicy: "fp32"}
	f.state.SetMetadata(x, meta)
	f.state.SetMetadata(root, meta)
	f.cfg.MaxEvidenceElements = 64
	return f, nil
}

func setup(t *testing.T) (*fixture, *Loop) {
	t.Helper()
	f, err := newFixture(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.archive.Close() })
	l, err := NewLoop(f.archive, f.cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := l.Close(); err != nil {
			t.Error(err)
		}
	})
	return f, l
}
func run(t *testing.T, l *Loop, f *fixture) *Result {
	t.Helper()
	r, err := l.RunEpoch(context.Background(), "epoch-1", f.state, nil, []string{"not-supported"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(r.Close)
	return r
}

// openOwners reports compiled executables that were never closed.
func (f *fixture) openOwners() int {
	n := 0
	for _, x := range f.compiled {
		if !x.closed {
			n++
		}
	}
	return n
}

func TestLoopCompletesWithFixedEnumerator(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			f, l := setup(t)
			l.cfg.Strategy = strategy
			r := run(t, l, f)
			if r.Search == nil || r.Search.Strategy != strategy || r.Search.Winner == nil || r.Search.Winner.ID != "host-stub:fast:tool-fixture-v1" || r.Promotion == nil || r.Confirmation == nil {
				t.Fatalf("result=%+v search=%+v", r, r.Search)
			}
			// The concrete owner evaluator is visible to Search, so its work
			// accounting and policy identity are the owner's, not "external".
			if !r.Search.PolicyComplete || r.Search.PolicyID.Options["evaluator_identity"] != f.cfg.EvaluatorID || r.Search.PolicyID.Options["evaluator"] != "executable-v1" || !r.Search.WorkAccounted || r.Search.Compilations == 0 {
				t.Fatalf("owner policy/accounting=%+v", r.Search)
			}
			// Every screening compile is one the owner counted, plus one
			// held-out confirmation compile.
			if len(f.compiled) != r.Search.Compilations+1 || f.openOwners() != 0 || f.cfg.Baseline.(*stubExecutable).closed {
				t.Fatalf("compiles=%d owner=%d open=%d", len(f.compiled), r.Search.Compilations, f.openOwners())
			}
		})
	}
}

func TestNoTimingRecordedBeforeVerification(t *testing.T) {
	f, l := setup(t)
	run(t, l, f)
	if f.runs["wrong"] != 0 {
		t.Fatal("fast wrong executable was timed")
	}
	for _, r := range f.archive.Records() {
		if r.Kind == "outcome" && strings.Contains(r.CandidateID, ":wrong:") && (len(r.Trials) != 0 || len(r.Samples) != 0 || !r.Unmeasured || r.Verification != "failed") {
			t.Fatalf("wrong candidate leaked timing: %+v", r)
		}
	}
}

func TestVerificationFailureQuarantinesCandidate(t *testing.T) {
	f, l := setup(t)
	run(t, l, f)
	found := false
	for _, r := range f.archive.Records() {
		if r.Kind == "outcome" && strings.Contains(r.CandidateID, ":wrong:") {
			found = true
			if r.Outcome != "quarantined" || r.Reason == "" || len(r.Inputs["x"]) != 2 || len(r.References["out"]) != 2 || r.Compile != "succeeded" {
				t.Fatalf("quarantine=%+v", r)
			}
		}
	}
	if !found {
		t.Fatal("missing failed candidate")
	}
}

func TestNumericalPolicyDeclaredPerOperation(t *testing.T) {
	for _, field := range []string{"operation", "oracle", "policy", "nan", "infinity", "zero", "empty", "mask", "tolerance"} {
		t.Run(field, func(t *testing.T) {
			f, _ := setup(t)
			c := f.cfg
			switch field {
			case "operation":
				c.Numerical.Operation = ""
			case "oracle":
				c.Numerical.Oracle = ""
			case "policy":
				c.Numerical.Policy = ""
			case "nan":
				c.Numerical.NaNs = ""
			case "infinity":
				c.Numerical.Infinities = ""
			case "zero":
				c.Numerical.SignedZero = ""
			case "empty":
				c.Numerical.Empty = ""
			case "mask":
				c.Numerical.Masks = ""
			case "tolerance":
				c.Numerical.Tolerance.Absolute = math.NaN()
			}
			if _, err := NewLoop(f.archive, c); err == nil {
				t.Fatal("missing numerical declaration accepted")
			}
		})
	}
}

func TestUninterleavedComparisonRejected(t *testing.T) {
	f, l := setup(t)
	r := run(t, l, f)
	trials := append([]cudatrial.Receipt(nil), r.Search.Winner.Trials...)
	trials[1], trials[3] = trials[3], trials[1]
	if err := validTrials(trials, r.Search.Winner.ID, f.cfg.Timing); err == nil {
		t.Fatal("uninterleaved evidence accepted")
	}
}

func TestClockAndTemperatureRecordedPerMeasurement(t *testing.T) {
	f, l := setup(t)
	run(t, l, f)
	count := 0
	for _, r := range f.archive.Records() {
		for _, s := range r.Samples {
			count++
			if s.Before.ClockMHz == nil || s.Before.TemperatureC == nil || s.After.PowerW == nil || s.After.PowerLimitW == nil || s.Before.Source != "synthetic" || s.After.Observed.IsZero() {
				t.Fatalf("missing conditions: %+v", s)
			}
		}
	}
	if count < 4 {
		t.Fatal("no interleaved state observations")
	}
	for _, missing := range []bool{false, true} {
		t.Run(fmt.Sprint(missing), func(t *testing.T) {
			g, m := setup(t)
			g.missingConditions = missing
			g.thermal = !missing
			if _, err := m.RunEpoch(context.Background(), "invalid", g.state, nil, nil); err == nil {
				t.Fatal("unavailable/throttled measurement admitted")
			}
			if len(g.runs) != 0 {
				t.Fatal("invalid conditions reached Run")
			}
		})
	}
}

func TestSubRooflineLatencyRejectedAsArtifact(t *testing.T) {
	f, l := setup(t)
	f.fastDuration = time.Nanosecond
	r := run(t, l, f)
	if r.Promotion != nil {
		t.Fatal("artifact promoted")
	}
	found := false
	for _, record := range f.archive.Records() {
		if strings.Contains(record.CandidateID, ":fast:") {
			for _, s := range record.Samples {
				if s.Floor.Verdict == cudaroofline.VerdictRejected {
					found = true
					if s.Floor.Operation != f.cfg.Operation || s.Status != "rejected" {
						t.Fatal("wrong floor authority")
					}
				}
			}
		}
	}
	if !found {
		t.Fatal("missing artifact refusal")
	}
}

func TestUnmeasuredWhenTimerUnavailable(t *testing.T) {
	f, l := setup(t)
	f.fastDuration = -1
	r := run(t, l, f)
	if r.Promotion != nil {
		t.Fatal("unmeasured candidate promoted")
	}
	found := false
	for _, record := range f.archive.Records() {
		if record.Kind == "outcome" && strings.Contains(record.CandidateID, ":fast:") {
			found = true
			if !record.Unmeasured || record.Outcome != "quarantined" {
				t.Fatalf("unmeasured=%+v", record)
			}
		}
	}
	if !found {
		t.Fatal("missing unmeasured record")
	}
}

func TestWallTimeNotUsedForCandidateComparison(t *testing.T) {
	f, l := setup(t)
	f.fastKind = cudatrial.TimingWallElapsed
	r := run(t, l, f)
	if r.Promotion != nil {
		t.Fatal("wall time promoted")
	}
	for _, c := range r.Search.AllCandidates {
		if c.Receipt.TimingKind == cudatrial.TimingWallElapsed {
			t.Fatal("wall objective admitted")
		}
	}
}

func TestGPULockRequiredForNativeMeasurement(t *testing.T) {
	f, l := setup(t)
	l.cfg.Identity.Mode = Native
	// No environment/global hook is changed. If this process genuinely inherited
	// a live lease, the test cannot test absence and must fail, not launch work.
	if nativeLease() {
		t.Fatal("native absence fixture requires an unleased host")
	}
	_, err := l.RunEpoch(context.Background(), "native-refusal", f.state, nil, nil)
	if !errors.Is(err, cudatrial.ErrHardwareLockRequired) || len(f.events) != 0 {
		t.Fatalf("native lock gate: %v %v", err, f.events)
	}
}

func TestCandidateIdentityReusesSearchOwnerScheme(t *testing.T) {
	f, l := setup(t)
	r := run(t, l, f)
	id := r.Search.Winner.ID
	found := false
	for _, record := range f.archive.Records() {
		if record.Kind == "outcome" && record.CandidateID == id {
			found = true
			if record.StructuralID != r.Search.Winner.Expr.Fingerprint() {
				t.Fatal("invented structural identity")
			}
		}
	}
	if !found || r.Promotion.CandidateID != id {
		t.Fatal("owner identity lost")
	}
}

type replyProposer struct{ reply Reply }

func (p replyProposer) Propose(context.Context, Request) (Reply, error) { return p.reply, nil }

func TestProposalOutputCannotBypassValidation(t *testing.T) {
	for _, edit := range []string{`{"rules":["../../escape"]}`, `{"rules":["fast; touch outside"]}`, `{"path":"../outside","rules":["fast"]}`, `{"source":"__global__ void evil(){}"}`, `{"rules":["unknown"]}`, `{"rules":["fast"],"rules":["wrong"]}`, `{"rules":["fast"]} {}`, `not JSON`} {
		t.Run(edit, func(t *testing.T) {
			f, l := setup(t)
			l.cfg.Proposer = replyProposer{Reply{Model: f.cfg.ExpectedModel, Edit: json.RawMessage(edit)}}
			if _, err := l.RunEpoch(context.Background(), "bad-proposal", f.state, nil, nil); err == nil {
				t.Fatal("invalid proposal accepted")
			}
			if len(f.events) != 0 {
				t.Fatal("invalid proposal reached compiler/executable")
			}
			records := f.archive.Records()
			last := records[len(records)-1]
			if last.Kind != "proposal" || last.Outcome != "rejected" {
				t.Fatal("missing proposal refusal")
			}
		})
	}
}

func TestProposalOutputCannotWriteOutsideCandidateDir(t *testing.T) {
	dir := t.TempDir()
	child := dir + "/candidate"
	if err := os.Mkdir(child, 0700); err != nil {
		t.Fatal(err)
	}
	f, err := newFixture(child)
	if err != nil {
		t.Fatal(err)
	}
	defer f.archive.Close()
	f.cfg.Proposer = replyProposer{Reply{Model: f.cfg.ExpectedModel, Edit: json.RawMessage(`{"path":"../escaped","rules":["fast"]}`)}}
	l, err := NewLoop(f.archive, f.cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = l.RunEpoch(context.Background(), "traversal", f.state, nil, nil); err == nil {
		t.Fatal("path accepted")
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 || entries[0].Name() != "candidate" {
		t.Fatalf("outside writes: %v %v", entries, err)
	}
}

func TestMalformedProposalRejected(t *testing.T) {
	for _, raw := range []string{`null`, `[]`, `{}`, `{"rules":1}`, `{"rules":[`, strings.Repeat("x", maxProposalBytes+1)} {
		if _, _, err := admitProposal([]byte(raw), nil); err == nil {
			t.Fatal("malformed edit accepted")
		}
	}
}

func TestModelIdentityRecordedWithCandidate(t *testing.T) {
	f, l := setup(t)
	identity := ModelIdentity{"model", "local-fixture-no-API", "v1"}
	l.cfg.ExpectedModel = identity
	l.cfg.Proposer = replyProposer{Reply{Model: identity, Edit: json.RawMessage(`{"rules":["fast"]}`)}}
	run(t, l, f)
	found := false
	for _, r := range f.archive.Records() {
		if r.Kind == "outcome" {
			found = true
			if r.Proposal == nil || r.Proposal.Model != identity || r.Proposal.Prompt != f.cfg.Prompt || r.Proposal.ReturnedEdit != `{"rules":["fast"]}` {
				t.Fatalf("model provenance=%+v", r.Proposal)
			}
		}
	}
	if !found {
		t.Fatal("no model fixture candidates")
	}
}

func TestCorrectSlowerCandidateNotPromoted(t *testing.T) {
	f, l := setup(t)
	l.cfg.Proposer = &FixedEnumerator{Edits: []json.RawMessage{json.RawMessage(`{"rules":["slow"]}`)}}
	r := run(t, l, f)
	if r.Promotion != nil || r.Reason == "" {
		t.Fatal("slower candidate promoted")
	}
	if f.runs["slow"] == 0 {
		t.Fatal("slower correct candidate not evaluated")
	}
}

func TestGPUFaultStopsEpochWithoutBlamingRule(t *testing.T) {
	f, l := setup(t)
	f.fault = true
	_, err := l.RunEpoch(context.Background(), "fault", f.state, nil, nil)
	if err == nil {
		t.Fatal("fault did not stop epoch")
	}
	found := false
	for _, r := range f.archive.Records() {
		if r.Kind == "outcome" && r.FaultCauseUnknown {
			found = true
			if !strings.Contains(r.Reason, "causal proposal unknown") || r.Outcome != "quarantined" {
				t.Fatal("fault attributed or lost")
			}
		}
	}
	if !found {
		t.Fatal("no fault quarantine")
	}
}

func TestNoSearchAlgorithmDuplicated(t *testing.T) {
	f, l := setup(t)
	r := run(t, l, f)
	if r.Search.ActionsAttempted == 0 || len(r.Search.Winner.Derivation) == 0 || r.Search.Strategy != "beam" {
		t.Fatal("owner transitions not used")
	}
	replayed, err := egraph.ReplaySearchState(f.state, []*egraph.Rewrite{f.cfg.Catalog["fast"], f.cfg.Catalog["wrong"], f.cfg.Catalog["slow"]}, r.Search.Winner.Derivation)
	if err != nil || replayed == nil {
		t.Fatalf("owner replay: %v", err)
	}
	if r.Search.PolicyID.RuleSet == "" || r.Search.PolicyID.Strategy != "beam" {
		t.Fatal("owner search identity absent")
	}
}

func TestOwnerWorkBudgetEnforced(t *testing.T) {
	f, l := setup(t)
	l.cfg.Budget.MaxCompilations = 1
	r, err := l.RunEpoch(context.Background(), "owner-budget", f.state, nil, nil)
	if err == nil {
		defer r.Close()
		if r.Promotion != nil || !r.Search.WorkAccounted || r.Search.Compilations != 1 {
			t.Fatalf("owner compile budget not enforced: %+v", r.Search)
		}
	}
	// The held-out evaluator is outside Search's budget; screening compiled once.
	if n := len(f.compiled); n < 1 || n > 2 || f.openOwners() != 0 {
		t.Fatalf("compiles=%d open=%d err=%v", n, f.openOwners(), err)
	}
	for _, x := range f.compiled[1:] {
		if x.op == "baseline" {
			t.Fatalf("second screening compile admitted: %v", x.id)
		}
	}
}

func TestNullCandidateRetainsNonAdditiveChannels(t *testing.T) {
	f, l := setup(t)
	l.cfg.Proposer = &FixedEnumerator{Edits: []json.RawMessage{json.RawMessage(`{"rules":[]}`)}}
	r := run(t, l, f)
	if r.Promotion != nil {
		t.Fatal("null comparison promoted")
	}
	for _, trial := range r.Search.Winner.Trials {
		if !trial.HasTiming(cudatrial.TimingSummedKernel) || !trial.HasTiming(cudatrial.TimingWallElapsed) {
			t.Fatal("channels lost")
		}
	}
	if reflect.DeepEqual(r.Search.Winner.Receipt.Timings, map[cudatrial.TimingKind]time.Duration{cudatrial.TimingWallElapsed: 100 * time.Nanosecond}) {
		t.Fatal("wall substitution")
	}
}

func TestBaselineVerificationFailureNamesBaseline(t *testing.T) {
	f, l := setup(t)
	base := l.cfg.Baseline.(*stubExecutable)
	l.cfg.Screen.Workload.ResetState = func() error { base.values = map[string][]float32{"out": {99, 99}}; return nil }
	_, err := l.RunEpoch(context.Background(), "baseline-failure", f.state, nil, nil)
	if err == nil || len(f.runs) != 0 {
		t.Fatalf("baseline failure timed: %v %v", err, f.runs)
	}
	for _, r := range f.archive.Records() {
		if r.Kind == "outcome" && (r.VerificationArm != "baseline" || r.VerificationIdentity != base.Identity() || r.Verification != "failed") {
			t.Fatalf("baseline blamed on candidate: %+v", r)
		}
	}
}

func TestCompileFailureQuarantineHasNoInventedIdentity(t *testing.T) {
	f, l := setup(t)
	compile := l.cfg.Compile
	l.cfg.Compile = func(ctx context.Context, e *egraph.Expr, b cudasearch.WorkloadBindings) (cudasearch.Executable, error) {
		if e.Op == "fast" {
			return nil, errors.New("stub compile failed")
		}
		return compile(ctx, e, b)
	}
	r := run(t, l, f)
	if r.Promotion != nil {
		t.Fatal("failed compilation promoted")
	}
	found := false
	for _, record := range f.archive.Records() {
		if record.Kind == "outcome" && record.StructuralID == "(fast x)" {
			found = true
			if record.CandidateID != "" || record.Compile != "failed" || !record.Unmeasured || record.Verification != "unverified" {
				t.Fatalf("compile failure=%+v", record)
			}
		}
	}
	if !found {
		t.Fatal("missing failed compile")
	}
}

func TestProposerIdentityMismatchRecordedAndRejected(t *testing.T) {
	f, l := setup(t)
	other := ModelIdentity{"model", "wrong-model", "v2"}
	l.cfg.Proposer = replyProposer{Reply{Model: other, Edit: json.RawMessage(`{"rules":["fast"]}`)}}
	if _, err := l.RunEpoch(context.Background(), "wrong-model", f.state, nil, nil); err == nil || len(f.events) != 0 {
		t.Fatal("mismatched model reached execution")
	}
	rs := f.archive.Records()
	last := rs[len(rs)-1]
	if last.Proposal.ReportedModel == nil || *last.Proposal.ReportedModel != other || last.Proposal.Model != f.cfg.ExpectedModel {
		t.Fatal("model mismatch evidence lost")
	}
}

func TestMeasuredZeroIsArtifactNotUnavailable(t *testing.T) {
	f, l := setup(t)
	f.fastDuration = 0
	r := run(t, l, f)
	if r.Promotion != nil {
		t.Fatal("measured zero promoted")
	}
	found := false
	for _, record := range f.archive.Records() {
		if record.Kind == "outcome" && strings.Contains(record.CandidateID, ":fast:") {
			found = true
			if record.Unmeasured || !strings.Contains(record.Reason, "claimed duration is zero") {
				t.Fatalf("zero confused with absence: %+v", record)
			}
		}
	}
	if !found {
		t.Fatal("zero candidate missing")
	}
}

func TestOwnerSpecificTimingFieldsRetained(t *testing.T) {
	f, l := setup(t)
	f.legacyTiming = true
	r := run(t, l, f)
	if r.Promotion == nil {
		t.Fatal("owner-specific positive duration rejected")
	}
	found := false
	for _, record := range f.archive.CandidateRecords(r.Search.Winner.ID) {
		for _, trial := range record.Trials {
			if trial.Name == "candidate" {
				found = true
				if trial.Timings[string(f.cfg.Timing)] != 50 {
					t.Fatalf("owner duration missing: %+v", trial)
				}
			}
		}
	}
	if !found {
		t.Fatal("no candidate duration")
	}
}
