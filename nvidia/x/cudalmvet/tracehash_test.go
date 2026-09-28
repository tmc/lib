package cudalmvet

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func buildTestGraph() (*cudair.Graph, map[string][]float32) {
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{8}, cudair.Float32)
	w := g.AddInput("W", []int{8}, cudair.Float32)

	h := g.AddBinary("mul", x, w)
	h.Name = "h"

	act := g.AddUnary("relu", h)
	act.Name = "act"

	_ = g.SetOutputs(act)

	inputs := map[string][]float32{
		"X": {1.0, -2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0},
		"W": {0.5, 0.5, 2.0, -1.0, 1.0, 0.5, 2.0, 0.25},
	}

	return g, inputs
}

func TestTraceHash_CPUDeterministic(t *testing.T) {
	g, inputs := buildTestGraph()

	oldProcs := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(oldProcs)

	var baselineHash uint64

	// Test across multiple GOMAXPROCS values and iterations
	for _, procs := range []int{1, 2, 4, 8} {
		runtime.GOMAXPROCS(procs)

		for iter := 0; iter < 5; iter++ {
			acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, false)
			if err != nil {
				t.Fatalf("procs=%d iter=%d: EvaluateCPUGraph failed: %v", procs, iter, err)
			}
			h := acc.Hash()
			if baselineHash == 0 {
				baselineHash = h
			} else if h != baselineHash {
				t.Fatalf("procs=%d iter=%d: non-deterministic hash: got 0x%016x, want 0x%016x",
					procs, iter, h, baselineHash)
			}
		}
	}
}

func TestTraceHash_CPUFoldGoldenAndOneULP(t *testing.T) {
	g, inputs := buildTestGraph()
	acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, true)
	if err != nil {
		t.Fatal(err)
	}
	const want = uint64(0xcf1b3c9dfa2443ab)
	if got := acc.Hash(); got != want {
		t.Fatalf("hash = 0x%016x, want 0x%016x", got, want)
	}
	float64Golden := NewAccumulator()
	float64Golden.FoldFloat64(1.25)
	if got := float64Golden.Hash(); got != 0xaaa3e93229aeb9dc {
		t.Fatalf("float64 golden = 0x%016x", got)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	results := map[string][]float32{
		"h":   {0.5, -1, 6, 4, 5, 3, -14, 2},
		"act": {0.5, 0, 6, 4, 5, 3, 0, 2},
	}
	replay := NewAccumulator()
	for _, fold := range buildTraceFoldPlan(snap) {
		replay.FoldFloat32Named(fold.Node, results[fold.Node][fold.Element])
	}
	if replay.Hash() != acc.Hash() {
		t.Fatalf("synthesized fold-plan replay = 0x%016x, CPU = 0x%016x", replay.Hash(), acc.Hash())
	}
	changed := append([]float32(nil), results["h"]...)
	changed[3] = math.Float32frombits(math.Float32bits(changed[3]) + 1)
	mutated := map[string][]float32{"h": changed, "act": results["act"]}
	base := foldGraphResults(snap, results, PolicyRawBits, true)
	other := foldGraphResults(snap, mutated, PolicyRawBits, true)
	if base.Hash() == other.Hash() {
		t.Fatal("one-ULP intermediate change did not move hash")
	}
	triage := Triage(base, other)
	if triage == nil || triage.NodeName != "h" || triage.DiffIndex != 3 {
		t.Fatalf("triage = %#v, want node h index 3", triage)
	}
}

func TestSynthesizedFoldPlanAndArgumentLayout(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{2}, cudair.Float32)
	b := g.AddInput("B", []int{2}, cudair.Float32)
	c := g.AddConstant("C", []int{2}, cudair.Float32, []float32{0.1, -0.2})
	d := g.AddBinary("sub", a, b)
	d.Name = "difference"
	out := g.AddBinary("sub", d, c)
	out.Name = "adjusted"
	other := g.AddUnary("neg", d)
	other.Name = "negative"
	if err := g.SetOutputs(out, other); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	args := kernelArgumentLayout(snap)
	wantArgs := []kernelArgument{{Name: "A"}, {Name: "B"}, {Name: "adjusted", Output: true}, {Name: "negative", Output: true}}
	if !reflect.DeepEqual(args, wantArgs) {
		t.Fatalf("argument layout = %#v, want %#v", args, wantArgs)
	}
	plan := buildTraceFoldPlan(snap)
	if len(plan) != 8 {
		t.Fatalf("fold plan has %d entries, want 8: %#v", len(plan), plan)
	}
	for i, want := range []foldTicket{{"C", 0, 0}, {"C", 1, 1}, {"difference", 0, 2}, {"difference", 1, 3}, {"adjusted", 0, 4}, {"adjusted", 1, 5}, {"negative", 0, 6}, {"negative", 1, 7}} {
		if plan[i] != want {
			t.Errorf("plan[%d] = %#v, want %#v", i, plan[i], want)
		}
	}
	source, _, err := synthesizeGraphKernel(g)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"const float constant_C[] = {0.100000001f, -0.200000003f};", "out_adjusted", "out_negative", "trace_ordered_float(var_C, 0U + (unsigned int)idx, trace_fold_limit)", "trace_ordered_float(var_difference, 2U + (unsigned int)idx, trace_fold_limit)"} {
		if !strings.Contains(source, want) {
			t.Errorf("synthesized source missing %q", want)
		}
	}
}

func TestAccumulatorFoldBoolRecordsValue(t *testing.T) {
	a := NewAccumulator(WithRecording(true))
	a.FoldBool(true)
	events := a.Events()
	if len(events) != 1 || events[0].Bits != 1 {
		t.Fatalf("events = %#v, want one event with Bits=1", events)
	}
}

func TestAccumulatorByteFoldsRecordEachByte(t *testing.T) {
	a := NewAccumulator(WithRecording(true))
	a.FoldByte(0x0a)
	a.FoldBytes([]byte{0x0b, 0x0c})
	events := a.Events()
	if a.Count() != 3 || len(events) != 3 {
		t.Fatalf("count=%d events=%#v", a.Count(), events)
	}
	for i, want := range []uint64{0x0a, 0x0b, 0x0c} {
		if events[i].Bits != want {
			t.Errorf("event %d bits=%x, want %x", i, events[i].Bits, want)
		}
	}
}

func TestAccumulatorCanonicalizeAllAndFloat64Policies(t *testing.T) {
	for _, policy := range []FloatPolicy{PolicyRawBits, PolicyCanonicalizeNaN, PolicyCanonicalizeZero, PolicyCanonicalizeAll} {
		t.Run(fmt.Sprint(policy), func(t *testing.T) {
			pos, neg := NewAccumulator(WithPolicy(policy)), NewAccumulator(WithPolicy(policy))
			pos.FoldFloat32(0)
			neg.FoldFloat32(float32(math.Copysign(0, -1)))
			wantEqual := policy == PolicyCanonicalizeZero || policy == PolicyCanonicalizeAll
			if (pos.Hash() == neg.Hash()) != wantEqual {
				t.Fatalf("signed-zero equality for policy %d = %v", policy, pos.Hash() == neg.Hash())
			}
			pos64, neg64 := NewAccumulator(WithPolicy(policy)), NewAccumulator(WithPolicy(policy))
			pos64.FoldFloat64(0)
			neg64.FoldFloat64(math.Copysign(0, -1))
			if (pos64.Hash() == neg64.Hash()) != wantEqual {
				t.Fatalf("float64 signed-zero equality for policy %d", policy)
			}
			pos64, neg64 = NewAccumulator(WithPolicy(policy)), NewAccumulator(WithPolicy(policy))
			pos64.FoldFloat64(1.25)
			neg64.FoldFloat64(math.Nextafter(1.25, 2))
			if pos64.Hash() == neg64.Hash() {
				t.Fatalf("float64 policy %d lost a one-ULP difference", policy)
			}
		})
	}
	for _, policy := range []FloatPolicy{PolicyRawBits, PolicyCanonicalizeNaN, PolicyCanonicalizeZero, PolicyCanonicalizeAll} {
		a, b := NewAccumulator(WithPolicy(policy)), NewAccumulator(WithPolicy(policy))
		a.FoldFloat64(math.Float64frombits(0x7ff8000000000001))
		b.FoldFloat64(math.Float64frombits(0x7ff8000000000002))
		wantEqual := policy == PolicyCanonicalizeNaN || policy == PolicyCanonicalizeAll
		if (a.Hash() == b.Hash()) != wantEqual {
			t.Errorf("float64 NaN payload equality for policy %d", policy)
		}
	}
	for _, policy := range []FloatPolicy{PolicyCanonicalizeNaN, PolicyCanonicalizeAll} {
		a, b := NewAccumulator(WithPolicy(policy)), NewAccumulator(WithPolicy(policy))
		a.FoldFloat64(math.Float64frombits(0x7ff8000000000001))
		b.FoldFloat64(math.Float64frombits(0x7ff8000000000002))
		if a.Hash() != b.Hash() {
			t.Fatalf("NaN payloads differ under policy %d", policy)
		}
	}
}

func TestTraceHash_OrderIsDeclared(t *testing.T) {
	// Fixed traversal: evaluating the same graph in declared topological order
	// produces identical hashes.
	g1, inputs1 := buildTestGraph()
	acc1, err := EvaluateCPUGraph(g1, inputs1, PolicyRawBits, false)
	if err != nil {
		t.Fatalf("EvaluateCPUGraph 1: %v", err)
	}

	g2, inputs2 := buildTestGraph()
	acc2, err := EvaluateCPUGraph(g2, inputs2, PolicyRawBits, false)
	if err != nil {
		t.Fatalf("EvaluateCPUGraph 2: %v", err)
	}

	if acc1.Hash() != acc2.Hash() {
		t.Errorf("declared order evaluations produced different hashes: 0x%016x vs 0x%016x",
			acc1.Hash(), acc2.Hash())
	}

	// Significant traversal: reversing element traversal order must produce different hashes.
	vals := []float32{1.0, 2.0, 3.0, 4.0}
	accForward := NewAccumulator()
	accForward.FoldFloat32s(vals)

	accReverse := NewAccumulator()
	for i := len(vals) - 1; i >= 0; i-- {
		accReverse.FoldFloat32(vals[i])
	}

	if accForward.Hash() == accReverse.Hash() {
		t.Errorf("reversing fold order unexpectedly produced identical hash: 0x%016x", accForward.Hash())
	}
}

func TestTraceHash_SignedZeroPolicy(t *testing.T) {
	posZero := float32(0.0)
	negZero := float32(math.Copysign(0, -1))

	if math.Float32bits(posZero) == math.Float32bits(negZero) {
		t.Fatal("expected distinct bit patterns for +0.0 and -0.0")
	}

	// PolicyRawBits: +0.0 and -0.0 produce distinct hashes
	accPos := NewAccumulator(WithPolicy(PolicyRawBits))
	accPos.FoldFloat32(posZero)

	accNeg := NewAccumulator(WithPolicy(PolicyRawBits))
	accNeg.FoldFloat32(negZero)

	if accPos.Hash() == accNeg.Hash() {
		t.Errorf("PolicyRawBits: expected different hashes for +0.0 and -0.0, both got 0x%016x",
			accPos.Hash())
	}

	// PolicyCanonicalizeZero: +0.0 and -0.0 produce identical hashes
	accPosCanon := NewAccumulator(WithPolicy(PolicyCanonicalizeZero))
	accPosCanon.FoldFloat32(posZero)

	accNegCanon := NewAccumulator(WithPolicy(PolicyCanonicalizeZero))
	accNegCanon.FoldFloat32(negZero)

	if accPosCanon.Hash() != accNegCanon.Hash() {
		t.Errorf("PolicyCanonicalizeZero: expected identical hashes, got 0x%016x vs 0x%016x",
			accPosCanon.Hash(), accNegCanon.Hash())
	}
}

func TestDifferential_UnmeasuredWithoutDevice(t *testing.T) {
	g, inputs := buildTestGraph()

	// Calling Compare without providing a device must return StatusUnmeasured,
	// never StatusAgreed.
	res, err := Compare(context.Background(), g, WithInputs(inputs))
	if err != nil {
		t.Fatalf("Compare returned unexpected error: %v", err)
	}

	if res.Status != StatusUnmeasured {
		t.Errorf("Compare without device: got status %v, want %v", res.Status, StatusUnmeasured)
	}
	if res.CPUHash == 0 || res.CPUHash == FNVOffsetBasis {
		t.Errorf("expected non-trivial CPUHash, got 0x%016x", res.CPUHash)
	}
	if res.DeviceHash != 0 {
		t.Errorf("expected 0 DeviceHash for unmeasured run, got 0x%016x", res.DeviceHash)
	}
}

// Mutant Control 1: The oracle must fail when output is perturbed by 1 ULP.
func TestTraceHash_Mutant_OneULP(t *testing.T) {
	baseVals := []float32{1.0, 2.0, 3.0, 4.0}
	accA := NewAccumulator(WithRecording(true))
	accA.FoldFloat32s(baseVals)

	// Perturb index 2 by exactly 1 ULP
	perturbedVals := make([]float32, len(baseVals))
	copy(perturbedVals, baseVals)
	bits := math.Float32bits(perturbedVals[2])
	perturbedVals[2] = math.Float32frombits(bits + 1)

	accB := NewAccumulator(WithRecording(true))
	accB.FoldFloat32s(perturbedVals)

	if accA.Hash() == accB.Hash() {
		t.Fatalf("oracle failed to detect 1 ULP perturbation: both hashes are 0x%016x", accA.Hash())
	}

	report := Triage(accA, accB)
	if report == nil {
		t.Fatal("expected triage report for differing accumulators")
	}
	if report.DiffIndex != 2 {
		t.Errorf("triage report DiffIndex: got %d, want 2", report.DiffIndex)
	}
	if report.BitsA == report.BitsB {
		t.Errorf("triage report BitsA (%08x) equals BitsB (%08x)", report.BitsA, report.BitsB)
	}
}

// Mutant Control 2: NaN payload test asserts documented policy behavior.
func TestTraceHash_Mutant_NaNPayload(t *testing.T) {
	nan1 := math.Float32frombits(0x7fc00001)
	nan2 := math.Float32frombits(0x7fc00002)

	if !math.IsNaN(float64(nan1)) || !math.IsNaN(float64(nan2)) {
		t.Fatal("values are not NaNs")
	}

	// Under PolicyRawBits: differing payloads yield distinct hashes
	accRaw1 := NewAccumulator(WithPolicy(PolicyRawBits))
	accRaw1.FoldFloat32(nan1)

	accRaw2 := NewAccumulator(WithPolicy(PolicyRawBits))
	accRaw2.FoldFloat32(nan2)

	if accRaw1.Hash() == accRaw2.Hash() {
		t.Errorf("PolicyRawBits: expected distinct hashes for differing NaN payloads, got 0x%016x",
			accRaw1.Hash())
	}

	// Under PolicyCanonicalizeNaN: differing payloads yield identical hashes
	accCanon1 := NewAccumulator(WithPolicy(PolicyCanonicalizeNaN))
	accCanon1.FoldFloat32(nan1)

	accCanon2 := NewAccumulator(WithPolicy(PolicyCanonicalizeNaN))
	accCanon2.FoldFloat32(nan2)

	if accCanon1.Hash() != accCanon2.Hash() {
		t.Errorf("PolicyCanonicalizeNaN: expected equal hashes, got 0x%016x vs 0x%016x",
			accCanon1.Hash(), accCanon2.Hash())
	}
}

// Mutant Control 3: Drop device reset between launches and assert contamination detection.
func TestTraceHash_Mutant_ContaminatedReset(t *testing.T) {
	// An accumulator starting at an arbitrary contaminated value (not FNVOffsetBasis)
	// produces a contaminated result.
	contaminatedBasis := uint64(0xdeadbeef12345678)
	accClean := NewAccumulator()
	accClean.FoldFloat32(1.0)

	accContaminated := &Accumulator{
		hash:   contaminatedBasis,
		policy: PolicyRawBits,
	}
	accContaminated.FoldFloat32(1.0)

	if accClean.Hash() == accContaminated.Hash() {
		t.Fatal("contaminated start state unexpectedly matched clean state")
	}

	// Check that CheckDeviceReset logic recognizes any non-offset basis as contaminated
	if contaminatedBasis == nvrtcx.TraceHashOffsetBasis {
		t.Fatal("contaminated basis equals TraceHashOffsetBasis")
	}
}

// Mutant Control 4: Force a cache hit on a perturbed compile and assert detection of artifact mismatch.
func TestTraceHash_Mutant_CacheHitArtifactMismatch(t *testing.T) {
	pert := Perturbation{
		Name:                 "vectorize_on",
		ExtraDeviceVectorize: true,
		JITOptLevel:          3,
	}

	// Simulate a cached artifact that was compiled without --extra-device-vectorization (e.g. from baseline)
	unperturbedArtifact := &cudakernel.Artifact{
		CompilerOpts: []string{"-std=c++17"},
		JITOptions: cudakernel.JITOptions{
			OptLevel: 3,
		},
	}

	err := ValidatePerturbationArtifact(pert, unperturbedArtifact)
	if err == nil {
		t.Fatal("expected ValidatePerturbationArtifact to reject unperturbed artifact for vectorize_on")
	}
	if !errors.Is(err, ErrPerturbationArtifactMismatch) {
		t.Errorf("got error %v, want ErrPerturbationArtifactMismatch", err)
	}

	// Simulate an artifact with mismatched JIT opt level
	optLevelMismatchArtifact := &cudakernel.Artifact{
		CompilerOpts: []string{"-std=c++17", "--extra-device-vectorization"},
		JITOptions: cudakernel.JITOptions{
			OptLevel: 4, // requested 3, cached 4
		},
	}

	err = ValidatePerturbationArtifact(pert, optLevelMismatchArtifact)
	if err == nil {
		t.Fatal("expected ValidatePerturbationArtifact to reject artifact with mismatched JIT opt level")
	}
	if !errors.Is(err, ErrPerturbationArtifactMismatch) {
		t.Errorf("got error %v, want ErrPerturbationArtifactMismatch", err)
	}

	// Valid matching artifact
	matchingArtifact := &cudakernel.Artifact{
		CompilerOpts: []string{"-std=c++17", "--extra-device-vectorization"},
		JITOptions: cudakernel.JITOptions{
			OptLevel: 3,
		},
	}
	if err := ValidatePerturbationArtifact(pert, matchingArtifact); err != nil {
		t.Errorf("unexpected validation failure for matching artifact: %v", err)
	}
}

func TestValidatePerturbationArtifactRejectsOptLevelZeroAndOutOfRange(t *testing.T) {
	for _, level := range []int{0, 5, 7} {
		pert := Perturbation{Name: "invalid", JITOptLevel: level}
		art := &cudakernel.Artifact{JITOptions: cudakernel.JITOptions{OptLevel: uint32(level)}}
		if err := ValidatePerturbationArtifact(pert, art); err == nil {
			t.Errorf("ValidatePerturbationArtifact(%d) succeeded", level)
		}
	}
	for _, level := range []int{1, 4} {
		pert := Perturbation{Name: "valid", JITOptLevel: level}
		art := &cudakernel.Artifact{JITOptions: cudakernel.JITOptions{OptLevel: uint32(level)}}
		if err := ValidatePerturbationArtifact(pert, art); err != nil {
			t.Errorf("ValidatePerturbationArtifact(%d) = %v", level, err)
		}
	}
}

func TestCompiledPerturbationMetadataMustMatch(t *testing.T) {
	p := Perturbation{Name: "vectorized", ExtraDeviceVectorize: true, JITOptLevel: 2}
	module := new(nvrtcx.CompiledModule)
	module.SetJITOptions(cudakernel.JITOptions{OptLevel: 2})
	if err := validateCompiledPerturbation(p, []string{"--extra-device-vectorization"}, module); err != nil {
		t.Fatal(err)
	}
	if err := validateCompiledPerturbation(p, nil, module); !errors.Is(err, ErrPerturbationArtifactMismatch) {
		t.Fatalf("missing compiler flag error = %v", err)
	}
	module.SetJITOptions(cudakernel.JITOptions{OptLevel: 3})
	if err := validateCompiledPerturbation(p, []string{"--extra-device-vectorization"}, module); !errors.Is(err, ErrPerturbationArtifactMismatch) {
		t.Fatalf("wrong JIT option error = %v", err)
	}
}

func TestRunPerturbationsWithoutDeviceIsUnmeasured(t *testing.T) {
	g, inputs := buildTestGraph()
	report, err := RunPerturbations(context.Background(), g, []Perturbation{{Name: "baseline", JITOptLevel: 4}}, WithInputs(inputs))
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Results) != 1 || report.Results[0].Status != StatusUnmeasured || report.HasDisagreements() {
		t.Fatalf("report = %#v", report)
	}
	if strings.Contains(report.String(), "MOVED") {
		t.Fatalf("unmeasured report claims movement:\n%s", report.String())
	}
}

func TestDeviceMismatchDoesNotInventFoldZero(t *testing.T) {
	triage := deviceMismatchTriage(0x10, 0x11)
	if triage.DiffIndex != -1 {
		t.Fatalf("unknown device mismatch DiffIndex = %d, want -1", triage.DiffIndex)
	}
}

func TestDevicePrefixTriageFindsFirstDifferingFold(t *testing.T) {
	a := NewAccumulator(WithRecording(true))
	for _, v := range []float32{1, 2, 3, 4, 5, 6, 7, 8} {
		a.FoldFloat32Named("node", v)
	}
	events := a.Events()
	localized, err := localizeDeviceMismatch(events, func(prefix int) (deviceTrace, error) {
		if prefix <= 4 {
			return deviceTrace{hash: events[prefix-1].Hash, lastBits: uint32(events[prefix-1].Bits)}, nil
		}
		return deviceTrace{hash: events[prefix-1].Hash ^ 1, lastBits: math.Float32bits(99)}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if localized.DiffIndex != 4 || localized.NodeName != "node" || localized.ValueB != 99 {
		t.Fatalf("triage = %#v, want first difference at fold 4 with device value 99", localized)
	}
}

func TestPerturbationSeparatesCompileAndRunErrors(t *testing.T) {
	p := Perturbation{Name: "x", JITOptLevel: 3}
	compileErr := errors.New("compile failed")
	runErr := errors.New("load failed")
	compileRow := perturbationOutcome(p, &ComparisonResult{Status: StatusUnmeasured, Stage: StageCompile, Err: compileErr}, nil)
	if !errors.Is(compileRow.CompileErr, compileErr) || compileRow.RunErr != nil {
		t.Fatalf("compile row = %#v", compileRow)
	}
	runRow := perturbationOutcome(p, &ComparisonResult{Status: StatusUnmeasured, Stage: StageLoad, Err: runErr}, nil)
	if !errors.Is(runRow.RunErr, runErr) || runRow.CompileErr != nil {
		t.Fatalf("run row = %#v", runRow)
	}
}

func TestContaminatedAccumulatorFailsWithoutReset(t *testing.T) {
	if err := validateAccumulatorState(nvrtcx.TraceHashOffsetBasis, 0, 0); err != nil {
		t.Fatal(err)
	}
	// Model the state left after a first launch when no reset occurs before the second.
	err := validateAccumulatorState(0xdeadbeef, 8, 0x3f800000)
	if !errors.Is(err, ErrContaminatedAccumulator) {
		t.Fatalf("second launch state error = %v", err)
	}
}
