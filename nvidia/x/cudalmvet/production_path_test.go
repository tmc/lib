package cudalmvet

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type fakeTraceModule struct {
	hash       uint64
	lastBits   uint32
	stateHash  uint64
	next       uint32
	last       uint32
	resetCalls int
	checkCalls int
	generated  []bool
	checkErr   error
}

func (m *fakeTraceModule) CheckReset() error {
	m.checkCalls++
	if m.checkErr != nil {
		return m.checkErr
	}
	return validateAccumulatorState(m.stateHash, m.next, m.last)
}

func (m *fakeTraceModule) Execute(_ context.Context, _ string, _ *CompareConfig, _ *cudair.Snapshot, _ uint32, reset, generated bool) (deviceTrace, error) {
	if reset {
		m.resetCalls++
		m.stateHash, m.next, m.last = FNVOffsetBasis, 0, 0
	}
	m.generated = append(m.generated, generated)
	return deviceTrace{hash: m.hash, lastBits: m.lastBits}, nil
}

func (*fakeTraceModule) Close() error { return nil }

func fakeGraphWithLength(n int) (*cudair.Graph, map[string][]float32) {
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{n}, cudair.Float32)
	y := g.AddUnary("neg", x)
	y.Name = "Y"
	_ = g.SetOutputs(y)
	return g, map[string][]float32{"X": make([]float32, n)}
}

func newFakeCompiled(jit cudakernel.JITOptions) *nvrtcx.CompiledModule {
	mod := new(nvrtcx.CompiledModule)
	mod.SetJITOptions(jit)
	return mod
}

func TestCompareRejectsNonRawDeviceFloatPolicy(t *testing.T) {
	g, inputs := fakeGraphWithLength(1)
	inputs["X"][0] = 0
	compileCalls, loadCalls := 0, 0
	compile := func(string, []nvrtcx.Option, cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
		compileCalls++
		return newFakeCompiled(cudakernel.JITOptions{}), nil
	}
	res, err := Compare(context.Background(), g,
		WithInputs(inputs),
		WithFloatPolicy(PolicyCanonicalizeZero),
		withCompileKernel(compile),
		withTraceModuleLoader(func(*nvrtcx.CompiledModule, *cuda.Device) (traceModule, error) {
			loadCalls++
			return &fakeTraceModule{stateHash: FNVOffsetBasis}, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusUnmeasured || res.Stage != StagePolicy || res.Err == nil || !strings.Contains(res.Err.Error(), "PolicyRawBits") {
		t.Fatalf("Compare result = %#v, want explicit unmeasured device policy refusal", res)
	}
	if compileCalls != 0 || loadCalls != 0 {
		t.Fatalf("policy refusal reached compile/load: compile=%d load=%d", compileCalls, loadCalls)
	}

	report, err := RunPerturbations(context.Background(), g, []Perturbation{{Name: "zero-policy", JITOptLevel: 2}},
		WithInputs(inputs), WithFloatPolicy(PolicyCanonicalizeZero), withCompileKernel(compile),
		withTraceModuleLoader(func(*nvrtcx.CompiledModule, *cuda.Device) (traceModule, error) {
			loadCalls++
			return &fakeTraceModule{stateHash: FNVOffsetBasis}, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	row := report.Results[0]
	if row.Status != StatusUnmeasured || row.PolicyErr == nil || row.ArtifactErr != nil || row.CompileErr != nil {
		t.Fatalf("non-RawBits row = %#v, want unmeasured policy error", row)
	}
	if compileCalls != 0 || loadCalls != 0 {
		t.Fatalf("RunPerturbations policy refusal reached compile/load: compile=%d load=%d", compileCalls, loadCalls)
	}
}

func TestCompareAndPerturbationsRefuseNaNTracesBeforeCompile(t *testing.T) {
	nanPayload := math.Float32frombits(0x7fc00001)
	type nanGraphCase struct {
		name   string
		graph  *cudair.Graph
		inputs map[string][]float32
	}
	makeUnary := func(name, op string, value float32) nanGraphCase {
		g := cudair.NewGraph()
		x := g.AddInput("X", []int{1}, cudair.Float32)
		y := g.AddUnary(op, x)
		y.Name = "Y"
		_ = g.SetOutputs(y)
		return nanGraphCase{name: name, graph: g, inputs: map[string][]float32{"X": {value}}}
	}
	mul := cudair.NewGraph()
	a := mul.AddInput("A", []int{1}, cudair.Float32)
	b := mul.AddInput("B", []int{1}, cudair.Float32)
	y := mul.AddBinary("mul", a, b)
	y.Name = "Y"
	_ = mul.SetOutputs(y)
	graphs := []nanGraphCase{
		makeUnary("neg-payload", "neg", nanPayload),
		makeUnary("relu-nan", "relu", nanPayload),
		{name: "mul-zero-infinity", graph: mul, inputs: map[string][]float32{"A": {0}, "B": {float32(math.Inf(1))}}},
	}
	for _, tc := range graphs {
		t.Run(tc.name, func(t *testing.T) {
			compileCalls, loadCalls := 0, 0
			compile := func(string, []nvrtcx.Option, cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
				compileCalls++
				return newFakeCompiled(cudakernel.JITOptions{}), nil
			}
			load := func(*nvrtcx.CompiledModule, *cuda.Device) (traceModule, error) {
				loadCalls++
				return &fakeTraceModule{stateHash: FNVOffsetBasis}, nil
			}
			res, err := Compare(context.Background(), tc.graph, WithInputs(tc.inputs), withCompileKernel(compile), withTraceModuleLoader(load))
			if err != nil {
				t.Fatal(err)
			}
			if res.Status != StatusUnmeasured || res.Stage != StagePolicy || !errors.Is(res.Err, ErrNaNTraceUnsupported) {
				t.Fatalf("Compare = %#v, want NaN policy refusal", res)
			}
			if compileCalls != 0 || loadCalls != 0 {
				t.Fatalf("Compare reached compile/load: %d/%d", compileCalls, loadCalls)
			}

			report, err := RunPerturbations(context.Background(), tc.graph,
				[]Perturbation{{Name: "one", JITOptLevel: 4}, {Name: "two", JITOptLevel: 2}},
				WithInputs(tc.inputs), withCompileKernel(compile), withTraceModuleLoader(load))
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Results) != 2 {
				t.Fatalf("rows = %d, want 2", len(report.Results))
			}
			for _, row := range report.Results {
				if row.Status != StatusUnmeasured || row.Stage != StagePolicy || !errors.Is(row.PolicyErr, ErrNaNTraceUnsupported) || row.CompileErr != nil || row.RunErr != nil {
					t.Errorf("row = %#v, want NaN policy refusal", row)
				}
			}
			if compileCalls != 0 || loadCalls != 0 {
				t.Fatalf("RunPerturbations reached compile/load: %d/%d", compileCalls, loadCalls)
			}
		})
	}
}

func TestRunPerturbationsProductionPathOptionsAndOutcomes(t *testing.T) {
	g, inputs := buildTestGraph()
	ref, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, false)
	if err != nil {
		t.Fatal(err)
	}
	perts := []Perturbation{
		{Name: "vectorized-opt3", ExtraDeviceVectorize: true, JITOptLevel: 3},
		{Name: "bad-artifact-opt3", ExtraDeviceVectorize: true, JITOptLevel: 3},
		{Name: "load-failure-opt2", JITOptLevel: 2},
	}
	var flagsByCall [][]string
	var jitByCall []cudakernel.JITOptions
	compile := func(_ string, opts []nvrtcx.Option, jit cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
		flags, err := nvrtcx.CompileFlags(opts...)
		if err != nil {
			return nil, err
		}
		flagsByCall = append(flagsByCall, flags)
		jitByCall = append(jitByCall, jit)
		if !containsString(flags, "--no-cache") {
			return nil, errors.New("perturbed compile did not disable cache")
		}
		return newFakeCompiled(jit), nil
	}
	loadCalls := 0
	load := func(mod *nvrtcx.CompiledModule, _ *cuda.Device) (traceModule, error) {
		loadCalls++
		if loadCalls == 2 {
			return nil, errors.New("injected module load failure")
		}
		return &fakeTraceModule{hash: ref.Hash(), stateHash: FNVOffsetBasis, lastBits: 0x3f800000}, nil
	}

	compileIndex := 0
	compileWithMismatch := func(src string, opts []nvrtcx.Option, jit cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
		mod, err := compile(src, opts, jit)
		if err != nil {
			return nil, err
		}
		if compileIndex == 1 {
			mod.SetJITOptions(cudakernel.JITOptions{OptLevel: 2})
		}
		compileIndex++
		return mod, nil
	}
	report, err := RunPerturbations(context.Background(), g, perts, WithInputs(inputs),
		withCompileKernel(compileWithMismatch), withTraceModuleLoader(load))
	if err != nil {
		t.Fatal(err)
	}
	if len(flagsByCall) != 3 || len(jitByCall) != 3 {
		t.Fatalf("compile calls = %d, JIT observations = %d, want 3 each", len(flagsByCall), len(jitByCall))
	}
	for i, want := range []uint32{3, 3, 2} {
		if jitByCall[i].OptLevel != want {
			t.Errorf("compile %d JIT opt level = %d, want %d", i, jitByCall[i].OptLevel, want)
		}
		if !containsString(flagsByCall[i], "--no-cache") {
			t.Errorf("compile %d flags omit --no-cache: %v", i, flagsByCall[i])
		}
	}
	for i := 0; i < 2; i++ {
		if !containsString(flagsByCall[i], "--extra-device-vectorization") {
			t.Errorf("compile %d flags omit vectorization: %v", i, flagsByCall[i])
		}
	}
	if containsString(flagsByCall[2], "--extra-device-vectorization") {
		t.Errorf("compile 2 unexpectedly has vectorization flag: %v", flagsByCall[2])
	}
	if report.Results[0].Status != StatusAgreed || report.Results[0].CompileErr != nil || report.Results[0].RunErr != nil {
		t.Errorf("successful production-path row = %#v", report.Results[0])
	}
	bad := report.Results[1]
	if bad.Status != StatusUnmeasured || bad.ArtifactErr == nil || bad.CompileErr != nil || bad.RunErr != nil {
		t.Errorf("artifact mismatch row = %#v", bad)
	}
	loadFailure := report.Results[2]
	if loadFailure.Status != StatusUnmeasured || loadFailure.RunErr == nil || loadFailure.CompileErr != nil || loadFailure.ArtifactErr != nil {
		t.Errorf("load failure row = %#v", loadFailure)
	}
}

func TestCompareChecksCleanResetAndPrefixResetsThroughRuntime(t *testing.T) {
	g, inputs := buildTestGraph()
	compile := func(string, []nvrtcx.Option, cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
		return newFakeCompiled(cudakernel.JITOptions{}), nil
	}
	dirty := &fakeTraceModule{hash: 1, stateHash: FNVOffsetBasis, next: 1, last: 0}
	res, err := Compare(context.Background(), g, WithInputs(inputs), withCompileKernel(compile),
		withTraceModuleLoader(func(*nvrtcx.CompiledModule, *cuda.Device) (traceModule, error) { return dirty, nil }))
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusUnmeasured || res.Stage != StageLaunch || !errors.Is(res.Err, ErrContaminatedAccumulator) {
		t.Fatalf("ticket-only contamination result = %#v", res)
	}
	if dirty.checkCalls != 1 || len(dirty.generated) != 0 {
		t.Fatalf("dirty state calls: checks=%d executes=%v", dirty.checkCalls, dirty.generated)
	}

	acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, false)
	if err != nil {
		t.Fatal(err)
	}
	runner := &fakeTraceModule{hash: acc.Hash() ^ 1, stateHash: FNVOffsetBasis, lastBits: 0x40000000}
	res, err = Compare(context.Background(), g, WithInputs(inputs), withCompileKernel(compile),
		withTraceModuleLoader(func(*nvrtcx.CompiledModule, *cuda.Device) (traceModule, error) { return runner, nil }))
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusDisagreed || runner.checkCalls != 1 || runner.resetCalls == 0 {
		t.Fatalf("prefix path result=%#v checks=%d resets=%d", res, runner.checkCalls, runner.resetCalls)
	}
}

func TestGeneratedElementLimitRunsBeforeCompile(t *testing.T) {
	for _, n := range []int{1024, 1025} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			g, inputs := fakeGraphWithLength(n)
			snap, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			got, err := generatedKernelElementCount(snap, inputs, true)
			if n == 1024 {
				if err != nil || got != n {
					t.Fatalf("limit check = (%d, %v), want (%d, nil)", got, err, n)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "1024") {
				t.Fatalf("1025-element limit error = %v, want named 1024 limit", err)
			}
			called := false
			res, err := Compare(context.Background(), g, WithInputs(inputs),
				withCompileKernel(func(string, []nvrtcx.Option, cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
					called = true
					return nil, errors.New("compiler should not be reached")
				}),
				withTraceModuleLoader(func(*nvrtcx.CompiledModule, *cuda.Device) (traceModule, error) { return nil, nil }))
			if err != nil || res.Stage != StageCompile || res.Err == nil || called {
				t.Fatalf("oversized Compare result=%#v err=%v compileCalled=%v", res, err, called)
			}
		})
	}
}

func TestCustomKernelOmitsGeneratedFoldArgumentAndLocalization(t *testing.T) {
	if got, want := launchScalarArguments(false), []launchScalarArgument{launchElementCount}; !reflect.DeepEqual(got, want) {
		t.Fatalf("custom scalar args = %v, want %v", got, want)
	}
	if got, want := launchScalarArguments(true), []launchScalarArgument{launchElementCount, launchTraceFoldLimit}; !reflect.DeepEqual(got, want) {
		t.Fatalf("generated scalar args = %v, want %v", got, want)
	}
	g, inputs := buildTestGraph()
	acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, false)
	if err != nil {
		t.Fatal(err)
	}
	runner := &fakeTraceModule{hash: acc.Hash() ^ 1, stateHash: FNVOffsetBasis}
	res, err := Compare(context.Background(), g, WithInputs(inputs), WithKernelSource("extern \"C\" __global__ void custom() {}", "custom"),
		withCompileKernel(func(string, []nvrtcx.Option, cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
			return newFakeCompiled(cudakernel.JITOptions{}), nil
		}),
		withTraceModuleLoader(func(*nvrtcx.CompiledModule, *cuda.Device) (traceModule, error) { return runner, nil }))
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusDisagreed || res.Triage == nil || res.Triage.DiffIndex != -1 {
		t.Fatalf("custom-kernel mismatch = %#v, want unlocalized disagreement", res)
	}
	if !reflect.DeepEqual(runner.generated, []bool{false}) {
		t.Fatalf("custom kernel generated flags = %v, want [false]", runner.generated)
	}
}

func TestPerturbationReportMovedUsesBaselineDeviceHash(t *testing.T) {
	report := &PerturbationReport{
		ReferenceHash:      0x111,
		BaselineDeviceHash: 0x222,
		Results: []PerturbationResult{
			{Perturbation: Perturbation{Name: "same"}, Status: StatusDisagreed, Hash: 0x222},
			{Perturbation: Perturbation{Name: "moved"}, Status: StatusDisagreed, Hash: 0x333},
			{Perturbation: Perturbation{Name: "cpu-agreement"}, Status: StatusAgreed, Hash: 0x111},
		},
	}
	got := report.String()
	if strings.Count(got, "HASH MOVED vs baseline device hash") != 2 {
		t.Fatalf("report movement annotations wrong:\n%s", got)
	}
	for _, line := range strings.Split(got, "\n") {
		if strings.Contains(line, "same ") && strings.Contains(line, "HASH MOVED") {
			t.Fatalf("same-device-hash row is marked MOVED:\n%s", got)
		}
		if (strings.Contains(line, "moved ") || strings.Contains(line, "cpu-agreement ")) && !strings.Contains(line, "HASH MOVED") {
			t.Fatalf("different baseline-device hash row is not marked MOVED:\n%s", got)
		}
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
