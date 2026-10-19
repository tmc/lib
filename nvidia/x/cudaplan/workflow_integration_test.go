package cudaplan_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const (
	workflowInputNodes   = 3
	workflowComputeNodes = 3
)

type workflowChainFixture struct {
	Snap      *cudair.Snapshot
	Schema    cudakernel.Schema
	Artifacts map[string]*cudakernel.Artifact
	NodeNames []string
	Output    string
}

// buildWorkflowChainFixture builds the shared deterministic three-compute-node
// FP32 chain (p = x * a; q = p + b; y = q * a) over 64 elements with separate
// multiplication and addition rounding and the three-pointer input/input/output
// ABI. All denominators distinguish the 3 input nodes (x, a, b) from the 3
// executable compute nodes (p, q, y).
func buildWorkflowChainFixture(t *testing.T, target string) workflowChainFixture {
	t.Helper()
	g := cudair.NewGraph()
	shape := []int{replayElements}
	x := g.AddInput("x", shape, cudair.Float32)
	a := g.AddInput("a", shape, cudair.Float32)
	b := g.AddInput("b", shape, cudair.Float32)

	p := g.AddBinary("mul", x, a)
	q := g.AddBinary("add", p, b)
	y := g.AddBinary("mul", q, a)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got := len(snap.Inputs()); got != workflowInputNodes {
		t.Fatalf("input nodes = %d, want %d", got, workflowInputNodes)
	}
	if got := len(snap.Order()) - len(snap.Inputs()); got != workflowComputeNodes {
		t.Fatalf("compute nodes = %d, want %d", got, workflowComputeNodes)
	}

	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "a", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "b", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	artifacts := make(map[string]*cudakernel.Artifact, 2)
	for _, op := range []string{"mul", "add"} {
		ptx := workflowBinaryPTX(target, op)
		art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(ptx), "apply", schema, target)
		if err != nil {
			t.Fatal(err)
		}
		artifacts[op] = art
	}
	return workflowChainFixture{
		Snap:      snap,
		Schema:    schema,
		Artifacts: artifacts,
		NodeNames: []string{p.Name, q.Name, y.Name},
		Output:    y.Name,
	}
}

func workflowBinaryPTX(target, op string) string {
	return fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry apply(.param .u64 a, .param .u64 b, .param .u64 out) {
 .reg .pred p;
 .reg .b32 tid;
 .reg .b64 pa, pb, po, offset;
 .reg .f32 va, vb, result;
 mov.u32 tid, %%tid.x;
 setp.ge.u32 p, tid, 64;
 @p bra done;
 ld.param.u64 pa, [a];
 ld.param.u64 pb, [b];
 ld.param.u64 po, [out];
 mul.wide.u32 offset, tid, 4;
 add.u64 pa, pa, offset;
 add.u64 pb, pb, offset;
 add.u64 po, po, offset;
 ld.global.f32 va, [pa];
 ld.global.f32 vb, [pb];
 %s.rn.f32 result, va, vb;
 st.global.f32 [po], result;
done:
 ret;
}
`, target, op)
}

// workflowChainOracle computes p = x * a, q = p + b, y = q * a element-by-element
// in float32 with separate rounding at each operation boundary.
func workflowChainOracle(data map[string][]float32) ([]float32, []float32, []float32) {
	x, a, b := data["x"], data["a"], data["b"]
	p := make([]float32, len(x))
	q := make([]float32, len(x))
	y := make([]float32, len(x))
	for i := range x {
		p[i] = float32(x[i] * a[i])
		q[i] = float32(p[i] + b[i])
		y[i] = float32(q[i] * a[i])
	}
	return p, q, y
}

func TestWorkflowThreeNodeIdentityAndCache(t *testing.T) {
	const target = "sm_121a"
	fx := buildWorkflowChainFixture(t, target)

	for round := range 3 {
		data := replayInputs(round)
		_, _, wantY := workflowChainOracle(data)
		refMap, err := cudaref.EvaluateSnapshot(fx.Snap, data)
		if err != nil {
			t.Fatalf("round %d EvaluateSnapshot: %v", round, err)
		}
		gotRef := refMap[fx.Output]
		if len(gotRef) != replayElements || !sameReplayOutput(gotRef, wantY) {
			t.Fatalf("round %d cudaref disagrees with independent FP32 oracle: got=%v want=%v", round, gotRef, wantY)
		}
		corrupt := append([]float32(nil), gotRef...)
		corrupt[replayElements/2] += 0.25
		if sameReplayOutput(corrupt, wantY) {
			t.Fatalf("round %d oracle comparison accepted perturbed element", round)
		}
	}

	tmpDir, err := os.MkdirTemp("", "cudaplan-workflow-cache-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	cache, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
	if err != nil {
		t.Fatal(err)
	}
	compilerID := cudacache.CompilerIdentity{
		NVRTCVersion: "13.0",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   target,
		Flags:        []string{"--std=c++20", "-O3"},
		OutputKind:   nvrtcx.OutputPTX,
	}
	wrongTargetID := compilerID
	wrongTargetID.TargetArch = "sm_100a"

	ctx := context.Background()
	reg := cudaplan.NewKernelRegistry()
	retrievedByOp := make(map[string]*cudakernel.Artifact, len(fx.Artifacts))
	for _, op := range []string{"mul", "add"} {
		orig := fx.Artifacts[op]
		key := cudacache.CompilerCacheKey("workflow_three_node_"+op, compilerID)
		wrongKey := cudacache.CompilerCacheKey("workflow_three_node_"+op, wrongTargetID)
		if key == wrongKey {
			t.Fatalf("cache keys collided across targets for %s", op)
		}
		if err := cache.PutArtifact(ctx, key, orig); err != nil {
			t.Fatalf("PutArtifact %s: %v", op, err)
		}
		if _, err := cache.GetArtifact(ctx, wrongKey); !errors.Is(err, cudacache.ErrNotFound) {
			t.Fatalf("wrong target key for %s: got err=%v, want ErrNotFound", op, err)
		}
		got, err := cache.GetArtifact(ctx, key)
		if err != nil {
			t.Fatalf("GetArtifact %s: %v", op, err)
		}
		if got == orig {
			t.Fatalf("GetArtifact %s returned original pointer instead of detached cache entry", op)
		}
		if got.Digest != orig.Digest || got.Target != target || got.EntryPoint != orig.EntryPoint || len(got.Schema.Params) != 3 {
			t.Fatalf("retrieved artifact identity mismatch for %s: got=%+v want=%+v", op, got, orig)
		}
		if err := got.ValidateTarget("sm_100a"); err == nil {
			t.Fatalf("artifact for %s accepted incompatible target sm_100a", op)
		}
		retrievedByOp[op] = got
		if err := reg.Register(op, got); err != nil {
			t.Fatalf("Register retrieved %s: %v", op, err)
		}
	}

	plan, err := cudaplan.Compile(fx.Snap, cudaplan.WithKernelRegistry(reg))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer plan.Close()

	if got := len(plan.RequiredInputs()); got != workflowInputNodes {
		t.Fatalf("required inputs = %d, want %d (%v)", got, workflowInputNodes, plan.RequiredInputs())
	}
	steps := plan.Steps()
	if len(steps) != workflowComputeNodes {
		t.Fatalf("compiled steps = %d, want %d", len(steps), workflowComputeNodes)
	}

	wantInputs := [][]string{
		{"x", "a"},
		{fx.NodeNames[0], "b"},
		{fx.NodeNames[1], "a"},
	}
	wantDeps := [][]int{
		nil,
		{0},
		{1},
	}
	wantOps := []string{"mul", "add", "mul"}
	for i, s := range steps {
		if s.NodeName != fx.NodeNames[i] || s.OutputName != fx.NodeNames[i] {
			t.Fatalf("step %d name=%q output=%q, want %q", i, s.NodeName, s.OutputName, fx.NodeNames[i])
		}
		if !slices.Equal(s.InputNames, wantInputs[i]) {
			t.Fatalf("step %d inputs=%v, want %v", i, s.InputNames, wantInputs[i])
		}
		if !slices.Equal(s.Dependencies, wantDeps[i]) {
			t.Fatalf("step %d dependencies=%v, want %v", i, s.Dependencies, wantDeps[i])
		}
		if s.IsOutput != (i == workflowComputeNodes-1) {
			t.Fatalf("step %d IsOutput=%v, want %v", i, s.IsOutput, i == workflowComputeNodes-1)
		}
		wantArt := retrievedByOp[wantOps[i]]
		if s.Artifact == nil || s.Artifact.Digest != wantArt.Digest || s.Artifact.Target != target {
			t.Fatalf("step %d artifact=%+v, want retrieved %+v", i, s.Artifact, wantArt)
		}
	}

	low, err := cudaplan.LowerRegion(steps, cudaplan.Region{Start: 0, End: len(steps)})
	if err != nil {
		t.Fatalf("LowerRegion: %v", err)
	}
	if low.Len() != workflowComputeNodes {
		t.Fatalf("lowered tasks = %d, want %d", low.Len(), workflowComputeNodes)
	}
	tasks := low.Graph().Tasks()
	if len(tasks) != workflowComputeNodes {
		t.Fatalf("task graph tasks = %d, want %d", len(tasks), workflowComputeNodes)
	}
	wantScopes := []cudataskgraph.Scope{
		cudataskgraph.ScopeDevice,
		cudataskgraph.ScopeDevice,
		cudataskgraph.ScopeSystem,
	}
	for i, task := range tasks {
		if task.ID != fx.NodeNames[i] || task.Rank != i || task.Scope != wantScopes[i] {
			t.Fatalf("task %d = %+v, want id=%q rank=%d scope=%v", i, task, fx.NodeNames[i], i, wantScopes[i])
		}
		if i == 0 && len(task.Deps) != 0 {
			t.Fatalf("task 0 deps=%v, want empty", task.Deps)
		}
		if i > 0 && !slices.Equal(task.Deps, []string{fx.NodeNames[i-1]}) {
			t.Fatalf("task %d deps=%v, want [%s]", i, task.Deps, fx.NodeNames[i-1])
		}
		b, ok := low.Binding(task.ID)
		if !ok {
			t.Fatalf("missing side-table binding for %q", task.ID)
		}
		if b.StepIndex != i || b.NodeName != steps[i].NodeName || b.OutputName != steps[i].OutputName || b.IsOutput != steps[i].IsOutput || b.Artifact.Digest != steps[i].Artifact.Digest || !slices.Equal(b.InputNames, wantInputs[i]) {
			t.Fatalf("binding %d = %+v, want match with step %+v", i, b, steps[i])
		}
	}

	baseGraphID := low.Graph().Identity()
	if baseGraphID == "" || baseGraphID == retrievedByOp["mul"].Digest || baseGraphID == retrievedByOp["add"].Digest {
		t.Fatalf("invalid or conflated task graph identity: %q", baseGraphID)
	}

	// Negative control 1: altered artifact bytes/digest must fail validation and
	// cache admission, and legitimately rebuilt altered bytes must change identity.
	corruptArt := *retrievedByOp["mul"]
	corruptArt.Raw = append([]byte(nil), corruptArt.Raw...)
	corruptArt.Raw[len(corruptArt.Raw)-2] = ' '
	if err := corruptArt.Validate(); err == nil {
		t.Fatal("Artifact.Validate accepted mutated raw bytes with old digest")
	}
	if err := cache.PutArtifact(ctx, cudacache.CompilerCacheKey("corrupt_mul", compilerID), &corruptArt); err == nil {
		t.Fatal("cache.PutArtifact accepted corrupted artifact")
	}
	rebuiltAlt, err := cudakernel.NewArtifact(cudakernel.KindPTX, append(append([]byte(nil), retrievedByOp["mul"].Raw...), '\n'), "apply", fx.Schema, target)
	if err != nil {
		t.Fatalf("NewArtifact altered bytes: %v", err)
	}
	if rebuiltAlt.Digest == retrievedByOp["mul"].Digest {
		t.Fatal("altered artifact bytes reused old digest")
	}

	// Negative control 2: changing task input edges changes topology identity.
	withExplicitInput := low.Graph().Tasks()
	withExplicitInput[1].Inputs = []string{fx.NodeNames[0]}
	gExplicit, err := cudataskgraph.New(withExplicitInput)
	if err != nil {
		t.Fatalf("cudataskgraph.New with explicit input: %v", err)
	}
	if gExplicit.Identity() == baseGraphID {
		t.Fatal("task graph identity unchanged when Task.Inputs changed")
	}
	rewiredTasks := low.Graph().Tasks()
	rewiredTasks[2].Deps = []string{fx.NodeNames[0], fx.NodeNames[1]}
	rewiredTasks[2].Inputs = []string{fx.NodeNames[0]}
	gRewired, err := cudataskgraph.New(rewiredTasks)
	if err != nil {
		t.Fatalf("cudataskgraph.New rewired: %v", err)
	}
	if gRewired.Identity() == baseGraphID || gRewired.Identity() == gExplicit.Identity() {
		t.Fatal("task graph identity did not change when dependency/input edges changed")
	}

	// Negative control 3: missing, cyclic, and rank-invalid dependencies must be
	// rejected through public cudataskgraph.New and cudaplan.LowerRegion paths.
	badGraphs := []struct {
		name  string
		tasks []cudataskgraph.Task
	}{
		{
			name: "missing_dependency",
			tasks: []cudataskgraph.Task{
				{ID: fx.NodeNames[0], Rank: 0, Scope: cudataskgraph.ScopeDevice},
				{ID: fx.NodeNames[1], Deps: []string{"nonexistent"}, Rank: 1, Scope: cudataskgraph.ScopeSystem},
			},
		},
		{
			name: "input_not_in_deps",
			tasks: []cudataskgraph.Task{
				{ID: fx.NodeNames[0], Rank: 0, Scope: cudataskgraph.ScopeDevice},
				{ID: fx.NodeNames[1], Deps: []string{fx.NodeNames[0]}, Inputs: []string{"x"}, Rank: 1, Scope: cudataskgraph.ScopeSystem},
			},
		},
		{
			name: "rank_invalid",
			tasks: []cudataskgraph.Task{
				{ID: fx.NodeNames[0], Rank: 1, Scope: cudataskgraph.ScopeDevice},
				{ID: fx.NodeNames[1], Deps: []string{fx.NodeNames[0]}, Rank: 1, Scope: cudataskgraph.ScopeSystem},
			},
		},
		{
			name: "cyclic_dependency",
			tasks: []cudataskgraph.Task{
				{ID: fx.NodeNames[0], Deps: []string{fx.NodeNames[1]}, Rank: 0, Scope: cudataskgraph.ScopeDevice},
				{ID: fx.NodeNames[1], Deps: []string{fx.NodeNames[0]}, Rank: 1, Scope: cudataskgraph.ScopeSystem},
			},
		},
	}
	for _, tc := range badGraphs {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := cudataskgraph.New(tc.tasks); err == nil {
				t.Fatalf("cudataskgraph.New accepted %s", tc.name)
			}
		})
	}

	badStepsMissing := plan.Steps()
	badStepsMissing[1].Dependencies = []int{99}
	if _, err := cudaplan.LowerRegion(badStepsMissing, cudaplan.Region{Start: 0, End: len(badStepsMissing)}); err == nil {
		t.Fatal("LowerRegion accepted out-of-range step dependency")
	}
	badStepsForward := plan.Steps()
	badStepsForward[0].Dependencies = []int{1}
	if _, err := cudaplan.LowerRegion(badStepsForward, cudaplan.Region{Start: 0, End: len(badStepsForward)}); err == nil {
		t.Fatal("LowerRegion accepted non-preceding step dependency")
	}
}

func buildWorkflowSearchState(fx workflowChainFixture) (*egraph.SearchState, []*egraph.Rewrite) {
	eg := egraph.New()
	x := eg.Add(egraph.ENode{Op: "x"})
	a := eg.Add(egraph.ENode{Op: "a"})
	b := eg.Add(egraph.ENode{Op: "b"})
	p := eg.Add(egraph.ENode{Op: "mul", Children: []egraph.Id{x, a}})
	q := eg.Add(egraph.ENode{Op: "add", Children: []egraph.Id{p, b}})
	y := eg.Add(egraph.ENode{Op: "mul", Children: []egraph.Id{q, a}})

	state := egraph.NewSearchState(eg, map[string]egraph.Id{"root": y})

	// Initial tree (mul (add (mul x a) b) a) has AstSizeCost = 7 (7 nodes).
	// Rule "a_opt_chain" rewrites (add (mul ?x ?a) ?b) into (fma ?x ?a ?b),
	// reducing the extracted tree to (mul (fma x a b) a) with AstSizeCost = 6.
	// Rule "z_slow_chain" rewrites the root into (slow_stage (slow_wrap ?q) ?a)
	// with AstSizeCost = 8.
	// Both are legal rewrites; profile priority can reorder exploration toward
	// "z_slow_chain" first, but with sufficient budget the independent evaluator's
	// lower-cost winner ("a_opt_chain") must still win.
	rules := []*egraph.Rewrite{
		egraph.MustRewrite("a_opt_chain", "(add (mul ?x ?a) ?b)", "(fma ?x ?a ?b)"),
		egraph.MustRewrite("z_slow_chain", "(mul ?q ?a)", "(slow_stage (slow_wrap ?q) ?a)"),
	}
	_ = fx
	return state, rules
}

func TestWorkflowProfileIsSoftAndAttributable(t *testing.T) {
	const targetArch = "sm_121a"
	fx := buildWorkflowChainFixture(t, targetArch)
	plan, err := cudaplan.Compile(fx.Snap, cudaplan.WithKernelRegistry(func() *cudaplan.KernelRegistry {
		r := cudaplan.NewKernelRegistry()
		_ = r.Register("mul", fx.Artifacts["mul"])
		_ = r.Register("add", fx.Artifacts["add"])
		return r
	}()))
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()

	shape, err := cudashape.New(replayElements)
	if err != nil {
		t.Fatal(err)
	}
	data := replayInputs(0)
	workload := cudasearch.WorkloadBindings{
		Name:            "workflow_three_node_fp32",
		Shape:           shape,
		DType:           cudair.Float32,
		NumericalPolicy: cudasearch.NumericalPolicyExact,
		InputNames:      []string{"x", "a", "b"},
		InputIdentities: map[string]string{
			"x": integrationInputsHash(map[string][]float32{"x": data["x"]}),
			"a": integrationInputsHash(map[string][]float32{"a": data["a"]}),
			"b": integrationInputsHash(map[string][]float32{"b": data["b"]}),
		},
	}
	if err := workload.Validate(); err != nil {
		t.Fatal(err)
	}

	step0 := plan.Steps()[0]
	now := time.Date(2026, 10, 18, 12, 0, 0, 0, time.UTC)
	profTarget := cudaprof.ProfileTarget{
		CodeDigest:    step0.Artifact.Digest,
		Region:        step0.NodeName,
		Workload:      workload.Key(),
		Device:        targetArch,
		CollectorMode: "synthetic_policy_test_input",
	}

	// Synthetic ProfileSnapshot is explicitly policy-test input only, never
	// measured hardware evidence.
	makePolicy := func() *cudasearch.ProfileActions {
		snap, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{
			Version:            1,
			Timestamp:          now.Add(-10 * time.Second),
			CodeDigest:         profTarget.CodeDigest,
			Region:             profTarget.Region,
			Workload:           profTarget.Workload,
			Device:             profTarget.Device,
			CollectorMode:      profTarget.CollectorMode,
			Coverage:           0.95,
			ClockUncertaintyNs: 5,
			Losses:             0,
			Metrics: map[string]int64{
				"smsp__pcsamp_warps_issue_stalled_long_scoreboard": 120,
			},
			MetricUnits: map[string]string{
				"smsp__pcsamp_warps_issue_stalled_long_scoreboard": "samples",
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return &cudasearch.ProfileActions{
			Snapshot:         snap,
			Target:           profTarget,
			Time:             now,
			MaxAge:           time.Minute,
			MinCoverage:      0.8,
			MaxLosses:        0,
			MaxUncertaintyNs: 20,
			Priors: []cudasearch.MetricRulePrior{
				{
					Metric:  "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
					Unit:    "samples",
					Minimum: 50,
					Rule:    "z_slow_chain",
				},
			},
		}
	}

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.AstSizeCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-workflow-v1"),
		Workload:  workload,
	}

	// 1. Tight budget (MaxEvaluations = 2: initial state + 1 transition) proves
	// profile action ordering is active and frozen: prioritized "z_slow_chain"
	// is explored first.
	tightBudget := cudasearch.SearchBudget{
		MaxStates:      2,
		MaxDepth:       1,
		MaxActions:     4,
		MaxEvaluations: 2,
		Timeout:        5 * time.Second,
	}
	stateTight, rulesTight := buildWorkflowSearchState(fx)
	policyTight := makePolicy()
	resTight, err := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget:    tightBudget,
		Profile:   policyTight,
	}).Search(context.Background(), stateTight, rulesTight, eval)
	if err != nil {
		t.Fatalf("tight BeamSearch: %v", err)
	}
	if resTight.ProfileFallback != "" || resTight.ProfileGuidance == nil || resTight.ProfileDecision == nil {
		t.Fatalf("expected active profile guidance, got fallback=%q decision=%+v", resTight.ProfileFallback, resTight.ProfileDecision)
	}
	if len(resTight.AllCandidates) != 2 || len(resTight.AllCandidates[1].Derivation) != 1 || resTight.AllCandidates[1].Derivation[0].RuleName != "z_slow_chain" {
		t.Fatalf("expected prioritized rule z_slow_chain to be evaluated first under tight budget, got candidates=%+v", resTight.AllCandidates)
	}

	// 2. Sufficient finite budget proves soft scoring: both legal candidates
	// remain legal and are evaluated with estimated costs only, and the
	// cost-inferior favored candidate ("z_slow_chain") does NOT override the
	// independent evaluator's winner ("a_opt_chain" -> "fused_mul_add_mul").
	fullBudget := cudasearch.SearchBudget{
		MaxStates:      6,
		MaxDepth:       1,
		MaxActions:     6,
		MaxEvaluations: 6,
		Timeout:        5 * time.Second,
	}
	stateFull, rulesFull := buildWorkflowSearchState(fx)
	policyFull := makePolicy()
	resFull, err := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 4,
		Budget:    fullBudget,
		Profile:   policyFull,
	}).Search(context.Background(), stateFull, rulesFull, eval)
	if err != nil {
		t.Fatalf("full BeamSearch: %v", err)
	}
	if resFull.ProfileFallback != "" || resFull.ProfileDecision == nil || resFull.ProfileDecision.ID == "" {
		t.Fatalf("missing profile decision on full search: %+v", resFull)
	}
	if resFull.ProfileDecision.Target != profTarget || !slices.Equal(resFull.ProfileDecision.ActiveRules, []string{"z_slow_chain"}) {
		t.Fatalf("unexpected profile decision target or active rules: %+v", resFull.ProfileDecision)
	}
	if resFull.PolicyID.Options["profile_decision"] != resFull.ProfileDecision.ID || resFull.PolicyID.Options["workload"] != workload.Key() || !strings.Contains(resFull.PolicyID.Options["budget"], `"MaxEvaluations":6`) {
		t.Fatalf("search policy ID did not record profile decision, workload, or finite budget: %+v", resFull.PolicyID.Options)
	}
	if resFull.Winner == nil || len(resFull.Winner.Derivation) != 1 || resFull.Winner.Derivation[0].RuleName != "a_opt_chain" || resFull.Winner.Expr.String() != "(mul (fma x a b) a)" {
		t.Fatalf("cost-inferior favored candidate overrode evaluator winner: got %+v", resFull.Winner)
	}
	for _, cand := range resFull.AllCandidates {
		if cand.CostSource != cudasearch.CostEstimated {
			t.Fatalf("host search candidate %s claimed CostSource=%q, want %q", cand.ID, cand.CostSource, cudasearch.CostEstimated)
		}
	}

	// Frozen policy check: mutating caller ProfileActions after Search does not
	// alter the recorded ProfileDecision.
	savedDecisionID := resFull.ProfileDecision.ID
	policyFull.Priors[0].Rule = "mutated_after_search"
	if resFull.ProfileDecision.Priors[0].Rule != "z_slow_chain" || resFull.ProfileDecision.ID != savedDecisionID {
		t.Fatalf("ProfileDecision aliased mutable caller input: %+v", resFull.ProfileDecision)
	}

	// Negative controls: stale timestamp, wrong artifact digest, wrong workload,
	// and wrong metric unit each produce an explicit ProfileFallback and preserve
	// static search winner.
	negatives := []struct {
		name       string
		mutate     func(*cudasearch.ProfileActions)
		wantSubstr string
	}{
		{
			name: "stale_timestamp",
			mutate: func(p *cudasearch.ProfileActions) {
				p.Snapshot.Timestamp = now.Add(-10 * time.Minute)
			},
			wantSubstr: "timestamp",
		},
		{
			name: "wrong_artifact_digest",
			mutate: func(p *cudasearch.ProfileActions) {
				sum := sha256.Sum256([]byte("wrong_artifact"))
				p.Target.CodeDigest = hex.EncodeToString(sum[:])
			},
			wantSubstr: "mismatch",
		},
		{
			name: "wrong_workload_target",
			mutate: func(p *cudasearch.ProfileActions) {
				p.Target.Workload = "wrong_workload_key"
			},
			wantSubstr: "workload",
		},
		{
			name: "wrong_metric_unit",
			mutate: func(p *cudasearch.ProfileActions) {
				p.Priors[0].Unit = "cycles"
			},
			wantSubstr: "unavailable",
		},
	}
	for _, tc := range negatives {
		t.Run(tc.name, func(t *testing.T) {
			badPolicy := makePolicy()
			tc.mutate(badPolicy)
			st, rl := buildWorkflowSearchState(fx)
			res, err := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
				BeamWidth: 4,
				Budget:    fullBudget,
				Profile:   badPolicy,
			}).Search(context.Background(), st, rl, eval)
			if err != nil {
				t.Fatalf("%s search failed instead of falling back: %v", tc.name, err)
			}
			if res.ProfileFallback == "" || !strings.Contains(res.ProfileFallback, tc.wantSubstr) {
				t.Fatalf("%s fallback=%q, want substring %q", tc.name, res.ProfileFallback, tc.wantSubstr)
			}
			if res.ProfileGuidance != nil {
				t.Fatalf("%s disguised fallback as accepted guidance: %+v", tc.name, res.ProfileGuidance)
			}
			if res.ProfileDecision == nil || res.ProfileDecision.Fallback != res.ProfileFallback || len(res.ProfileDecision.ActiveRules) != 0 {
				t.Fatalf("%s decision did not record fallback cleanly: %+v", tc.name, res.ProfileDecision)
			}
			if res.Winner == nil || res.Winner.Expr.String() != "(mul (fma x a b) a)" || res.Winner.Cost != resFull.Winner.Cost {
				t.Fatalf("%s altered static search winner or invented latency: got=%+v want=%+v", tc.name, res.Winner, resFull.Winner)
			}
		})
	}
}

// TestWorkflowCancellationRetainsEpochOwnership exercises the host task-graph
// epoch ownership and cancellation protocol over the lowered three-compute-node
// IDs. This is a host task-protocol contract, not proof of a GPU join.
func TestWorkflowCancellationRetainsEpochOwnership(t *testing.T) {
	fx := buildWorkflowChainFixture(t, "sm_121a")
	reg := cudaplan.NewKernelRegistry()
	_ = reg.Register("mul", fx.Artifacts["mul"])
	_ = reg.Register("add", fx.Artifacts["add"])

	plan, err := cudaplan.Compile(fx.Snap, cudaplan.WithKernelRegistry(reg))
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()

	steps := plan.Steps()
	low, err := cudaplan.LowerRegion(steps, cudaplan.Region{Start: 0, End: len(steps)})
	if err != nil {
		t.Fatal(err)
	}
	ids := low.TaskIDs()
	if len(ids) != workflowComputeNodes {
		t.Fatalf("lowered task IDs = %d, want %d", len(ids), workflowComputeNodes)
	}
	g := low.Graph()

	epoch1 := cudataskgraph.NewEpoch(1)
	for _, id := range ids {
		if err := epoch1.Claim(id, g); err != nil {
			t.Fatalf("Claim %s: %v", id, err)
		}
	}
	wantRetained := append([]string(nil), ids...)
	slices.Sort(wantRetained)
	if got := epoch1.Retained(); !slices.Equal(got, wantRetained) {
		t.Fatalf("initial Retained = %v, want %v", got, wantRetained)
	}

	// Negative control 1: publish with old/stale epoch number fails.
	if err := epoch1.PublishEpoch(0, ids[0], g); err == nil {
		t.Fatal("PublishEpoch accepted stale epoch 0")
	}
	if err := epoch1.PublishEpoch(2, ids[0], g); err == nil {
		t.Fatal("PublishEpoch accepted mismatched epoch 2")
	}

	// Negative control 2: out-of-order dependency completion fails before
	// predecessor publishes.
	if err := epoch1.Publish(ids[1], g); err == nil || !strings.Contains(err.Error(), "unpublished") {
		t.Fatalf("out-of-order Publish(%s) = %v, want unpublished dependency error", ids[1], err)
	}
	if err := epoch1.Publish(ids[2], g); err == nil || !strings.Contains(err.Error(), "unpublished") {
		t.Fatalf("out-of-order Publish(%s) = %v, want unpublished dependency error", ids[2], err)
	}

	// Ordered publish of first two compute nodes succeeds.
	if err := epoch1.Publish(ids[0], g); err != nil {
		t.Fatalf("Publish %s: %v", ids[0], err)
	}
	if err := epoch1.Publish(ids[1], g); err != nil {
		t.Fatalf("Publish %s: %v", ids[1], err)
	}
	if !epoch1.Published(ids[0]) || !epoch1.Published(ids[1]) || epoch1.Published(ids[2]) {
		t.Fatalf("unexpected published state before abort")
	}

	// Abort while the third task is still uncompleted.
	epoch1.Abort()
	if !epoch1.Aborted() || epoch1.Drained() {
		t.Fatal("expected epoch1 to be aborted and not yet drained")
	}

	// Negative control 3: late completion after Abort fails.
	if err := epoch1.Publish(ids[2], g); err == nil || !strings.Contains(err.Error(), "aborted") {
		t.Fatalf("late Publish after Abort = %v, want aborted error", err)
	}
	if epoch1.Published(ids[2]) {
		t.Fatal("late Publish mutated published state after Abort")
	}

	// Negative control 4: Release after Abort before Drain fails and Retained
	// remains unchanged for every claimed task.
	for _, id := range ids {
		if err := epoch1.Release(id); err == nil {
			t.Fatalf("Release(%s) succeeded after Abort before Drain", id)
		}
	}
	if got := epoch1.Retained(); !slices.Equal(got, wantRetained) {
		t.Fatalf("Retained changed after refused Release: got %v, want %v", got, wantRetained)
	}

	// Drain then Release succeeds and clears Retained.
	epoch1.Drain()
	if !epoch1.Drained() {
		t.Fatal("epoch1 did not report Drained after Drain()")
	}
	for _, id := range ids {
		if err := epoch1.Release(id); err != nil {
			t.Fatalf("Release(%s) after Drain: %v", id, err)
		}
	}
	if got := epoch1.Retained(); len(got) != 0 {
		t.Fatalf("Retained after Drain+Release = %v, want empty", got)
	}

	// Subsequent epoch remains independent and completes ordered publish cleanly.
	epoch2 := cudataskgraph.NewEpoch(2)
	for _, id := range ids {
		if err := epoch2.Claim(id, g); err != nil {
			t.Fatalf("epoch2 Claim %s: %v", id, err)
		}
		if err := epoch2.Publish(id, g); err != nil {
			t.Fatalf("epoch2 ordered Publish %s: %v", id, err)
		}
	}
	for _, id := range ids {
		if err := epoch2.Release(id); err != nil {
			t.Fatalf("epoch2 Release %s: %v", id, err)
		}
	}
	if got := epoch2.Retained(); len(got) != 0 {
		t.Fatalf("epoch2 Retained = %v, want empty", got)
	}
}

// TestWorkflowNativeCleanupCustody injects drain and release failures into the
// same cleanup owner used by the native fixtures, without touching a driver.
func TestWorkflowNativeCleanupCustody(t *testing.T) {
	for _, failKind := range []int{-1, workflowCompletion, workflowGraph, workflowPlan, workflowBuffer, workflowModule, workflowStream, workflowDevice, workflowLease} {
		name := fmt.Sprintf("release_%d", failKind)
		if failKind == -1 {
			name = "drain"
		}
		t.Run(name, func(t *testing.T) {
			failed := errors.New("injected cleanup failure")
			var calls []string
			refuse := true
			owners := &workflowNativeOwners{drain: func() error {
				calls = append(calls, "drain")
				if refuse && failKind == -1 {
					return failed
				}
				return nil
			}}
			// Registration deliberately runs backwards: dependency order must
			// come from the owner, independent of allocation order or t.Cleanup.
			for kind := workflowLease; kind >= workflowCompletion; kind-- {
				label := fmt.Sprintf("owner_%d", kind)
				owners.add(kind, label, func() error {
					calls = append(calls, label)
					if refuse && kind == failKind {
						return failed
					}
					return nil
				})
			}
			if err := owners.close(); !errors.Is(err, failed) {
				t.Fatalf("close = %v, want injected failure", err)
			}
			want := []string{"drain"}
			for kind := workflowCompletion; kind <= failKind; kind++ {
				want = append(want, fmt.Sprintf("owner_%d", kind))
			}
			if !slices.Equal(calls, want) {
				t.Fatalf("failed cleanup calls = %v, want %v", calls, want)
			}
			// Failed and dependent owners must still be callable on retry;
			// successfully released owners must never be released twice.
			for kind := workflowCompletion; kind <= workflowLease; kind++ {
				wantCount := 1
				if failKind >= 0 && kind < failKind {
					wantCount = 0
				}
				if got := len(owners.resources[kind]); got != wantCount {
					t.Fatalf("retained owner %d count = %d, want %d", kind, got, wantCount)
				}
			}
			refuse = false
			calls = nil
			if err := owners.close(); err != nil {
				t.Fatal(err)
			}
			want = nil
			if failKind == -1 {
				want = append(want, "drain")
			}
			start := max(failKind, workflowCompletion)
			for kind := start; kind <= workflowLease; kind++ {
				want = append(want, fmt.Sprintf("owner_%d", kind))
			}
			if !slices.Equal(calls, want) {
				t.Fatalf("retry cleanup calls = %v, want %v", calls, want)
			}
		})
	}
}
