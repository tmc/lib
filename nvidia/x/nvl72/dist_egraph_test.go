package nvl72

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// tpBlock builds (dist_rmsnorm (dist_add (dist_allreduce_sum (dist_gemm x w)) r) g),
// the tensor-parallel down-projection, residual and norm.
func tpBlock() (*egraph.EGraph, egraph.Id) {
	g := egraph.New()
	leaf := func(s string) egraph.Id { return g.Add(egraph.NewLeaf(s)) }
	mm := g.Add(egraph.NewNode(DistGemm, leaf("x"), leaf("w")))
	ar := g.Add(egraph.NewNode(DistAllReduceSum, mm))
	add := g.Add(egraph.NewNode(DistAdd, ar, leaf("r")))
	return g, g.Add(egraph.NewNode(DistRMSNorm, add, leaf("g")))
}

func nvl72CostParams(m int) DistFusionCostParams {
	return DistFusionCostParams{
		WorldSize: 8, SwitchBWGBps: NVL72NVLink5UniGBps, SwitchLatencyUs: 0.8,
		TensorCoreTFLOPS: NVL72BlackwellTensorTFLOPS, BatchTokens: m,
	}
}

// TestDistRegimeSwitching is the regime oracle: the same e-graph extracts
// the pulled multimem.ld_reduce form at decode batch sizes and the pushed
// per-tile switch reduction at prefill sizes, each at least 38% faster
// than the unfused schedule under the model.
func TestDistRegimeSwitching(t *testing.T) {
	tests := []struct {
		m    int
		want DistRewriteKind
	}{
		{1, RulePullMultimemLdReduce},
		{16, RulePullMultimemLdReduce},
		{4096, RuleTilePushFabricRed},
		{16384, RuleTilePushFabricRed},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprint(tc.m), func(t *testing.T) {
			g, root := tpBlock()
			s, err := ExtractOptimalDistSchedule(g, root, nvl72CostParams(tc.m))
			if err != nil {
				t.Fatal(err)
			}
			gain := 1 - s.EstLatencyUs/s.UnfusedLatencyUs
			t.Logf("M=%d: %s %.2f us vs unfused %.2f us (%.0f%% lower): %v", tc.m, s.Rule, s.EstLatencyUs, s.UnfusedLatencyUs, 100*gain, s.Expr)
			if s.Rule != tc.want {
				t.Fatalf("rule %q, want %q", s.Rule, tc.want)
			}
			if gain < 0.38 {
				t.Fatalf("latency %.2f us is %.0f%% below unfused %.2f us, want >= 38%%", s.EstLatencyUs, 100*gain, s.UnfusedLatencyUs)
			}
			if s.PTX == "" {
				t.Fatal("no PTX for the chosen rule")
			}
		})
	}
}

// TestDistRegimeCrossover checks that the choice switches once, from pull
// to push, as the batch grows.
func TestDistRegimeCrossover(t *testing.T) {
	prev, switches := RulePullMultimemLdReduce, 0
	for m := 1; m <= 1<<16; m *= 2 {
		g, root := tpBlock()
		s, err := ExtractOptimalDistSchedule(g, root, nvl72CostParams(m))
		if err != nil {
			t.Fatal(err)
		}
		if s.Rule != RulePullMultimemLdReduce && s.Rule != RuleTilePushFabricRed {
			t.Fatalf("M=%d: rule %q", m, s.Rule)
		}
		if s.Rule != prev {
			t.Logf("switches to %s at M=%d", s.Rule, m)
			switches++
			prev = s.Rule
		}
	}
	if switches != 1 || prev != RuleTilePushFabricRed {
		t.Fatalf("%d switches ending at %s, want one switch to push", switches, prev)
	}
}

// distEnv returns random per-rank inputs for the pattern variables.
func distEnv(rng *rand.Rand, P int, shapes map[string][2]int) map[string][]DistMatrix {
	env := map[string][]DistMatrix{}
	for name, s := range shapes {
		for range P {
			m := newDistMatrix(s[0], s[1])
			for i := range m.Data {
				m.Data[i] = float32(rng.NormFloat64())
			}
			env[name] = append(env[name], m)
		}
	}
	return env
}

func maxRelDiff(a, b []DistMatrix) float64 {
	worst := 0.0
	for r := range a {
		for i := range a[r].Data {
			x, y := float64(a[r].Data[i]), float64(b[r].Data[i])
			worst = max(worst, math.Abs(x-y)/max(1, math.Abs(y)))
		}
	}
	return worst
}

func ruleShapes(P, M, K, N int) map[string]map[string][2]int {
	return map[string]map[string][2]int{
		string(RuleTilePushFabricRed):      {"x": {M, K}, "w": {K, N}, "r": {M, N}},
		string(RulePullMultimemLdReduce):   {"p": {M, N}, "r": {M, N}, "g": {1, N}},
		string(RuleAGMatmulToRingPrefetch): {"x": {M / P, K}, "w": {K, N}},
		string(RuleMatmulRSAssociative):    {"x": {M, K}, "w": {K, N}},
	}
}

// TestDistRulesSound evaluates both sides of every rule on random multi-rank
// inputs: the tiled, pulled and pipelined forms equal the unfused
// GEMM, collective, residual and RMSNorm sequence.
func TestDistRulesSound(t *testing.T) {
	const P, M, K, N = 4, 8, 6, 12
	shapes := ruleShapes(P, M, K, N)
	for _, rule := range AlgebraicCollectiveRules() {
		t.Run(rule.Name, func(t *testing.T) {
			env := distEnv(rand.New(rand.NewPCG(31, 32)), P, shapes[rule.Name])
			cfg := DistEvalConfig{Ranks: P, TileM: 3, TileN: 5}
			lhs, err := EvalDistExpr(PatternExpr(rule.LHS), cfg, env)
			if err != nil {
				t.Fatal(err)
			}
			rhs, err := EvalDistExpr(PatternExpr(rule.RHS), cfg, env)
			if err != nil {
				t.Fatal(err)
			}
			if d := maxRelDiff(rhs, lhs); !(d <= 1e-5) {
				t.Fatalf("rewrite changes the result by %g", d)
			}
		})
	}
}

// TestDistPullRuleDoesNotCrossRMSNorm: RMSNorm inside the all-reduce is not
// linear in the partials, so no rule may fuse it; and the rewrite that
// would, if added, is caught by the evaluator.
func TestDistPullRuleDoesNotCrossRMSNorm(t *testing.T) {
	g := egraph.New()
	leaf := func(s string) egraph.Id { return g.Add(egraph.NewLeaf(s)) }
	add := g.Add(egraph.NewNode(DistAdd, leaf("p"), leaf("r")))
	norm := g.Add(egraph.NewNode(DistRMSNorm, add, leaf("g")))
	root := g.Add(egraph.NewNode(DistAllReduceSum, norm))
	s, err := ExtractOptimalDistSchedule(g, root, nvl72CostParams(16))
	if err != nil {
		t.Fatal(err)
	}
	if s.Rule != "" {
		t.Fatalf("rule %q fired across RMSNorm: %v", s.Rule, s.Expr)
	}
	for _, c := range g.Classes() {
		for _, n := range c.Nodes {
			if _, fused := distRuleForOp[n.Op]; fused {
				t.Fatalf("saturation produced %s", n.Op)
			}
		}
	}

	bad := egraph.MustRewrite("mutant_pull_across_norm",
		"(dist_allreduce_sum (dist_rmsnorm (dist_add ?p ?r) ?g))",
		"(fused_multimem_ld_reduce_rmsnorm ?p ?r ?g)")
	env := distEnv(rand.New(rand.NewPCG(33, 34)), 4, ruleShapes(4, 8, 6, 12)[string(RulePullMultimemLdReduce)])
	cfg := DistEvalConfig{Ranks: 4}
	lhs, err := EvalDistExpr(PatternExpr(bad.LHS), cfg, env)
	if err != nil {
		t.Fatal(err)
	}
	rhs, err := EvalDistExpr(PatternExpr(bad.RHS), cfg, env)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxRelDiff(rhs, lhs); d < 1e-2 {
		t.Fatalf("mutant rewrite changes the result by only %g; evaluator cannot reject it", d)
	}
}

func TestDistFusionCostParamsErrors(t *testing.T) {
	for name, edit := range map[string]func(*DistFusionCostParams){
		"one_rank":   func(p *DistFusionCostParams) { p.WorldSize = 1 },
		"no_tokens":  func(p *DistFusionCostParams) { p.BatchTokens = 0 },
		"no_bw":      func(p *DistFusionCostParams) { p.SwitchBWGBps = 0 },
		"bad_target": func(p *DistFusionCostParams) { p.Target = "sm_80" },
	} {
		t.Run(name, func(t *testing.T) {
			p := nvl72CostParams(16)
			edit(&p)
			g, root := tpBlock()
			if _, err := ExtractOptimalDistSchedule(g, root, p); err == nil {
				t.Fatal("ExtractOptimalDistSchedule succeeded, want error")
			}
		})
	}
}

func TestDistRulePTX(t *testing.T) {
	for _, tc := range []struct {
		rule   DistRewriteKind
		target string
		want   string
	}{
		{RulePullMultimemLdReduce, "sm_100a", "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2"},
		{RuleTilePushFabricRed, "sm_100a", "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"},
		{RuleTilePushFabricRed, "sm_107a", "fabric.try_red.async.tensor.2d.multimem"},
		{RuleMatmulRSAssociative, "sm_90a", "multimem.red.relaxed.sys.global.add.v4.bf16x2"},
	} {
		ptx, err := DistRulePTX(tc.rule, tc.target)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(ptx, tc.want) {
			t.Fatalf("%s on %s: PTX missing %q:\n%s", tc.rule, tc.target, tc.want, ptx)
		}
	}
}

// multimemRMSNormRef is the float64 reference of MultimemRMSNormKernel:
// per element the rank sum rounded to BF16, plus the BF16 residual, then
// RMSNorm with the BF16 weight.
func multimemRMSNormRef(sym [][]uint16, res, w []uint16, rows, cols int, eps float32) [][]float64 {
	out := make([][]float64, len(sym))
	for r := range sym {
		out[r] = make([]float64, rows*cols)
		for i := range rows {
			h := make([]float64, cols)
			var sq float64
			for j := range cols {
				idx := i*cols + j
				var s float32
				for q := range sym {
					s += FromBF16Bits(sym[q][idx])
				}
				h[j] = float64(FromBF16Bits(BF16Bits(s))) + float64(FromBF16Bits(res[r*rows*cols+idx]))
				sq += h[j] * h[j]
			}
			inv := 1 / math.Sqrt(sq/float64(cols)+float64(eps))
			for j := range cols {
				out[r][i*cols+j] = h[j] * inv * float64(FromBF16Bits(w[r*cols+j]))
			}
		}
	}
	return out
}

func TestMultimemRMSNormKernelHostEmu(t *testing.T) {
	const P, rows, cols = 4, 3, 1024
	k := MultimemRMSNormKernel{Name: "mm_rmsnorm", Cols: cols, Ranks: P, Threads: 64}
	src, err := k.Source()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(35, 36))
	bf := func(n int) []uint16 {
		v := make([]uint16, n)
		for i := range v {
			v[i] = BF16Bits(float32(rng.NormFloat64()))
		}
		return v
	}
	sym := make([][]uint16, P)
	var symFlat []uint16
	for r := range P {
		sym[r] = bf(rows * cols)
		symFlat = append(symFlat, sym[r]...)
	}
	res, w := bf(P*rows*cols), bf(P*cols)
	want := multimemRMSNormRef(sym, res, w, rows, cols, 1e-6)
	out := cudahostemu.Poison(4 * P * rows * cols)
	grid, block := k.Grid(rows)
	err = cudahostemu.RunMultiRank(context.Background(), src, P, cudahostemu.Launch{
		Kernel: k.Name,
		Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
		Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
	},
		cudahostemu.IntArg(rows),
		cudahostemu.BufferArg("sym_in", "uint16_t", u16Bytes(symFlat)),
		cudahostemu.BufferArg("residual", "uint16_t", u16Bytes(res)),
		cudahostemu.BufferArg("weight", "uint16_t", u16Bytes(w)),
		cudahostemu.BufferArg("out", "float", out),
	)
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatalf("RunMultiRank: %v", err)
	}
	got := bytesF32(out)
	for r := range P {
		for i := range rows * cols {
			if d := math.Abs(float64(got[r*rows*cols+i]) - want[r][i]); !(d <= 1e-4) {
				t.Fatalf("rank %d elem %d: %v, want %v", r, i, got[r*rows*cols+i], want[r][i])
			}
		}
	}
}

// TestMultimemRMSNormKernelNative assembles the pulled kernel with nvcc for
// sm_90a, sm_100a and the native target; it does not execute it, since
// multimem needs a multicast object spanning GPUs.
func TestMultimemRMSNormKernelNative(t *testing.T) {
	nvcc := requireNative(t)
	src, err := MultimemRMSNormKernel{Name: "mm_rmsnorm", Cols: 7168, Ranks: 8}.Source()
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{"sm_90a", "sm_100a", nativeTarget()} {
		t.Run(target, func(t *testing.T) {
			ptx := t.TempDir() + "/k.ptx"
			nativeCompile(t, nvcc, target, src, ptx, "-ptx")
			nativeCompile(t, nvcc, target, src, t.TempDir()+"/k.cubin", "-cubin")
		})
	}
}

func ExampleExtractOptimalDistSchedule() {
	for _, m := range []int{16, 4096} {
		g, root := tpBlock()
		s, err := ExtractOptimalDistSchedule(g, root, DistFusionCostParams{
			WorldSize: 8, SwitchBWGBps: 900, SwitchLatencyUs: 0.8,
			TensorCoreTFLOPS: 2250, BatchTokens: m,
		})
		if err != nil {
			panic(err)
		}
		fmt.Println(m, s.Rule)
	}
	// Output:
	// 16 allreduce_rmsnorm_to_multimem_ld_reduce_fused
	// 4096 gemm_allreduce_to_tile_fabric_try_red_multimem
}
