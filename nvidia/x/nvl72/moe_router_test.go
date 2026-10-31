package nvl72

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func randomLogits(rng *rand.Rand, tokens, experts int, scale float32) [][]float32 {
	l := make([][]float32, tokens)
	for t := range l {
		l[t] = make([]float32, experts)
		for e := range l[t] {
			l[t][e] = scale * float32(rng.NormFloat64())
		}
	}
	return l
}

// zipfLogits returns logits whose top-K is a draw without replacement from
// a Zipf(alpha) distribution over experts: log(w_e) plus Gumbel noise.
func zipfLogits(rng *rand.Rand, tokens, experts int, alpha float64) [][]float32 {
	l := make([][]float32, tokens)
	for t := range l {
		l[t] = make([]float32, experts)
		for e := range l[t] {
			g := -math.Log(-math.Log(rng.Float64()))
			l[t][e] = float32(-alpha*math.Log(float64(e+1)) + g)
		}
	}
	return l
}

func checkPlanInvariants(t *testing.T, cfg MoERouterConfig, p RoutingPlan, T int) {
	t.Helper()
	cfg, _ = cfg.norm()
	per := cfg.NumExperts / cfg.NumGroups
	total := 0
	for tok, experts := range p.SelectedExperts {
		if len(experts) != cfg.TopK {
			t.Fatalf("token %d: %d experts, want %d", tok, len(experts), cfg.TopK)
		}
		groups := map[int]bool{}
		seen := map[int]bool{}
		var sum float64
		for k, e := range experts {
			if seen[e] {
				t.Fatalf("token %d: expert %d selected twice", tok, e)
			}
			seen[e] = true
			groups[e/per] = true
			sum += float64(p.SelectedWeights[tok][k])
		}
		if len(groups) > cfg.TopGroups {
			t.Fatalf("token %d: %d groups used, limit %d", tok, len(groups), cfg.TopGroups)
		}
		if !(math.Abs(sum-1) <= 1e-5) {
			t.Fatalf("token %d: weights sum %v, want 1", tok, sum)
		}
	}
	for _, c := range p.ExpertCounts {
		total += c
	}
	if total != T*cfg.TopK {
		t.Fatalf("counts sum %d, want %d", total, T*cfg.TopK)
	}
	// Every (token, expert) pair appears exactly once in its expert's rows.
	for e := range cfg.NumExperts {
		lo, hi := p.ExpertRowOffsets[e], p.ExpertRowOffsets[e+1]
		if (hi-lo)%cfg.MTile != 0 {
			t.Fatalf("expert %d: %d rows, not a multiple of MTile %d", e, hi-lo, cfg.MTile)
		}
		var got []int32
		for r := lo; r < hi; r++ {
			if v := p.Gather4Indices[r/4][r%4]; v != int32(T) {
				got = append(got, v)
			}
		}
		var want []int32
		for tok, experts := range p.SelectedExperts {
			if slices.Contains(experts, e) {
				want = append(want, int32(tok))
			}
		}
		if !slices.Equal(got, want) {
			t.Fatalf("expert %d gather rows %v, want %v", e, got, want)
		}
	}
}

func TestComputeRoutingPlan(t *testing.T) {
	tests := []struct {
		name string
		cfg  MoERouterConfig
	}{
		{"sigmoid_grouped", MoERouterConfig{NumExperts: 64, NumGroups: 8, TopGroups: 4, TopK: 6, Scoring: RouterSigmoidBiasFree, BiasUpdateRate: 0.01, MTile: 8}},
		{"sigmoid_ungrouped", MoERouterConfig{NumExperts: 32, TopK: 2, Scoring: RouterSigmoidBiasFree, MTile: 4}},
		{"softmax", MoERouterConfig{NumExperts: 64, NumGroups: 4, TopGroups: 2, TopK: 4, Scoring: RouterSoftmaxTopK, MTile: 8}},
		{"sinkhorn", MoERouterConfig{NumExperts: 64, NumGroups: 1, TopK: 4, Scoring: RouterSinkhornKnopp, MTile: 16}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))
			const T = 40
			logits := randomLogits(rng, T, tc.cfg.NumExperts, 2)
			bias := make([]float32, tc.cfg.NumExperts)
			for e := range bias {
				bias[e] = 0.1 * float32(rng.NormFloat64())
			}
			p, err := ComputeRoutingPlan(tc.cfg, logits, bias, 8)
			if err != nil {
				t.Fatal(err)
			}
			checkPlanInvariants(t, tc.cfg, p, T)
			mean := float64(T*tc.cfg.TopK) / float64(tc.cfg.NumExperts)
			for e, c := range p.ExpertCounts {
				want := bias[e]
				if float64(c) < mean {
					want += tc.cfg.BiasUpdateRate
				} else if float64(c) > mean {
					want -= tc.cfg.BiasUpdateRate
				}
				if p.UpdatedBiases[e] != want {
					t.Fatalf("expert %d count %d: bias %v -> %v, want %v", e, c, bias[e], p.UpdatedBiases[e], want)
				}
			}
		})
	}
}

func TestComputeRoutingPlanErrors(t *testing.T) {
	good := MoERouterConfig{NumExperts: 64, NumGroups: 8, TopGroups: 4, TopK: 8}
	logits := randomLogits(rand.New(rand.NewPCG(3, 4)), 4, 64, 1)
	tests := []struct {
		name  string
		cfg   MoERouterConfig
		ranks int
	}{
		{"experts_not_multiple_of_32", MoERouterConfig{NumExperts: 60, TopK: 2}, 4},
		{"groups_do_not_divide", MoERouterConfig{NumExperts: 64, NumGroups: 3, TopK: 2}, 4},
		{"topk_exceeds_groups", MoERouterConfig{NumExperts: 64, NumGroups: 8, TopGroups: 1, TopK: 9}, 4},
		{"bad_mtile", MoERouterConfig{NumExperts: 64, TopK: 2, MTile: 6}, 4},
		{"zero_ranks", good, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ComputeRoutingPlan(tc.cfg, logits, nil, tc.ranks); err == nil {
				t.Fatal("ComputeRoutingPlan succeeded, want error")
			}
		})
	}
}

// TestDeepSeekV3BiasOnlySelects checks sigmoid routing against an
// independent float64 DeepSeek-V3 reference: biases pick experts but the
// combine weights are the unbiased sigmoids. Weights built from the biased
// scores (the classic mistake) must not match.
func TestDeepSeekV3BiasOnlySelects(t *testing.T) {
	cfg := MoERouterConfig{NumExperts: 64, NumGroups: 8, TopGroups: 4, TopK: 8, Scoring: RouterSigmoidBiasFree, MTile: 4}
	rng := rand.New(rand.NewPCG(5, 6))
	logits := randomLogits(rng, 32, 64, 1.5)
	bias := make([]float32, 64)
	for e := range bias {
		bias[e] = 0.5 * float32(rng.NormFloat64())
	}
	p, err := ComputeRoutingPlan(cfg, logits, bias, 8)
	if err != nil {
		t.Fatal(err)
	}
	mutantDiffers := false
	for tok, experts := range p.SelectedExperts {
		var s, sb []float64
		for _, e := range experts {
			x := 1 / (1 + math.Exp(-float64(logits[tok][e])))
			s = append(s, x)
			sb = append(sb, x+float64(bias[e]))
		}
		var sum, sumb float64
		for k := range s {
			sum += s[k]
			sumb += sb[k]
		}
		for k := range s {
			if got, want := float64(p.SelectedWeights[tok][k]), s[k]/sum; !(math.Abs(got-want) <= 1e-6) {
				t.Fatalf("token %d expert %d weight %v, want unbiased %v", tok, experts[k], got, want)
			}
			if !(math.Abs(sb[k]/sumb-s[k]/sum) <= 1e-3) {
				mutantDiffers = true
			}
		}
	}
	if !mutantDiffers {
		t.Fatal("biased-weight mutant matches the reference; test cannot detect it")
	}
}

func TestSinkhornBalance(t *testing.T) {
	cfg := MoERouterConfig{NumExperts: 64, TopK: 4, MTile: 4}
	rng := rand.New(rand.NewPCG(7, 8))
	const T = 256
	logits := zipfLogits(rng, T, 64, 1.0)
	m, err := SinkhornBalance(cfg, logits)
	if err != nil {
		t.Fatal(err)
	}
	for tok, row := range m {
		var s float64
		for _, x := range row {
			s += float64(x)
		}
		if !(math.Abs(s-1) <= 1e-5) {
			t.Fatalf("row %d sums to %v, want 1", tok, s)
		}
	}
	// Three rounds end on a row step, so columns are close to T/E but not
	// exact; softmax columns of the same logits are far from it.
	colDev := func(m [][]float32) float64 {
		dev := 0.0
		for e := range 64 {
			var c float64
			for tok := range T {
				c += float64(m[tok][e])
			}
			dev = max(dev, math.Abs(c-T/64.0)/(T/64.0))
		}
		return dev
	}
	soft := make([][]float32, T)
	for tok := range soft {
		soft[tok] = softmaxRow(logits[tok])
	}
	if d, ds := colDev(m), colDev(soft); d > 0.15 || d*10 > ds {
		t.Fatalf("column deviation from T/E: sinkhorn %.3f, softmax %.3f; want sinkhorn <= 0.15 and 10x below softmax", d, ds)
	}

	maxLoad := func(mode RouterScoringMode) int {
		c := cfg
		c.Scoring = mode
		p, err := ComputeRoutingPlan(c, logits, nil, 8)
		if err != nil {
			t.Fatal(err)
		}
		return slices.Max(p.ExpertCounts)
	}
	softLoad, sinkLoad := maxLoad(RouterSoftmaxTopK), maxLoad(RouterSinkhornKnopp)
	if sinkLoad*2 > softLoad {
		t.Fatalf("Sinkhorn max expert load %d, softmax %d: want Sinkhorn at most half", sinkLoad, softLoad)
	}
	t.Logf("max expert load: softmax %d, sinkhorn %d (mean %d)", softLoad, sinkLoad, T*cfg.TopK/64)
}

// TestHotExpertMirroring is the load-skew oracle: Zipf(1.4) routing of 4096
// tokens to 256 experts on 72 ranks is badly skewed, and two shadow slots
// per rank flatten it while every (token, expert) pair is still served
// exactly once by a copy of that expert.
//
// With HotSkewThreshold 2 mirroring stops at a skew near 1.65: the most
// loaded rank then holds only experts below twice the mean, none of which
// qualify. The 1.35 target needs the threshold at 1.25.
func TestHotExpertMirroring(t *testing.T) {
	cfg := MoERouterConfig{NumExperts: 256, NumGroups: 8, TopGroups: 8, TopK: 8, Scoring: RouterSigmoidBiasFree, MTile: 64, HotSkewThreshold: 1.25}
	rng := rand.New(rand.NewPCG(9, 10))
	const T, ranks = 4096, 72
	logits := zipfLogits(rng, T, 256, 1.4)

	base, err := ComputeRoutingPlan(cfg, logits, nil, ranks)
	if err != nil {
		t.Fatal(err)
	}
	if base.MaxRankLoadSkew != base.UnmirroredRankLoadSkew || len(base.Instances) != 256 {
		t.Fatalf("no shadow slots: skew %v -> %v with %d instances", base.UnmirroredRankLoadSkew, base.MaxRankLoadSkew, len(base.Instances))
	}
	if base.UnmirroredRankLoadSkew <= 4.2 {
		t.Fatalf("unmirrored skew %.3f, want > 4.2 for this workload", base.UnmirroredRankLoadSkew)
	}

	cfg.MaxShadowSlots = 2
	p, err := ComputeRoutingPlan(cfg, logits, nil, ranks)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("rank load skew %.3f -> %.3f with %d shadow instances", p.UnmirroredRankLoadSkew, p.MaxRankLoadSkew, len(p.Instances)-256)
	if p.MaxRankLoadSkew > 1.35 {
		t.Fatalf("mirrored skew %.3f, want <= 1.35", p.MaxRankLoadSkew)
	}
	at2 := cfg
	at2.HotSkewThreshold = 2
	p2, err := ComputeRoutingPlan(at2, logits, nil, ranks)
	if err != nil {
		t.Fatal(err)
	}
	if p2.MaxRankLoadSkew > 1.7 || p2.MaxRankLoadSkew < p.MaxRankLoadSkew {
		t.Fatalf("threshold 2: skew %.3f, want in [%.3f, 1.7]", p2.MaxRankLoadSkew, p.MaxRankLoadSkew)
	}
	perRank := make([]int, ranks)
	for _, in := range p.Instances {
		if in.Shadow {
			perRank[in.Rank]++
		}
	}
	if slices.Max(perRank) > 2 {
		t.Fatalf("a rank uses %d shadow slots, limit 2", slices.Max(perRank))
	}

	// Equivalence: serve each instance's gather rows with its own copy of
	// the expert and combine; the result must equal unmirrored routing.
	const D = 4
	expert := func(e, tok int) [D]float64 {
		var y [D]float64
		for d := range y {
			y[d] = math.Sin(float64(e*31+tok*7+d)) * float64(e+1)
		}
		return y
	}
	served := make(map[[2]int]int)
	out := make([][D]float64, T)
	weightOf := func(tok, e int) float64 {
		k := slices.Index(p.SelectedExperts[tok], e)
		return float64(p.SelectedWeights[tok][k])
	}
	for i, in := range p.Instances {
		n := 0
		for _, q := range p.InstanceGather4[i] {
			for _, tok := range q {
				if tok == T {
					continue
				}
				n++
				served[[2]int{int(tok), in.Expert}]++
				y := expert(in.Expert, int(tok))
				for d := range D {
					out[tok][d] += weightOf(int(tok), in.Expert) * y[d]
				}
			}
		}
		if n != p.MirroredCounts[i] {
			t.Fatalf("instance %d serves %d rows, MirroredCounts says %d", i, n, p.MirroredCounts[i])
		}
	}
	for tok, experts := range p.SelectedExperts {
		var want [D]float64
		for k, e := range experts {
			if served[[2]int{tok, e}] != 1 {
				t.Fatalf("token %d expert %d served %d times", tok, e, served[[2]int{tok, e}])
			}
			y := expert(e, tok)
			for d := range D {
				want[d] += float64(p.SelectedWeights[tok][k]) * y[d]
			}
		}
		for d := range D {
			if !(math.Abs(out[tok][d]-want[d]) <= 1e-9) {
				t.Fatalf("token %d dim %d: mirrored %v, unmirrored %v", tok, d, out[tok][d], want[d])
			}
		}
	}
	if len(served) != T*cfg.TopK {
		t.Fatalf("%d pairs served, want %d", len(served), T*cfg.TopK)
	}
}

func TestRoutingPlanGather4Specs(t *testing.T) {
	cfg := MoERouterConfig{NumExperts: 32, TopK: 2, MTile: 4}
	p, err := ComputeRoutingPlan(cfg, randomLogits(rand.New(rand.NewPCG(1, 1)), 8, 32, 1), nil, 4)
	if err != nil {
		t.Fatal(err)
	}
	specs, err := p.Gather4Specs("sm_121a", 64, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != len(p.Gather4Indices) {
		t.Fatalf("%d specs for %d quads", len(specs), len(p.Gather4Indices))
	}
	ptx, err := specs[0].FormatPTX()
	if err != nil || !strings.Contains(ptx, "tile.gather4") {
		t.Fatalf("FormatPTX = %q, %v", ptx, err)
	}
	if _, err := p.Gather4Specs("sm_90a", 64, 2); err == nil {
		t.Fatal("Gather4Specs(sm_90a) succeeded, want error: Hopper has no .gather4")
	}
}

// TestFusedMoERouterHostEmu runs the emitted router under cudahostemu and
// requires its selections, histogram and gather rows to equal
// ComputeRoutingPlan bit for bit, and its weights to within 1e-6.
func TestFusedMoERouterHostEmu(t *testing.T) {
	tests := []struct {
		name string
		cfg  MoERouterConfig
	}{
		{"sigmoid", MoERouterConfig{NumExperts: 64, NumGroups: 8, TopGroups: 4, TopK: 6, Scoring: RouterSigmoidBiasFree, MTile: 8}},
		{"softmax", MoERouterConfig{NumExperts: 64, NumGroups: 4, TopGroups: 2, TopK: 4, Scoring: RouterSoftmaxTopK, MTile: 4}},
		{"sinkhorn", MoERouterConfig{NumExperts: 32, TopK: 3, Scoring: RouterSinkhornKnopp, MTile: 4}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			const T = 24
			rng := rand.New(rand.NewPCG(11, 12))
			logits := randomLogits(rng, T, tc.cfg.NumExperts, 2)
			bias := make([]float32, tc.cfg.NumExperts)
			for e := range bias {
				bias[e] = 0.2 * float32(rng.NormFloat64())
			}
			want, err := ComputeRoutingPlan(tc.cfg, logits, bias, 4)
			if err != nil {
				t.Fatal(err)
			}
			got := runRouterEmu(t, FusedMoERouter{Name: "router_" + tc.name, Config: tc.cfg, Tokens: T, Warps: 4}, logits, bias)
			for tok := range T {
				for k, e := range want.SelectedExperts[tok] {
					if got.SelectedExperts[tok][k] != e {
						t.Fatalf("token %d slot %d: kernel expert %d, plan %d", tok, k, got.SelectedExperts[tok][k], e)
					}
					if d := math.Abs(float64(got.SelectedWeights[tok][k] - want.SelectedWeights[tok][k])); !(d <= 1e-6) {
						t.Fatalf("token %d slot %d: kernel weight %v, plan %v", tok, k, got.SelectedWeights[tok][k], want.SelectedWeights[tok][k])
					}
				}
			}
			if !slices.Equal(got.ExpertCounts, want.ExpertCounts) {
				t.Fatalf("counts %v, want %v", got.ExpertCounts, want.ExpertCounts)
			}
			if !slices.Equal(got.ExpertRowOffsets, want.ExpertRowOffsets) {
				t.Fatalf("row offsets %v, want %v", got.ExpertRowOffsets, want.ExpertRowOffsets)
			}
			if !slices.Equal(got.Gather4Indices, want.Gather4Indices) {
				t.Fatalf("gather quads differ:\nkernel %v\nplan   %v", got.Gather4Indices, want.Gather4Indices)
			}
		})
	}
}

func runRouterEmu(t *testing.T, k FusedMoERouter, logits [][]float32, bias []float32) RoutingPlan {
	t.Helper()
	src, err := k.Source()
	if err != nil {
		t.Fatal(err)
	}
	c, _ := k.Config.norm()
	T, E, K := k.Tokens, c.NumExperts, c.TopK
	flat := make([]float32, 0, T*E)
	for _, row := range logits {
		flat = append(flat, row...)
	}
	rows := k.GatherRows()
	sel := cudahostemu.Poison(4 * T * K)
	w := cudahostemu.Poison(4 * T * K)
	counts := cudahostemu.Poison(4 * E)
	gather := cudahostemu.Poison(4 * rows)
	offsets := cudahostemu.Poison(4 * (E + 1))
	grid, block := k.Grid()
	err = cudahostemu.RunMultiRank(context.Background(), src, 1, cudahostemu.Launch{
		Kernel: k.Name,
		Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
		Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
	},
		cudahostemu.BufferArg("logits", "float", f32Bytes(flat)),
		cudahostemu.BufferArg("bias", "float", f32Bytes(bias)),
		cudahostemu.BufferArg("sel_experts", "int32_t", sel),
		cudahostemu.BufferArg("sel_weights", "float", w),
		cudahostemu.BufferArg("counts", "int32_t", counts),
		cudahostemu.BufferArg("gather", "int32_t", gather),
		cudahostemu.BufferArg("row_offsets", "int32_t", offsets),
		cudahostemu.BufferArg("work", "float", cudahostemu.Poison(4*T*E)),
	)
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatalf("RunMultiRank: %v", err)
	}
	var p RoutingPlan
	selI, wF := bytesI32(sel), bytesF32(w)
	for tok := range T {
		var experts []int
		for _, e := range selI[tok*K : (tok+1)*K] {
			experts = append(experts, int(e))
		}
		p.SelectedExperts = append(p.SelectedExperts, experts)
		p.SelectedWeights = append(p.SelectedWeights, wF[tok*K:(tok+1)*K])
	}
	for _, x := range bytesI32(counts) {
		p.ExpertCounts = append(p.ExpertCounts, int(x))
	}
	for _, x := range bytesI32(offsets) {
		p.ExpertRowOffsets = append(p.ExpertRowOffsets, int(x))
	}
	p.Gather4Indices = quads(bytesI32(gather)[:p.ExpertRowOffsets[E]])
	return p
}

func ExampleComputeRoutingPlan() {
	cfg := MoERouterConfig{NumExperts: 32, TopK: 2, Scoring: RouterSigmoidBiasFree, MTile: 4}
	logits := make([][]float32, 4)
	for t := range logits {
		logits[t] = make([]float32, 32)
		logits[t][t] = 3  // each token prefers expert t
		logits[t][31] = 1 // and then expert 31
	}
	p, err := ComputeRoutingPlan(cfg, logits, nil, 4)
	if err != nil {
		panic(err)
	}
	fmt.Println(p.SelectedExperts)
	fmt.Println(p.ExpertCounts[31], p.Gather4Indices[len(p.Gather4Indices)-1])
	// Output:
	// [[0 31] [1 31] [2 31] [3 31]]
	// 4 [0 1 2 3]
}

// TestFusedMoERouterNative runs the emitted router on a GPU and compares it
// with ComputeRoutingPlan. Device expf may differ from the host by an ulp,
// so weights are compared to 1e-5; selections must match exactly.
func TestFusedMoERouterNative(t *testing.T) {
	nvcc := requireNative(t)
	for _, mode := range []RouterScoringMode{RouterSigmoidBiasFree, RouterSoftmaxTopK, RouterSinkhornKnopp} {
		t.Run(mode.String(), func(t *testing.T) {
			cfg := MoERouterConfig{NumExperts: 256, NumGroups: 8, TopGroups: 4, TopK: 8, Scoring: mode, MTile: 64}
			const T = 64
			rng := rand.New(rand.NewPCG(13, 14))
			logits := randomLogits(rng, T, 256, 2)
			bias := make([]float32, 256)
			for e := range bias {
				bias[e] = 0.2 * float32(rng.NormFloat64())
			}
			want, err := ComputeRoutingPlan(cfg, logits, bias, 72)
			if err != nil {
				t.Fatal(err)
			}
			k := FusedMoERouter{Name: "native_router", Config: cfg, Tokens: T, Warps: 8}
			src, err := k.Source()
			if err != nil {
				t.Fatal(err)
			}
			E, K, rows := 256, cfg.TopK, k.GatherRows()
			src += "#include <cstdio>\n"
			src += cArray("float", "h_logits", T*E, func(i int) string { return fmt.Sprintf("%.9g", logits[i/E][i%E]) })
			src += cArray("float", "h_bias", E, func(i int) string { return fmt.Sprintf("%.9g", bias[i]) })
			src += fmt.Sprintf(`
int main() {
    const int T = %d, E = %d, K = %d, ROWS = %d;
    float *logits, *bias, *w, *work;
    int32_t *sel, *counts, *gather, *offsets;
    cudaMalloc(&logits, sizeof h_logits); cudaMalloc(&bias, sizeof h_bias);
    cudaMalloc(&w, 4 * T * K); cudaMalloc(&work, 4 * T * E);
    cudaMalloc(&sel, 4 * T * K); cudaMalloc(&counts, 4 * E);
    cudaMalloc(&gather, 4 * ROWS); cudaMalloc(&offsets, 4 * (E + 1));
    cudaMemcpy(logits, h_logits, sizeof h_logits, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, h_bias, sizeof h_bias, cudaMemcpyHostToDevice);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    native_router_launch(logits, bias, sel, w, counts, gather, offsets, work, 0);
    cudaEventRecord(a);
    native_router_launch(logits, bias, sel, w, counts, gather, offsets, work, 0);
    cudaEventRecord(b);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed\n"); return 1; }
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    static int32_t hs[T * K], hc[E], ho[E + 1], hg[ROWS]; static float hw[T * K];
    cudaMemcpy(hs, sel, sizeof hs, cudaMemcpyDeviceToHost);
    cudaMemcpy(hw, w, sizeof hw, cudaMemcpyDeviceToHost);
    cudaMemcpy(hc, counts, sizeof hc, cudaMemcpyDeviceToHost);
    cudaMemcpy(ho, offsets, sizeof ho, cudaMemcpyDeviceToHost);
    cudaMemcpy(hg, gather, sizeof hg, cudaMemcpyDeviceToHost);
    printf("us %%.3f\n", ms * 1000.0f);
    printf("sel"); for (int i = 0; i < T * K; ++i) printf(" %%d", hs[i]); printf("\n");
    printf("w"); for (int i = 0; i < T * K; ++i) printf(" %%.9g", hw[i]); printf("\n");
    printf("counts"); for (int i = 0; i < E; ++i) printf(" %%d", hc[i]); printf("\n");
    printf("offsets"); for (int i = 0; i <= E; ++i) printf(" %%d", ho[i]); printf("\n");
    printf("gather"); for (int i = 0; i < ho[E]; ++i) printf(" %%d", hg[i]); printf("\n");
    return 0;
}
`, T, E, K, rows)
			out := nativeRun(t, nvcc, src)
			t.Logf("router %s, %d tokens: %.3f us (one launch, includes launch overhead)", mode, T, parseFields(t, out, "us")[0])
			sel, w := parseFields(t, out, "sel"), parseFields(t, out, "w")
			for tok := range T {
				for kk, e := range want.SelectedExperts[tok] {
					if int(sel[tok*K+kk]) != e {
						t.Fatalf("token %d slot %d: device expert %v, plan %d", tok, kk, sel[tok*K+kk], e)
					}
					if d := math.Abs(w[tok*K+kk] - float64(want.SelectedWeights[tok][kk])); !(d <= 1e-5) {
						t.Fatalf("token %d slot %d: device weight %v, plan %v", tok, kk, w[tok*K+kk], want.SelectedWeights[tok][kk])
					}
				}
			}
			for e, c := range parseFields(t, out, "counts") {
				if int(c) != want.ExpertCounts[e] {
					t.Fatalf("expert %d: device count %v, plan %d", e, c, want.ExpertCounts[e])
				}
			}
			gather := parseFields(t, out, "gather")
			for i, q := range want.Gather4Indices {
				for j, v := range q {
					if int32(gather[4*i+j]) != v {
						t.Fatalf("gather row %d: device %v, plan %d", 4*i+j, gather[4*i+j], v)
					}
				}
			}
		})
	}
}
