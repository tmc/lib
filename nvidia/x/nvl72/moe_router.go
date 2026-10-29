package nvl72

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatma"
)

// RouterScoringMode selects the MoE gating function and load-balancing
// mechanism used by [ComputeRoutingPlan] and [FusedMoERouter].
type RouterScoringMode uint8

const (
	// RouterSigmoidBiasFree is DeepSeek-V3 auxiliary-loss-free routing:
	// experts are selected by sigmoid(u)+b and combined with the unbiased
	// sigmoid(u) renormalized over the selected experts.
	RouterSigmoidBiasFree RouterScoringMode = iota

	// RouterSoftmaxTopK selects experts by softmax(u) and renormalizes the
	// selected probabilities.
	RouterSoftmaxTopK

	// RouterSinkhornKnopp selects experts by a Sinkhorn-Knopp normalization
	// of exp(u) over the whole token batch (rows sum to 1, columns to
	// tokens/experts) and combines them with renormalized softmax(u).
	RouterSinkhornKnopp
)

func (m RouterScoringMode) String() string {
	switch m {
	case RouterSigmoidBiasFree:
		return "sigmoid_bias_free"
	case RouterSoftmaxTopK:
		return "softmax_topk"
	case RouterSinkhornKnopp:
		return "sinkhorn_knopp"
	}
	return fmt.Sprintf("RouterScoringMode(%d)", uint8(m))
}

// SinkhornIterations is the number of column/row normalization rounds the
// Sinkhorn-Knopp router runs.
const SinkhornIterations = 3

// MoERouterConfig configures the fused router, the TMA .gather4 index
// layout, and the hot-expert mirroring planner.
type MoERouterConfig struct {
	NumExperts       int // routed experts; a multiple of 32 (8 per lane at 256)
	NumGroups        int // expert groups; divides NumExperts, at most 32
	TopGroups        int // groups each token may route to
	TopK             int // experts per token, at most 32
	Scoring          RouterScoringMode
	BiasUpdateRate   float32 // gamma in b_e += gamma * sign(mean - count_e)
	HotSkewThreshold float64 // expert load above this multiple of the mean is mirrored
	MaxShadowSlots   int     // spare expert slots per rank
	MTile            int     // gather tile rows; a multiple of 4 (default 128)
}

func (c MoERouterConfig) norm() (MoERouterConfig, error) {
	if c.NumGroups == 0 {
		c.NumGroups = 1
	}
	if c.TopGroups == 0 {
		c.TopGroups = c.NumGroups
	}
	if c.MTile == 0 {
		c.MTile = 128
	}
	if c.HotSkewThreshold == 0 {
		c.HotSkewThreshold = 2
	}
	switch {
	case c.NumExperts < 32 || c.NumExperts%32 != 0 || c.NumExperts > 1024:
		return c, fmt.Errorf("moe router: NumExperts %d must be a multiple of 32 in 32..1024", c.NumExperts)
	case c.NumGroups < 1 || c.NumGroups > 32 || c.NumExperts%c.NumGroups != 0:
		return c, fmt.Errorf("moe router: NumGroups %d must divide NumExperts %d and be in 1..32", c.NumGroups, c.NumExperts)
	case c.TopGroups < 1 || c.TopGroups > c.NumGroups:
		return c, fmt.Errorf("moe router: TopGroups %d must be in 1..%d", c.TopGroups, c.NumGroups)
	case c.TopK < 1 || c.TopK > 32 || c.TopK > c.TopGroups*(c.NumExperts/c.NumGroups):
		return c, fmt.Errorf("moe router: TopK %d must be in 1..32 and fit in %d groups of %d experts",
			c.TopK, c.TopGroups, c.NumExperts/c.NumGroups)
	case c.Scoring > RouterSinkhornKnopp:
		return c, fmt.Errorf("moe router: unknown scoring mode %v", c.Scoring)
	case c.MTile < 4 || c.MTile%4 != 0:
		return c, fmt.Errorf("moe router: MTile %d must be a positive multiple of 4", c.MTile)
	case c.HotSkewThreshold < 1:
		return c, fmt.Errorf("moe router: HotSkewThreshold %g must be >= 1", c.HotSkewThreshold)
	case c.MaxShadowSlots < 0:
		return c, fmt.Errorf("moe router: MaxShadowSlots %d must be >= 0", c.MaxShadowSlots)
	}
	return c, nil
}

// ExpertInstance is one copy of an expert's weights on a rank: the primary
// placement or a shadow-slot mirror. It serves tokens [Start, Start+Count)
// of the expert's token list in token order.
type ExpertInstance struct {
	Expert int
	Rank   int
	Shadow bool
	Start  int
	Count  int
}

// RoutingPlan is the result of [ComputeRoutingPlan].
type RoutingPlan struct {
	SelectedExperts [][]int     // [tokens][TopK] in selection order
	SelectedWeights [][]float32 // [tokens][TopK] combine weights, summing to 1
	ExpertCounts    []int       // [NumExperts] tokens per expert before mirroring
	UpdatedBiases   []float32   // [NumExperts] biases after one update step

	// Gather4Indices lists token rows grouped by expert in token order,
	// each expert padded to a multiple of MTile with the pad row (the
	// token count, a zero row appended to the activations), cut into the
	// 4-row quads of cp.async.bulk.tensor.2d...tile::gather4.
	Gather4Indices   [][4]int32
	ExpertRowOffsets []int // [NumExperts+1] first gather row of each expert

	Instances       []ExpertInstance // primaries in expert order, then shadows
	MirroredCounts  []int            // [len(Instances)] tokens per instance
	InstanceGather4 [][][4]int32     // [len(Instances)] quads padded per instance
	RankLoads       []int            // [numRanks] tokens after mirroring

	UnmirroredRankLoadSkew float64 // max/mean rank load with primaries only
	MaxRankLoadSkew        float64 // max/mean rank load after mirroring
}

// ExpertHomeRank returns the rank that holds expert e's primary weights:
// experts are placed in contiguous blocks, e*numRanks/numExperts.
func ExpertHomeRank(e, numExperts, numRanks int) int {
	return e * numRanks / numExperts
}

// ComputeRoutingPlan routes logits ([tokens][NumExperts]) with biases
// ([NumExperts], used only for selection), computes one bias update, packs
// the expert-sorted gather quads, and mirrors hot experts into shadow slots
// of the least-loaded of numRanks ranks.
//
// The float32 arithmetic follows the operation order of the kernel that
// [FusedMoERouter] emits, including the warp butterfly order of every
// cross-lane sum, so selections agree with it exactly.
func ComputeRoutingPlan(cfg MoERouterConfig, logits [][]float32, biases []float32, numRanks int) (RoutingPlan, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return RoutingPlan{}, err
	}
	E := cfg.NumExperts
	T := len(logits)
	if T == 0 {
		return RoutingPlan{}, fmt.Errorf("moe router: no tokens")
	}
	for t, row := range logits {
		if len(row) != E {
			return RoutingPlan{}, fmt.Errorf("moe router: logits[%d] has %d experts, want %d", t, len(row), E)
		}
	}
	if biases == nil {
		biases = make([]float32, E)
	}
	if len(biases) != E {
		return RoutingPlan{}, fmt.Errorf("moe router: %d biases, want %d", len(biases), E)
	}
	if numRanks < 1 || numRanks > E {
		return RoutingPlan{}, fmt.Errorf("moe router: numRanks %d must be in 1..%d", numRanks, E)
	}

	sel, weightBase := routerScores(cfg, logits, biases)
	p := RoutingPlan{
		SelectedExperts: make([][]int, T),
		SelectedWeights: make([][]float32, T),
		ExpertCounts:    make([]int, E),
	}
	for t := range T {
		experts := selectExperts(cfg, sel[t])
		w := make([]float32, len(experts))
		var sum float32
		for k, e := range experts {
			w[k] = weightBase[t][e]
			sum += w[k]
		}
		for k := range w {
			w[k] /= sum
		}
		for _, e := range experts {
			p.ExpertCounts[e]++
		}
		p.SelectedExperts[t] = experts
		p.SelectedWeights[t] = w
	}

	mean := float64(T*cfg.TopK) / float64(E)
	p.UpdatedBiases = make([]float32, E)
	for e, b := range biases {
		switch c := float64(p.ExpertCounts[e]); {
		case c < mean:
			b += cfg.BiasUpdateRate
		case c > mean:
			b -= cfg.BiasUpdateRate
		}
		p.UpdatedBiases[e] = b
	}

	tokens := expertTokens(p.SelectedExperts, E)
	pad := int32(T)
	p.ExpertRowOffsets = make([]int, E+1)
	var rows []int32
	for e := range E {
		p.ExpertRowOffsets[e] = len(rows)
		rows = appendPadded(rows, tokens[e], cfg.MTile, pad)
	}
	p.ExpertRowOffsets[E] = len(rows)
	p.Gather4Indices = quads(rows)

	p.mirror(cfg, tokens, numRanks, mean)
	return p, nil
}

// routerScores returns the selection scores and the unnormalized combine
// weights for every token and expert.
func routerScores(cfg MoERouterConfig, logits [][]float32, biases []float32) (sel, weight [][]float32) {
	T, E := len(logits), cfg.NumExperts
	sel = make([][]float32, T)
	weight = make([][]float32, T)
	for t, u := range logits {
		sel[t] = make([]float32, E)
		weight[t] = make([]float32, E)
		switch cfg.Scoring {
		case RouterSigmoidBiasFree:
			for e, x := range u {
				s := 1 / (1 + expf(-x))
				weight[t][e] = s
				sel[t][e] = s + biases[e]
			}
		default:
			copy(weight[t], softmaxRow(u))
			copy(sel[t], weight[t])
		}
	}
	if cfg.Scoring != RouterSinkhornKnopp {
		return sel, weight
	}
	// Sinkhorn-Knopp on q = exp(u - rowmax), shared with the softmax
	// numerator so that the kernel computes it once.
	for t, u := range logits {
		m := warpMax(u)
		for e, x := range u {
			sel[t][e] = expf(x - m)
		}
	}
	target := float32(T) / float32(E)
	for range SinkhornIterations {
		for e := range E {
			var c float32
			for t := range T {
				c += sel[t][e]
			}
			scale := target / c
			for t := range T {
				sel[t][e] = float32(sel[t][e] * scale)
			}
		}
		for t := range T {
			r := warpSum(sel[t])
			for e := range E {
				sel[t][e] /= r
			}
		}
	}
	return sel, weight
}

// SinkhornBalance runs only the Sinkhorn-Knopp normalization of [routerScores]
// and returns the matrix; its rows sum to 1 and columns to tokens/experts.
func SinkhornBalance(cfg MoERouterConfig, logits [][]float32) ([][]float32, error) {
	cfg.Scoring = RouterSinkhornKnopp
	cfg, err := cfg.norm()
	if err != nil {
		return nil, err
	}
	for t, row := range logits {
		if len(row) != cfg.NumExperts {
			return nil, fmt.Errorf("moe router: logits[%d] has %d experts, want %d", t, len(row), cfg.NumExperts)
		}
	}
	sel, _ := routerScores(cfg, logits, make([]float32, cfg.NumExperts))
	return sel, nil
}

func softmaxRow(u []float32) []float32 {
	m := warpMax(u)
	q := make([]float32, len(u))
	for e, x := range u {
		q[e] = expf(x - m)
	}
	s := warpSum(q)
	for e := range q {
		q[e] /= s
	}
	return q
}

func expf(x float32) float32 { return float32(math.Exp(float64(x))) }

// warpSum sums v the way a warp does when lane l owns v[l], v[l+32], ...:
// sequential lane partials, then a shfl.bfly reduction with masks 16..1.
func warpSum(v []float32) float32 {
	var lane [32]float32
	for i, x := range v {
		lane[i%32] += x
	}
	for m := 16; m > 0; m >>= 1 {
		var next [32]float32
		for l := range 32 {
			next[l] = lane[l] + lane[l^m]
		}
		lane = next
	}
	return lane[0]
}

func warpMax(v []float32) float32 {
	m := float32(math.Inf(-1))
	for _, x := range v {
		m = max(m, x)
	}
	return m
}

// better reports whether score a at index ia ranks above b at ib: higher
// score first, lower index on ties.
func better(a float32, ia int, b float32, ib int) bool {
	return a > b || (a == b && ia < ib)
}

// selectExperts returns the group-limited top-K experts for one token's
// selection scores, in selection order. A group scores the sum of its two
// best experts (one when a group has a single expert).
func selectExperts(cfg MoERouterConfig, sel []float32) []int {
	E, G := cfg.NumExperts, cfg.NumGroups
	per := E / G
	gscore := make([]float32, G)
	for g := range G {
		b1, b2 := float32(math.Inf(-1)), float32(math.Inf(-1))
		for e := g * per; e < (g+1)*per; e++ {
			switch s := sel[e]; {
			case s > b1:
				b1, b2 = s, b1
			case s > b2:
				b2 = s
			}
		}
		gscore[g] = b1
		if per > 1 {
			gscore[g] = b1 + b2
		}
	}
	allowed := make([]bool, G)
	for range cfg.TopGroups {
		best := -1
		for g := range G {
			if !allowed[g] && (best < 0 || better(gscore[g], g, gscore[best], best)) {
				best = g
			}
		}
		allowed[best] = true
	}
	taken := make([]bool, E)
	out := make([]int, 0, cfg.TopK)
	for range cfg.TopK {
		best := -1
		for e := range E {
			if allowed[e/per] && !taken[e] && (best < 0 || better(sel[e], e, sel[best], best)) {
				best = e
			}
		}
		taken[best] = true
		out = append(out, best)
	}
	return out
}

func expertTokens(selected [][]int, numExperts int) [][]int32 {
	tokens := make([][]int32, numExperts)
	for t, experts := range selected {
		for _, e := range experts {
			tokens[e] = append(tokens[e], int32(t))
		}
	}
	return tokens
}

func appendPadded(rows, tokens []int32, tile int, pad int32) []int32 {
	rows = append(rows, tokens...)
	for n := len(tokens); n%tile != 0; n++ {
		rows = append(rows, pad)
	}
	return rows
}

func quads(rows []int32) [][4]int32 {
	q := make([][4]int32, len(rows)/4)
	for i := range q {
		copy(q[i][:], rows[4*i:])
	}
	return q
}

// mirror places primaries, then repeatedly takes the most loaded rank and,
// if it serves a hot expert, mirrors that expert into a free shadow slot on
// the least loaded rank that does not hold it, re-splitting the expert's
// tokens across its instances in quads so as to level rank loads.
func (p *RoutingPlan) mirror(cfg MoERouterConfig, tokens [][]int32, numRanks int, mean float64) {
	E := cfg.NumExperts
	byExpert := make([][]int, E) // instance indices per expert
	for e := range E {
		byExpert[e] = []int{len(p.Instances)}
		p.Instances = append(p.Instances, ExpertInstance{
			Expert: e, Rank: ExpertHomeRank(e, E, numRanks), Count: len(tokens[e]),
		})
	}
	load := func() []int {
		l := make([]int, numRanks)
		for _, in := range p.Instances {
			l[in.Rank] += in.Count
		}
		return l
	}
	p.UnmirroredRankLoadSkew = skew(load())

	slots := make([]int, numRanks)
	for r := range slots {
		slots[r] = cfg.MaxShadowSlots
	}
	hot := func(e int) bool { return float64(len(tokens[e])) > cfg.HotSkewThreshold*mean }
	for {
		loads := load()
		top := 0
		for r, l := range loads {
			if l > loads[top] {
				top = r
			}
		}
		victim := -1
		for i, in := range p.Instances {
			if in.Rank == top && hot(in.Expert) && (victim < 0 || in.Count > p.Instances[victim].Count) {
				victim = i
			}
		}
		if victim < 0 {
			break
		}
		e := p.Instances[victim].Expert
		holds := make([]bool, numRanks)
		for _, i := range byExpert[e] {
			holds[p.Instances[i].Rank] = true
		}
		dst := -1
		for r := range numRanks {
			if slots[r] > 0 && !holds[r] && (dst < 0 || loads[r] < loads[dst]) {
				dst = r
			}
		}
		if dst < 0 {
			break
		}
		slots[dst]--
		byExpert[e] = append(byExpert[e], len(p.Instances))
		p.Instances = append(p.Instances, ExpertInstance{Expert: e, Rank: dst, Shadow: true})
		p.split(byExpert[e], len(tokens[e]), loads)
	}

	pad := int32(len(p.SelectedExperts))
	p.MirroredCounts = make([]int, len(p.Instances))
	p.InstanceGather4 = make([][][4]int32, len(p.Instances))
	for i, in := range p.Instances {
		p.MirroredCounts[i] = in.Count
		run := tokens[in.Expert][in.Start : in.Start+in.Count]
		p.InstanceGather4[i] = quads(appendPadded(nil, run, cfg.MTile, pad))
	}
	p.RankLoads = load()
	p.MaxRankLoadSkew = skew(p.RankLoads)
}

// split water-fills n tokens of one expert, a quad at a time, onto the
// instance whose rank carries the least other load, then lays the shares
// out in instance order.
func (p *RoutingPlan) split(inst []int, n int, loads []int) {
	base := make([]int, len(inst))
	share := make([]int, len(inst))
	for j, i := range inst {
		in := p.Instances[i]
		base[j] = loads[in.Rank] - in.Count
	}
	for left := n; left > 0; {
		best := 0
		for j := range inst {
			if base[j]+share[j] < base[best]+share[best] {
				best = j
			}
		}
		q := min(4, left)
		share[best] += q
		left -= q
	}
	start := 0
	for j, i := range inst {
		p.Instances[i].Start = start
		p.Instances[i].Count = share[j]
		start += share[j]
	}
}

func skew(loads []int) float64 {
	total, top := 0, 0
	for _, l := range loads {
		total += l
		top = max(top, l)
	}
	if total == 0 {
		return 1
	}
	return float64(top) * float64(len(loads)) / float64(total)
}

// Gather4Specs returns the TMA .gather4 transfer for each quad of
// p.Gather4Indices, loading cols elements of elemBytes per row on target.
func (p RoutingPlan) Gather4Specs(target string, cols, elemBytes int) ([]cudatma.TMAGather4Spec, error) {
	specs := make([]cudatma.TMAGather4Spec, len(p.Gather4Indices))
	for i, q := range p.Gather4Indices {
		specs[i] = cudatma.TMAGather4Spec{
			Target: target, Mode: cudatma.TMAGather4Load,
			ColDim: cols, ElemBytes: elemBytes, RowIndices: q,
		}
		if err := specs[i].Validate(); err != nil {
			return nil, fmt.Errorf("moe router: gather quad %d: %w", i, err)
		}
	}
	return specs, nil
}

// FusedMoERouter emits a single-CTA router kernel that scores, group-limits,
// selects top-K, builds the per-expert histogram in shared memory, and
// writes the expert-sorted gather rows, with no intermediate kernel.
//
// One warp routes one token at a time; lane l owns experts l, l+32, ....
// Cross-lane maxima and sums use shfl.sync.bfly (__shfl_xor_sync), the
// histogram uses shared-memory atomics, and expert ranking is a butterfly
// arg-max. Sinkhorn mode needs column sums over every token, so the whole
// batch is routed by one CTA; the kernel is therefore for decode-sized
// batches, and multi-CTA Sinkhorn (a cluster DSMEM reduction) is not emitted.
//
// The emitted CUDA C++ runs under cudahostemu. It uses only warp shuffles,
// shared atomics and __syncthreads, so it also compiles for sm_90a through
// sm_121a; the gather quads it writes are the indices a TMA .gather4 consumer
// ([RoutingPlan.Gather4Specs]) loads, and that consumer is not emitted here.
type FusedMoERouter struct {
	Name   string
	Config MoERouterConfig
	Tokens int // tokens per launch
	Warps  int // warps in the CTA (default 4)
}

func (k FusedMoERouter) norm() (FusedMoERouter, error) {
	if k.Name == "" {
		k.Name = "nvl72_fused_moe_router"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("moe router: bad kernel name %q", k.Name)
	}
	if k.Warps == 0 {
		k.Warps = 4
	}
	if k.Warps < 1 || k.Warps > 32 {
		return k, fmt.Errorf("moe router: Warps %d must be in 1..32", k.Warps)
	}
	if k.Tokens < 1 {
		return k, fmt.Errorf("moe router: Tokens %d must be >= 1", k.Tokens)
	}
	cfg, err := k.Config.norm()
	if err != nil {
		return k, err
	}
	k.Config = cfg
	return k, nil
}

// GatherRows returns the upper bound on gather rows the kernel writes: every
// expert's tokens padded to MTile.
func (k FusedMoERouter) GatherRows() int {
	k, err := k.norm()
	if err != nil {
		return 0
	}
	c := k.Config
	return k.Tokens*c.TopK + c.NumExperts*(c.MTile-1)
}

// Grid returns the launch configuration: one CTA of Warps warps.
func (k FusedMoERouter) Grid() (grid, block [3]int) {
	w := k.Warps
	if w == 0 {
		w = 4
	}
	return [3]int{1, 1, 1}, [3]int{32 * w, 1, 1}
}

// CUDA returns the kernel definition without [Prelude]. Its parameters are
// logits [T][E] f32, bias [E] f32, sel_experts [T][K] i32, sel_weights
// [T][K] f32, counts [E] i32, gather [GatherRows] i32, row_offsets [E+1]
// i32, and work [T][E] f32 scratch.
func (k FusedMoERouter) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	c := k.Config
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: fused %s MoE router, %d tokens x %d experts, %d groups (top %d), top-%d, MTile %d.\n",
		k.Name, c.Scoring, k.Tokens, c.NumExperts, c.NumGroups, c.TopGroups, c.TopK, c.MTile)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n", 32*k.Warps, k.Name)
	b.WriteString(`    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int32_t* __restrict__ sel_experts,
    float* __restrict__ sel_weights,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ gather,
    int32_t* __restrict__ row_offsets,
    float* __restrict__ work) {
`)
	fmt.Fprintf(&b, "    constexpr int T = %d, E = %d, G = %d, TOPG = %d, K = %d, MTILE = %d, WARPS = %d, MODE = %d, ITERS = %d;\n",
		k.Tokens, c.NumExperts, c.NumGroups, c.TopGroups, c.TopK, c.MTile, k.Warps, c.Scoring, SinkhornIterations)
	b.WriteString(moeRouterBody)
	return b.String(), nil
}

const moeRouterBody = `    constexpr int EPL = E / 32, PER = E / G;
    __shared__ int32_t s_counts[E];
    __shared__ int32_t s_offsets[E + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int e = threadIdx.x; e < E; e += blockDim.x) s_counts[e] = 0;
    __syncthreads();

    // Phase 1 (Sinkhorn only): q = exp(u - rowmax), then ITERS rounds of
    // column scaling to T/E and row scaling to 1.
    if constexpr (MODE == 2) {
        for (int t = warp; t < T; t += WARPS) {
            const float* u = logits + (size_t)t * E;
            float m = -INFINITY;
            for (int j = 0; j < EPL; ++j) m = fmaxf(m, u[lane + 32 * j]);
            for (int x = 16; x > 0; x >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, x));
            for (int j = 0; j < EPL; ++j) work[(size_t)t * E + lane + 32 * j] = expf(u[lane + 32 * j] - m);
        }
        __syncthreads();
        for (int it = 0; it < ITERS; ++it) {
            for (int e = threadIdx.x; e < E; e += blockDim.x) {
                float c = 0.0f;
                for (int t = 0; t < T; ++t) c += work[(size_t)t * E + e];
                const float scale = ((float)T / (float)E) / c;
                for (int t = 0; t < T; ++t) work[(size_t)t * E + e] = work[(size_t)t * E + e] * scale;
            }
            __syncthreads();
            for (int t = warp; t < T; t += WARPS) {
                float r = 0.0f;
                for (int j = 0; j < EPL; ++j) r += work[(size_t)t * E + lane + 32 * j];
                for (int x = 16; x > 0; x >>= 1) r += __shfl_xor_sync(0xffffffffu, r, x);
                for (int j = 0; j < EPL; ++j) work[(size_t)t * E + lane + 32 * j] /= r;
            }
            __syncthreads();
        }
    }

    // Phase 2: per-token scoring, group limit, top-K, histogram.
    for (int t = warp; t < T; t += WARPS) {
        const float* u = logits + (size_t)t * E;
        float wv[EPL];
        float* srow = work + (size_t)t * E;
        if constexpr (MODE == 0) {
            for (int j = 0; j < EPL; ++j) {
                const int e = lane + 32 * j;
                wv[j] = 1.0f / (1.0f + expf(-u[e]));
                srow[e] = wv[j] + bias[e];
            }
        } else {
            float m = -INFINITY;
            for (int j = 0; j < EPL; ++j) m = fmaxf(m, u[lane + 32 * j]);
            for (int x = 16; x > 0; x >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, x));
            float s = 0.0f;
            for (int j = 0; j < EPL; ++j) {
                wv[j] = expf(u[lane + 32 * j] - m);
                s += wv[j];
            }
            for (int x = 16; x > 0; x >>= 1) s += __shfl_xor_sync(0xffffffffu, s, x);
            for (int j = 0; j < EPL; ++j) {
                wv[j] /= s;
                if constexpr (MODE == 1) srow[lane + 32 * j] = wv[j];
            }
        }
        __syncwarp();

        // Group score: sum of the two best selection scores in the group.
        float gs = -INFINITY;
        if (lane < G) {
            float b1 = -INFINITY, b2 = -INFINITY;
            for (int e = lane * PER; e < (lane + 1) * PER; ++e) {
                const float s = srow[e];
                if (s > b1) { b2 = b1; b1 = s; } else if (s > b2) { b2 = s; }
            }
            gs = PER > 1 ? b1 + b2 : b1;
        }
        unsigned allowed = 0u;
        for (int i = 0; i < TOPG; ++i) {
            float bs = (lane < G && !((allowed >> lane) & 1u)) ? gs : -INFINITY;
            int bi = (lane < G && !((allowed >> lane) & 1u)) ? lane : 0x7fffffff;
            for (int x = 16; x > 0; x >>= 1) {
                const float os = __shfl_xor_sync(0xffffffffu, bs, x);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, x);
                if (os > bs || (os == bs && oi < bi)) { bs = os; bi = oi; }
            }
            allowed |= 1u << bi;
        }

        unsigned taken = 0u;
        float topw[K];
        int tope[K];
        for (int k = 0; k < K; ++k) {
            float bs = -INFINITY;
            int bi = 0x7fffffff;
            for (int j = 0; j < EPL; ++j) {
                const int e = lane + 32 * j;
                if (((allowed >> (e / PER)) & 1u) && !((taken >> j) & 1u)) {
                    const float s = srow[e];
                    if (bi == 0x7fffffff || s > bs || (s == bs && e < bi)) { bs = s; bi = e; }
                }
            }
            for (int x = 16; x > 0; x >>= 1) {
                const float os = __shfl_xor_sync(0xffffffffu, bs, x);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, x);
                if (oi != 0x7fffffff && (bi == 0x7fffffff || os > bs || (os == bs && oi < bi))) { bs = os; bi = oi; }
            }
            const int owner = bi & 31, slot = bi >> 5;
            float w = 0.0f;
            for (int j = 0; j < EPL; ++j) if (j == slot) w = wv[j];
            w = __shfl_sync(0xffffffffu, w, owner);
            if (lane == owner) taken |= 1u << slot;
            topw[k] = w;
            tope[k] = bi;
        }
        if (lane == 0) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += topw[k];
            for (int k = 0; k < K; ++k) {
                sel_experts[t * K + k] = tope[k];
                sel_weights[t * K + k] = topw[k] / sum;
                atomicAdd(&s_counts[tope[k]], 1);
            }
        }
        __syncwarp();
    }
    __syncthreads();

    // Phase 3: padded expert offsets and expert-sorted gather rows.
    if (threadIdx.x == 0) {
        int off = 0;
        for (int e = 0; e < E; ++e) {
            s_offsets[e] = off;
            off += (s_counts[e] + MTILE - 1) / MTILE * MTILE;
        }
        s_offsets[E] = off;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        counts[e] = s_counts[e];
        row_offsets[e] = s_offsets[e];
        int pos = s_offsets[e];
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k)
                if (sel_experts[t * K + k] == e) gather[pos++] = t;
        while (pos < s_offsets[e + 1]) gather[pos++] = T;
    }
    if (threadIdx.x == 0) row_offsets[E] = s_offsets[E];
}
`

// Launcher returns the host launcher function for k.
func (k FusedMoERouter) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(`inline void %[1]s_launch(const float* logits, const float* bias, int32_t* sel_experts, float* sel_weights,
    int32_t* counts, int32_t* gather, int32_t* row_offsets, float* work, cudaStream_t stream) {
    EG_LAUNCH(%[1]s, dim3(1), dim3(%[2]d), 0, stream, logits, bias, sel_experts, sel_weights, counts, gather, row_offsets, work);
}
`, k.Name, 32*k.Warps), nil
}

// Source returns a complete translation unit with [Prelude], the kernel, and
// its launcher.
func (k FusedMoERouter) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// EmitFusedMoERouter returns the [FusedMoERouter.Source] for cfg routing
// tokens per launch under the kernel name kernelName.
func EmitFusedMoERouter(cfg MoERouterConfig, tokens int, kernelName string) (string, error) {
	return FusedMoERouter{Name: kernelName, Config: cfg, Tokens: tokens}.Source()
}
