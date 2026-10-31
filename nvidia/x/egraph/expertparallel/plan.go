package expertparallel

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// Config describes one MoE layer and its expert-parallel split.
type Config struct {
	Ranks   int // EP ranks; divides Tokens and Experts
	Tokens  int // tokens across all ranks
	Hidden  int // model width, a multiple of 32
	Inter   int // routed expert intermediate width
	Experts int // routed experts, 32 to 1024; at most 32 per rank
	TopK    int // routes per token, 1 to 32

	// SharedInter is the shared expert's intermediate width; 0 means no
	// shared expert.
	SharedInter int

	// Route configures top-K routing. D, K and X are set from the layer;
	// Bias is 0 (no bias) or hopper.F32.
	Route hopper.RowTopK

	// Eps is the post-expert norm epsilon; 0 means 1e-8.
	Eps float32
}

// TokensPerRank returns Tokens/Ranks.
func (c Config) TokensPerRank() int { return c.Tokens / c.Ranks }

// LocalExperts returns Experts/Ranks.
func (c Config) LocalExperts() int { return c.Experts / c.Ranks }

// Capacity returns the receive rows per rank: every route of every token,
// the worst case of all routes landing on one rank.
func (c Config) Capacity() int { return c.Tokens * c.TopK }

func (c Config) validate() error {
	switch {
	case c.Ranks < 1 || c.Tokens < 1 || c.Tokens%c.Ranks != 0:
		return fmt.Errorf("expertparallel: %d tokens over %d ranks", c.Tokens, c.Ranks)
	case c.Experts < 32 || c.Experts > 1024 || c.Experts%32 != 0 || c.Experts%c.Ranks != 0:
		return fmt.Errorf("expertparallel: %d experts over %d ranks", c.Experts, c.Ranks)
	case c.LocalExperts() > 32:
		return fmt.Errorf("expertparallel: %d local experts exceed 32", c.LocalExperts())
	case c.TopK < 1 || c.TopK > 32:
		return fmt.Errorf("expertparallel: top-%d routing", c.TopK)
	case c.Hidden < 32 || c.Hidden%32 != 0 || c.Inter < 16 || c.Inter%16 != 0:
		return fmt.Errorf("expertparallel: hidden %d, inter %d", c.Hidden, c.Inter)
	case c.SharedInter < 0 || c.SharedInter%16 != 0:
		return fmt.Errorf("expertparallel: shared inter %d", c.SharedInter)
	case c.Route.Bias != 0 && c.Route.Bias != hopper.F32:
		return fmt.Errorf("expertparallel: route bias must be F32")
	}
	return nil
}

func (c Config) eps() float32 {
	if c.Eps == 0 {
		return 1e-8
	}
	return c.Eps
}

// A Tensor names one input of the layer and its full (EP=1) shape. Ranks
// read their slice of the expert weights.
type Tensor = hopper.Tensor

// Inputs returns the layer's inputs in [Plan.Source] parameter order, with
// their EP=1 shapes: logits [Tokens, Experts], bias [Experts] when routing
// has one, x [Tokens, Hidden], the NVFP4 w13 [Experts, 2·Inter, Hidden] and
// w2 [Experts, Hidden, Inter] (packed bytes, scale bytes, global scales),
// post_w [Hidden], and, with a shared expert, its NVFP4 s13 and s2 as one
// expert.
func (c Config) Inputs() []Tensor {
	T, E, H, I, J := c.Tokens, c.Experts, c.Hidden, c.Inter, c.SharedInter
	ts := []Tensor{{Name: "logits", DType: hopper.F32, Shape: []int{T, E}}}
	if c.Route.Bias != 0 {
		ts = append(ts, Tensor{Name: "bias", DType: hopper.F32, Shape: []int{E}})
	}
	ts = append(ts, Tensor{Name: "x", DType: hopper.BF16, Shape: []int{T, H}})
	ts = append(ts, nvfp4Inputs("w13", E, 2*I, H)...)
	ts = append(ts, nvfp4Inputs("w2", E, H, I)...)
	ts = append(ts, Tensor{Name: "post_w", DType: hopper.BF16, Shape: []int{H}})
	if J > 0 {
		ts = append(ts, nvfp4Inputs("s13", 1, 2*J, H)...)
		ts = append(ts, nvfp4Inputs("s2", 1, H, J)...)
	}
	return ts
}

func nvfp4Inputs(prefix string, experts, n, k int) []Tensor {
	return []Tensor{
		{Name: prefix + "_q", DType: hopper.U8, Shape: []int{experts, n, k / 2}},
		{Name: prefix + "_sf", DType: hopper.U8, Shape: []int{experts, n, k / 16}},
		{Name: prefix + "_gs", DType: hopper.F32, Shape: []int{experts}},
	}
}

func nvfp4(b *hopper.Builder, prefix string, experts, n, k int) egraph.Id {
	ts := nvfp4Inputs(prefix, experts, n, k)
	var ids [3]egraph.Id
	for i, t := range ts {
		ids[i] = b.Input(t.Name, t.DType, t.Shape...)
	}
	return b.NVFP4(ids[0], ids[1], ids[2])
}

func (c Config) route() hopper.RowTopK {
	r := c.Route
	r.D, r.K, r.X = c.Experts, c.TopK, hopper.F32
	return r
}

// Dense returns the EP=1 routed MoE term over all tokens and experts, the
// single-rank formulation EP=N must reproduce: top-K routing of logits,
// expert alignment, grouped W13+SwiGLU and W2 GEMMs, post-expert norm and
// the weighted top-K sum. It reads [Config.Inputs] by name (plus
// local_start [1], zero) and has no shared expert.
func (c Config) Dense() (*hopper.Builder, egraph.Id, error) {
	if err := c.validate(); err != nil {
		return nil, 0, err
	}
	T, E, H, I := c.Tokens, c.Experts, c.Hidden, c.Inter
	b := hopper.NewBuilder()
	logits := b.Input("logits", hopper.F32, T, E)
	var bias egraph.Id
	if c.Route.Bias != 0 {
		bias = b.Input("bias", hopper.F32, E)
	}
	ls := b.Input("local_start", hopper.I32, 1)
	x := b.Input("x", hopper.BF16, T, H)
	w13 := nvfp4(b, "w13", E, 2*I, H)
	w2 := nvfp4(b, "w2", E, H, I)
	postW := b.Input("post_w", hopper.BF16, H)
	ids, weights := b.RowTopK(c.route(), logits, bias)
	offsets, rows, slots := b.ExpertAlign(hopper.ExpertAlign{K: c.TopK, Experts: E, Local: E, Warps: 4}, ids, ls)
	inter := b.SwiGLU(b.GroupedMatMul(x, w13, offsets, rows))
	norm := b.PostExpertNorm(b.GroupedMatMul(inter, w2, offsets), postW, c.eps())
	return b, b.WeightedTopKSum(norm, slots, weights), nil
}

// expertTerm is one rank's expert stage over the Capacity received rows,
// each a single route (K = 1) with weight 1: the output row is the
// normalized expert output, which combine weights.
func (c Config) expertTerm() (*hopper.Builder, egraph.Id) {
	C, E, H, I, L := c.Capacity(), c.Experts, c.Hidden, c.Inter, c.LocalExperts()
	b := hopper.NewBuilder()
	x := b.Input("recv_x", hopper.BF16, C, H)
	ids := b.Input("recv_ids", hopper.I32, C, 1)
	ls := b.Input("local_start", hopper.I32, 1)
	w13 := nvfp4(b, "w13", L, 2*I, H)
	w2 := nvfp4(b, "w2", L, H, I)
	postW := b.Input("post_w", hopper.BF16, H)
	ones := b.Input("recv_w", hopper.F32, C, 1)
	offsets, rows, slots := b.ExpertAlign(hopper.ExpertAlign{K: 1, Experts: E, Local: L, Warps: 4}, ids, ls)
	inter := b.SwiGLU(b.GroupedMatMul(x, w13, offsets, rows))
	norm := b.PostExpertNorm(b.GroupedMatMul(inter, w2, offsets), postW, c.eps())
	return b, b.WeightedTopKSum(norm, slots, ones)
}

// sharedTerm is one rank's shared expert over its Tokens/Ranks rows, as a
// one-expert grouped GEMM pair with offsets {0, rows}.
func (c Config) sharedTerm() (*hopper.Builder, egraph.Id) {
	S, H, J := c.TokensPerRank(), c.Hidden, c.SharedInter
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.BF16, S, H)
	offsets := b.Input("shared_offsets", hopper.I32, 2)
	s13 := nvfp4(b, "s13", 1, 2*J, H)
	s2 := nvfp4(b, "s2", 1, H, J)
	return b, b.GroupedMatMul(b.SwiGLU(b.GroupedMatMul(x, s13, offsets)), s2, offsets)
}

// A Plan is a compiled expert-parallel MoE layer.
type Plan struct {
	Config
	Name string

	Route  hopper.RowTopK // the route stage kernel
	Expert *hopper.Plan   // one rank's expert stage; output [Capacity, Hidden] F32
	Shared *hopper.Plan   // one rank's shared expert, or nil; output [Tokens/Ranks, Hidden] F32

	expertExpr, sharedExpr *egraph.Expr
}

// New plans cfg. Every kernel and host function it emits is prefixed with
// name, a C identifier.
func New(cfg Config, name string) (*Plan, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if !cIdent(name) {
		return nil, fmt.Errorf("expertparallel: bad name %q", name)
	}
	p := &Plan{Config: cfg, Name: name, Route: cfg.route()}
	p.Route.Name = name + "_route"
	b, root := cfg.expertTerm()
	var err error
	if p.expertExpr, err = hopper.Optimize(b.G, root); err != nil {
		return nil, fmt.Errorf("expertparallel: expert stage: %w", err)
	}
	if p.expertExpr.Op != hopper.OpExpertSum {
		return nil, fmt.Errorf("expertparallel: expert stage optimized to %s, want %s", p.expertExpr.Op, hopper.OpExpertSum)
	}
	if p.Expert, err = hopper.Compile(p.expertExpr, name+"_expert", hopper.F32); err != nil {
		return nil, fmt.Errorf("expertparallel: expert stage: %w", err)
	}
	if cfg.SharedInter > 0 {
		b, root := cfg.sharedTerm()
		if p.sharedExpr, err = hopper.Optimize(b.G, root); err != nil {
			return nil, fmt.Errorf("expertparallel: shared expert: %w", err)
		}
		if p.Shared, err = hopper.Compile(p.sharedExpr, name+"_shared", hopper.F32); err != nil {
			return nil, fmt.Errorf("expertparallel: shared expert: %w", err)
		}
	}
	return p, nil
}

// A Stage is one step of a rank's schedule.
type Stage struct {
	Name string
	// Side reports whether the stage runs on the rank's side stream.
	Side bool
	// Collective reports whether the stage reads other ranks' buffers;
	// it waits for the preceding stage on every rank.
	Collective bool
	// Kernels is the number of launches.
	Kernels int
}

// Stages returns one rank's schedule in issue order. The shared expert, on
// the side stream, overlaps every main-stream stage before combine.
func (p *Plan) Stages() []Stage {
	st := []Stage{
		{Name: "route", Kernels: 1},
		{Name: "layout", Kernels: 1},
	}
	if p.Shared != nil {
		st = append(st, Stage{Name: "shared", Side: true, Kernels: len(p.Shared.Kernels)})
	}
	return append(st,
		Stage{Name: "dispatch", Collective: true, Kernels: 1},
		Stage{Name: "expert", Kernels: len(p.Expert.Kernels)},
		Stage{Name: "combine", Collective: true, Kernels: 1},
	)
}

// Collectives returns rank's dispatch and combine all-to-alls as collective
// intents on communicator comm at generation gen. Counts are the worst-case
// per-rank payloads: Capacity rows of Hidden BF16 values out, F32 back.
func (p *Plan) Collectives(comm string, gen uint64, rank int) ([]egraph.CollectiveIntent, error) {
	n := uint64(p.Capacity() * p.Hidden)
	cs := []egraph.CollectiveIntent{
		{Communicator: comm, Generation: gen, Rank: rank, WorldSize: p.Ranks, Operation: "alltoall", DType: "bf16", Count: n, Stream: 1, EffectToken: 1},
		{Communicator: comm, Generation: gen, Rank: rank, WorldSize: p.Ranks, Operation: "alltoall", DType: "f32", Count: n, Stream: 1, EffectToken: 2},
	}
	for i, c := range cs {
		if err := c.Validate(); err != nil {
			return nil, fmt.Errorf("expertparallel: collective %d: %w", i, err)
		}
	}
	return cs, nil
}

func cIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if r != '_' && (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (i == 0 || r < '0' || r > '9') {
			return false
		}
	}
	return true
}
