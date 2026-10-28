package tensorparallel

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// A Layer is the shape of one dense LLaMA decoder layer over Rows token
// rows.
type Layer struct {
	Hidden       int
	Heads        int // query heads
	KVHeads      int
	HeadDim      int
	Intermediate int
	Eps          float32
	QKVBias      bool
	Rows         int // 0 means 1

	// QKVOut is the dtype of the QKV projection output, and AttnDType the
	// dtype of the attention output the o_proj reads. Zero means F32.
	QKVOut    hopper.DType
	AttnDType hopper.DType
}

func (l Layer) norm() (Layer, error) {
	if l.Rows == 0 {
		l.Rows = 1
	}
	if l.QKVOut == 0 {
		l.QKVOut = hopper.F32
	}
	if l.AttnDType == 0 {
		l.AttnDType = hopper.F32
	}
	if l.Hidden <= 0 || l.Heads <= 0 || l.KVHeads <= 0 || l.HeadDim <= 0 || l.Intermediate <= 0 || l.Rows < 0 {
		return l, fmt.Errorf("tensorparallel: bad layer shape %+v", l)
	}
	if l.Heads%l.KVHeads != 0 {
		return l, fmt.Errorf("tensorparallel: %d query heads not a multiple of %d kv heads", l.Heads, l.KVHeads)
	}
	return l, nil
}

// A Fusion says where a layer's all-reduces run.
type Fusion uint8

const (
	// FuseNone writes each rank's partial to its own buffer; a separate
	// collective sums them.
	FuseNone Fusion = iota
	// FuseReduceAdd adds every rank's partial into one buffer with
	// red.global.add: ranks on one device, or peers of one address space.
	FuseReduceAdd
	// FuseMultimem adds every rank's partial into every rank's copy of a
	// multicast buffer with multimem.red (NVLS, sm_90 and later).
	FuseMultimem
)

func (f Fusion) String() string {
	switch f {
	case FuseNone:
		return "none"
	case FuseReduceAdd:
		return "red.add"
	case FuseMultimem:
		return "multimem"
	}
	return fmt.Sprintf("Fusion(%d)", uint8(f))
}

func (f Fusion) store() hopper.GEMMStore {
	switch f {
	case FuseReduceAdd:
		return hopper.StoreReduceAdd
	case FuseMultimem:
		return hopper.StoreMultimemReduceAdd
	}
	return hopper.StoreWrite
}

// Options configure [Shard].
type Options struct {
	TP     int    // ranks; must divide the KV heads and the intermediate size
	Prefix string // kernel name prefix; empty means "tp"
	Fusion Fusion
	// Target is the SM target, such as "sm_121a" or "sm_103a". FuseMultimem
	// needs sm_90 or later; other fusions ignore it.
	Target string
}

// A Range is the half-open interval [Lo, Hi).
type Range struct{ Lo, Hi int }

// Len returns Hi - Lo.
func (r Range) Len() int { return r.Hi - r.Lo }

// A RankPlan is one rank's share of a sharded layer.
type RankPlan struct {
	Rank int

	// AttnIn computes the rank's q, k and v heads from the full x: inputs
	// x, attn_norm, w_qkv and, with a bias, b_qkv.
	AttnIn *hopper.Plan
	// OProj computes the rank's o_proj partial from its attention heads:
	// inputs attn, w_o and, on rank 0, the residual x.
	OProj *hopper.Plan
	// FFN computes the rank's MLP partial from the reduced h: inputs h,
	// ffn_norm, w_gu, w_down; rank 0 adds h.
	FFN *hopper.Plan

	// QHeads and KVHeads are the rank's query and KV heads; Inter its
	// columns of the intermediate dimension.
	QHeads, KVHeads, Inter Range

	headDim, hidden, inter int
}

// A CollectiveNode is an all-reduce between plans: every rank's output is
// the sum over ranks of its input, Count F32 values.
type CollectiveNode struct {
	Name  string // "attn" after OProj, "ffn" after FFN
	Count int
	// Fused reports that the producing GEMM's epilogue performs the sum
	// ([Fusion] other than FuseNone): no separate collective runs, but
	// the output must be zeroed before and fenced across ranks after.
	Fused bool
	index int
}

// Intents returns the node's collective intents in layer layer, one per
// rank of a tp-rank communicator named comm at generation gen. Each layer
// and node has its own effect token.
func (c CollectiveNode) Intents(comm string, gen uint64, layer, tp int) []egraph.CollectiveIntent {
	out := make([]egraph.CollectiveIntent, tp)
	for r := range out {
		out[r] = egraph.CollectiveIntent{
			Communicator: comm,
			Generation:   gen,
			Rank:         r,
			WorldSize:    tp,
			Operation:    "allreduce",
			DType:        "f32",
			Count:        uint64(c.Count),
			EffectToken:  uint64(2*layer + c.index + 1),
		}
	}
	return out
}

// A Plan is a [Layer] sharded over TP ranks.
type Plan struct {
	TP     int
	Layer  Layer // the full layer
	Local  Layer // one rank's shard
	Fusion Fusion
	Ranks  []RankPlan

	// Attn reduces the o_proj partials into h; FFN reduces the MLP
	// partials into the layer output.
	Attn, FFN CollectiveNode
}

// Collectives returns the plan's all-reduces in execution order.
func (p *Plan) Collectives() []CollectiveNode { return []CollectiveNode{p.Attn, p.FFN} }

// Shard shards l over opt.TP ranks and builds every rank's plans. TP must
// divide the KV heads (so each rank keeps whole GQA groups) and the
// intermediate size; TP 1 yields the unsharded layer split at the same
// points.
func Shard(l Layer, opt Options) (*Plan, error) {
	l, err := l.norm()
	if err != nil {
		return nil, err
	}
	tp := opt.TP
	if tp < 1 {
		return nil, fmt.Errorf("tensorparallel: tp %d < 1", tp)
	}
	if l.KVHeads%tp != 0 || l.Intermediate%tp != 0 {
		return nil, fmt.Errorf("tensorparallel: tp %d does not divide %d kv heads and intermediate size %d", tp, l.KVHeads, l.Intermediate)
	}
	if opt.Fusion > FuseMultimem {
		return nil, fmt.Errorf("tensorparallel: bad fusion %v", opt.Fusion)
	}
	if opt.Fusion != FuseNone && l.Rows > 16 {
		return nil, fmt.Errorf("tensorparallel: %v fusion needs at most 16 rows, have %d", opt.Fusion, l.Rows)
	}
	if opt.Fusion == FuseMultimem {
		sm, err := smVersion(opt.Target)
		if err != nil {
			return nil, err
		}
		if sm < 90 {
			return nil, fmt.Errorf("tensorparallel: multimem fusion needs sm_90 or later, target %s", opt.Target)
		}
	}
	prefix := opt.Prefix
	if prefix == "" {
		prefix = "tp"
	}
	local := l
	local.Heads, local.KVHeads, local.Intermediate = l.Heads/tp, l.KVHeads/tp, l.Intermediate/tp

	p := &Plan{
		TP:     tp,
		Layer:  l,
		Local:  local,
		Fusion: opt.Fusion,
		Attn:   CollectiveNode{Name: "attn", Count: l.Rows * l.Hidden, Fused: opt.Fusion != FuseNone, index: 0},
		FFN:    CollectiveNode{Name: "ffn", Count: l.Rows * l.Hidden, Fused: opt.Fusion != FuseNone, index: 1},
	}
	attnIn, err := buildAttnIn(local, prefix+"_attn_in")
	if err != nil {
		return nil, err
	}
	leadO, leadF, err := buildPost(local, prefix, true, opt.Fusion)
	if err != nil {
		return nil, err
	}
	peerO, peerF := leadO, leadF
	if tp > 1 {
		if peerO, peerF, err = buildPost(local, prefix+"_peer", false, opt.Fusion); err != nil {
			return nil, err
		}
	}
	for r := range tp {
		rp := RankPlan{
			Rank:    r,
			AttnIn:  attnIn,
			OProj:   leadO,
			FFN:     leadF,
			QHeads:  Range{r * local.Heads, (r + 1) * local.Heads},
			KVHeads: Range{r * local.KVHeads, (r + 1) * local.KVHeads},
			Inter:   Range{r * local.Intermediate, (r + 1) * local.Intermediate},
			headDim: l.HeadDim,
			hidden:  l.Hidden,
			inter:   l.Intermediate,
		}
		if r > 0 {
			rp.OProj, rp.FFN = peerO, peerF
		}
		p.Ranks = append(p.Ranks, rp)
	}
	return p, nil
}

func smVersion(target string) (int, error) {
	s, ok := strings.CutPrefix(target, "sm_")
	if !ok {
		return 0, fmt.Errorf("tensorparallel: bad target %q", target)
	}
	s = strings.TrimRight(s, "af")
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("tensorparallel: bad target %q", target)
	}
	return n, nil
}

func compile(name string, b *hopper.Builder, root egraph.Id, out hopper.DType) (*hopper.Plan, error) {
	e, err := hopper.Optimize(b.G, root)
	if err != nil {
		return nil, fmt.Errorf("tensorparallel: optimize %s: %w", name, err)
	}
	p, err := hopper.Compile(e, name, out)
	if err != nil {
		return nil, fmt.Errorf("tensorparallel: %w", err)
	}
	return p, nil
}

// buildAttnIn builds the column-parallel QKV projection of local's heads.
func buildAttnIn(local Layer, name string) (*hopper.Plan, error) {
	m, h := local.Rows, local.Hidden
	qkvDim := (local.Heads + 2*local.KVHeads) * local.HeadDim
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, m, h)
	norm := b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("attn_norm", hopper.F32, h), h, local.Eps))
	qkv := b.MatMul(norm, b.Input("w_qkv", hopper.BF16, qkvDim, h))
	if local.QKVBias {
		qkv = b.Map(hopper.MustLambda("(add $0 $1)"), qkv, b.Input("b_qkv", hopper.F32, qkvDim))
	}
	return compile(name, b, qkv, local.QKVOut)
}

// buildPost builds the row-parallel o_proj and the MLP of local's shard,
// named name_o_proj and name_ffn. With residual, each adds its input
// residual (x, h).
func buildPost(local Layer, name string, residual bool, fuse Fusion) (oproj, ffn *hopper.Plan, err error) {
	m, h := local.Rows, local.Hidden
	qDim := local.Heads * local.HeadDim
	add := hopper.MustLambda("(add $0 $1)")

	b := hopper.NewBuilder()
	ao := b.Input("attn", local.AttnDType, m, qDim)
	if local.AttnDType != hopper.BF16 {
		ao = b.Cast(hopper.BF16, ao)
	}
	o := b.MatMul(ao, b.Input("w_o", hopper.BF16, h, qDim))
	if residual {
		o = b.Map(add, o, b.Input("x", hopper.F32, m, h))
	}
	if oproj, err = compile(name+"_o_proj", b, o, hopper.F32); err != nil {
		return nil, nil, err
	}

	b = hopper.NewBuilder()
	hid := b.Input("h", hopper.F32, m, h)
	norm := b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, h), h, local.Eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", hopper.BF16, 2*local.Intermediate, h)))
	out := b.MatMul(act, b.Input("w_down", hopper.BF16, h, local.Intermediate))
	if residual {
		out = b.Map(add, out, hid)
	}
	if ffn, err = compile(name+"_ffn", b, out, hopper.F32); err != nil {
		return nil, nil, err
	}
	if fuse != FuseNone {
		for _, p := range []*hopper.Plan{oproj, ffn} {
			if err := fuseStore(p, fuse.store()); err != nil {
				return nil, nil, err
			}
		}
	}
	return oproj, ffn, nil
}

// fuseStore makes the GEMM writing p's output store with s.
func fuseStore(p *hopper.Plan, s hopper.GEMMStore) error {
	for i := range p.Kernels {
		k := &p.Kernels[i]
		if len(k.Args) == 0 || k.Args[len(k.Args)-1] != p.Output.Name {
			continue
		}
		if k.GEMM == nil {
			return fmt.Errorf("tensorparallel: %s: output not written by a skinny gemm; cannot fuse the all-reduce", p.Name)
		}
		g := *k.GEMM
		g.Store = s
		k.GEMM = &g
		return nil
	}
	return errors.New("tensorparallel: " + p.Name + ": no kernel writes the output")
}
