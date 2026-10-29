package hopper

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// Model-level term operators for paged attention, routing, grouped MoE
// projections, and cross-operator fusions.
const (
	OpPagedAttn        = "paged_attn"
	OpPagedAttnGated   = "paged_attn_gated"
	OpRMSNormGate      = "rms_norm_gate"
	OpRowTopK          = "row_topk"
	OpTopKIDs          = "topk_ids"
	OpTopKWeights      = "topk_weights"
	OpExpertAlign      = "expert_align"
	OpAlignOffsets     = "align_offsets"
	OpAlignRows        = "align_rows"
	OpAlignSlots       = "align_slots"
	OpGroupedMatMul    = "grouped_matmul"
	OpGroupedGEMM      = "grouped_gemm"
	OpGroupedGEMMGLU   = "grouped_gemm_glu"
	OpGroupedGEMMSumSq = "grouped_gemm_sumsq"
	OpSwiGLU           = "swiglu"
	OpPostExpertNorm   = "post_expert_norm"
	OpWeightedTopKSum  = "weighted_topk_sum"
	OpExpertSum        = "expert_sum"
	OpNone             = "none"
)

// PagedAttention adds a paged FP8 KV-cache grouped-query attention term:
// q [tokens, Hq, 64] over kCache, vCache [cacheRows, Hkv, 64], pageTable
// [batch, maxPages], seqLens [batch], kScale [1], and vScale [1].
func (b *Builder) PagedAttention(cfg PagedAttention, q, kCache, vCache, pageTable, seqLens, kScale, vScale egraph.Id) egraph.Id {
	cfg.Name = ""
	cfg.Out = 0
	cfg.Gate = false
	cfg.GateStride = 0
	cfg.GateOffset = 0
	cfg.GateEps = 0
	if cfg.QLen == 0 {
		cfg.QLen = 1
	}
	if cfg.Page == 0 {
		cfg.Page = 1
	}
	cfgNode := b.G.Add(egraph.ENode{
		Op:   fmt.Sprintf("attn:%d/%d/q%d/w%d/p%d", cfg.Hq, cfg.Hkv, cfg.QLen, cfg.Window, cfg.Page),
		Data: cfg,
	})
	return b.node(OpPagedAttn, cfgNode, q, kCache, vCache, pageTable, seqLens, kScale, vScale)
}

// RMSNormGate applies per-row RMSNorm (over the last axis) multiplied by
// sigmoid(gate): rms_norm(x, eps) * sigmoid(gate). When x is a [PagedAttention]
// term, [Optimize] rewrites the pair to a single gated [PagedAttention] launch.
func (b *Builder) RMSNormGate(x, gate egraph.Id, eps float32) egraph.Id {
	if eps == 0 {
		eps = 1e-5
	}
	return b.node(OpRMSNormGate, x, gate, b.G.Add(egraph.NewLeaf(eps)))
}

// RowTopK adds a top-K routing term over x [rows, D] and optional bias [D]
// (pass 0 when unused) and returns the selected expert indices I32 [rows, K]
// and routing weights F32 [rows, K].
func (b *Builder) RowTopK(cfg RowTopK, x, bias egraph.Id) (ids, weights egraph.Id) {
	cfg.Name = ""
	if bias == 0 {
		bias = b.G.Add(egraph.ENode{Op: OpNone})
	}
	cfgNode := b.G.Add(egraph.ENode{
		Op:   fmt.Sprintf("topk:%d/%d", cfg.D, cfg.K),
		Data: cfg,
	})
	tk := b.node(OpRowTopK, cfgNode, x, bias)
	return b.node(OpTopKIDs, tk), b.node(OpTopKWeights, tk)
}

// ExpertAlign sorts top-K route ids [tokens, K] (or [tokens*K]) by local
// expert starting at localStart [1] and returns the expert row offsets
// [Local+1], gathered token rows [tokens*K], and route-to-row slots [tokens*K].
func (b *Builder) ExpertAlign(cfg ExpertAlign, ids, localStart egraph.Id) (offsets, rows, slots egraph.Id) {
	cfg.Name = ""
	cfgNode := b.G.Add(egraph.ENode{
		Op:   fmt.Sprintf("align:%d/%d/%d", cfg.K, cfg.Local, cfg.Experts),
		Data: cfg,
	})
	al := b.node(OpExpertAlign, cfgNode, ids, localStart)
	return b.node(OpAlignOffsets, al), b.node(OpAlignRows, al), b.node(OpAlignSlots, al)
}

// GroupedMatMul multiplies expert-grouped rows of x by per-expert weights w
// using expert row bounds offsets [Local+1]. When rows is provided, row m of
// the grouped GEMM gathers x[rows[m]] before multiplying.
func (b *Builder) GroupedMatMul(x, w, offsets egraph.Id, rows ...egraph.Id) egraph.Id {
	var r egraph.Id
	if len(rows) > 0 && rows[0] != 0 {
		r = rows[0]
	} else {
		r = b.G.Add(egraph.ENode{Op: OpNone})
	}
	return b.node(OpGroupedMatMul, x, w, offsets, r)
}

// SwiGLU splits the last axis of x [..., 2*D] into gate [..., :D] and up
// [..., D:] halves and computes silu(gate) * up rounded to BF16.
func (b *Builder) SwiGLU(x egraph.Id) egraph.Id {
	return b.node(OpSwiGLU, x)
}

// PostExpertNorm applies the MoE post-expert gain and RMSNorm over each row of
// y [rows, width] with per-column gain postW [width]:
//
//	g = y * (1 + postW)
//	out = g * rsqrt(mean(g^2) + eps)
func (b *Builder) PostExpertNorm(y, postW egraph.Id, eps float32) egraph.Id {
	if eps == 0 {
		eps = 1e-8
	}
	return b.node(OpPostExpertNorm, y, postW, b.G.Add(egraph.NewLeaf(eps)))
}

// WeightedTopKSum accumulates the routed rows of y [tokens*K, width] for each
// token using ExpertAlign's slots [tokens*K] and RowTopK's weights [tokens, K]
// in increasing sorted row (local expert) order.
func (b *Builder) WeightedTopKSum(y, slots, weights egraph.Id) egraph.Id {
	return b.node(OpWeightedTopKSum, y, slots, weights)
}

func modelRules() []*egraph.Rewrite {
	return []*egraph.Rewrite{
		{
			Name: "paged-attn-gate",
			LHS: egraph.Pat(OpRMSNormGate,
				egraph.Pat(OpPagedAttn, egraph.Var("cfg"), egraph.Var("q"), egraph.Var("k"), egraph.Var("v"), egraph.Var("pt"), egraph.Var("sl"), egraph.Var("ks"), egraph.Var("vs")),
				egraph.Var("gate"),
				egraph.Var("eps"),
			),
			RHS: egraph.Pat(OpPagedAttnGated,
				egraph.Var("cfg"), egraph.Var("q"), egraph.Var("k"), egraph.Var("v"), egraph.Var("pt"), egraph.Var("sl"), egraph.Var("ks"), egraph.Var("vs"),
				egraph.Var("gate"),
				egraph.Var("eps"),
			),
		},
		{
			Name: "grouped-matmul-gemm",
			LHS:  egraph.Pat(OpGroupedMatMul, egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows")),
			RHS:  egraph.Pat(OpGroupedGEMM, egraph.Pat("fn:$0"), egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows")),
		},
		{
			Name: "grouped-matmul-swiglu",
			LHS: egraph.Pat(OpSwiGLU,
				egraph.Pat(OpGroupedMatMul, egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows")),
			),
			RHS: egraph.Pat(OpGroupedGEMMGLU, egraph.Pat("fn:(mul (silu $0) $1)"), egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows")),
		},
		{
			Name: "grouped-gemm-swiglu",
			LHS: egraph.Pat(OpSwiGLU,
				egraph.Pat(OpGroupedGEMM, egraph.Pat("fn:$0"), egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows")),
			),
			RHS: egraph.Pat(OpGroupedGEMMGLU, egraph.Pat("fn:(mul (silu $0) $1)"), egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows")),
		},
		{
			Name: "grouped-w2-postnorm-expertsum",
			LHS: egraph.Pat(OpWeightedTopKSum,
				egraph.Pat(OpPostExpertNorm,
					egraph.Pat(OpGroupedMatMul, egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows")),
					egraph.Var("post_w"),
					egraph.Var("eps"),
				),
				egraph.Var("slots"),
				egraph.Var("weights"),
			),
			RHS: egraph.Pat(OpExpertSum,
				egraph.Pat(OpGroupedGEMMSumSq, egraph.Pat("fn:(mul $0 (add 1 $1))"), egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows"), egraph.Var("post_w")),
				egraph.Var("slots"),
				egraph.Var("weights"),
				egraph.Var("eps"),
			),
		},
		{
			Name: "grouped-gemm-w2-postnorm-expertsum",
			LHS: egraph.Pat(OpWeightedTopKSum,
				egraph.Pat(OpPostExpertNorm,
					egraph.Pat(OpGroupedGEMM, egraph.Pat("fn:$0"), egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows")),
					egraph.Var("post_w"),
					egraph.Var("eps"),
				),
				egraph.Var("slots"),
				egraph.Var("weights"),
			),
			RHS: egraph.Pat(OpExpertSum,
				egraph.Pat(OpGroupedGEMMSumSq, egraph.Pat("fn:(mul $0 (add 1 $1))"), egraph.Var("x"), egraph.Var("w"), egraph.Var("offsets"), egraph.Var("rows"), egraph.Var("post_w")),
				egraph.Var("slots"),
				egraph.Var("weights"),
				egraph.Var("eps"),
			),
		},
	}
}

// fuseModelPatterns recognizes expanded Map/ReduceSum/Slice patterns for
// per-head RMSNormGate, SwiGLU, and PostExpertNorm so callers may express them
// either with high-level Builder helpers or with primitive Map/ReduceSum/Slice
// nodes.
func fuseModelPatterns(g *egraph.EGraph) (int, error) {
	added := 0
	for _, class := range g.Classes() {
		for _, n := range class.Nodes {
			if n.Op != OpMap || len(n.Children) != 3 {
				continue
			}
			fnOp := g.Class(n.Children[0]).Nodes[0].Op
			switch {
			case fnOp == "fn:(mul (silu $0) $1)":
				// Match (map fn:(mul (silu $0) $1) (slice x 0 d) (slice x d d)) -> (swiglu x)
				for _, s0 := range g.Class(n.Children[1]).Nodes {
					if s0.Op != OpSlice || len(s0.Children) != 3 {
						continue
					}
					st0, ok0 := g.Class(s0.Children[1]).Nodes[0].Data.(int)
					ln0, ok1 := g.Class(s0.Children[2]).Nodes[0].Data.(int)
					if !ok0 || !ok1 || st0 != 0 || ln0 < 1 {
						continue
					}
					for _, s1 := range g.Class(n.Children[2]).Nodes {
						if s1.Op != OpSlice || len(s1.Children) != 3 || g.Find(s1.Children[0]) != g.Find(s0.Children[0]) {
							continue
						}
						st1, ok2 := g.Class(s1.Children[1]).Nodes[0].Data.(int)
						ln1, ok3 := g.Class(s1.Children[2]).Nodes[0].Data.(int)
						if ok2 && ok3 && st1 == ln0 && ln1 == ln0 {
							id := g.Add(egraph.NewNode(OpSwiGLU, s0.Children[0]))
							if _, merged := g.Union(class.ID, id); merged {
								added++
							}
						}
					}
				}
			case fnOp == "fn:(mul $0 (sigmoid $1))":
				// Match (map fn:(mul $0 (sigmoid $1)) (map fn:(mul $0 (rsqrt (add (div $1 64) EPS))) x (reduce_sum (map fn:(mul $0 $0) x))) gate)
				// when x's class contains OpPagedAttn -> (rms_norm_gate x gate EPS).
				for _, normNode := range g.Class(n.Children[1]).Nodes {
					if normNode.Op != OpMap || len(normNode.Children) != 3 {
						continue
					}
					eps, ok := parseRMSNormLambda(g.Class(normNode.Children[0]).Nodes[0].Op, HeadDim)
					if !ok {
						continue
					}
					xID := g.Find(normNode.Children[1])
					if !hasReduceSumSq(g, normNode.Children[2], xID) || !classHasOp(g, xID, OpPagedAttn) {
						continue
					}
					id := g.Add(egraph.NewNode(OpRMSNormGate, xID, n.Children[2], g.Add(egraph.NewLeaf(eps))))
					if _, merged := g.Union(class.ID, id); merged {
						added++
					}
				}
			default:
				// Match (map fn:(mul $0 (rsqrt (add (div $1 W) EPS))) g (reduce_sum (map fn:(mul $0 $0) g)))
				// where g is (map fn:(mul $0 (add 1 $1)) y post_w) -> (post_expert_norm y post_w EPS).
				eps, ok := parseAnyRMSNormLambda(g.Class(n.Children[0]).Nodes[0].Op)
				if !ok {
					continue
				}
				gID := g.Find(n.Children[1])
				if !hasReduceSumSq(g, n.Children[2], gID) {
					continue
				}
				for _, gn := range g.Class(gID).Nodes {
					if gn.Op == OpMap && len(gn.Children) == 3 && g.Class(gn.Children[0]).Nodes[0].Op == "fn:(mul $0 (add 1 $1))" {
						id := g.Add(egraph.NewNode(OpPostExpertNorm, gn.Children[1], gn.Children[2], g.Add(egraph.NewLeaf(eps))))
						if _, merged := g.Union(class.ID, id); merged {
							added++
						}
					}
				}
			}
		}
	}
	if added > 0 {
		g.Rebuild()
	}
	return added, nil
}

func parseRMSNormLambda(op string, d int) (float32, bool) {
	prefix := fmt.Sprintf("fn:(mul $0 (rsqrt (add (div $1 %d) ", d)
	if !strings.HasPrefix(op, prefix) || !strings.HasSuffix(op, ")))") {
		return 0, false
	}
	s := strings.TrimSuffix(strings.TrimPrefix(op, prefix), ")))")
	v, err := strconv.ParseFloat(s, 32)
	return float32(v), err == nil
}

func parseAnyRMSNormLambda(op string) (float32, bool) {
	const prefix = "fn:(mul $0 (rsqrt (add (div $1 "
	if !strings.HasPrefix(op, prefix) || !strings.HasSuffix(op, ")))") {
		return 0, false
	}
	rest := strings.TrimSuffix(strings.TrimPrefix(op, prefix), ")))")
	parts := strings.Split(rest, ") ")
	if len(parts) != 2 {
		return 0, false
	}
	v, err := strconv.ParseFloat(parts[1], 32)
	return float32(v), err == nil
}

func hasReduceSumSq(g *egraph.EGraph, redID, xID egraph.Id) bool {
	for _, rn := range g.Class(redID).Nodes {
		if rn.Op != OpReduceSum || len(rn.Children) != 1 {
			continue
		}
		for _, mn := range g.Class(rn.Children[0]).Nodes {
			if mn.Op == OpMap && len(mn.Children) == 2 &&
				g.Class(mn.Children[0]).Nodes[0].Op == "fn:(mul $0 $0)" &&
				g.Find(mn.Children[1]) == g.Find(xID) {
				return true
			}
		}
	}
	return false
}

func classHasOp(g *egraph.EGraph, id egraph.Id, op string) bool {
	for _, n := range g.Class(id).Nodes {
		if n.Op == op {
			return true
		}
	}
	return false
}

func isModelOp(op string) bool {
	switch op {
	case OpPagedAttn, OpPagedAttnGated, OpRMSNormGate,
		OpRowTopK, OpTopKIDs, OpTopKWeights,
		OpExpertAlign, OpAlignOffsets, OpAlignRows, OpAlignSlots,
		OpGroupedMatMul, OpGroupedGEMM, OpGroupedGEMMGLU, OpGroupedGEMMSumSq,
		OpSwiGLU, OpPostExpertNorm, OpWeightedTopKSum, OpExpertSum:
		return true
	}
	return false
}

func defaultTempDType(e *egraph.Expr) DType {
	switch e.Op {
	case OpCast:
		if d, ok := e.Children[0].Data.(DType); ok {
			return d
		}
	case OpGroupedGEMMGLU, OpSwiGLU:
		return BF16
	case OpTopKIDs, OpAlignOffsets, OpAlignRows, OpAlignSlots:
		return I32
	}
	return F32
}

// shapeOfModel infers the output tensor shape of a model-level operator.
func shapeOfModel(e *egraph.Expr) ([]int, error) {
	switch e.Op {
	case OpPagedAttn, OpPagedAttnGated:
		return shapeOf(e.Children[1])
	case OpRMSNormGate:
		return shapeOf(e.Children[0])
	case OpRowTopK:
		cfg, ok := e.Children[0].Data.(RowTopK)
		if !ok {
			return nil, fmt.Errorf("row_topk: missing RowTopK config")
		}
		xs, err := shapeOf(e.Children[1])
		if err != nil {
			return nil, err
		}
		return []int{xs[0], cfg.K}, nil
	case OpTopKIDs, OpTopKWeights:
		return shapeOf(e.Children[0])
	case OpExpertAlign:
		cfg, ok := e.Children[0].Data.(ExpertAlign)
		if !ok {
			return nil, fmt.Errorf("expert_align: missing ExpertAlign config")
		}
		idsShape, err := shapeOf(e.Children[1])
		if err != nil {
			return nil, err
		}
		return []int{size(idsShape), cfg.Local}, nil
	case OpAlignOffsets:
		al := e.Children[0]
		cfg, ok := al.Children[0].Data.(ExpertAlign)
		if !ok {
			return nil, fmt.Errorf("align_offsets: missing ExpertAlign config")
		}
		return []int{cfg.Local + 1}, nil
	case OpAlignRows, OpAlignSlots:
		al := e.Children[0]
		idsShape, err := shapeOf(al.Children[1])
		if err != nil {
			return nil, err
		}
		return []int{size(idsShape)}, nil
	case OpGroupedMatMul, OpGroupedGEMM, OpGroupedGEMMGLU, OpGroupedGEMMSumSq:
		xi := 0
		if e.Op != OpGroupedMatMul {
			xi = 1
		}
		m, err := groupedRowsCount(e.Children[xi], e.Children[xi+3])
		if err != nil {
			return nil, err
		}
		_, n, _, err := groupedWeightDims(e.Children[xi+1], e.Children[xi+2])
		if err != nil {
			return nil, err
		}
		if e.Op == OpGroupedGEMMGLU {
			return []int{m, n / 2}, nil
		}
		return []int{m, n}, nil
	case OpSwiGLU:
		s, err := shapeOf(e.Children[0])
		if err != nil {
			return nil, err
		}
		last := len(s) - 1
		if s[last]%2 != 0 {
			return nil, fmt.Errorf("swiglu: odd width %d", s[last])
		}
		out := slices.Clone(s)
		out[last] /= 2
		return out, nil
	case OpPostExpertNorm:
		return shapeOf(e.Children[0])
	case OpWeightedTopKSum, OpExpertSum:
		yShape, err := shapeOf(e.Children[0])
		if err != nil {
			return nil, err
		}
		wShape, err := shapeOf(e.Children[2])
		if err != nil {
			return nil, err
		}
		tokens := wShape[0]
		if len(wShape) == 1 {
			k := size(wShape) / max(yShape[0], 1)
			if k > 0 {
				tokens = size(wShape) / k
			}
		}
		return []int{tokens, yShape[len(yShape)-1]}, nil
	}
	return nil, fmt.Errorf("no shape for operator %q", e.Op)
}

func groupedRowsCount(xExpr, rowsExpr *egraph.Expr) (int, error) {
	if rowsExpr != nil && rowsExpr.Op != OpNone {
		rs, err := shapeOf(rowsExpr)
		if err != nil {
			return 0, err
		}
		return size(rs), nil
	}
	xs, err := shapeOf(xExpr)
	if err != nil {
		return 0, err
	}
	return xs[0], nil
}

func groupedWeightDims(wExpr, offsetsExpr *egraph.Expr) (local, n, k int, err error) {
	os, err := shapeOf(offsetsExpr)
	if err != nil {
		return 0, 0, 0, err
	}
	local = size(os) - 1
	if local < 1 {
		return 0, 0, 0, fmt.Errorf("grouped weight: bad offsets shape %v", os)
	}
	switch wExpr.Op {
	case OpNVFP4:
		ps, err := shapeOf(wExpr.Children[0])
		if err != nil {
			return 0, 0, 0, err
		}
		if len(ps) == 3 {
			return ps[0], ps[1], 2 * ps[2], nil
		}
		if len(ps) == 2 && ps[0]%local == 0 {
			return local, ps[0] / local, 2 * ps[1], nil
		}
		return 0, 0, 0, fmt.Errorf("grouped nvfp4 weight: bad packed shape %v for %d experts", ps, local)
	default:
		ws, err := shapeOf(wExpr)
		if err != nil {
			return 0, 0, 0, err
		}
		if len(ws) == 3 {
			return ws[0], ws[1], ws[2], nil
		}
		if len(ws) == 2 && ws[0]%local == 0 {
			return local, ws[0] / local, ws[1], nil
		}
		return 0, 0, 0, fmt.Errorf("grouped weight: bad shape %v for %d experts", ws, local)
	}
}

// evalModel evaluates a model-level operator on the CPU.
func evalModel(e *egraph.Expr, env map[string]Value, memo map[*egraph.Expr]Value) (Value, error) {
	kid := func(i int) (Value, error) {
		if i >= len(e.Children) {
			return Value{}, fmt.Errorf("eval %s: missing operand %d", e.Op, i)
		}
		return eval(e.Children[i], env, memo)
	}
	if isSSMOp(e.Op) {
		return evalSSM(e, env, memo)
	}
	switch e.Op {
	case OpPagedAttn, OpPagedAttnGated:
		cfg, ok := e.Children[0].Data.(PagedAttention)
		if !ok {
			return Value{}, fmt.Errorf("eval %s: missing PagedAttention config", e.Op)
		}
		q, err := kid(1)
		if err != nil {
			return Value{}, err
		}
		kv, err := kid(2)
		if err != nil {
			return Value{}, err
		}
		vv, err := kid(3)
		if err != nil {
			return Value{}, err
		}
		pt, err := kid(4)
		if err != nil {
			return Value{}, err
		}
		sl, err := kid(5)
		if err != nil {
			return Value{}, err
		}
		ks, err := kid(6)
		if err != nil {
			return Value{}, err
		}
		vs, err := kid(7)
		if err != nil {
			return Value{}, err
		}
		cfg.Name = "eval_attn"
		if cfg.Q == 0 {
			cfg.Q = BF16
		}
		cfg.Out = F32
		kb := make([]uint8, len(kv.Data))
		for i, f := range kv.Data {
			kb[i] = uint8(f)
		}
		vb := make([]uint8, len(vv.Data))
		for i, f := range vv.Data {
			vb[i] = uint8(f)
		}
		pti := make([]int32, len(pt.Data))
		for i, f := range pt.Data {
			pti[i] = int32(f)
		}
		sli := make([]int32, len(sl.Data))
		for i, f := range sl.Data {
			sli[i] = int32(f)
		}
		cache := PagedCache{
			K:         kb,
			V:         vb,
			PageTable: pti,
			MaxPages:  pt.Shape[len(pt.Shape)-1],
			SeqLens:   sli,
			KScale:    float32(ks.Data[0]),
			VScale:    float32(vs.Data[0]),
		}
		if e.Op == OpPagedAttnGated {
			gv, err := kid(8)
			if err != nil {
				return Value{}, err
			}
			eps, _ := e.Children[9].Data.(float32)
			cfg.Gate = true
			cfg.GateStride = cfg.Hq * HeadDim
			cfg.GateOffset = 0
			cfg.GateEps = eps
			cache.Gate = make([]float32, len(gv.Data))
			for i, f := range gv.Data {
				cache.Gate[i] = BF16.Round(float32(f))
			}
		}
		qf := make([]float32, len(q.Data))
		for i, f := range q.Data {
			qf[i] = float32(f)
		}
		out, err := cfg.Reference(qf, cache)
		if err != nil {
			return Value{}, err
		}
		return Value{Shape: slices.Clone(q.Shape), Data: out}, nil

	case OpRMSNormGate:
		x, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		g, err := kid(1)
		if err != nil {
			return Value{}, err
		}
		eps, _ := e.Children[2].Data.(float32)
		if eps == 0 {
			eps = 1e-5
		}
		d := x.Shape[len(x.Shape)-1]
		out := Value{Shape: slices.Clone(x.Shape), Data: make([]float64, len(x.Data))}
		for r := 0; r < len(x.Data)/d; r++ {
			row := x.Data[r*d : (r+1)*d]
			grow := g.Data[r*d : (r+1)*d]
			var ss float64
			for _, v := range row {
				ss += v * v
			}
			inv := 1 / math.Sqrt(ss/float64(d)+float64(eps))
			for c, v := range row {
				gf := float64(BF16.Round(float32(grow[c])))
				out.Data[r*d+c] = v * inv / (1 + math.Exp(-gf))
			}
		}
		return out, nil

	case OpTopKIDs, OpTopKWeights:
		ids, weights, shape, err := evalRowTopK(e.Children[0], env, memo)
		if err != nil {
			return Value{}, err
		}
		if e.Op == OpTopKIDs {
			return Value{Shape: shape, Data: ids}, nil
		}
		return Value{Shape: shape, Data: weights}, nil

	case OpAlignOffsets, OpAlignRows, OpAlignSlots:
		offsets, rows, slots, err := evalExpertAlign(e.Children[0], env, memo)
		if err != nil {
			return Value{}, err
		}
		switch e.Op {
		case OpAlignOffsets:
			return offsets, nil
		case OpAlignRows:
			return rows, nil
		default:
			return slots, nil
		}

	case OpGroupedMatMul, OpGroupedGEMM, OpGroupedGEMMGLU, OpGroupedGEMMSumSq:
		val, _, err := evalGroupedGEMM(e, env, memo)
		return val, err

	case OpSwiGLU:
		x, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		d := x.Shape[len(x.Shape)-1]
		if d%2 != 0 {
			return Value{}, fmt.Errorf("eval swiglu: odd width %d", d)
		}
		half := d / 2
		rows := len(x.Data) / d
		outShape := slices.Clone(x.Shape)
		outShape[len(outShape)-1] = half
		out := Value{Shape: outShape, Data: make([]float64, rows*half)}
		silu := MustLambda("(mul (silu $0) $1)")
		for r := range rows {
			for c := range half {
				gv := float32(x.Data[r*d+c])
				uv := float32(x.Data[r*d+half+c])
				out.Data[r*half+c] = float64(BF16.Round(silu.Eval(gv, uv)))
			}
		}
		return out, nil

	case OpPostExpertNorm:
		y, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		pw, err := kid(1)
		if err != nil {
			return Value{}, err
		}
		eps, _ := e.Children[2].Data.(float32)
		if eps == 0 {
			eps = 1e-8
		}
		w := y.Shape[len(y.Shape)-1]
		rows := len(y.Data) / w
		out := Value{Shape: slices.Clone(y.Shape), Data: make([]float64, len(y.Data))}
		for r := range rows {
			var sum float32
			for g := 0; g < w/32; g++ {
				var gsum float32
				for j := range 32 {
					c := g*32 + j
					v := float32(y.Data[r*w+c]) * (1 + float32(pw.Data[c]))
					gsum += v * v
				}
				sum += gsum
			}
			inv := float32(1 / math.Sqrt(float64(sum/float32(w)+eps)))
			for c := range w {
				v := BF16.Round(float32(y.Data[r*w+c]) * (1 + float32(pw.Data[c])))
				out.Data[r*w+c] = float64(v * inv)
			}
		}
		return out, nil

	case OpWeightedTopKSum:
		y, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		slots, err := kid(1)
		if err != nil {
			return Value{}, err
		}
		weights, err := kid(2)
		if err != nil {
			return Value{}, err
		}
		w := y.Shape[len(y.Shape)-1]
		tokens := weights.Shape[0]
		k := len(weights.Data) / tokens
		es := ExpertSum{Name: "eval_esum", Width: w, K: k, Y: F32, Out: F32, PreNormed: true}
		yf := make([]float32, len(y.Data))
		for i, f := range y.Data {
			yf[i] = float32(f)
		}
		si := make([]int32, len(slots.Data))
		for i, f := range slots.Data {
			si[i] = int32(f)
		}
		wf := make([]float32, len(weights.Data))
		for i, f := range weights.Data {
			wf[i] = float32(f)
		}
		res, err := es.Reference(tokens, yf, nil, si, wf)
		if err != nil {
			return Value{}, err
		}
		out := make([]float64, len(res))
		for i, f := range res {
			out[i] = float64(f)
		}
		return Value{Shape: []int{tokens, w}, Data: out}, nil

	case OpExpertSum:
		yv, ssv, err := evalGroupedGEMM(e.Children[0], env, memo)
		if err != nil {
			return Value{}, err
		}
		slots, err := kid(1)
		if err != nil {
			return Value{}, err
		}
		weights, err := kid(2)
		if err != nil {
			return Value{}, err
		}
		eps, _ := e.Children[3].Data.(float32)
		if eps == 0 {
			eps = 1e-8
		}
		w := yv.Shape[len(yv.Shape)-1]
		tokens := weights.Shape[0]
		k := len(weights.Data) / tokens
		es := ExpertSum{Name: "eval_esum", Width: w, K: k, Y: BF16, Out: F32, Eps: eps}
		yf := make([]float32, len(yv.Data))
		for i, f := range yv.Data {
			yf[i] = float32(f)
		}
		ssf := make([]float32, len(ssv.Data))
		for i, f := range ssv.Data {
			ssf[i] = float32(f)
		}
		si := make([]int32, len(slots.Data))
		for i, f := range slots.Data {
			si[i] = int32(f)
		}
		wf := make([]float32, len(weights.Data))
		for i, f := range weights.Data {
			wf[i] = float32(f)
		}
		res, err := es.Reference(tokens, yf, ssf, si, wf)
		if err != nil {
			return Value{}, err
		}
		out := make([]float64, len(res))
		for i, f := range res {
			out[i] = float64(f)
		}
		return Value{Shape: []int{tokens, w}, Data: out}, nil
	}
	return Value{}, fmt.Errorf("eval: unsupported operator %q", e.Op)
}

func evalRowTopK(tk *egraph.Expr, env map[string]Value, memo map[*egraph.Expr]Value) (ids, weights []float64, shape []int, err error) {
	cfg, ok := tk.Children[0].Data.(RowTopK)
	if !ok {
		return nil, nil, nil, fmt.Errorf("eval row_topk: missing RowTopK config")
	}
	cfg.Name = "eval_topk"
	xv, err := eval(tk.Children[1], env, memo)
	if err != nil {
		return nil, nil, nil, err
	}
	rows := xv.Shape[0]
	xf := make([]float32, len(xv.Data))
	for i, f := range xv.Data {
		xf[i] = float32(f)
	}
	var bf []float32
	if tk.Children[2].Op != OpNone {
		bv, err := eval(tk.Children[2], env, memo)
		if err != nil {
			return nil, nil, nil, err
		}
		bf = make([]float32, len(bv.Data))
		for i, f := range bv.Data {
			bf[i] = float32(f)
		}
	}
	idOut, wOut, err := cfg.Reference(rows, xf, bf)
	if err != nil {
		return nil, nil, nil, err
	}
	ids = make([]float64, len(idOut))
	for i, v := range idOut {
		ids[i] = float64(v)
	}
	weights = make([]float64, len(wOut))
	for i, v := range wOut {
		weights[i] = float64(v)
	}
	return ids, weights, []int{rows, cfg.K}, nil
}

func evalExpertAlign(al *egraph.Expr, env map[string]Value, memo map[*egraph.Expr]Value) (offsets, rows, slots Value, err error) {
	cfg, ok := al.Children[0].Data.(ExpertAlign)
	if !ok {
		return Value{}, Value{}, Value{}, fmt.Errorf("eval expert_align: missing ExpertAlign config")
	}
	cfg.Name = "eval_align"
	idv, err := eval(al.Children[1], env, memo)
	if err != nil {
		return Value{}, Value{}, Value{}, err
	}
	lsv, err := eval(al.Children[2], env, memo)
	if err != nil {
		return Value{}, Value{}, Value{}, err
	}
	ids := make([]int32, len(idv.Data))
	for i, f := range idv.Data {
		ids[i] = int32(f)
	}
	offI, rowI, slotI, err := cfg.Reference(ids, int32(lsv.Data[0]))
	if err != nil {
		return Value{}, Value{}, Value{}, err
	}
	offF := make([]float64, len(offI))
	for i, v := range offI {
		offF[i] = float64(v)
	}
	rowF := make([]float64, len(ids))
	for i, v := range rowI {
		rowF[i] = float64(v)
	}
	slotF := make([]float64, len(slotI))
	for i, v := range slotI {
		slotF[i] = float64(v)
	}
	return Value{Shape: []int{cfg.Local + 1}, Data: offF},
		Value{Shape: []int{len(ids)}, Data: rowF},
		Value{Shape: []int{len(ids)}, Data: slotF}, nil
}

func evalGroupedGEMM(e *egraph.Expr, env map[string]Value, memo map[*egraph.Expr]Value) (out, ss Value, err error) {
	xi := 0
	epi := Identity
	if e.Op != OpGroupedMatMul {
		xi = 1
		var ok bool
		epi, ok = lambdaOf(egraph.ENode{Op: e.Children[0].Op})
		if !ok {
			return Value{}, Value{}, fmt.Errorf("eval %s: bad lambda %q", e.Op, e.Children[0].Op)
		}
	}
	xv, err := eval(e.Children[xi], env, memo)
	if err != nil {
		return Value{}, Value{}, err
	}
	wExpr := e.Children[xi+1]
	offv, err := eval(e.Children[xi+2], env, memo)
	if err != nil {
		return Value{}, Value{}, err
	}
	local, n, k, err := groupedWeightDims(wExpr, e.Children[xi+2])
	if err != nil {
		return Value{}, Value{}, err
	}
	var rows []int32
	m := xv.Shape[0]
	if rExpr := e.Children[xi+3]; rExpr.Op != OpNone {
		rv, err := eval(rExpr, env, memo)
		if err != nil {
			return Value{}, Value{}, err
		}
		m = len(rv.Data)
		rows = make([]int32, m)
		for i, f := range rv.Data {
			rows[i] = int32(f)
		}
	}
	offsets := make([]int32, len(offv.Data))
	for i, f := range offv.Data {
		offsets[i] = int32(f)
	}
	weights, err := evalGroupedWeights(wExpr, local, n, k, env, memo)
	if err != nil {
		return Value{}, Value{}, err
	}
	raw := make([]float32, m*n)
	for ex := range local {
		wMat := weights[ex]
		for r := int(offsets[ex]); r < int(offsets[ex+1]); r++ {
			srcRow := r
			if rows != nil {
				srcRow = int(rows[r])
			}
			for col := range n {
				var acc float64
				for c := range k {
					acc += float64(BF16.Round(float32(xv.Data[srcRow*k+c]))) * wMat[col*k+c]
				}
				raw[r*n+col] = float32(acc)
			}
		}
	}
	switch e.Op {
	case OpGroupedGEMMGLU:
		half := n / 2
		data := make([]float64, m*half)
		for r := range int(offsets[local]) {
			for c := range half {
				v := epi.Eval(raw[r*n+c], raw[r*n+half+c])
				data[r*half+c] = float64(BF16.Round(v))
			}
		}
		return Value{Shape: []int{m, half}, Data: data}, Value{}, nil
	case OpGroupedGEMMSumSq:
		pw, err := eval(e.Children[xi+4], env, memo)
		if err != nil {
			return Value{}, Value{}, err
		}
		data := make([]float64, m*n)
		ssData := make([]float64, m*(n/32))
		for r := range int(offsets[local]) {
			for g := 0; g < n/32; g++ {
				var sq float32
				for j := range 32 {
					c := g*32 + j
					y := epi.Eval(raw[r*n+c], float32(BF16.Round(float32(pw.Data[c]))))
					sq += y * y
					data[r*n+c] = float64(BF16.Round(y))
				}
				ssData[r*(n/32)+g] = float64(sq)
			}
		}
		return Value{Shape: []int{m, n}, Data: data}, Value{Shape: []int{m, n / 32}, Data: ssData}, nil
	default:
		data := make([]float64, m*n)
		for r := range int(offsets[local]) {
			for c := range n {
				data[r*n+c] = float64(epi.Eval(raw[r*n+c]))
			}
		}
		return Value{Shape: []int{m, n}, Data: data}, Value{}, nil
	}
}

func evalGroupedWeights(wExpr *egraph.Expr, local, n, k int, env map[string]Value, memo map[*egraph.Expr]Value) ([][]float64, error) {
	out := make([][]float64, local)
	switch wExpr.Op {
	case OpNVFP4:
		packed, err := eval(wExpr.Children[0], env, memo)
		if err != nil {
			return nil, err
		}
		sf, err := eval(wExpr.Children[1], env, memo)
		if err != nil {
			return nil, err
		}
		gs, err := eval(wExpr.Children[2], env, memo)
		if err != nil {
			return nil, err
		}
		row := make([]float32, k)
		pb, sb := make([]uint8, k/2), make([]uint8, k/16)
		for e := range local {
			w := make([]float64, n*k)
			gScale := float32(gs.Data[min(e, len(gs.Data)-1)])
			for r := range n {
				baseRow := e*n + r
				for i := range pb {
					pb[i] = uint8(packed.Data[baseRow*(k/2)+i])
				}
				for i := range sb {
					sb[i] = uint8(sf.Data[baseRow*(k/16)+i])
				}
				if err := DequantNVFP4(row, pb, sb, gScale); err != nil {
					return nil, err
				}
				for i, f := range row {
					w[r*k+i] = float64(f)
				}
			}
			out[e] = w
		}
		return out, nil
	default:
		wv, err := eval(wExpr, env, memo)
		if err != nil {
			return nil, err
		}
		for e := range local {
			w := make([]float64, n*k)
			for i := range w {
				w[i] = float64(BF16.Round(float32(wv.Data[e*n*k+i])))
			}
			out[e] = w
		}
		return out, nil
	}
}

func (c *compiler) freshTemp(d DType, shape []int) Tensor {
	c.temps++
	t := Tensor{
		Name:  fmt.Sprintf("%s_t%d", c.plan.Name, c.temps),
		DType: d,
		Shape: slices.Clone(shape),
	}
	c.plan.Temps = append(c.plan.Temps, t)
	return t
}

func (c *compiler) modelProjectionBuffer(e *egraph.Expr) (Tensor, bool, error) {
	switch e.Op {
	case OpTopKIDs, OpTopKWeights:
		ts, err := c.ensureRowTopK(e.Children[0])
		if err != nil {
			return Tensor{}, true, err
		}
		if e.Op == OpTopKIDs {
			return ts[0], true, nil
		}
		return ts[1], true, nil
	case OpAlignOffsets, OpAlignRows, OpAlignSlots:
		ts, err := c.ensureExpertAlign(e.Children[0])
		if err != nil {
			return Tensor{}, true, err
		}
		switch e.Op {
		case OpAlignOffsets:
			return ts[0], true, nil
		case OpAlignRows:
			return ts[1], true, nil
		default:
			return ts[2], true, nil
		}
	}
	return Tensor{}, false, nil
}

func (c *compiler) ensureRowTopK(tk *egraph.Expr) ([]Tensor, error) {
	if ts, ok := c.multi[tk.Class]; ok && tk.Class != 0 {
		return ts, nil
	}
	cfg, ok := tk.Children[0].Data.(RowTopK)
	if !ok {
		return nil, fmt.Errorf("row_topk: missing config")
	}
	x, err := c.buffer(tk.Children[1])
	if err != nil {
		return nil, err
	}
	rows := x.Shape[0]
	cfg.Name = fmt.Sprintf("%s_topk%d", c.plan.Name, len(c.plan.Kernels)+1)
	cfg.X = x.DType
	args := []string{x.Name}
	if tk.Children[2].Op != OpNone {
		bias, err := c.buffer(tk.Children[2])
		if err != nil {
			return nil, err
		}
		cfg.Bias = bias.DType
		args = append(args, bias.Name)
	} else {
		cfg.Bias = 0
	}
	ids := c.freshTemp(I32, []int{rows, cfg.K})
	weights := c.freshTemp(F32, []int{rows, cfg.K})
	args = append(args, ids.Name, weights.Name)
	if _, err := cfg.norm(); err != nil {
		return nil, err
	}
	c.plan.Kernels = append(c.plan.Kernels, PlanKernel{
		TopK: &cfg,
		Rows: rows,
		Args: args,
	})
	ts := []Tensor{ids, weights}
	if tk.Class != 0 {
		c.multi[tk.Class] = ts
	}
	return ts, nil
}

func (c *compiler) ensureExpertAlign(al *egraph.Expr) ([]Tensor, error) {
	if ts, ok := c.multi[al.Class]; ok && al.Class != 0 {
		return ts, nil
	}
	cfg, ok := al.Children[0].Data.(ExpertAlign)
	if !ok {
		return nil, fmt.Errorf("expert_align: missing config")
	}
	ids, err := c.buffer(al.Children[1])
	if err != nil {
		return nil, err
	}
	ls, err := c.buffer(al.Children[2])
	if err != nil {
		return nil, err
	}
	routes := size(ids.Shape)
	tokens := routes / cfg.K
	cfg.Name = fmt.Sprintf("%s_align%d", c.plan.Name, len(c.plan.Kernels)+1)
	if _, err := cfg.norm(); err != nil {
		return nil, err
	}
	offsets := c.freshTemp(I32, []int{cfg.Local + 1})
	rows := c.freshTemp(I32, []int{routes})
	slots := c.freshTemp(I32, []int{routes})
	c.plan.Kernels = append(c.plan.Kernels, PlanKernel{
		Align: &cfg,
		Rows:  tokens,
		Args:  []string{ids.Name, ls.Name, offsets.Name, rows.Name, slots.Name},
	})
	ts := []Tensor{offsets, rows, slots}
	if al.Class != 0 {
		c.multi[al.Class] = ts
	}
	return ts, nil
}

func (c *compiler) emitModel(e *egraph.Expr, out Tensor) error {
	switch e.Op {
	case OpPagedAttn, OpPagedAttnGated:
		return c.pagedAttention(e, out)
	case OpRMSNormGate:
		return c.rmsNormGate(e, out)
	case OpGroupedMatMul, OpGroupedGEMM, OpGroupedGEMMGLU:
		return c.groupedGEMM(e, out, nil)
	case OpSwiGLU:
		return c.swiGLU(e, out)
	case OpPostExpertNorm:
		return c.postExpertNorm(e, out)
	case OpWeightedTopKSum:
		return c.weightedTopKSum(e, out)
	case OpExpertSum:
		return c.expertSum(e, out)
	}
	return fmt.Errorf("compile: unsupported model operator %q", e.Op)
}

func (c *compiler) pagedAttention(e *egraph.Expr, out Tensor) error {
	cfg, ok := e.Children[0].Data.(PagedAttention)
	if !ok {
		return fmt.Errorf("paged_attn: missing config")
	}
	q, err := c.buffer(e.Children[1])
	if err != nil {
		return err
	}
	kCache, err := c.buffer(e.Children[2])
	if err != nil {
		return err
	}
	vCache, err := c.buffer(e.Children[3])
	if err != nil {
		return err
	}
	pageTable, err := c.buffer(e.Children[4])
	if err != nil {
		return err
	}
	seqLens, err := c.buffer(e.Children[5])
	if err != nil {
		return err
	}
	kScale, err := c.buffer(e.Children[6])
	if err != nil {
		return err
	}
	vScale, err := c.buffer(e.Children[7])
	if err != nil {
		return err
	}
	cfg.Name = c.plan.Name + "_" + out.Name
	cfg.Q = q.DType
	cfg.Out = out.DType
	args := []string{q.Name, kCache.Name, vCache.Name, pageTable.Name, seqLens.Name, kScale.Name, vScale.Name}
	if e.Op == OpPagedAttnGated {
		gateBuf, stride, offset, err := c.gateView(e.Children[8], cfg.Hq)
		if err != nil {
			return err
		}
		eps, _ := e.Children[9].Data.(float32)
		cfg.Gate = true
		cfg.GateStride = stride
		cfg.GateOffset = offset
		cfg.GateEps = eps
		args = append(args, gateBuf.Name)
	}
	normed, _, err := cfg.norm()
	if err != nil {
		return err
	}
	cfg = normed
	batch := seqLens.Shape[0]
	maxPages := pageTable.Shape[len(pageTable.Shape)-1]
	maxKeys := maxPages * cfg.Page
	if cfg.Window > 0 && cfg.Window+cfg.QLen-1 < maxKeys {
		maxKeys = cfg.Window + cfg.QLen - 1
	}
	wsBytes := max(cfg.WorkspaceBytes(batch, maxKeys), 64)
	ws := c.freshTemp(U8, []int{wsBytes})
	args = append(args, out.Name, ws.Name)
	c.plan.Kernels = append(c.plan.Kernels, PlanKernel{
		Attention: &cfg,
		Batch:     batch,
		MaxKeys:   maxKeys,
		MaxPages:  maxPages,
		Args:      args,
	})
	return nil
}

func (c *compiler) gateView(e *egraph.Expr, hq int) (buf Tensor, stride, offset int, err error) {
	rb := &rowBuilder{c: c}
	v, err := rb.view(e)
	if err != nil {
		return Tensor{}, 0, 0, err
	}
	if v.buf.DType != BF16 {
		return Tensor{}, 0, 0, fmt.Errorf("paged_attn_gated: gate buffer %s must be BF16, have %v", v.buf.Name, v.buf.DType)
	}
	stride = hq * HeadDim
	if len(v.buf.Shape) >= 2 {
		stride = size(v.buf.Shape[1:])
	}
	return v.buf, stride, v.offset, nil
}

func (c *compiler) rmsNormGate(e *egraph.Expr, out Tensor) error {
	x := e.Children[0]
	gate := e.Children[1]
	eps, _ := e.Children[2].Data.(float32)
	if eps == 0 {
		eps = 1e-5
	}
	d := out.Shape[len(out.Shape)-1]
	sq := &egraph.Expr{Op: OpMap, Children: []*egraph.Expr{{Op: "fn:(mul $0 $0)"}, x}}
	ss := &egraph.Expr{Op: OpReduceSum, Children: []*egraph.Expr{sq}}
	normFn := fmt.Sprintf("fn:(mul $0 (rsqrt (add (div $1 %d) %s)))", d, strconv.FormatFloat(float64(eps), 'g', -1, 32))
	norm := &egraph.Expr{Op: OpMap, Children: []*egraph.Expr{{Op: normFn}, x, ss}}
	gated := &egraph.Expr{Op: OpMap, Children: []*egraph.Expr{{Op: "fn:(mul $0 (sigmoid $1))"}, norm, gate}}
	return c.rows(gated, out)
}

// swiGLU lowers e to a row kernel over its gemm's output or, when only e
// reads a plain skinny gemm, to that gemm with a paired SwiGLU epilogue.
func (c *compiler) swiGLU(e *egraph.Expr, out Tensor) error {
	x := e.Children[0]
	if g := plainGEMM(x); g != nil && c.uses[x.Class] <= 1 {
		if xs, err := shapeOf(g.Children[1]); err == nil && c.skinny(xs) {
			glu := &egraph.Expr{Op: OpGEMM, Class: x.Class, Children: []*egraph.Expr{{Op: "fn:(mul (silu $0) $1)"}, g.Children[1], g.Children[2]}}
			return c.gemm(glu, out, true)
		}
	}
	d := out.Shape[len(out.Shape)-1]
	s0 := &egraph.Expr{Op: OpSlice, Children: []*egraph.Expr{x, {Data: 0}, {Data: d}}}
	s1 := &egraph.Expr{Op: OpSlice, Children: []*egraph.Expr{x, {Data: d}, {Data: d}}}
	mapped := &egraph.Expr{Op: OpMap, Children: []*egraph.Expr{{Op: "fn:(mul (silu $0) $1)"}, s0, s1}}
	return c.rows(mapped, out)
}

// plainGEMM returns e as a gemm term with the identity epilogue and no
// epilogue operands, or nil.
func plainGEMM(e *egraph.Expr) *egraph.Expr {
	switch {
	case e.Op == OpMatMul && len(e.Children) == 2:
		return &egraph.Expr{Op: OpGEMM, Class: e.Class, Children: []*egraph.Expr{{Op: "fn:$0"}, e.Children[0], e.Children[1]}}
	case e.Op == OpGEMM && len(e.Children) == 3 && e.Children[0].Op == "fn:$0":
		return e
	}
	return nil
}

func (c *compiler) postExpertNorm(e *egraph.Expr, out Tensor) error {
	y := e.Children[0]
	pw := e.Children[1]
	eps, _ := e.Children[2].Data.(float32)
	if eps == 0 {
		eps = 1e-8
	}
	w := out.Shape[len(out.Shape)-1]
	gain := &egraph.Expr{Op: OpMap, Children: []*egraph.Expr{{Op: "fn:(mul $0 (add 1 $1))"}, y, pw}}
	gainBF16 := &egraph.Expr{Op: OpCast, Children: []*egraph.Expr{{Data: BF16}, gain}}
	sq := &egraph.Expr{Op: OpMap, Children: []*egraph.Expr{{Op: "fn:(mul $0 $0)"}, gain}}
	ss := &egraph.Expr{Op: OpReduceSum, Children: []*egraph.Expr{sq}}
	normFn := fmt.Sprintf("fn:(mul $0 (rsqrt (add (div $1 %d) %s)))", w, strconv.FormatFloat(float64(eps), 'g', -1, 32))
	norm := &egraph.Expr{Op: OpMap, Children: []*egraph.Expr{{Op: normFn}, gainBF16, ss}}
	return c.rows(norm, out)
}

func (c *compiler) groupedGEMM(e *egraph.Expr, out Tensor, ssOut *Tensor) error {
	xi := 0
	epi := Identity
	if e.Op != OpGroupedMatMul {
		xi = 1
		var ok bool
		epi, ok = lambdaOf(egraph.ENode{Op: e.Children[0].Op})
		if !ok {
			return fmt.Errorf("%s: bad epilogue %q", e.Op, e.Children[0].Op)
		}
	}
	x, err := c.buffer(e.Children[xi])
	if err != nil {
		return err
	}
	if x.DType != BF16 {
		return fmt.Errorf("%s: x must be BF16, have %v", e.Op, x.DType)
	}
	wExpr := e.Children[xi+1]
	offsets, err := c.buffer(e.Children[xi+2])
	if err != nil {
		return err
	}
	local, n, k, err := groupedWeightDims(wExpr, e.Children[xi+2])
	if err != nil {
		return err
	}
	args := []string{x.Name}
	gather := false
	m := x.Shape[0]
	if rExpr := e.Children[xi+3]; rExpr.Op != OpNone {
		rows, err := c.buffer(rExpr)
		if err != nil {
			return err
		}
		gather = true
		m = size(rows.Shape)
		args = append(args, rows.Name)
	}
	var wf WeightFormat
	switch wExpr.Op {
	case OpNVFP4:
		wf = WeightNVFP4
		for _, ch := range wExpr.Children {
			wt, ok := ch.Data.(Tensor)
			if !ok {
				return fmt.Errorf("%s: nvfp4 operands must be inputs", e.Op)
			}
			c.input(wt)
			args = append(args, wt.Name)
		}
	default:
		wt, ok := wExpr.Data.(Tensor)
		if !ok || wt.DType != BF16 {
			return fmt.Errorf("%s: weight must be BF16 input or nvfp4", e.Op)
		}
		c.input(wt)
		wf = WeightBF16
		args = append(args, wt.Name)
	}
	args = append(args, offsets.Name)
	g := TileGEMM{
		Name:     c.plan.Name + "_" + out.Name,
		N:        n,
		K:        k,
		Weight:   wf,
		Out:      out.DType,
		Epilogue: epi,
		Grouped:  true,
		Gather:   gather,
		GLU:      e.Op == OpGroupedGEMMGLU,
		SumSq:    ssOut != nil,
		BM:       64,
		BN:       64,
	}
	if !g.GLU && n%128 == 0 || g.GLU && (n/2)%64 == 0 && n%128 == 0 {
		g.BN = 128
	}
	if e.Op == OpGroupedGEMMSumSq {
		pw, err := c.buffer(e.Children[xi+4])
		if err != nil {
			return err
		}
		g.Extras = []Operand{{Name: "post_w", DType: pw.DType, Broadcast: PerCol}}
		args = append(args, pw.Name)
	}
	args = append(args, out.Name)
	if ssOut != nil {
		args = append(args, ssOut.Name)
	}
	if _, err := g.norm(); err != nil {
		return err
	}
	c.plan.Kernels = append(c.plan.Kernels, PlanKernel{
		Tile:    &g,
		M:       m,
		MaxRows: m,
		Experts: local,
		Args:    args,
	})
	return nil
}

func (c *compiler) weightedTopKSum(e *egraph.Expr, out Tensor) error {
	y, err := c.buffer(e.Children[0])
	if err != nil {
		return err
	}
	slots, err := c.buffer(e.Children[1])
	if err != nil {
		return err
	}
	weights, err := c.buffer(e.Children[2])
	if err != nil {
		return err
	}
	w := out.Shape[len(out.Shape)-1]
	tokens := out.Shape[0]
	k := size(weights.Shape) / tokens
	es := ExpertSum{
		Name:      c.plan.Name + "_" + out.Name,
		Width:     w,
		K:         k,
		Y:         y.DType,
		Out:       out.DType,
		PreNormed: true,
	}
	if _, err := es.norm(); err != nil {
		return err
	}
	c.plan.Kernels = append(c.plan.Kernels, PlanKernel{
		ExpertSum: &es,
		Rows:      tokens,
		Args:      []string{y.Name, slots.Name, weights.Name, out.Name},
	})
	return nil
}

func (c *compiler) expertSum(e *egraph.Expr, out Tensor) error {
	gemmExpr := e.Children[0]
	if gemmExpr.Op != OpGroupedGEMMSumSq {
		return fmt.Errorf("expert_sum: child 0 must be %s, have %s", OpGroupedGEMMSumSq, gemmExpr.Op)
	}
	var y, ss Tensor
	if ts, ok := c.multi[gemmExpr.Class]; ok && gemmExpr.Class != 0 {
		y, ss = ts[0], ts[1]
	} else {
		shape, err := shapeOf(gemmExpr)
		if err != nil {
			return err
		}
		y = c.freshTemp(BF16, shape)
		ss = c.freshTemp(F32, []int{shape[0], shape[1] / 32})
		if err := c.groupedGEMM(gemmExpr, y, &ss); err != nil {
			return err
		}
		if gemmExpr.Class != 0 {
			c.multi[gemmExpr.Class] = []Tensor{y, ss}
		}
	}
	slots, err := c.buffer(e.Children[1])
	if err != nil {
		return err
	}
	weights, err := c.buffer(e.Children[2])
	if err != nil {
		return err
	}
	eps, _ := e.Children[3].Data.(float32)
	if eps == 0 {
		eps = 1e-8
	}
	w := out.Shape[len(out.Shape)-1]
	tokens := out.Shape[0]
	k := size(weights.Shape) / tokens
	es := ExpertSum{
		Name:  c.plan.Name + "_" + out.Name,
		Width: w,
		K:     k,
		Y:     y.DType,
		Out:   out.DType,
		Eps:   eps,
	}
	if _, err := es.norm(); err != nil {
		return err
	}
	c.plan.Kernels = append(c.plan.Kernels, PlanKernel{
		ExpertSum: &es,
		Rows:      tokens,
		Args:      []string{y.Name, ss.Name, slots.Name, weights.Name, out.Name},
	})
	return nil
}
