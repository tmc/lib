package main

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/tensorparallel"
)

// tpMegakernelOptions configures a cross-layer tensor-parallel megakernel group.
type tpMegakernelOptions struct {
	TP          int
	Prefix      string
	Fusion      tensorparallel.Fusion
	Target      string
	BarrierMode hopper.SysBarrierMode
	QKVBias     bool
	Rows        int
}

// A tpMegakernelGroup holds the per-rank cross-layer tensor-parallel megakernel
// plans for a LLaMA decoder model.
type tpMegakernelGroup struct {
	cfg         *ModelConfig
	tp          int
	fusion      tensorparallel.Fusion
	barrierMode hopper.SysBarrierMode
	target      string
	qkvBias     bool
	rows        int
	ranks       []*tensorparallel.TPMegakernelPlan
}

// newTPMegakernelGroup builds a cross-layer tensor-parallel megakernel group
// for cfg across opt.TP ranks using [tensorparallel.BuildCrossLayerTPMegakernel].
func newTPMegakernelGroup(cfg *ModelConfig, opt tpMegakernelOptions) (*tpMegakernelGroup, error) {
	if cfg == nil {
		return nil, errors.New("tp megakernel group: nil model config")
	}
	if cfg.NumHiddenLayers < 2 {
		return nil, fmt.Errorf("tp megakernel group: need at least 2 layers, have %d", cfg.NumHiddenLayers)
	}
	if opt.TP < 1 {
		return nil, fmt.Errorf("tp megakernel group: tp %d < 1", opt.TP)
	}
	rows := opt.Rows
	if rows == 0 {
		rows = 1
	}
	if rows < 0 {
		return nil, fmt.Errorf("tp megakernel group: rows %d < 0", rows)
	}

	layer := tensorparallel.Layer{
		Hidden:       cfg.HiddenSize,
		Heads:        cfg.NumAttentionHeads,
		KVHeads:      cfg.NumKeyValueHeads,
		HeadDim:      cfg.HeadDim(),
		Intermediate: cfg.IntermediateSize,
		Eps:          cfg.RmsNormEps,
		QKVBias:      opt.QKVBias,
		Rows:         rows,
		QKVOut:       hopper.BF16,
		AttnDType:    hopper.F32,
	}
	tpCfg := tensorparallel.TPMegakernelConfig{
		TP:          opt.TP,
		Prefix:      opt.Prefix,
		Fusion:      opt.Fusion,
		Target:      opt.Target,
		BarrierMode: opt.BarrierMode,
	}

	var ranks []*tensorparallel.TPMegakernelPlan
	var err error
	if cfg.NumHiddenLayers == 2 {
		ranks, err = tensorparallel.BuildCrossLayerTPMegakernel(tpCfg, layer, layer)
	} else {
		layers := make([]tensorparallel.Layer, cfg.NumHiddenLayers)
		for i := range layers {
			layers[i] = layer
		}
		ranks, err = tensorparallel.BuildTPMegakernel(tpCfg, layers)
	}
	if err != nil {
		return nil, err
	}

	return &tpMegakernelGroup{
		cfg:         cfg,
		tp:          opt.TP,
		fusion:      ranks[0].Fusion,
		barrierMode: ranks[0].BarrierMode,
		target:      ranks[0].Target,
		qkvBias:     opt.QKVBias,
		rows:        rows,
		ranks:       ranks,
	}, nil
}

// Sources returns each rank's emitted CUDA C++ translation unit containing
// [hopper.ReadySysPrelude] and system-scope (.sys) barriers at the o_proj -> ffn
// and down_proj -> next_attn_in boundaries.
func (g *tpMegakernelGroup) Sources() ([]string, error) {
	if g == nil || len(g.ranks) == 0 {
		return nil, errors.New("tp megakernel group: empty group")
	}
	out := make([]string, len(g.ranks))
	for r, rp := range g.ranks {
		src, err := rp.Source()
		if err != nil {
			return nil, fmt.Errorf("rank %d source: %w", r, err)
		}
		out[r] = src
	}
	return out, nil
}

// evalCrossLayerHost executes the cross-layer boundary
// (Layer[L].OProj + Layer[L].FFN + Layer[L+1].AttnIn) across all ranks in g,
// performing the two cross-rank reductions at the .sys barrier points and
// returning the reduced h, reduced x_next, and reassembled Q, K, V projections
// for Layer[L+1].
func (g *tpMegakernelGroup) evalCrossLayerHost(wL, wNext *LayerWeights, x, fullAttn []float32) (hOut, xNextOut, qOut, kOut, vOut []float32, err error) {
	if g == nil || len(g.ranks) == 0 {
		return nil, nil, nil, nil, nil, errors.New("tp megakernel group: empty group")
	}
	if wL == nil || wNext == nil {
		return nil, nil, nil, nil, nil, errors.New("tp megakernel group: nil layer weights")
	}
	hid := g.cfg.HiddenSize
	hd := g.cfg.HeadDim()
	qDim := g.cfg.NumAttentionHeads * hd
	if len(x) != hid {
		return nil, nil, nil, nil, nil, fmt.Errorf("tp megakernel group: x len %d != hidden %d", len(x), hid)
	}
	if len(fullAttn) != qDim {
		return nil, nil, nil, nil, nil, fmt.Errorf("tp megakernel group: attn len %d != qDim %d", len(fullAttn), qDim)
	}

	eps := float64(g.cfg.RmsNormEps)
	hAcc := make([]float64, hid)

	// Stage 1: Each rank computes its row-parallel OProj shard and reduces into h
	// before the first .sys barrier (o_proj -> ffn).
	for r, plan := range g.ranks {
		rp := plan.RankShardL
		attnShard := fullAttn[rp.QHeads.Lo*hd : rp.QHeads.Hi*hd]
		woShard := tensorparallel.O(rp, wL.WO)
		part := tpMatVecF64(woShard, attnShard, hid, rp.QHeads.Len()*hd)
		if r == 0 {
			for i := range part {
				part[i] += float64(x[i])
			}
		}
		for i := range hAcc {
			hAcc[i] += part[i]
		}
	}

	// Stage 2: After the first .sys barrier, each rank normalizes h, runs its
	// column-parallel GateUp + SwiGLU and row-parallel Down shard, and reduces
	// into x_next before the second .sys barrier (down_proj -> next_attn_in).
	xNextAcc := make([]float64, hid)
	hNorm := tpRMSNormF64(hAcc, wL.PostNorm, eps)
	for r, plan := range g.ranks {
		rp := plan.RankShardL
		interLen := rp.Inter.Len()
		wguShard := tensorparallel.GateUp(rp, wL.WGate, wL.WUp)
		gu := tpMatVecF64FromF64(wguShard, hNorm, 2*interLen, hid)
		act := make([]float64, interLen)
		for i := range act {
			gate := gu[i]
			up := gu[interLen+i]
			act[i] = (gate / (1.0 + math.Exp(-gate))) * up
		}
		wdShard := tensorparallel.Down(rp, wL.WDown)
		downPart := tpMatVecF64FromF64(wdShard, act, hid, interLen)
		if r == 0 {
			for i := range downPart {
				downPart[i] += hAcc[i]
			}
		}
		for i := range xNextAcc {
			xNextAcc[i] += downPart[i]
		}
	}

	// Stage 3: After the second .sys barrier, each rank normalizes x_next and
	// computes its column-parallel Layer[L+1].AttnIn QKV shard.
	xNextNorm := tpRMSNormF64(xNextAcc, wNext.InputNorm, eps)
	for _, plan := range g.ranks {
		rp := plan.RankShardNext
		qLen := rp.QHeads.Len() * hd
		kvLen := rp.KVHeads.Len() * hd
		qkvDim := qLen + 2*kvLen
		wqkvShard := tensorparallel.QKV(rp, wNext.WQ, wNext.WK, wNext.WV, hid)
		qkv := tpMatVecF64FromF64(wqkvShard, xNextNorm, qkvDim, hid)
		if g.qkvBias && len(wNext.QBias) > 0 && len(wNext.KBias) > 0 && len(wNext.VBias) > 0 {
			bqkv := tensorparallel.QKV(rp, wNext.QBias, wNext.KBias, wNext.VBias, 1)
			for i := range qkv {
				qkv[i] += float64(bqkv[i])
			}
		}
		for i := 0; i < qLen; i++ {
			qOut = append(qOut, float32(qkv[i]))
		}
		for i := 0; i < kvLen; i++ {
			kOut = append(kOut, float32(qkv[qLen+i]))
		}
		for i := 0; i < kvLen; i++ {
			vOut = append(vOut, float32(qkv[qLen+kvLen+i]))
		}
	}

	hOut = make([]float32, hid)
	xNextOut = make([]float32, hid)
	for i := range hid {
		hOut[i] = float32(hAcc[i])
		xNextOut[i] = float32(xNextAcc[i])
	}
	return hOut, xNextOut, qOut, kOut, vOut, nil
}

// verifyTPMegakernelParity compares 1-rank unsharded cross-layer execution
// against R-rank tensor-parallel execution for cfg and layer weights wL, wNext.
func verifyTPMegakernelParity(cfg *ModelConfig, wL, wNext *LayerWeights, x, fullAttn []float32, tp int, fusion tensorparallel.Fusion, qkvBias bool) error {
	g1, err := newTPMegakernelGroup(cfg, tpMegakernelOptions{
		TP:      1,
		Fusion:  fusion,
		QKVBias: qkvBias,
	})
	if err != nil {
		return fmt.Errorf("build 1-rank reference group: %w", err)
	}
	gR, err := newTPMegakernelGroup(cfg, tpMegakernelOptions{
		TP:      tp,
		Fusion:  fusion,
		QKVBias: qkvBias,
	})
	if err != nil {
		return fmt.Errorf("build %d-rank tp group: %w", tp, err)
	}

	h1, x1, q1, k1, v1, err := g1.evalCrossLayerHost(wL, wNext, x, fullAttn)
	if err != nil {
		return fmt.Errorf("eval 1-rank: %w", err)
	}
	hR, xR, qR, kR, vR, err := gR.evalCrossLayerHost(wL, wNext, x, fullAttn)
	if err != nil {
		return fmt.Errorf("eval %d-rank: %w", tp, err)
	}

	for _, chk := range []struct {
		name      string
		got, want []float32
	}{
		{"h", hR, h1},
		{"x_next", xR, x1},
		{"q", qR, q1},
		{"k", kR, k1},
		{"v", vR, v1},
	} {
		if len(chk.got) != len(chk.want) {
			return fmt.Errorf("tp=%d %s length %d != want %d", tp, chk.name, len(chk.got), len(chk.want))
		}
		for i := range chk.want {
			diff := math.Abs(float64(chk.got[i] - chk.want[i]))
			tol := 1e-5 * (1.0 + math.Abs(float64(chk.want[i])))
			if diff > tol {
				return fmt.Errorf("tp=%d %s[%d] = %g, want %g (diff %g > tol %g)",
					tp, chk.name, i, chk.got[i], chk.want[i], diff, tol)
			}
		}
	}
	return nil
}

func tpRMSNormF64(x []float64, weight []float32, eps float64) []float64 {
	var ss float64
	for _, v := range x {
		ss += v * v
	}
	inv := 1.0 / math.Sqrt(ss/float64(len(x))+eps)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * inv * float64(weight[i])
	}
	return out
}

func tpMatVecF64(w, x []float32, rows, cols int) []float64 {
	out := make([]float64, rows)
	for r := 0; r < rows; r++ {
		var sum float64
		row := w[r*cols : (r+1)*cols]
		for c, xv := range x {
			sum += float64(row[c]) * float64(xv)
		}
		out[r] = sum
	}
	return out
}

func tpMatVecF64FromF64(w []float32, x []float64, rows, cols int) []float64 {
	out := make([]float64, rows)
	for r := 0; r < rows; r++ {
		var sum float64
		row := w[r*cols : (r+1)*cols]
		for c, xv := range x {
			sum += float64(row[c]) * xv
		}
		out[r] = sum
	}
	return out
}
