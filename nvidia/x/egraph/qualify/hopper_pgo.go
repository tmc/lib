package qualify

import (
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// Step is an alias for [hopper.PlanKernel] used by profile-guided refinement
// and qualification gates.
type Step = hopper.PlanKernel

// MegakernelSource is an alias for [hopper.Megakernel] returned when a plan is
// promoted to a persistent megakernel.
type MegakernelSource = hopper.Megakernel

// A CandidateStepProposal is a legal schedule rewrite candidate proposed by
// [ProposeStepCandidates], distinguishing direct causal counter/compiler
// refinements from stall-sample hypotheses.
type CandidateStepProposal struct {
	Step       Step
	Causal     bool
	Hypothesis BottleneckKind
	Share      float64
	Rationale  []string
}

// ProposeStepCandidates returns legal, distinct candidate step rewrites for
// Phase 1 ABBA evaluation. If causal has classified bottlenecks, the causal
// refinement is emitted first with Causal: true. Then each hypothesis in
// [HypothesesFromStalls](stalls) is evaluated on a synthetic single-bottleneck
// trial observation without mutating causal; distinct valid steps are appended
// with Causal: false.
func ProposeStepCandidates(step Step, causal KernelObservation, stalls StallSample) []CandidateStepProposal {
	var proposals []CandidateStepProposal
	if bns := ClassifyBottlenecks(causal); len(bns) > 0 {
		cand, reasons := RefineStep(step, causal)
		if len(reasons) > 0 && stepValid(cand) && !stepSameSchedule(cand, step) {
			proposals = append(proposals, CandidateStepProposal{
				Step:      cand,
				Causal:    true,
				Rationale: reasons,
			})
		}
	}

	name := causal.Name
	if name == "" {
		name = stepKernelName(step)
	}
	for _, h := range HypothesesFromStalls(stalls) {
		trialObs := syntheticObservationForBottleneck(name, causal.DeviceSMs, h.Bottleneck)
		cand, reasons := RefineStep(step, trialObs)
		if len(reasons) == 0 || !stepValid(cand) || stepSameSchedule(cand, step) {
			continue
		}
		dup := false
		for _, prev := range proposals {
			if stepSameSchedule(cand, prev.Step) {
				dup = true
				break
			}
		}
		if dup {
			continue
		}
		proposals = append(proposals, CandidateStepProposal{
			Step:       cand,
			Causal:     false,
			Hypothesis: h.Bottleneck,
			Share:      h.Share,
			Rationale:  reasons,
		})
	}
	return proposals
}

func syntheticObservationForBottleneck(name string, deviceSMs int, b BottleneckKind) KernelObservation {
	obs := KernelObservation{
		Name:      name,
		DeviceSMs: deviceSMs,
	}
	switch b {
	case BottleneckBankConflict:
		obs.SharedBankConflict = 0.25
	case BottleneckWGMMASerial:
		obs.WGMMASerialized = true
	case BottleneckRegisterSpill:
		obs.SpillStoreBytes = 16
	case BottleneckLowOccupancy:
		obs.AchievedOccupancy = 0.15
		obs.Registers = 192
	case BottleneckMemoryBoundKV:
		obs.MemoryBoundKV = true
	case BottleneckWaveTail:
		sms := max(deviceSMs, 132)
		obs.DeviceSMs = sms
		obs.GridBlocks = sms + max(sms/5, 1)
	case BottleneckLaunchOverhead:
		obs.DurationNs = 10000
		obs.LaunchGapNs = 2000
	}
	return obs
}

func stepValid(k Step) bool {
	switch {
	case k.Tile != nil:
		_, err := k.Tile.Norm()
		return err == nil
	case k.GEMM != nil:
		_, err := k.GEMM.Norm()
		return err == nil
	case k.Attention != nil:
		_, err := k.Attention.Norm()
		return err == nil
	case k.Row != nil:
		return k.Row.Validate() == nil
	default:
		return false
	}
}

func stepSameSchedule(a, b Step) bool {
	switch {
	case a.Tile != nil && b.Tile != nil:
		ta, tb := *a.Tile, *b.Tile
		return ta.BM == tb.BM &&
			ta.BN == tb.BN &&
			ta.Stages == tb.Stages &&
			ta.WGMMA == tb.WGMMA &&
			ta.Swizzle128B == tb.Swizzle128B &&
			ta.WarpSpecialized == tb.WarpSpecialized &&
			ta.Repacked == tb.Repacked &&
			ta.DequantWords == tb.DequantWords &&
			ta.StreamK == tb.StreamK &&
			ta.GroupM == tb.GroupM
	case a.GEMM != nil && b.GEMM != nil:
		ga, gb := *a.GEMM, *b.GEMM
		return ga.RowsPerBlock == gb.RowsPerBlock &&
			ga.Warps == gb.Warps &&
			ga.SplitK == gb.SplitK &&
			ga.Unroll == gb.Unroll &&
			ga.Prefetch == gb.Prefetch &&
			ga.BlackwellFP4 == gb.BlackwellFP4
	case a.Attention != nil && b.Attention != nil:
		aa, ab := *a.Attention, *b.Attention
		return aa.Chunk == ab.Chunk &&
			aa.Depth == ab.Depth &&
			aa.Stages == ab.Stages &&
			aa.TableAhead == ab.TableAhead &&
			aa.WGMMA == ab.WGMMA &&
			aa.QKFP8 == ab.QKFP8 &&
			aa.Swizzle == ab.Swizzle &&
			aa.ClusterCombine == ab.ClusterCombine &&
			aa.KVHeadsPerBlock == ab.KVHeadsPerBlock &&
			aa.DeviceSplits == ab.DeviceSplits
	case a.Row != nil && b.Row != nil:
		return a.Row.Cluster == b.Row.Cluster
	default:
		return false
	}
}

// RefineStep deterministically rewrites step's schedule parameters to cure each
// bottleneck classified in obs, returning the refined step and human-readable
// rewrite rationales.
func RefineStep(step Step, obs KernelObservation) (Step, []string) {
	out := cloneStep(step)
	bns := ClassifyBottlenecks(obs)
	if len(bns) == 0 {
		return out, nil
	}
	switch {
	case out.Tile != nil:
		g := *out.Tile
		apply := func(k TileGEMMKnobs) hopper.TileGEMM {
			cand := g
			cand.BM = k.BM
			cand.BN = k.BN
			cand.Stages = k.Stages
			cand.WGMMA = k.WGMMA
			cand.Swizzle128B = k.Swizzle128B
			cand.WarpSpecialized = k.WarpSpecialized
			cand.Repacked = k.Repacked
			cand.DequantWords = k.DequantWords
			cand.StreamK = k.StreamK
			cand.GroupM = k.GroupM
			return cand
		}
		knobs, reasons := RefineTileGEMMKnobs(TileGEMMKnobs{
			Name:            g.Name,
			BM:              g.BM,
			BN:              g.BN,
			Stages:          g.Stages,
			WGMMA:           g.WGMMA,
			Swizzle128B:     g.Swizzle128B,
			WarpSpecialized: g.WarpSpecialized,
			Repacked:        g.Repacked,
			DequantWords:    g.DequantWords,
			StreamK:         g.StreamK,
			GroupM:          g.GroupM,
			Grouped:         g.Grouped,
			IsNVFP4:         g.Weight == hopper.WeightNVFP4,
			LinearEpilogue:  g.Epilogue.IsZero() || g.Epilogue.IsIdentity(),
		}, obs, bns, func(ck TileGEMMKnobs) bool {
			_, err := apply(ck).Norm()
			return err == nil
		})
		t := apply(knobs)
		out.Tile = &t
		return out, reasons

	case out.GEMM != nil:
		g := *out.GEMM
		normed, err := g.Norm()
		normR, normW, normS, normU := g.RowsPerBlock, g.Warps, g.SplitK, g.Unroll
		chunk := g.Weight.Chunk()
		if err == nil {
			normR, normW, normS, normU = normed.RowsPerBlock, normed.Warps, normed.SplitK, normed.Unroll
			chunk = normed.Weight.Chunk()
		}
		apply := func(k SkinnyGEMMKnobs) hopper.SkinnyGEMM {
			cand := g
			cand.RowsPerBlock = k.RowsPerBlock
			cand.Unroll = k.Unroll
			cand.Prefetch = k.Prefetch
			cand.BlackwellFP4 = k.BlackwellFP4
			return cand
		}
		knobs, reasons := RefineSkinnyGEMMKnobs(SkinnyGEMMKnobs{
			Name:             g.Name,
			N:                g.N,
			K:                g.K,
			RowsPerBlock:     g.RowsPerBlock,
			NormRowsPerBlock: normR,
			Warps:            g.Warps,
			NormWarps:        normW,
			SplitK:           g.SplitK,
			NormSplitK:       normS,
			Unroll:           g.Unroll,
			NormUnroll:       normU,
			Prefetch:         g.Prefetch,
			BlackwellFP4:     g.BlackwellFP4,
			IsNVFP4:          g.Weight == hopper.WeightNVFP4,
			WeightChunk:      chunk,
		}, obs, bns, func(ck SkinnyGEMMKnobs) bool {
			_, err := apply(ck).Norm()
			return err == nil
		})
		sg := apply(knobs)
		out.GEMM = &sg
		return out, reasons

	case out.Attention != nil:
		a := *out.Attention
		normStages := a.Stages
		if normStages == 0 {
			if normed, err := a.Norm(); err == nil {
				normStages = normed.Stages
			} else {
				normStages = 2
			}
		}
		apply := func(k PagedAttentionKnobs) hopper.PagedAttention {
			cand := a
			cand.Chunk = k.Chunk
			cand.Depth = k.Depth
			cand.Stages = k.Stages
			cand.TableAhead = k.TableAhead
			cand.WGMMA = k.WGMMA
			cand.QKFP8 = k.QKFP8
			cand.Swizzle = k.Swizzle
			cand.ClusterCombine = k.ClusterCombine
			return cand
		}
		knobs, reasons := RefinePagedAttentionKnobs(PagedAttentionKnobs{
			Name:            a.Name,
			Hq:              a.Hq,
			Hkv:             a.Hkv,
			QLen:            a.QLen,
			Window:          a.Window,
			QIsE4M3:         a.Q == hopper.E4M3,
			KVHeadsPerBlock: a.KVHeadsPerBlock,
			Chunk:           a.Chunk,
			Depth:           a.Depth,
			Stages:          a.Stages,
			NormStages:      normStages,
			TableAhead:      a.TableAhead,
			WGMMA:           a.WGMMA,
			QKFP8:           a.QKFP8,
			Swizzle:         a.Swizzle,
			ClusterCombine:  a.ClusterCombine,
		}, out.Batch, out.MaxKeys, obs, bns, func(ck PagedAttentionKnobs) bool {
			_, err := apply(ck).Norm()
			return err == nil
		})
		pa := apply(knobs)
		out.Attention = &pa
		return out, reasons

	case out.Row != nil:
		r := *out.Row
		knobs, reasons := RefineRowKernelKnobs(RowKernelKnobs{
			Name:       r.Name,
			Cluster:    r.Cluster,
			CanCluster: r.CanCluster(),
		}, bns, func(ck RowKernelKnobs) bool {
			cand := r
			cand.Cluster = ck.Cluster
			return cand.Validate() == nil
		})
		r.Cluster = knobs.Cluster
		out.Row = &r
		return out, reasons

	default:
		return out, nil
	}
}

// RefinePlan matches plan's kernel steps to receipt.Observations, applies
// [RefineStep] to each step, updates any affected workspace tensors, and if
// [BottleneckLaunchOverhead] is present on at least one consumer step of a
// megakernel-compatible plan, lowers the refined plan to a persistent
// [MegakernelSource] named plan.Name+"_mega".
func RefinePlan(plan *hopper.Plan, receipt ProfileReceipt) (*hopper.Plan, *MegakernelSource, []string, error) {
	if plan == nil {
		return nil, nil, nil, fmt.Errorf("refine plan: nil plan")
	}
	refined := clonePlan(plan)
	defaultSMs := 0
	if receipt.Device != "" {
		if cfg, err := (hopper.CostConfig{Device: receipt.Device}).Norm(); err == nil {
			defaultSMs = cfg.SMs
		}
	}

	var (
		actions      []string
		wantMega     bool
		consumerHits int
	)
	matched := make([]bool, len(receipt.Observations))
	for i := range refined.Kernels {
		obs, ok := MatchObservation(stepKernelName(refined.Kernels[i]), i, receipt.Observations, matched)
		if !ok {
			continue
		}
		if obs.DeviceSMs <= 0 && defaultSMs > 0 {
			obs.DeviceSMs = defaultSMs
		}
		bns := ClassifyBottlenecks(obs)
		if slices.Contains(bns, BottleneckLaunchOverhead) && (i >= 1 || len(refined.Kernels) > 1) {
			consumerHits++
			wantMega = true
		}

		prevTileStreamK := refined.Kernels[i].Tile != nil && refined.Kernels[i].Tile.StreamK > 0
		step, stepLog := RefineStep(refined.Kernels[i], obs)
		refined.Kernels[i] = step
		actions = append(actions, stepLog...)

		if step.Attention != nil && len(step.Args) > 0 {
			maxKeys := step.MaxKeys
			if maxKeys <= 0 {
				if step.Attention.Window > 0 {
					maxKeys = step.Attention.Window + max(step.Attention.QLen, 1) - 1
				} else {
					maxKeys = 1024
				}
			}
			wsBytes := max(step.Attention.WorkspaceBytes(max(step.Batch, 1), maxKeys), 64)
			wsName := step.Args[len(step.Args)-1]
			for j := range refined.Temps {
				if refined.Temps[j].Name == wsName {
					refined.Temps[j].Shape = []int{wsBytes}
				}
			}
		}
		if step.Tile != nil && step.Tile.StreamK > 0 && !prevTileStreamK {
			if wsBytes := step.Tile.WorkspaceBytes(); wsBytes > 0 {
				wsName := fmt.Sprintf("%s_sk_ws%d", refined.Name, i+1)
				refined.Temps = append(refined.Temps, hopper.Tensor{
					Name:  wsName,
					DType: hopper.F32,
					Shape: []int{(wsBytes + 3) / 4},
				})
				refined.Kernels[i].Args = append(slices.Clone(refined.Kernels[i].Args), wsName)
			}
		}
	}

	if wantMega && consumerHits >= 1 && len(refined.Kernels) >= 2 {
		megaName := refined.Name + "_mega"
		if refined.Name == "" {
			megaName = "plan_mega"
		}
		mega, err := refined.Megakernel(megaName)
		if err != nil {
			compat := clonePlan(refined)
			for i := range compat.Kernels {
				if compat.Kernels[i].Row != nil && compat.Kernels[i].Row.Cluster {
					compat.Kernels[i].Row.Cluster = false
				}
				if compat.Kernels[i].Tile != nil && compat.Kernels[i].Tile.WarpSpecialized {
					compat.Kernels[i].Tile.WarpSpecialized = false
				}
			}
			if m2, err2 := compat.Megakernel(megaName); err2 == nil {
				refined, mega, err = compat, m2, nil
			}
		}
		if err == nil {
			actions = append(actions, fmt.Sprintf("plan %s: promote %d stages to persistent megakernel %s to eliminate launch overhead", refined.Name, len(refined.Kernels), mega.Name))
			return refined, &mega, actions, nil
		}
	}

	return refined, nil, actions, nil
}

// RefineFromProfile is a convenience alias for [RefinePlan].
func RefineFromProfile(plan *hopper.Plan, receipt ProfileReceipt) (*hopper.Plan, *MegakernelSource, []string, error) {
	return RefinePlan(plan, receipt)
}

func stepKernelName(k hopper.PlanKernel) string {
	switch {
	case k.GEMM != nil:
		return k.GEMM.Name
	case k.Tile != nil:
		return k.Tile.Name
	case k.MMA != nil:
		return k.MMA.Name
	case k.Attention != nil:
		return k.Attention.Name
	case k.Row != nil:
		return k.Row.Name
	case k.TopK != nil:
		return k.TopK.Name
	case k.Align != nil:
		return k.Align.Name
	case k.ExpertSum != nil:
		return k.ExpertSum.Name
	default:
		return ""
	}
}

func cloneStep(k hopper.PlanKernel) hopper.PlanKernel {
	out := k
	out.Args = slices.Clone(k.Args)
	if k.GEMM != nil {
		g := *k.GEMM
		g.Extras = slices.Clone(k.GEMM.Extras)
		out.GEMM = &g
	}
	if k.Tile != nil {
		t := *k.Tile
		t.Extras = slices.Clone(k.Tile.Extras)
		out.Tile = &t
	}
	if k.Attention != nil {
		a := *k.Attention
		out.Attention = &a
	}
	if k.Row != nil {
		r := *k.Row
		r.Inputs = slices.Clone(k.Row.Inputs)
		r.Values = make([]hopper.RowValue, len(k.Row.Values))
		for i, v := range k.Row.Values {
			v.Args = slices.Clone(v.Args)
			r.Values[i] = v
		}
		r.Outputs = slices.Clone(k.Row.Outputs)
		out.Row = &r
	}
	if k.TopK != nil {
		tk := *k.TopK
		out.TopK = &tk
	}
	if k.Align != nil {
		al := *k.Align
		out.Align = &al
	}
	if k.ExpertSum != nil {
		es := *k.ExpertSum
		out.ExpertSum = &es
	}
	return out
}

func clonePlan(p *hopper.Plan) *hopper.Plan {
	if p == nil {
		return nil
	}
	out := &hopper.Plan{
		Name:    p.Name,
		Inputs:  make([]hopper.Tensor, len(p.Inputs)),
		Temps:   make([]hopper.Tensor, len(p.Temps)),
		Output:  hopper.Tensor{Name: p.Output.Name, DType: p.Output.DType, Shape: slices.Clone(p.Output.Shape)},
		Kernels: make([]hopper.PlanKernel, len(p.Kernels)),
	}
	for i, t := range p.Inputs {
		out.Inputs[i] = hopper.Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, t := range p.Temps {
		out.Temps[i] = hopper.Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, k := range p.Kernels {
		out.Kernels[i] = cloneStep(k)
	}
	return out
}
