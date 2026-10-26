package hopper

import (
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// BottleneckKind classifies a measured or compiled hardware bottleneck.
type BottleneckKind = qualify.BottleneckKind

const (
	BottleneckBankConflict   = qualify.BottleneckBankConflict
	BottleneckWGMMASerial    = qualify.BottleneckWGMMASerial
	BottleneckRegisterSpill  = qualify.BottleneckRegisterSpill
	BottleneckLowOccupancy   = qualify.BottleneckLowOccupancy
	BottleneckWaveTail       = qualify.BottleneckWaveTail
	BottleneckLaunchOverhead = qualify.BottleneckLaunchOverhead
	BottleneckMemoryBoundKV  = qualify.BottleneckMemoryBoundKV
)

// KernelObservation holds measured profiling and compiler telemetry for a
// single kernel or plan step.
type KernelObservation = qualify.KernelObservation

// ProfileReceipt aggregates per-step observations and device metadata.
type ProfileReceipt = qualify.ProfileReceipt

// Step is an alias for [PlanKernel] used by profile-guided refinement.
type Step = PlanKernel

// MegakernelSource is an alias for [Megakernel] returned when a plan is
// promoted to a persistent megakernel.
type MegakernelSource = Megakernel

// ClassifyBottlenecks classifies the hardware bottlenecks present in obs in a
// deterministic order.
func ClassifyBottlenecks(obs KernelObservation) []BottleneckKind {
	return qualify.ClassifyBottlenecks(obs)
}

// ObservationFromPTXAS constructs a [KernelObservation] from a parsed
// [ptxasinfo.Function] report entry and launch geometry.
func ObservationFromPTXAS(fn ptxasinfo.Function, gridBlocks, deviceSMs int) KernelObservation {
	return qualify.ObservationFromPTXAS(fn, gridBlocks, deviceSMs)
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
		apply := func(k qualify.TileGEMMKnobs) TileGEMM {
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
		knobs, reasons := qualify.RefineTileGEMMKnobs(qualify.TileGEMMKnobs{
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
			IsNVFP4:         g.Weight == WeightNVFP4,
			LinearEpilogue:  g.Epilogue.root == nil || g.Epilogue.IsIdentity(),
		}, obs, bns, func(ck qualify.TileGEMMKnobs) bool {
			_, err := apply(ck).norm()
			return err == nil
		})
		t := apply(knobs)
		out.Tile = &t
		return out, reasons

	case out.GEMM != nil:
		g := *out.GEMM
		normed, err := g.norm()
		normR, normW, normS, normU := g.RowsPerBlock, g.Warps, g.SplitK, g.Unroll
		chunk := g.Weight.chunk()
		if err == nil {
			normR, normW, normS, normU = normed.RowsPerBlock, normed.Warps, normed.SplitK, normed.Unroll
			chunk = normed.Weight.chunk()
		}
		apply := func(k qualify.SkinnyGEMMKnobs) SkinnyGEMM {
			cand := g
			cand.RowsPerBlock = k.RowsPerBlock
			cand.Unroll = k.Unroll
			cand.Prefetch = k.Prefetch
			cand.BlackwellFP4 = k.BlackwellFP4
			return cand
		}
		knobs, reasons := qualify.RefineSkinnyGEMMKnobs(qualify.SkinnyGEMMKnobs{
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
			IsNVFP4:          g.Weight == WeightNVFP4,
			WeightChunk:      chunk,
		}, obs, bns, func(ck qualify.SkinnyGEMMKnobs) bool {
			_, err := apply(ck).norm()
			return err == nil
		})
		sg := apply(knobs)
		out.GEMM = &sg
		return out, reasons

	case out.Attention != nil:
		a := *out.Attention
		normStages := a.Stages
		if normStages == 0 {
			if normed, _, err := a.norm(); err == nil {
				normStages = normed.Stages
			} else {
				normStages = 2
			}
		}
		apply := func(k qualify.PagedAttentionKnobs) PagedAttention {
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
		knobs, reasons := qualify.RefinePagedAttentionKnobs(qualify.PagedAttentionKnobs{
			Name:            a.Name,
			Hq:              a.Hq,
			Hkv:             a.Hkv,
			QLen:            a.QLen,
			Window:          a.Window,
			QIsE4M3:         a.Q == E4M3,
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
		}, out.Batch, out.MaxKeys, obs, bns, func(ck qualify.PagedAttentionKnobs) bool {
			_, _, err := apply(ck).norm()
			return err == nil
		})
		pa := apply(knobs)
		out.Attention = &pa
		return out, reasons

	case out.Row != nil:
		r := *out.Row
		knobs, reasons := qualify.RefineRowKernelKnobs(qualify.RowKernelKnobs{
			Name:       r.Name,
			Cluster:    r.Cluster,
			CanCluster: r.CanCluster(),
		}, bns, func(ck qualify.RowKernelKnobs) bool {
			cand := r
			cand.Cluster = ck.Cluster
			_, err := cand.check()
			return err == nil
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
func RefinePlan(plan *Plan, receipt ProfileReceipt) (*Plan, *MegakernelSource, []string, error) {
	if plan == nil {
		return nil, nil, nil, fmt.Errorf("refine plan: nil plan")
	}
	refined := clonePlan(plan)
	defaultSMs := 0
	if receipt.Device != "" {
		if cfg, err := (CostConfig{Device: receipt.Device}).norm(); err == nil {
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
		obs, ok := qualify.MatchObservation(stepKernelName(refined.Kernels[i]), i, receipt.Observations, matched)
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
				refined.Temps = append(refined.Temps, Tensor{
					Name:  wsName,
					DType: F32,
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
func RefineFromProfile(plan *Plan, receipt ProfileReceipt) (*Plan, *MegakernelSource, []string, error) {
	return RefinePlan(plan, receipt)
}

func stepKernelName(k PlanKernel) string {
	switch {
	case k.GEMM != nil:
		return k.GEMM.Name
	case k.Tile != nil:
		return k.Tile.Name
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

func cloneStep(k PlanKernel) PlanKernel {
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
		r.Values = make([]RowValue, len(k.Row.Values))
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

func clonePlan(p *Plan) *Plan {
	if p == nil {
		return nil
	}
	out := &Plan{
		Name:    p.Name,
		Inputs:  make([]Tensor, len(p.Inputs)),
		Temps:   make([]Tensor, len(p.Temps)),
		Output:  Tensor{Name: p.Output.Name, DType: p.Output.DType, Shape: slices.Clone(p.Output.Shape)},
		Kernels: make([]PlanKernel, len(p.Kernels)),
	}
	for i, t := range p.Inputs {
		out.Inputs[i] = Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, t := range p.Temps {
		out.Temps[i] = Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, k := range p.Kernels {
		out.Kernels[i] = cloneStep(k)
	}
	return out
}
