package qualify

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// A StructuralBeamCandidate is a whole-graph lowering candidate in a structural
// e-graph beam search. It binds the extracted e-graph expression, the lowered
// multi-launch [hopper.Plan], an optional persistent [hopper.Megakernel]
// lowering, the canonical SHA-256 structural signature over the kernel sequence
// and schedules, the analytical roofline latency estimate in nanoseconds, and
// any profile-guided step schedule proposals.
type StructuralBeamCandidate struct {
	Name          string                  `json:"name"`
	Expr          *egraph.Expr            `json:"-"`
	Plan          *hopper.Plan            `json:"-"`
	Megakernel    *hopper.Megakernel      `json:"-"`
	PlanSignature string                  `json:"plan_signature"`
	EstimatedNs   float64                 `json:"estimated_ns"`
	StepProposals []CandidateStepProposal `json:"step_proposals,omitempty"`
	CostConfig    hopper.CostConfig       `json:"-"`
}

// Handle returns a stable identifier for c suitable for timing pairs and
// evidence map keys.
func (c StructuralBeamCandidate) Handle() string {
	if c.Name != "" {
		return c.Name
	}
	if c.Megakernel != nil && c.Megakernel.Name != "" {
		return c.Megakernel.Name
	}
	if c.Plan != nil && c.Plan.Name != "" {
		return c.Plan.Name
	}
	return c.PlanSignature
}

// Source returns the self-contained CUDA C++ translation unit for c (using
// c.Megakernel.Source when c.Megakernel is non-nil, or c.Plan.Source
// otherwise) along with its SHA-256 digest.
func (c StructuralBeamCandidate) Source() (string, [32]byte, error) {
	if c.Megakernel != nil {
		src, err := c.Megakernel.Source()
		if err != nil {
			return "", [32]byte{}, err
		}
		return src, sha256.Sum256([]byte(src)), nil
	}
	if c.Plan == nil {
		return "", [32]byte{}, errors.New("structural beam candidate: nil plan")
	}
	src, err := c.Plan.Source()
	if err != nil {
		return "", [32]byte{}, err
	}
	return src, sha256.Sum256([]byte(src)), nil
}

// A RejectedBeamCandidate is a [StructuralBeamCandidate] pruned by
// [FilterStructuralBeamByPTXAS] because at least one of its kernels or
// megakernel stages violated [StaticGate].
type RejectedBeamCandidate struct {
	Candidate        StructuralBeamCandidate `json:"candidate"`
	Reasons          []string                `json:"reasons"`
	RejectedVariants []RejectedHopperVariant `json:"rejected_variants,omitempty"`
}

// Reason returns the semicolon-separated rejection reasons for r.
func (r RejectedBeamCandidate) Reason() string {
	return strings.Join(r.Reasons, "; ")
}

// A StructuralBeamEvaluation records the outcome of evaluating a single
// [StructuralBeamCandidate] against the baseline in
// [SelectStructuralBeamWithHoldout].
type StructuralBeamEvaluation struct {
	Candidate StructuralBeamCandidate `json:"candidate"`
	Handle    string                  `json:"handle"`
	Result    HoldoutPromotionResult  `json:"result"`
	Promoted  bool                    `json:"promoted"`
}

// A StructuralBeamWinner reports the whole-graph tournament winner selected by
// [SelectStructuralBeamWithHoldout]. When no candidate in the beam passes the
// two-stage screen + held-out ABBA + A/A null + greedy token-ID gate, Winner
// and Selected fall back to the baseline candidate with Promoted=false and
// FallbackToBaseline=true.
type StructuralBeamWinner struct {
	Winner             StructuralBeamCandidate    `json:"winner"`
	Selected           StructuralBeamCandidate    `json:"selected"`
	Promoted           bool                       `json:"promoted"`
	FallbackToBaseline bool                       `json:"fallback_to_baseline"`
	Status             string                     `json:"status"`
	Reason             string                     `json:"reason,omitempty"`
	Screen             pairedtiming.Summary       `json:"screen"`
	Holdout            pairedtiming.Summary       `json:"holdout"`
	ScreenObserved     *pairedtiming.Summary      `json:"screen_observed,omitempty"`
	HoldoutObserved    *pairedtiming.Summary      `json:"holdout_observed,omitempty"`
	AANullObserved     *pairedtiming.Summary      `json:"aa_null_observed,omitempty"`
	OneSidedSignP      float64                    `json:"one_sided_sign_p"`
	TokensMatched      bool                       `json:"tokens_matched"`
	Evaluations        []StructuralBeamEvaluation `json:"evaluations,omitempty"`
}

// ComputePlanSignature computes a deterministic SHA-256 hex digest of plan's
// kernel sequence, step schedules, argument bindings, tail fusions, and
// optional persistent megakernel stage topology.
func ComputePlanSignature(plan *hopper.Plan, mega *hopper.Megakernel) string {
	type stepSig struct {
		Index     int                     `json:"index"`
		Op        string                  `json:"op"`
		M         int                     `json:"m,omitempty"`
		MaxRows   int                     `json:"max_rows,omitempty"`
		Experts   int                     `json:"experts,omitempty"`
		Batch     int                     `json:"batch,omitempty"`
		MaxKeys   int                     `json:"max_keys,omitempty"`
		MaxPages  int                     `json:"max_pages,omitempty"`
		Rows      int                     `json:"rows,omitempty"`
		Args      []string                `json:"args,omitempty"`
		GEMM      *hopper.SkinnyGEMM      `json:"gemm,omitempty"`
		Tile      *hopper.TileGEMM        `json:"tile,omitempty"`
		MMA       *hopper.SkinnyMMA       `json:"mma,omitempty"`
		UMMA      *hopper.UMMAGEMM        `json:"umma,omitempty"`
		Quant     *hopper.NVFP4Quant      `json:"quant,omitempty"`
		Row       *hopper.RowKernel       `json:"row,omitempty"`
		Attention *hopper.PagedAttention  `json:"attention,omitempty"`
		TopK      *hopper.RowTopK         `json:"topk,omitempty"`
		Align     *hopper.ExpertAlign     `json:"align,omitempty"`
		ExpertSum *hopper.ExpertSum       `json:"expert_sum,omitempty"`
		Tail      *hopper.RowKernel       `json:"tail,omitempty"`
	}
	type megaStageSig struct {
		Index      int    `json:"index"`
		Grid       [3]int `json:"grid"`
		BlockBase  int    `json:"block_base"`
		Blocks     int    `json:"blocks"`
		Shared     int    `json:"shared"`
		WaitSlots  []int  `json:"wait_slots,omitempty"`
		SignalSlot int    `json:"signal_slot"`
	}
	type planSig struct {
		OutDType   string         `json:"out_dtype,omitempty"`
		OutShape   []int          `json:"out_shape,omitempty"`
		Steps      []stepSig      `json:"steps"`
		Megakernel bool           `json:"megakernel"`
		MegaStages []megaStageSig `json:"mega_stages,omitempty"`
	}

	var sig planSig
	if plan != nil {
		sig.OutDType = plan.Output.DType.String()
		sig.OutShape = slices.Clone(plan.Output.Shape)
		sig.Steps = make([]stepSig, len(plan.Kernels))
		for i, k := range plan.Kernels {
			st := stepSig{
				Index:    i,
				Op:       stepOpKind(k),
				M:        k.M,
				MaxRows:  k.MaxRows,
				Experts:  k.Experts,
				Batch:    k.Batch,
				MaxKeys:  k.MaxKeys,
				MaxPages: k.MaxPages,
				Rows:     k.Rows,
				Args:     slices.Clone(k.Args),
			}
			switch {
			case k.GEMM != nil:
				g := *k.GEMM
				if normed, err := g.Norm(); err == nil {
					g = normed
				}
				g.Name = ""
				if g.Tail != nil {
					tailCopy := *g.Tail
					tailCopy.Name = ""
					st.Tail = &tailCopy
					g.Tail = nil
				}
				st.GEMM = &g
			case k.Tile != nil:
				t := *k.Tile
				if normed, err := t.Norm(); err == nil {
					t = normed
				}
				t.Name = ""
				st.Tile = &t
			case k.MMA != nil:
				m := *k.MMA
				m.Name = ""
				st.MMA = &m
			case k.UMMA != nil:
				u := *k.UMMA
				u.Name = ""
				st.UMMA = &u
			case k.Quant != nil:
				q := *k.Quant
				q.Name = ""
				st.Quant = &q
			case k.Row != nil:
				r := *k.Row
				r.Name = ""
				st.Row = &r
			case k.Attention != nil:
				a := *k.Attention
				if normed, err := a.Norm(); err == nil {
					a = normed
				}
				a.Name = ""
				st.Attention = &a
			case k.TopK != nil:
				tk := *k.TopK
				if normed, err := tk.Norm(); err == nil {
					tk = normed
				}
				tk.Name = ""
				st.TopK = &tk
			case k.Align != nil:
				al := *k.Align
				if normed, err := al.Norm(); err == nil {
					al = normed
				}
				al.Name = ""
				st.Align = &al
			case k.ExpertSum != nil:
				es := *k.ExpertSum
				if normed, err := es.Norm(); err == nil {
					es = normed
				}
				es.Name = ""
				st.ExpertSum = &es
			}
			sig.Steps[i] = st
		}
	}
	if mega != nil {
		sig.Megakernel = true
		sig.MegaStages = make([]megaStageSig, len(mega.Stages))
		for i, st := range mega.Stages {
			sig.MegaStages[i] = megaStageSig{
				Index:      i,
				Grid:       st.Grid,
				BlockBase:  st.BlockBase,
				Blocks:     st.Blocks,
				Shared:     st.Shared,
				WaitSlots:  slices.Clone(st.WaitSlots),
				SignalSlot: st.SignalSlot,
			}
		}
	}
	raw, err := json.Marshal(sig)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// diversifiedCostModel perturbs node and epilogue costs to extract structurally
// distinct expressions (e.g. unfused GEMM + RowKernel vs partially fused vs
// fully epilogue-fused) from a saturated e-graph.
type diversifiedCostModel struct {
	epilogueArityPenalty float64
	nonIdentityPenalty   float64
	mapCost              float64
	reduceCost           float64
	modelFusedBonus      float64
}

func (m diversifiedCostModel) NodeCost(n egraph.ENode, children []float64) float64 {
	sum := 0.0
	for _, c := range children {
		sum += c
	}
	switch n.Op {
	case hopper.OpMatMul:
		return sum + 1e12
	case hopper.OpGEMM:
		cost := sum + 2000
		if len(n.Children) > 3 {
			cost += float64(len(n.Children)-3) * m.epilogueArityPenalty
		}
		if len(n.Children) > 0 && n.Children[0] != 0 {
			cost += m.nonIdentityPenalty
		}
		return cost
	case hopper.OpGroupedMatMul:
		return sum + 2500 + m.modelFusedBonus
	case hopper.OpGroupedGEMM, hopper.OpGroupedGEMMGLU, hopper.OpGroupedGEMMSumSq:
		return sum + max(2000-m.modelFusedBonus, 100)
	case hopper.OpPagedAttn:
		return sum + 2000 + m.modelFusedBonus
	case hopper.OpPagedAttnGated:
		return sum + max(2000-m.modelFusedBonus, 100)
	case hopper.OpRMSNormGate:
		return sum + m.mapCost
	case hopper.OpRowTopK, hopper.OpExpertAlign, hopper.OpSwiGLU, hopper.OpPostExpertNorm, hopper.OpWeightedTopKSum:
		return sum + m.mapCost
	case hopper.OpExpertSum:
		return sum + max(1500-m.modelFusedBonus, 100)
	case hopper.OpMap:
		return sum + m.mapCost
	case hopper.OpReduceSum, hopper.OpReduceMax:
		return sum + m.reduceCost
	}
	return sum
}

// ExtractStructuralPlanBeam runs equality saturation snapshots across rounds
// (before and after [hopper.FuseEpilogues], combined with top-K and
// cost-diversified extractions plus multi-launch, tail-fused, and persistent
// megakernel lowerings) to produce up to beamWidth structurally distinct,
// valid [StructuralBeamCandidate] plans, deduplicated by [ComputePlanSignature]
// and sorted by analytical EstimatedNs.
func ExtractStructuralPlanBeam(g *egraph.EGraph, root egraph.Id, name string, out hopper.DType, cfg hopper.CostConfig, beamWidth int) ([]StructuralBeamCandidate, error) {
	if g == nil {
		return nil, errors.New("extract structural plan beam: nil e-graph")
	}
	if beamWidth <= 0 {
		return nil, nil
	}
	if strings.TrimSpace(name) == "" {
		name = "beam_plan"
	}
	normCfg, err := cfg.Norm()
	if err != nil {
		normCfg = cfg
	}

	models := []egraph.CostModel{
		hopper.Cost{},
		// Prefer unfused GEMM + separate RowKernel trees over epilogue fusion.
		diversifiedCostModel{
			epilogueArityPenalty: 6000,
			nonIdentityPenalty:   4000,
			mapCost:              100,
			reduceCost:           150,
			modelFusedBonus:      -1500,
		},
		// Prefer intermediate fusion (fuse 1-input epilogues, keep multi-input maps as row kernels).
		diversifiedCostModel{
			epilogueArityPenalty: 2500,
			nonIdentityPenalty:   0,
			mapCost:              1800,
			reduceCost:           1800,
			modelFusedBonus:      500,
		},
		// Strongly reward maximum epilogue & cross-op fusion.
		diversifiedCostModel{
			epilogueArityPenalty: 0,
			nonIdentityPenalty:   0,
			mapCost:              4000,
			reduceCost:           4000,
			modelFusedBonus:      1000,
		},
	}

	var exprs []*egraph.Expr
	seenExpr := make(map[string]bool)
	addExpr := func(e *egraph.Expr) {
		if e == nil {
			return
		}
		fp := e.Fingerprint()
		if seenExpr[fp] {
			return
		}
		seenExpr[fp] = true
		exprs = append(exprs, e)
	}

	harvestSnapshot := func(snap *egraph.EGraph) {
		if err := snap.RebuildChecked(); err != nil {
			return
		}
		limits := egraph.ExtractionLimits{MaxWork: 1 << 30, MaxNodes: 1 << 24, MaxDepth: 1 << 20}
		topKLimit := max(beamWidth*4, 12)
		for _, m := range models {
			if e, _, _, err := egraph.ExtractSegmented(context.Background(), snap, root, m, limits); err == nil {
				addExpr(e)
			}
			if cands, err := egraph.ExtractTopK(snap, root, topKLimit, m); err == nil {
				for _, c := range cands {
					addExpr(c.Expr)
				}
			}
		}
	}

	work := g.Clone()
	harvestSnapshot(work)

	for range 8 {
		if _, err := egraph.SaturateChecked(work, hopper.Rules(), egraph.SaturationOptions{MaxIterations: 16}); err != nil {
			return nil, fmt.Errorf("extract structural plan beam: saturate: %w", err)
		}
		// Snapshot after algebraic/model rewrites before epilogue folding.
		harvestSnapshot(work)

		n, err := hopper.FuseEpilogues(work)
		if err != nil {
			return nil, fmt.Errorf("extract structural plan beam: fuse epilogues: %w", err)
		}
		harvestSnapshot(work)
		if n == 0 {
			break
		}
	}

	var (
		candidates []StructuralBeamCandidate
		seenSig    = make(map[string]bool)
	)
	addCandidate := func(expr *egraph.Expr, p *hopper.Plan, mega *hopper.Megakernel, tag string) {
		if p == nil || len(p.Kernels) == 0 {
			return
		}
		if _, err := p.Source(); err != nil {
			return
		}
		if mega != nil {
			if _, err := mega.Source(); err != nil {
				return
			}
		}
		sig := ComputePlanSignature(p, mega)
		if sig == "" || seenSig[sig] {
			return
		}
		seenSig[sig] = true
		candName := fmt.Sprintf("%s_%s_%d", name, tag, len(candidates)+1)
		renamed := renamePlanForCandidate(p, candName)
		var megaCopy *hopper.Megakernel
		if mega != nil {
			if m, err := renamed.Megakernel(candName + "_mega"); err == nil {
				megaCopy = &m
			} else {
				mc := *mega
				mc.Name = candName + "_mega"
				megaCopy = &mc
			}
		}
		estNs := estimateCandidateNs(renamed, megaCopy, normCfg)
		candidates = append(candidates, StructuralBeamCandidate{
			Name:          candName,
			Expr:          expr,
			Plan:          renamed,
			Megakernel:    megaCopy,
			PlanSignature: sig,
			EstimatedNs:   estNs,
			CostConfig:    normCfg,
		})
	}

	for _, expr := range exprs {
		// 1. Hardware-scheduled plan under cfg.
		if scheduled, err := hopper.CompileWithCost(expr, name, out, normCfg); err == nil {
			addCandidate(expr, scheduled, nil, "sched")

			// 2. Persistent megakernel promotion when multi-stage compatible.
			if len(scheduled.Kernels) >= 2 {
				if compat, mega, ok := promotePlanToMegakernel(scheduled, name+"_mega"); ok {
					addCandidate(expr, compat, &mega, "mega")
				}
			}

			// 3. Tail-fused plan (SkinnyGEMM + single-CTA RowKernel tail).
			if len(scheduled.Kernels) >= 2 {
				tailPlan := clonePlan(scheduled)
				if tailPlan.FuseTails() > 0 {
					addCandidate(expr, tailPlan, nil, "tail")
				}
			}
		}

		// 4. Unscheduled / conservative baseline lowering (Compile without SelectSchedules).
		if basePlan, err := hopper.Compile(expr, name, out); err == nil {
			addCandidate(expr, basePlan, nil, "base")
			if len(basePlan.Kernels) >= 2 {
				if compat, mega, ok := promotePlanToMegakernel(basePlan, name+"_base_mega"); ok {
					addCandidate(expr, compat, &mega, "base_mega")
				}
			}
		}
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("extract structural plan beam %s: no valid lowerable candidates for root %d", name, root)
	}

	slices.SortStableFunc(candidates, compareBeamCandidates)
	if len(candidates) > beamWidth {
		candidates = candidates[:beamWidth]
	}
	return candidates, nil
}

func compareBeamCandidates(a, b StructuralBeamCandidate) int {
	if a.EstimatedNs < b.EstimatedNs {
		return -1
	}
	if a.EstimatedNs > b.EstimatedNs {
		return 1
	}
	return strings.Compare(a.PlanSignature, b.PlanSignature)
}

func estimateCandidateNs(plan *hopper.Plan, mega *hopper.Megakernel, cfg hopper.CostConfig) float64 {
	if plan == nil {
		return math.Inf(1)
	}
	normCfg, err := cfg.Norm()
	if err != nil {
		normCfg = cfg
	}
	est := plan.EstimateNs(normCfg)
	if mega != nil && len(plan.Kernels) > 1 {
		launchNs := normCfg.LaunchNs
		if launchNs <= 0 {
			launchNs = 2000.0
		}
		// Persistent megakernel replaces (N-1) host stream launches with an
		// in-device atomic ready-counter barrier (~150 ns per stage transition).
		const deviceBarrierNs = 150.0
		savedPerTransition := max(launchNs-deviceBarrierNs, 0)
		est = max(est-float64(len(plan.Kernels)-1)*savedPerTransition, 1.0)
	}
	return est
}

func promotePlanToMegakernel(plan *hopper.Plan, megaName string) (*hopper.Plan, hopper.Megakernel, bool) {
	if plan == nil || len(plan.Kernels) < 2 {
		return nil, hopper.Megakernel{}, false
	}
	if mega, err := plan.Megakernel(megaName); err == nil {
		return clonePlan(plan), mega, true
	}
	compat := clonePlan(plan)
	for i := range compat.Kernels {
		k := &compat.Kernels[i]
		if k.Row != nil && k.Row.Cluster {
			k.Row.Cluster = false
		}
		if k.Tile != nil && k.Tile.WarpSpecialized {
			k.Tile.WarpSpecialized = false
		}
		if k.GEMM != nil && k.GEMM.Warps != 8 {
			cand := *k.GEMM
			cand.Warps = 8
			if normed, err := cand.Norm(); err == nil {
				*k.GEMM = normed
			}
		}
	}
	mega, err := compat.Megakernel(megaName)
	if err != nil {
		return nil, hopper.Megakernel{}, false
	}
	return compat, mega, true
}

func renamePlanForCandidate(plan *hopper.Plan, candName string) *hopper.Plan {
	cp := clonePlan(plan)
	if cp == nil || candName == "" {
		return cp
	}
	oldPrefix := cp.Name + "_"
	cp.Name = candName
	for i := range cp.Kernels {
		k := &cp.Kernels[i]
		rename := func(cur string) string {
			if strings.HasPrefix(cur, oldPrefix) {
				return candName + "_" + strings.TrimPrefix(cur, oldPrefix)
			}
			if cur == "" {
				return fmt.Sprintf("%s_step%d", candName, i+1)
			}
			return candName + "_" + cur
		}
		switch {
		case k.GEMM != nil:
			k.GEMM.Name = rename(k.GEMM.Name)
		case k.Tile != nil:
			k.Tile.Name = rename(k.Tile.Name)
		case k.MMA != nil:
			k.MMA.Name = rename(k.MMA.Name)
		case k.UMMA != nil:
			k.UMMA.Name = rename(k.UMMA.Name)
		case k.Quant != nil:
			k.Quant.Name = rename(k.Quant.Name)
		case k.Row != nil:
			k.Row.Name = rename(k.Row.Name)
		case k.Attention != nil:
			k.Attention.Name = rename(k.Attention.Name)
		case k.TopK != nil:
			k.TopK.Name = rename(k.TopK.Name)
		case k.Align != nil:
			k.Align.Name = rename(k.Align.Name)
		case k.ExpertSum != nil:
			k.ExpertSum.Name = rename(k.ExpertSum.Name)
		}
	}
	return cp
}

// ExpandBeamWithProfile expands each candidate in beam with step-level schedule
// variants proposed by [ProposeStepCandidates] from causal counter observations
// and PC-sampling stalls, deduplicating by structural signature and keeping up
// to maxTotal candidates.
func ExpandBeamWithProfile(beam []StructuralBeamCandidate, causal map[string]KernelObservation, stalls map[string]StallSample, maxTotal int) []StructuralBeamCandidate {
	if len(beam) == 0 || maxTotal <= 0 {
		return nil
	}

	var (
		expanded []StructuralBeamCandidate
		seenSig  = make(map[string]int) // signature -> index in expanded
	)
	addOrMerge := func(c StructuralBeamCandidate) {
		if c.Plan == nil || len(c.Plan.Kernels) == 0 {
			return
		}
		if c.PlanSignature == "" {
			c.PlanSignature = ComputePlanSignature(c.Plan, c.Megakernel)
		}
		if c.PlanSignature == "" {
			return
		}
		if idx, ok := seenSig[c.PlanSignature]; ok {
			if len(expanded[idx].StepProposals) == 0 && len(c.StepProposals) > 0 {
				expanded[idx].StepProposals = slices.Clone(c.StepProposals)
			}
			return
		}
		seenSig[c.PlanSignature] = len(expanded)
		expanded = append(expanded, c)
	}

	for _, base := range beam {
		if base.Plan == nil {
			continue
		}
		baseCopy := cloneBeamCandidate(base)
		var allProposals []CandidateStepProposal
		stepProposalsByIdx := make([][]CandidateStepProposal, len(baseCopy.Plan.Kernels))
		launchOverheadSeen := false

		for i, step := range baseCopy.Plan.Kernels {
			obs := lookupStepCausal(baseCopy, step, i, causal)
			st := lookupStepStalls(baseCopy, step, i, stalls)
			if slices.Contains(ClassifyBottlenecks(obs), BottleneckLaunchOverhead) {
				launchOverheadSeen = true
			}
			props := ProposeStepCandidates(step, obs, st)
			for _, p := range props {
				if p.Hypothesis == BottleneckLaunchOverhead {
					launchOverheadSeen = true
				}
			}
			stepProposalsByIdx[i] = props
			allProposals = append(allProposals, props...)
		}
		if len(allProposals) > 0 {
			baseCopy.StepProposals = slices.Clone(allProposals)
		}
		addOrMerge(baseCopy)

		// 1. Single-step schedule expansions for each step proposal.
		for i, props := range stepProposalsByIdx {
			for pIdx, prop := range props {
				modPlan := clonePlan(baseCopy.Plan)
				applyStepProposalToPlan(modPlan, i, prop.Step)
				variantName := fmt.Sprintf("%s_s%dp%d", baseCopy.Handle(), i+1, pIdx+1)
				modPlan = renamePlanForCandidate(modPlan, variantName)

				var modMega *hopper.Megakernel
				if baseCopy.Megakernel != nil {
					if compat, mega, ok := promotePlanToMegakernel(modPlan, variantName+"_mega"); ok {
						modPlan = compat
						modMega = &mega
					}
				}
				sig := ComputePlanSignature(modPlan, modMega)
				estNs := estimateCandidateNs(modPlan, modMega, baseCopy.CostConfig)
				addOrMerge(StructuralBeamCandidate{
					Name:          variantName,
					Expr:          baseCopy.Expr,
					Plan:          modPlan,
					Megakernel:    modMega,
					PlanSignature: sig,
					EstimatedNs:   estNs,
					StepProposals: []CandidateStepProposal{prop},
					CostConfig:    baseCopy.CostConfig,
				})
			}
		}

		// 2. Combined multi-step expansion applying the top proposal for every step.
		stepsChanged := 0
		combPlan := clonePlan(baseCopy.Plan)
		var combProps []CandidateStepProposal
		for i, props := range stepProposalsByIdx {
			if len(props) == 0 {
				continue
			}
			applyStepProposalToPlan(combPlan, i, props[0].Step)
			combProps = append(combProps, props[0])
			stepsChanged++
		}
		if stepsChanged >= 2 {
			combName := baseCopy.Handle() + "_combined"
			combPlan = renamePlanForCandidate(combPlan, combName)
			var combMega *hopper.Megakernel
			if baseCopy.Megakernel != nil {
				if compat, mega, ok := promotePlanToMegakernel(combPlan, combName+"_mega"); ok {
					combPlan = compat
					combMega = &mega
				}
			}
			addOrMerge(StructuralBeamCandidate{
				Name:          combName,
				Expr:          baseCopy.Expr,
				Plan:          combPlan,
				Megakernel:    combMega,
				PlanSignature: ComputePlanSignature(combPlan, combMega),
				EstimatedNs:   estimateCandidateNs(combPlan, combMega, baseCopy.CostConfig),
				StepProposals: combProps,
				CostConfig:    baseCopy.CostConfig,
			})
		}

		// 3. If launch overhead was observed on a multi-launch candidate, also
		// propose promoting the refined (or base) plan to a persistent megakernel.
		if launchOverheadSeen && baseCopy.Megakernel == nil && len(baseCopy.Plan.Kernels) >= 2 {
			srcPlan := baseCopy.Plan
			if stepsChanged >= 1 {
				srcPlan = combPlan
			}
			megaCandName := baseCopy.Handle() + "_pgo_mega"
			if compat, mega, ok := promotePlanToMegakernel(renamePlanForCandidate(srcPlan, megaCandName), megaCandName+"_mega"); ok {
				addOrMerge(StructuralBeamCandidate{
					Name:          megaCandName,
					Expr:          baseCopy.Expr,
					Plan:          compat,
					Megakernel:    &mega,
					PlanSignature: ComputePlanSignature(compat, &mega),
					EstimatedNs:   estimateCandidateNs(compat, &mega, baseCopy.CostConfig),
					StepProposals: slices.Clone(allProposals),
					CostConfig:    baseCopy.CostConfig,
				})
			}
		}
	}

	if len(expanded) > maxTotal {
		expanded = expanded[:maxTotal]
	}
	return expanded
}

func applyStepProposalToPlan(plan *hopper.Plan, idx int, newStep Step) {
	if plan == nil || idx < 0 || idx >= len(plan.Kernels) {
		return
	}
	prev := plan.Kernels[idx]
	prevStreamK := prev.Tile != nil && prev.Tile.StreamK > 0
	oldName := stepKernelName(prev)

	step := cloneStep(newStep)
	if oldName != "" {
		setStepKernelName(&step, oldName)
	}
	plan.Kernels[idx] = step

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
		for j := range plan.Temps {
			if plan.Temps[j].Name == wsName {
				plan.Temps[j].Shape = []int{wsBytes}
			}
		}
	}
	if step.Tile != nil && step.Tile.StreamK > 0 && !prevStreamK {
		if wsBytes := step.Tile.WorkspaceBytes(); wsBytes > 0 {
			wsName := fmt.Sprintf("%s_sk_ws%d", plan.Name, idx+1)
			plan.Temps = append(plan.Temps, hopper.Tensor{
				Name:  wsName,
				DType: hopper.F32,
				Shape: []int{(wsBytes + 3) / 4},
			})
			plan.Kernels[idx].Args = append(slices.Clone(plan.Kernels[idx].Args), wsName)
		}
	}
}

func setStepKernelName(k *hopper.PlanKernel, name string) {
	switch {
	case k.GEMM != nil:
		k.GEMM.Name = name
	case k.Tile != nil:
		k.Tile.Name = name
	case k.MMA != nil:
		k.MMA.Name = name
	case k.UMMA != nil:
		k.UMMA.Name = name
	case k.Quant != nil:
		k.Quant.Name = name
	case k.Row != nil:
		k.Row.Name = name
	case k.Attention != nil:
		k.Attention.Name = name
	case k.TopK != nil:
		k.TopK.Name = name
	case k.Align != nil:
		k.Align.Name = name
	case k.ExpertSum != nil:
		k.ExpertSum.Name = name
	}
}

func stepOpKind(k hopper.PlanKernel) string {
	switch {
	case k.GEMM != nil:
		return "skinny_gemm"
	case k.Tile != nil:
		return "tile_gemm"
	case k.MMA != nil:
		return "skinny_mma"
	case k.UMMA != nil:
		return "umma_gemm"
	case k.Quant != nil:
		return "nvfp4_quant"
	case k.Row != nil:
		return "row_kernel"
	case k.Attention != nil:
		return "paged_attention"
	case k.TopK != nil:
		return "row_topk"
	case k.Align != nil:
		return "expert_align"
	case k.ExpertSum != nil:
		return "expert_sum"
	default:
		return "unknown"
	}
}

func lookupStepCausal(cand StructuralBeamCandidate, step Step, stepIdx int, causal map[string]KernelObservation) KernelObservation {
	if len(causal) == 0 {
		return KernelObservation{}
	}
	kName := stepKernelName(step)
	opKind := stepOpKind(step)
	keys := []string{
		kName,
		cand.Name + ":" + kName,
		fmt.Sprintf("%s:%d", cand.Name, stepIdx),
		fmt.Sprintf("step_%d", stepIdx),
		opKind,
		"*",
		"",
	}
	for _, key := range keys {
		if obs, ok := causal[key]; ok {
			if obs.Name == "" {
				obs.Name = kName
			}
			return obs
		}
	}
	for key, obs := range causal {
		if key != "" && (strings.HasSuffix(kName, "_"+key) || strings.HasPrefix(kName, key+"_") || strings.Contains(kName, key)) {
			if obs.Name == "" {
				obs.Name = kName
			}
			return obs
		}
	}
	return KernelObservation{Name: kName}
}

func lookupStepStalls(cand StructuralBeamCandidate, step Step, stepIdx int, stalls map[string]StallSample) StallSample {
	if len(stalls) == 0 {
		return StallSample{}
	}
	kName := stepKernelName(step)
	opKind := stepOpKind(step)
	keys := []string{
		kName,
		cand.Name + ":" + kName,
		fmt.Sprintf("%s:%d", cand.Name, stepIdx),
		fmt.Sprintf("step_%d", stepIdx),
		opKind,
		"*",
		"",
	}
	for _, key := range keys {
		if st, ok := stalls[key]; ok {
			return st
		}
	}
	for key, st := range stalls {
		if key != "" && (strings.HasSuffix(kName, "_"+key) || strings.HasPrefix(kName, key+"_") || strings.Contains(kName, key)) {
			return st
		}
	}
	return StallSample{}
}

// Variants returns the [hopper.Variant] descriptors for every kernel step (and
// optional persistent megakernel entry) in c.
func (c StructuralBeamCandidate) Variants() []hopper.Variant {
	var out []hopper.Variant
	if c.Plan != nil {
		for _, k := range c.Plan.Kernels {
			if v, ok := stepToHopperVariant(k); ok {
				out = append(out, v)
			}
		}
	}
	if c.Megakernel != nil && c.Megakernel.Name != "" {
		maxShared := 0
		for _, st := range c.Megakernel.Stages {
			maxShared = max(maxShared, st.Shared)
		}
		out = append(out, hopper.Variant{
			Name:  c.Megakernel.Name,
			Op:    "megakernel",
			Entry: c.Megakernel.Name + "_launch",
			Launch: hopper.Launch{
				Block:       256,
				Warps:       8,
				SharedBytes: maxShared,
			},
		})
	}
	return out
}

func stepToHopperVariant(k hopper.PlanKernel) (hopper.Variant, bool) {
	switch {
	case k.GEMM != nil:
		if v, err := k.GEMM.Variant(); err == nil {
			return v, true
		}
		g := *k.GEMM
		warps := max(g.Warps, 8)
		return hopper.Variant{
			Name:  g.Name,
			Op:    "skinny_gemm",
			Entry: g.Name + "_launch",
			Launch: hopper.Launch{
				Block:        32 * warps,
				Warps:        warps,
				SplitK:       g.SplitK,
				RowsPerBlock: g.RowsPerBlock,
				Unroll:       g.Unroll,
				Prefetch:     g.Prefetch,
			},
		}, g.Name != ""
	case k.Tile != nil:
		if v, err := k.Tile.Variant(); err == nil {
			return v, true
		}
		t := *k.Tile
		block, warps := 256, 8
		if t.WarpSpecialized {
			block, warps = 384, 12
		}
		mma := "mma.sync"
		if t.WGMMA {
			mma = "wgmma"
		}
		return hopper.Variant{
			Name:  t.Name,
			Op:    "tile_gemm",
			Entry: t.Name + "_launch",
			Launch: hopper.Launch{
				Block:           block,
				Warps:           warps,
				Stages:          t.Stages,
				BM:              t.BM,
				BN:              t.BN,
				GroupM:          t.GroupM,
				StreamK:         t.StreamK,
				Swizzle128B:     t.Swizzle128B,
				WarpSpecialized: t.WarpSpecialized,
				SharedBytes:     t.SharedBytes(),
				MMA:             mma,
			},
		}, t.Name != ""
	case k.Attention != nil:
		if v, err := k.Attention.Variant(); err == nil {
			return v, true
		}
		a := *k.Attention
		return hopper.Variant{
			Name:  a.Name,
			Op:    "paged_attention",
			Entry: a.Name + "_launch",
			Launch: hopper.Launch{
				Block:          128,
				Warps:          4,
				Stages:         a.Stages,
				TableAhead:     a.TableAhead,
				SplitKV:        a.Chunk,
				KVDepth:        a.Depth,
				KVHeads:        a.KVHeadsPerBlock,
				DeviceSplits:   a.DeviceSplits,
				Swizzle:        a.Swizzle,
				ClusterCombine: a.ClusterCombine,
			},
		}, a.Name != ""
	case k.Row != nil:
		_, block := k.Row.Grid(max(k.Rows, 1))
		threads := max(block[0], 256)
		return hopper.Variant{
			Name:  k.Row.Name,
			Op:    "row_kernel",
			Entry: k.Row.Name + "_launch",
			Launch: hopper.Launch{
				Block: threads,
				Warps: (threads + 31) / 32,
			},
		}, k.Row.Name != ""
	case k.TopK != nil:
		_, block := k.TopK.Grid(max(k.Rows, 1))
		threads := max(block[0], 256)
		return hopper.Variant{
			Name:  k.TopK.Name,
			Op:    "row_topk",
			Entry: k.TopK.Name + "_launch",
			Launch: hopper.Launch{
				Block: threads,
				Warps: (threads + 31) / 32,
			},
		}, k.TopK.Name != ""
	case k.Align != nil:
		threads := max(k.Align.Warps, 4) * 32
		return hopper.Variant{
			Name:  k.Align.Name,
			Op:    "expert_align",
			Entry: k.Align.Name + "_launch",
			Launch: hopper.Launch{
				Block: threads,
				Warps: (threads + 31) / 32,
			},
		}, k.Align.Name != ""
	case k.ExpertSum != nil:
		_, block := k.ExpertSum.Grid(max(k.Rows, 1))
		threads := max(block[0], 256)
		return hopper.Variant{
			Name:  k.ExpertSum.Name,
			Op:    "expert_sum",
			Entry: k.ExpertSum.Name + "_launch",
			Launch: hopper.Launch{
				Block: threads,
				Warps: (threads + 31) / 32,
			},
		}, k.ExpertSum.Name != ""
	default:
		name := stepKernelName(k)
		if name == "" {
			return hopper.Variant{}, false
		}
		return hopper.Variant{
			Name:   name,
			Op:     stepOpKind(k),
			Entry:  name + "_launch",
			Launch: hopper.Launch{Block: 256, Warps: 8},
		}, true
	}
}

// FilterStructuralBeamByPTXAS filters a beam of structural plan candidates
// against parsed ptxas -v reports for arch, pruning any candidate whose kernel
// steps or persistent megakernel stages violate gate (register spills, WGMMA
// serialization C7519, MaxRegisters, or MinCTAsPerSM occupancy).
func FilterStructuralBeamByPTXAS(beam []StructuralBeamCandidate, reports map[string]ptxasinfo.Report, arch string, gate StaticGate) (kept []StructuralBeamCandidate, rejected []RejectedBeamCandidate) {
	for _, cand := range beam {
		variants := cand.Variants()
		if len(variants) == 0 {
			rejected = append(rejected, RejectedBeamCandidate{
				Candidate: cand,
				Reasons:   []string{"candidate has no lowerable kernel variants"},
			})
			continue
		}
		report := resolveCandidatePTXASReport(cand, variants, reports)
		_, rejVars := FilterHopperVariantsByPTXAS(variants, report, arch, gate)
		if len(rejVars) > 0 {
			reasons := make([]string, 0, len(rejVars))
			for _, rv := range rejVars {
				reasons = append(reasons, fmt.Sprintf("%s: %s", rv.Variant.Name, rv.Reason()))
			}
			rejected = append(rejected, RejectedBeamCandidate{
				Candidate:        cand,
				Reasons:          reasons,
				RejectedVariants: rejVars,
			})
			continue
		}
		kept = append(kept, cand)
	}
	return kept, rejected
}

func resolveCandidatePTXASReport(cand StructuralBeamCandidate, variants []hopper.Variant, reports map[string]ptxasinfo.Report) ptxasinfo.Report {
	if len(reports) == 0 {
		return ptxasinfo.Report{}
	}
	var merged ptxasinfo.Report
	appendReport := func(r ptxasinfo.Report) {
		merged.Targets = append(merged.Targets, r.Targets...)
	}

	candKeys := []string{cand.Name, cand.PlanSignature}
	if cand.Megakernel != nil && cand.Megakernel.Name != "" {
		candKeys = append(candKeys, cand.Megakernel.Name)
	}
	if cand.Plan != nil && cand.Plan.Name != "" {
		candKeys = append(candKeys, cand.Plan.Name)
	}
	seenKey := make(map[string]bool)
	foundCandReport := false
	for _, k := range candKeys {
		if k == "" || seenKey[k] {
			continue
		}
		seenKey[k] = true
		if r, ok := reports[k]; ok {
			appendReport(r)
			foundCandReport = true
		}
	}
	for _, v := range variants {
		if v.Name == "" || seenKey[v.Name] {
			continue
		}
		seenKey[v.Name] = true
		if r, ok := reports[v.Name]; ok {
			appendReport(r)
			foundCandReport = true
		}
	}
	if foundCandReport {
		return merged
	}
	for _, wildcard := range []string{"*", ""} {
		if r, ok := reports[wildcard]; ok {
			appendReport(r)
			return merged
		}
	}
	// Fall back to any report containing a function matching one of cand's variants.
	for k, r := range reports {
		if seenKey[k] {
			continue
		}
		matched := false
		for _, t := range r.Targets {
			for _, fn := range t.Functions {
				for _, v := range variants {
					if MatchesVariantFunction(fn.Name, v.Name) {
						matched = true
						break
					}
				}
				if matched {
					break
				}
			}
			if matched {
				break
			}
		}
		if matched {
			appendReport(r)
		}
	}
	return merged
}

// SelectStructuralBeamWithHoldout evaluates each surviving candidate in beam
// against baseline using the two-stage promotion gate ([SelectHoldoutPromoted] /
// SelectMeasuredWithHoldout: 3-block ABBA screen, >= 6-block held-out ABBA with
// one-sided sign-test p <= 0.05, A/A null calibration, caller admission/digest
// verification, and exact frozen-prompt greedy token-ID parity), returning the
// verified candidate with the highest Holdout.Ratio or falling back to baseline
// if none pass the holdout gate.
func SelectStructuralBeamWithHoldout(query MeasuredIdentity, baseline StructuralBeamCandidate, beam []StructuralBeamCandidate, evidence map[string]HoldoutCandidateEvidence, verify func(MeasuredPair) error) (StructuralBeamWinner, error) {
	fallback := func(status, reason string, evals []StructuralBeamEvaluation) StructuralBeamWinner {
		return StructuralBeamWinner{
			Winner:             baseline,
			Selected:           baseline,
			Promoted:           false,
			FallbackToBaseline: true,
			Status:             status,
			Reason:             reason,
			Holdout:            pairedtiming.Summary{Ratio: 1.0, MinRatio: 1.0},
			Evaluations:        evals,
		}
	}

	if query.Context == ([32]byte{}) || query.Admission == ([32]byte{}) {
		return fallback("NOT_EVALUATED", "complete context and trusted admission identities required", nil), nil
	}
	if verify == nil {
		return fallback("NOT_EVALUATED", "trusted caller admission verifier absent", nil), nil
	}
	if len(beam) == 0 {
		return fallback("FALLBACK_BASELINE", "empty structural candidate beam", nil), nil
	}

	var (
		evals        []StructuralBeamEvaluation
		haveBest     bool
		bestCand     StructuralBeamCandidate
		bestRes      HoldoutPromotionResult
		rejectNotes  []string
	)

	for _, cand := range beam {
		ev, handle, ok := lookupCandidateHoldoutEvidence(cand, baseline, evidence)
		if !ok {
			if cand.PlanSignature != "" && cand.PlanSignature == baseline.PlanSignature {
				continue
			}
			res := HoldoutPromotionResult{
				Status:         "NOT_EVALUATED",
				Reason:         fmt.Sprintf("no holdout evidence for candidate %q", cand.Handle()),
				CandidateState: "NOT_EVALUATED",
			}
			evals = append(evals, StructuralBeamEvaluation{
				Candidate: cand,
				Handle:    cand.Handle(),
				Result:    res,
				Promoted:  false,
			})
			rejectNotes = append(rejectNotes, fmt.Sprintf("%s: %s", cand.Handle(), res.Reason))
			continue
		}

		res, err := SelectHoldoutPromoted(query, ev, verify)
		if err != nil {
			return fallback("NOT_EVALUATED", err.Error(), evals), err
		}
		promoted := res.Status == "EXACT_HOLDOUT_PROMOTED" &&
			res.CandidateState == "PROMOTED_HOLDOUT_VERIFIED" &&
			res.TokensMatched &&
			res.HoldoutObserved != nil

		evals = append(evals, StructuralBeamEvaluation{
			Candidate: cand,
			Handle:    handle,
			Result:    res,
			Promoted:  promoted,
		})

		if !promoted {
			reason := res.Reason
			if reason == "" {
				reason = res.Status
			}
			rejectNotes = append(rejectNotes, fmt.Sprintf("%s: %s", handle, reason))
			continue
		}

		if !haveBest ||
			res.HoldoutObserved.Ratio > bestRes.HoldoutObserved.Ratio ||
			(res.HoldoutObserved.Ratio == bestRes.HoldoutObserved.Ratio && cand.EstimatedNs < bestCand.EstimatedNs) {
			haveBest = true
			bestCand = cand
			bestRes = res
		}
	}

	if !haveBest {
		reason := "no structural beam candidate passed holdout promotion gate"
		if len(rejectNotes) > 0 {
			reason = strings.Join(rejectNotes, "; ")
		}
		return fallback("FALLBACK_BASELINE", reason, evals), nil
	}

	winner := StructuralBeamWinner{
		Winner:             bestCand,
		Selected:           bestCand,
		Promoted:           true,
		FallbackToBaseline: false,
		Status:             bestRes.Status,
		Reason:             bestRes.Reason,
		ScreenObserved:     bestRes.ScreenObserved,
		HoldoutObserved:    bestRes.HoldoutObserved,
		AANullObserved:     bestRes.AANullObserved,
		OneSidedSignP:      bestRes.OneSidedSignP,
		TokensMatched:      bestRes.TokensMatched,
		Evaluations:        evals,
	}
	if bestRes.ScreenObserved != nil {
		winner.Screen = *bestRes.ScreenObserved
	}
	if bestRes.HoldoutObserved != nil {
		winner.Holdout = *bestRes.HoldoutObserved
	}
	return winner, nil
}

func lookupCandidateHoldoutEvidence(cand, baseline StructuralBeamCandidate, evidence map[string]HoldoutCandidateEvidence) (HoldoutCandidateEvidence, string, bool) {
	if len(evidence) == 0 {
		return HoldoutCandidateEvidence{}, "", false
	}
	keys := []string{cand.Name, cand.PlanSignature}
	if cand.Megakernel != nil && cand.Megakernel.Name != "" {
		keys = append(keys, cand.Megakernel.Name)
	}
	if cand.Plan != nil && cand.Plan.Name != "" {
		keys = append(keys, cand.Plan.Name)
	}
	for _, k := range keys {
		if k == "" {
			continue
		}
		if ev, ok := evidence[k]; ok {
			return normalizeHoldoutEvidenceHandles(ev, baseline, k), k, true
		}
	}
	for k, ev := range evidence {
		for _, ck := range keys {
			if ck != "" && ev.Screen.Candidate == ck {
				return normalizeHoldoutEvidenceHandles(ev, baseline, ck), k, true
			}
		}
	}
	return HoldoutCandidateEvidence{}, "", false
}

func normalizeHoldoutEvidenceHandles(ev HoldoutCandidateEvidence, baseline StructuralBeamCandidate, candHandle string) HoldoutCandidateEvidence {
	if ev.Screen.Reference == "" {
		ref := baseline.Handle()
		if ref == "" || ref == candHandle {
			ref = "baseline_plan"
		}
		ev.Screen.Reference = ref
	}
	if ev.Screen.Candidate == "" {
		ev.Screen.Candidate = candHandle
	}
	return ev
}

func cloneBeamCandidate(c StructuralBeamCandidate) StructuralBeamCandidate {
	out := c
	if c.Plan != nil {
		out.Plan = clonePlan(c.Plan)
	}
	if c.Megakernel != nil {
		m := *c.Megakernel
		m.Inputs = slices.Clone(c.Megakernel.Inputs)
		m.Temps = slices.Clone(c.Megakernel.Temps)
		m.Stages = slices.Clone(c.Megakernel.Stages)
		out.Megakernel = &m
	}
	out.StepProposals = slices.Clone(c.StepProposals)
	return out
}
