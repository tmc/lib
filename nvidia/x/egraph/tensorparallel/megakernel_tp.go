package tensorparallel

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// TPMegakernelConfig configures cross-layer tensor-parallel megakernel
// construction across TP ranks.
type TPMegakernelConfig struct {
	// TP is the tensor-parallel world size (number of ranks; must be >= 1).
	TP int

	// Prefix is the base identifier prefix for generated plans and kernels.
	// When empty, it defaults to "tp_mega".
	Prefix string

	// Fusion selects the in-GEMM epilogue reduction mode (FuseReduceAdd or
	// FuseMultimem). When FuseNone (0), it defaults to FuseReduceAdd (or
	// FuseMultimem when BarrierMode == hopper.SysBarrierMultimem).
	Fusion Fusion

	// Target is the SM target architecture (e.g. "sm_90a", "sm_100a", "sm_121a").
	// When empty and Fusion == FuseMultimem, it defaults to "sm_90a".
	Target string

	// BarrierMode selects the system-scope (.sys) ready-counter synchronization
	// mode across ranks (SysBarrierPeerAtomic or SysBarrierMultimem). When
	// Fusion == FuseMultimem, BarrierMode automatically uses SysBarrierMultimem.
	BarrierMode hopper.SysBarrierMode
}

func (c TPMegakernelConfig) norm() (TPMegakernelConfig, error) {
	if c.TP < 1 {
		return c, fmt.Errorf("tensorparallel megakernel: tp %d < 1", c.TP)
	}
	c.Prefix = strings.TrimSpace(c.Prefix)
	if c.Prefix == "" {
		c.Prefix = "tp_mega"
	}
	if c.BarrierMode > hopper.SysBarrierMultimem {
		return c, fmt.Errorf("tensorparallel megakernel: bad barrier mode %v", c.BarrierMode)
	}
	if c.Fusion == FuseNone {
		if c.BarrierMode == hopper.SysBarrierMultimem {
			c.Fusion = FuseMultimem
		} else {
			c.Fusion = FuseReduceAdd
		}
	}
	if c.Fusion != FuseReduceAdd && c.Fusion != FuseMultimem {
		return c, fmt.Errorf("tensorparallel megakernel: unsupported fusion %v (want FuseReduceAdd or FuseMultimem)", c.Fusion)
	}
	if c.Fusion == FuseMultimem {
		c.BarrierMode = hopper.SysBarrierMultimem
		if strings.TrimSpace(c.Target) == "" {
			c.Target = "sm_90a"
		}
	}
	return c, nil
}

// A TPCrossLayerBoundary holds one rank's fused cross-layer plan and persistent
// megakernel bridging Layer[L].OProj + Layer[L].FFN + Layer[L+1].AttnIn.
type TPCrossLayerBoundary struct {
	LayerL              int
	LayerNext           int
	PostPlan            *hopper.Plan
	FusedPlan           *hopper.Plan
	Megakernel          hopper.Megakernel
	OProjBarrierSlot    int
	DownProjBarrierSlot int
	SysBarrierSlots     []int
	SysBarriers         []hopper.SysEpochBarrierConfig
}

// A TPMegakernelPlan holds one tensor-parallel rank's sharded plans, fused
// cross-layer megakernel, and system-scope (.sys) ready barrier descriptors.
type TPMegakernelPlan struct {
	Rank          int
	WorldSize     int
	Prefix        string
	Fusion        Fusion
	Target        string
	BarrierMode   hopper.SysBarrierMode
	ShardL        *Plan
	ShardNext     *Plan
	RankShardL    RankPlan
	RankShardNext RankPlan

	// FirstAttnIn is rank r's pre-attention QKV projection plan for the first
	// layer in the stack.
	FirstAttnIn *hopper.Plan

	// PostPlan is rank r's fused Layer[L].OProj + Layer[L].FFN plan before
	// cross-layer fusion with Layer[L+1].AttnIn.
	PostPlan *hopper.Plan

	// FusedPlan is rank r's fused cross-layer plan
	// (Layer[L].OProj + Layer[L].FFN + Layer[L+1].AttnIn).
	FusedPlan *hopper.Plan

	// Megakernel is the persistent megakernel lowered from FusedPlan.
	Megakernel hopper.Megakernel

	// OProjBarrierSlot and DownProjBarrierSlot are the ready counter indices in
	// Megakernel that synchronize across all WorldSize ranks at system scope
	// (.sys) after o_proj and down_proj respectively.
	OProjBarrierSlot    int
	DownProjBarrierSlot int
	SysBarrierSlots     []int
	SysBarriers         []hopper.SysEpochBarrierConfig

	// Boundaries holds all adjacent cross-layer boundaries when built from a
	// multi-layer stack via BuildTPMegakernel.
	Boundaries []TPCrossLayerBoundary

	// FinalPost is rank r's fused OProj + FFN plan for the final layer in the stack.
	FinalPost *hopper.Plan

	// BaseReadyWords is the per-rank ready counter stride, and TotalReadyWords
	// is WorldSize * BaseReadyWords for a shared multi-rank ready buffer.
	BaseReadyWords  int
	TotalReadyWords int
}

// BuildCrossLayerTPMegakernel shards layerL and layerNext across cfg.TP ranks
// and builds one persistent cross-layer [*TPMegakernelPlan] per rank r in
// [0, cfg.TP-1], fusing Layer[L].OProj + Layer[L].FFN + Layer[L+1].AttnIn with
// in-GEMM epilogue reductions (StoreReduceAdd or StoreMultimemReduceAdd) and
// system-scope (.sys) ready barriers between o_proj -> ffn and
// down_proj -> next_attn_in.
func BuildCrossLayerTPMegakernel(cfg TPMegakernelConfig, layerL, layerNext Layer) ([]*TPMegakernelPlan, error) {
	return BuildTPMegakernel(cfg, []Layer{layerL, layerNext})
}

// BuildTPMegakernel shards each layer in layers across cfg.TP ranks and builds
// each rank's cross-layer [*TPMegakernelPlan] across adjacent decoder layers.
// At least two layers are required to form a cross-layer boundary.
func BuildTPMegakernel(cfg TPMegakernelConfig, layers []Layer) ([]*TPMegakernelPlan, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return nil, err
	}
	if len(layers) < 2 {
		return nil, fmt.Errorf("tensorparallel megakernel: need at least 2 layers for cross-layer fusion, have %d", len(layers))
	}

	sharded := make([]*Plan, len(layers))
	for i, l := range layers {
		layerPrefix := fmt.Sprintf("%s_l%d", cfg.Prefix, i)
		sp, err := Shard(l, Options{
			TP:     cfg.TP,
			Prefix: layerPrefix,
			Fusion: cfg.Fusion,
			Target: cfg.Target,
		})
		if err != nil {
			return nil, fmt.Errorf("tensorparallel megakernel layer %d: %w", i, err)
		}
		sharded[i] = sp
	}

	numBoundaries := len(layers) - 1
	plans := make([]*TPMegakernelPlan, cfg.TP)
	for r := range cfg.TP {
		boundaries := make([]TPCrossLayerBoundary, numBoundaries)
		for bIdx := 0; bIdx < numBoundaries; bIdx++ {
			bnd, err := buildRankCrossLayerBoundary(cfg, r, bIdx, bIdx+1, sharded[bIdx], sharded[bIdx+1])
			if err != nil {
				return nil, fmt.Errorf("tensorparallel megakernel rank %d boundary %d->%d: %w", r, bIdx, bIdx+1, err)
			}
			boundaries[bIdx] = bnd
		}

		lastIdx := len(layers) - 1
		finalPost, err := fuseRankOProjAndFFN(sharded[lastIdx].Ranks[r], fmt.Sprintf("%s_r%d_l%d_post", cfg.Prefix, r, lastIdx), fmt.Sprintf("l%d_", lastIdx))
		if err != nil {
			return nil, fmt.Errorf("tensorparallel megakernel rank %d final post: %w", r, err)
		}

		primary := boundaries[0]
		baseWords := primary.Megakernel.ReadyWords()
		plans[r] = &TPMegakernelPlan{
			Rank:                r,
			WorldSize:           cfg.TP,
			Prefix:              cfg.Prefix,
			Fusion:              cfg.Fusion,
			Target:              cfg.Target,
			BarrierMode:         cfg.BarrierMode,
			ShardL:              sharded[0],
			ShardNext:           sharded[1],
			RankShardL:          sharded[0].Ranks[r],
			RankShardNext:       sharded[1].Ranks[r],
			FirstAttnIn:         cloneHopperPlan(sharded[0].Ranks[r].AttnIn),
			PostPlan:            primary.PostPlan,
			FusedPlan:           primary.FusedPlan,
			Megakernel:          primary.Megakernel,
			OProjBarrierSlot:    primary.OProjBarrierSlot,
			DownProjBarrierSlot: primary.DownProjBarrierSlot,
			SysBarrierSlots:     slices.Clone(primary.SysBarrierSlots),
			SysBarriers:         slices.Clone(primary.SysBarriers),
			Boundaries:          boundaries,
			FinalPost:           finalPost,
			BaseReadyWords:      baseWords,
			TotalReadyWords:     cfg.TP * baseWords,
		}
	}
	return plans, nil
}

func buildRankCrossLayerBoundary(cfg TPMegakernelConfig, rank, l0Idx, l1Idx int, shardL, shardNext *Plan) (TPCrossLayerBoundary, error) {
	rpL := shardL.Ranks[rank]
	rpNext := shardNext.Ranks[rank]

	l0Prefix := fmt.Sprintf("l%d_", l0Idx)
	l1Prefix := fmt.Sprintf("l%d_", l1Idx)
	postName := fmt.Sprintf("%s_r%d_l%d_post", cfg.Prefix, rank, l0Idx)

	postPlan, err := fuseRankOProjAndFFN(rpL, postName, l0Prefix)
	if err != nil {
		return TPCrossLayerBoundary{}, err
	}

	nextAttnIn := cloneHopperPlan(rpNext.AttnIn)
	if len(nextAttnIn.Kernels) == 2 {
		setHopperKernelName(&nextAttnIn.Kernels[0], l1Prefix+"attn_norm")
		setHopperKernelName(&nextAttnIn.Kernels[1], l1Prefix+"qkv_proj")
	}
	renameMap := map[string]string{
		nextAttnIn.Output.Name: l1Prefix + "qkv",
	}
	for _, in := range nextAttnIn.Inputs {
		if in.Name != "x" {
			renameMap[in.Name] = l1Prefix + in.Name
		}
	}
	for _, tmp := range nextAttnIn.Temps {
		renameMap[tmp.Name] = l1Prefix + tmp.Name
	}
	renameHopperPlanTensors(nextAttnIn, renameMap)

	megaName := fmt.Sprintf("%s_r%d_l%d_l%d_mega", cfg.Prefix, rank, l0Idx, l1Idx)
	glue := hopper.CrossLayerGlue{
		Name:               megaName,
		LayerLPrefix:       l0Prefix,
		LayerNextPrefix:    l1Prefix,
		BridgeInput:        "x",
		RetainBridgeOutput: true,
	}
	fusedPlan, mega, err := hopper.BuildCrossLayerMegakernel(postPlan, nextAttnIn, glue)
	if err != nil {
		return TPCrossLayerBoundary{}, err
	}

	var sysSlots []int
	var sysBarriers []hopper.SysEpochBarrierConfig
	oSlot := -1
	downSlot := -1

	for i, st := range mega.Stages {
		if st.Kernel.GEMM != nil && (st.Kernel.GEMM.Store == hopper.StoreReduceAdd || st.Kernel.GEMM.Store == hopper.StoreMultimemReduceAdd) {
			if st.SignalSlot < 0 {
				return TPCrossLayerBoundary{}, fmt.Errorf("stage %d (%s) has reducing store %v but no signal slot", i, st.Name, st.Kernel.GEMM.Store)
			}
			slot := st.SignalSlot
			sysSlots = append(sysSlots, slot)
			if oSlot < 0 {
				oSlot = slot
			} else {
				downSlot = slot
			}
			sysBarriers = append(sysBarriers, hopper.SysEpochBarrierConfig{
				Name:          fmt.Sprintf("%s_slot%d_barrier", mega.Name, slot),
				WorldSize:     cfg.TP,
				BlocksPerRank: st.Blocks,
				Slot:          slot,
				Mode:          cfg.BarrierMode,
			})
		}
	}
	if oSlot < 0 || downSlot < 0 {
		return TPCrossLayerBoundary{}, fmt.Errorf("expected 2 reducing GEMM stages (o_proj and down_proj), got slots %v", sysSlots)
	}

	return TPCrossLayerBoundary{
		LayerL:              l0Idx,
		LayerNext:           l1Idx,
		PostPlan:            postPlan,
		FusedPlan:           fusedPlan,
		Megakernel:          mega,
		OProjBarrierSlot:    oSlot,
		DownProjBarrierSlot: downSlot,
		SysBarrierSlots:     sysSlots,
		SysBarriers:         sysBarriers,
	}, nil
}

// fuseRankOProjAndFFN fuses one rank's OProj and FFN plans across the reduced
// intermediate hidden buffer (prefix+"h").
func fuseRankOProjAndFFN(rp RankPlan, name, prefix string) (*hopper.Plan, error) {
	if rp.OProj == nil || len(rp.OProj.Kernels) == 0 {
		return nil, errors.New("empty OProj plan")
	}
	if rp.FFN == nil || len(rp.FFN.Kernels) == 0 {
		return nil, errors.New("empty FFN plan")
	}

	oproj := cloneHopperPlan(rp.OProj)
	ffn := cloneHopperPlan(rp.FFN)

	hBridgeName := prefix + "h"
	xOutName := prefix + "x_out"

	oRename := map[string]string{
		oproj.Output.Name: hBridgeName,
	}
	for _, in := range oproj.Inputs {
		oRename[in.Name] = prefix + in.Name
	}
	for _, tmp := range oproj.Temps {
		oRename[tmp.Name] = prefix + "o_" + tmp.Name
	}
	renameHopperPlanTensors(oproj, oRename)

	if len(oproj.Kernels) == 2 {
		setHopperKernelName(&oproj.Kernels[0], prefix+"attn_cast")
		setHopperKernelName(&oproj.Kernels[1], prefix+"o_proj")
	} else if len(oproj.Kernels) == 1 {
		setHopperKernelName(&oproj.Kernels[0], prefix+"o_proj")
	}

	fRename := map[string]string{
		"h":             hBridgeName,
		ffn.Output.Name: xOutName,
	}
	for _, in := range ffn.Inputs {
		if in.Name != "h" {
			fRename[in.Name] = prefix + in.Name
		}
	}
	for _, tmp := range ffn.Temps {
		fRename[tmp.Name] = prefix + "ffn_" + tmp.Name
	}
	renameHopperPlanTensors(ffn, fRename)

	if len(ffn.Kernels) == 3 {
		setHopperKernelName(&ffn.Kernels[0], prefix+"ffn_norm")
		setHopperKernelName(&ffn.Kernels[1], prefix+"gate_up")
		setHopperKernelName(&ffn.Kernels[2], prefix+"down_proj")
	}

	var inputs []hopper.Tensor
	seenIn := make(map[string]bool)
	for _, in := range oproj.Inputs {
		seenIn[in.Name] = true
		inputs = append(inputs, in)
	}
	for _, in := range ffn.Inputs {
		if in.Name == hBridgeName || seenIn[in.Name] {
			continue
		}
		seenIn[in.Name] = true
		inputs = append(inputs, in)
	}

	var temps []hopper.Tensor
	temps = append(temps, oproj.Temps...)
	temps = append(temps, oproj.Output)
	temps = append(temps, ffn.Temps...)

	kernels := make([]hopper.PlanKernel, 0, len(oproj.Kernels)+len(ffn.Kernels))
	kernels = append(kernels, oproj.Kernels...)
	kernels = append(kernels, ffn.Kernels...)

	return &hopper.Plan{
		Name:    name,
		Inputs:  inputs,
		Temps:   temps,
		Output:  ffn.Output,
		Kernels: kernels,
	}, nil
}

// ReadySlot returns the index in the shared ready counter buffer for stageSlot
// on rank p.Rank. System-scope barrier slots (o_proj and down_proj) are shared
// across all ranks at index stageSlot; intra-rank slots are offset by
// p.Rank * p.BaseReadyWords.
func (p *TPMegakernelPlan) ReadySlot(stageSlot int) int {
	if slices.Contains(p.SysBarrierSlots, stageSlot) {
		return stageSlot
	}
	return p.Rank*p.BaseReadyWords + stageSlot
}

// CUDA emits the rank's persistent cross-layer megakernel device source with
// [hopper.ReadySysPrelude] and system-scope (.sys) barriers
// (eg_ready_signal_sys / eg_ready_wait_sys across WorldSize * Blocks) at the
// o_proj -> ffn and down_proj -> next_attn_in boundaries.
func (p *TPMegakernelPlan) CUDA() (string, error) {
	if p == nil {
		return "", errors.New("tensorparallel megakernel: nil plan")
	}
	raw, err := p.Megakernel.CUDA()
	if err != nil {
		return "", err
	}
	return p.rewriteSysBarriers(raw), nil
}

// Launcher emits the host launch helpers (<Name>_launch and <Name>_run) for
// this rank's persistent megakernel.
func (p *TPMegakernelPlan) Launcher() (string, error) {
	if p == nil {
		return "", errors.New("tensorparallel megakernel: nil plan")
	}
	return p.Megakernel.Launcher()
}

// Source emits the complete CUDA translation unit ([hopper.Prelude],
// [hopper.ReadySysPrelude], stage device functions, persistent cross-layer
// megakernel with .sys barriers, and host launchers).
func (p *TPMegakernelPlan) Source() (string, error) {
	if p == nil {
		return "", errors.New("tensorparallel megakernel: nil plan")
	}
	cu, err := p.CUDA()
	if err != nil {
		return "", err
	}
	l, err := p.Launcher()
	if err != nil {
		return "", err
	}
	return hopper.Prelude + "\n" + cu + "\n" + l, nil
}

func (p *TPMegakernelPlan) rewriteSysBarriers(src string) string {
	// Replace single-GPU ReadyPrelude with ReadySysPrelude(p.BarrierMode).
	src = strings.Replace(src, hopper.ReadyPrelude, hopper.ReadySysPrelude(p.BarrierMode), 1)

	// Rewrite cross-rank .sys barrier slots (o_proj and down_proj) to use
	// eg_ready_signal_sys and eg_ready_wait_sys with WorldSize * st.Blocks.
	for _, slot := range p.SysBarrierSlots {
		if slot < 0 || slot >= len(p.Megakernel.Stages) {
			continue
		}
		st := p.Megakernel.Stages[slot]
		totalNeed := p.WorldSize * st.Blocks

		for _, out := range st.Outputs {
			oldStage := fmt.Sprintf("EG_READY_STAGE(&ready[%d], epoch, %d, %s,", slot, st.Blocks, out.Name)
			newStage := fmt.Sprintf("EG_READY_STAGE(&ready[%d], epoch, %d, %s,", slot, totalNeed, out.Name)
			src = strings.ReplaceAll(src, oldStage, newStage)
		}

		oldSig := fmt.Sprintf("eg_ready_signal(&ready[%d], epoch, 1);", slot)
		newSig := fmt.Sprintf("eg_ready_signal_sys(&ready[%d], epoch, 1);", slot)
		src = strings.ReplaceAll(src, oldSig, newSig)

		oldWait := fmt.Sprintf("eg_ready_wait(&ready[%d], epoch, %d);", slot, st.Blocks)
		newWait := fmt.Sprintf("eg_ready_wait_sys(&ready[%d], epoch, %d);", slot, totalNeed)
		src = strings.ReplaceAll(src, oldWait, newWait)
	}

	// Offset rank-local intra-rank slots when Rank > 0 so all TP ranks can
	// share a single ready counter array without intra-rank slot collisions.
	if p.Rank > 0 && p.BaseReadyWords > 0 {
		epochSlot := p.Megakernel.EpochSlot()
		oldEpoch := fmt.Sprintf("eg_ready_resolve_epoch(&ready[%d],", epochSlot)
		newEpoch := fmt.Sprintf("eg_ready_resolve_epoch(&ready[%d],", p.ReadySlot(epochSlot))
		src = strings.ReplaceAll(src, oldEpoch, newEpoch)

		for s := len(p.Megakernel.Stages) - 1; s >= 0; s-- {
			if slices.Contains(p.SysBarrierSlots, s) {
				continue
			}
			mapped := p.ReadySlot(s)
			st := p.Megakernel.Stages[s]
			for _, out := range st.Outputs {
				oldStage := fmt.Sprintf("EG_READY_STAGE(&ready[%d], epoch, %d, %s,", s, st.Blocks, out.Name)
				newStage := fmt.Sprintf("EG_READY_STAGE(&ready[%d], epoch, %d, %s,", mapped, st.Blocks, out.Name)
				src = strings.ReplaceAll(src, oldStage, newStage)
			}
			oldSig := fmt.Sprintf("eg_ready_signal(&ready[%d], epoch, 1);", s)
			newSig := fmt.Sprintf("eg_ready_signal(&ready[%d], epoch, 1);", mapped)
			src = strings.ReplaceAll(src, oldSig, newSig)

			oldWait := fmt.Sprintf("eg_ready_wait(&ready[%d], epoch, %d);", s, st.Blocks)
			newWait := fmt.Sprintf("eg_ready_wait(&ready[%d], epoch, %d);", mapped, st.Blocks)
			src = strings.ReplaceAll(src, oldWait, newWait)
		}
	}

	return src
}

func cloneHopperPlan(p *hopper.Plan) *hopper.Plan {
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
	for i, in := range p.Inputs {
		out.Inputs[i] = hopper.Tensor{Name: in.Name, DType: in.DType, Shape: slices.Clone(in.Shape)}
	}
	for i, t := range p.Temps {
		out.Temps[i] = hopper.Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, k := range p.Kernels {
		ck := k
		ck.Args = slices.Clone(k.Args)
		if k.GEMM != nil {
			g := *k.GEMM
			g.Extras = slices.Clone(k.GEMM.Extras)
			if k.GEMM.Tail != nil {
				tr := *k.GEMM.Tail
				tr.Inputs = slices.Clone(tr.Inputs)
				tr.Values = slices.Clone(tr.Values)
				tr.Outputs = slices.Clone(tr.Outputs)
				g.Tail = &tr
			}
			ck.GEMM = &g
		}
		if k.Row != nil {
			r := *k.Row
			r.Inputs = slices.Clone(k.Row.Inputs)
			r.Values = make([]hopper.RowValue, len(k.Row.Values))
			for j, v := range k.Row.Values {
				v.Args = slices.Clone(v.Args)
				r.Values[j] = v
			}
			r.Outputs = slices.Clone(k.Row.Outputs)
			ck.Row = &r
		}
		out.Kernels[i] = ck
	}
	return out
}

func setHopperKernelName(k *hopper.PlanKernel, name string) {
	switch {
	case k.GEMM != nil:
		k.GEMM.Name = name
	case k.Row != nil:
		k.Row.Name = name
	case k.Tile != nil:
		k.Tile.Name = name
	}
}

func renameHopperPlanTensors(p *hopper.Plan, rename map[string]string) {
	mapName := func(s string) string {
		if r, ok := rename[s]; ok && r != "" {
			return r
		}
		return s
	}
	for i := range p.Inputs {
		p.Inputs[i].Name = mapName(p.Inputs[i].Name)
	}
	for i := range p.Temps {
		p.Temps[i].Name = mapName(p.Temps[i].Name)
	}
	if p.Output.Name != "" {
		p.Output.Name = mapName(p.Output.Name)
	}
	for i := range p.Kernels {
		k := &p.Kernels[i]
		for j, a := range k.Args {
			k.Args[j] = mapName(a)
		}
		if k.GEMM != nil {
			for j := range k.GEMM.Extras {
				k.GEMM.Extras[j].Name = mapName(k.GEMM.Extras[j].Name)
			}
		}
		if k.Row != nil {
			for j := range k.Row.Inputs {
				buf := k.Row.Inputs[j].Buffer
				if buf == "" {
					buf = k.Row.Inputs[j].Name
				}
				mapped := mapName(buf)
				if k.Row.Inputs[j].Buffer != "" || mapped != k.Row.Inputs[j].Name {
					k.Row.Inputs[j].Buffer = mapped
				}
			}
			for j := range k.Row.Outputs {
				k.Row.Outputs[j].Name = mapName(k.Row.Outputs[j].Name)
				if k.Row.Outputs[j].Index != "" {
					k.Row.Outputs[j].Index = mapName(k.Row.Outputs[j].Index)
				}
			}
		}
	}
}
