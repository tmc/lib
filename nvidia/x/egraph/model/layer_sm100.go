package model

import (
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// ExtendedLayerBundle holds the base [CompiledLayer] together with the
// hardware-specialized kernels, PDL launch chain, MTP attention descriptor,
// verified sm_100a TMEM traces, CLC persistent kernels, and multi-GPU NVLink
// collective sources produced by [CompileExtendedLayer].
type ExtendedLayerBundle struct {
	Base             *CompiledLayer
	Mode             string
	PDL              *hopper.PDLPlanSource
	MTPAttention     *hopper.MTPPagedAttention
	TCGen05Kernels   []string
	TCGen05Traces    [][]cudatmem.TraceEvent
	CLCKernels       []string
	CollectiveSource string
}

// CompileExtendedLayer compiles a transformer decoder layer under cfg and opts,
// wiring together Programmatic Dependent Launch (PDL), speculative tree-verify
// MTP attention, Blackwell sm_100a tcgen05 (with formal cudatmem trace
// verification) and CLC persistent work-stealing GEMMs, and NVLink multi-GPU
// TP/EP collectives.
func CompileExtendedLayer(cfg ArchConfig, opts ExtendedArchOptions) (*ExtendedLayerBundle, error) {
	var (
		err    error
		sm100a bool
	)
	cfg, opts, sm100a, err = ValidateExtendedArchOptions(cfg, opts)
	if err != nil {
		return nil, err
	}
	dev := opts.Device
	seqLen := opts.SeqLen
	baseCfg := cfg
	if baseCfg.WeightDType == E4M3 && baseCfg.KVDType == E4M3 {
		baseCfg.WeightDType = BF16
	}
	base, err := CompileLayer(baseCfg, opts.BatchM, seqLen, dev)
	if err != nil && sm100a {
		base, err = CompileLayer(baseCfg, opts.BatchM, seqLen, "NVIDIA GB10")
		if err == nil {
			base.Device = dev
			prof, err := cudadevicetable.LookupProfile("sm_100a")
			if err != nil {
				return nil, err
			}
			ceil := SM100ACostCeiling(dev)
			sm100Cost := hopper.CostConfig{
				Device:       dev,
				Residency:    ceil.Residency,
				Ceiling:      ceil,
				TensorTFLOPS: 2250.0,
				SMs:          int(prof.SMs.Value),
				LaunchNs:     2000.0,
				MaxKeys:      seqLen,
			}
			if err := selectArchSchedules(base.Plan, cfg, opts.BatchM, sm100Cost); err != nil {
				return nil, err
			}
			base.Estimates = estimatePlanSteps(base.Plan, sm100Cost)
			var totalNs float64
			for _, est := range base.Estimates {
				totalNs += est.EstimatedNs
			}
			base.TotalNs = totalNs
		}
	}
	if err != nil {
		return nil, err
	}
	base.Config = cfg

	bundle := &ExtendedLayerBundle{
		Base: base,
		Mode: base.Mode,
	}

	hopperOrBlackwell := sm100a || IsHopperOrBlackwellDevice(dev)

	// 1. Programmatic Dependent Launch (PDL) chain for medium-batch Hopper/Blackwell.
	if opts.EnablePDL || (hopperOrBlackwell && opts.BatchM > 8 && opts.BatchM < 64) {
		pdlSrc, err := emitLayerPDL(base.Plan, base.Plan.Name+"_pdl")
		if err != nil {
			return nil, err
		}
		bundle.PDL = pdlSrc
		bundle.Mode = ModePDLChain
	}

	// 2. Multi-Token Prediction (MTP) / speculative tree-verify paged attention.
	if opts.QTokensPerSeq > 1 {
		headDim := cfg.HeadDim
		if headDim <= 0 {
			headDim = cfg.HiddenSize / cfg.NumAttentionHeads
		}
		const pageSize = 16
		maxPages := max((seqLen+pageSize-1)/pageSize, 1)
		g := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
		wgmma := cfg.KVDType == E4M3 && headDim%32 == 0 && g*opts.QTokensPerSeq >= 8
		mtp := hopper.MTPPagedAttention{
			Name:          base.Plan.Name + "_mtp_paged_attn",
			NumHeads:      cfg.NumAttentionHeads,
			KVHeads:       cfg.NumKeyValueHeads,
			HeadDim:       headDim,
			PageSize:      pageSize,
			MaxPages:      maxPages,
			QTokensPerSeq: opts.QTokensPerSeq,
			KVFP8:         cfg.KVDType == E4M3,
			Gate:          cfg.AttentionGate,
			Swizzle:       true,
			WGMMA:         wgmma,
		}
		mtp, err = mtp.Norm()
		if err != nil {
			return nil, err
		}
		if _, err := mtp.Source(); err != nil {
			return nil, err
		}
		bundle.MTPAttention = &mtp
	}

	// 3. Datacenter Blackwell sm_100a (NVIDIA B200 / NVIDIA GB200) tcgen05 & CLC.
	if sm100a {
		tcKernels, tcTraces, ctaGroup, err := EmitSM100TCGen05Suite(base.Plan.Name, cfg, opts)
		if err != nil {
			return nil, err
		}
		bundle.TCGen05Kernels = tcKernels
		bundle.TCGen05Traces = tcTraces

		if (opts.EnableCLC && opts.BatchM >= 64) || opts.BatchM >= 128 {
			var clcTiles []CLCTileSpec
			for idx, pk := range base.Plan.Kernels {
				if pk.Tile != nil && !pk.Tile.Grouped {
					clcTiles = append(clcTiles, CLCTileSpec{
						Index:  idx,
						Name:   pk.Tile.Name,
						N:      pk.Tile.N,
						K:      pk.Tile.K,
						GLU:    pk.Tile.GLU,
						Weight: WeightFormat(pk.Tile.Weight),
					})
				}
			}
			clcKernels, err := EmitLayerCLCKernels(base.Plan.Name, clcTiles, cfg)
			if err != nil {
				return nil, err
			}
			bundle.CLCKernels = clcKernels
		}

		if ctaGroup == 2 && len(bundle.CLCKernels) > 0 {
			bundle.Mode = ModeBlackwellCTA2CLC
		} else if bundle.PDL == nil || !opts.EnablePDL {
			bundle.Mode = ModeBlackwellTCGen05
		}
	}

	// 4. Multi-GPU NVLink TP and EP collectives.
	if opts.TPRanks > 1 || opts.EPRanks > 1 {
		colSrc, err := EmitLayerCollectives(hopper.Prelude, cfg, opts.TPRanks, opts.EPRanks)
		if err != nil {
			return nil, err
		}
		bundle.CollectiveSource = colSrc
	}

	base.Mode = bundle.Mode
	base.ExecutionMode = bundle.Mode
	return bundle, nil
}

// emitLayerPDL lowers p (or its longest PDL-compatible non-attention segment)
// via Plan.EmitPDL.
func emitLayerPDL(p *hopper.Plan, pdlName string) (*hopper.PDLPlanSource, error) {
	if p == nil || len(p.Kernels) == 0 {
		return nil, fmt.Errorf("compile extended layer: empty plan for PDL")
	}
	if src, err := p.EmitPDL(pdlName); err == nil {
		return src, nil
	}

	tensorByName := make(map[string]hopper.Tensor)
	for _, t := range p.Inputs {
		tensorByName[t.Name] = t
	}
	for _, t := range p.Temps {
		tensorByName[t.Name] = t
	}
	if p.Output.Name != "" {
		tensorByName[p.Output.Name] = p.Output
	}

	var segments [][]hopper.PlanKernel
	var cur []hopper.PlanKernel
	flush := func() {
		if len(cur) >= 2 {
			segments = append(segments, slices.Clone(cur))
		}
		cur = nil
	}
	for _, pk := range p.Kernels {
		if pk.Attention != nil || pk.TopK != nil || pk.Align != nil || (pk.Tile != nil && pk.Tile.Grouped) {
			flush()
			continue
		}
		cp := pk
		if cp.Row != nil {
			rk := *cp.Row
			rk.Cluster = false
			cp.Row = &rk
		}
		if cp.Tile != nil {
			tk := *cp.Tile
			tk.WarpSpecialized = false
			if tk.Repacked {
				tk.Repacked = false
				tk.DequantWords = 0
				if tk.N%128 != 0 || (tk.GLU && (tk.N/2)%64 != 0) {
					tk.BN = 64
				}
			}
			cp.Tile = &tk
		}
		cur = append(cur, cp)
	}
	flush()

	if len(segments) == 0 {
		return nil, fmt.Errorf("compile extended layer: no PDL-compatible segment in plan %s", p.Name)
	}
	best := segments[len(segments)-1]
	for _, seg := range segments {
		if len(seg) > len(best) {
			best = seg
		}
	}
	sub := buildSubPlan(pdlName, best, tensorByName)
	return sub.EmitPDL(pdlName)
}
