package hopper

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// Extended execution mode names selected by [CompileExtendedLayer].
const (
	ModePDLChain         = "pdl_chain"
	ModeBlackwellTCGen05 = "blackwell_tcgen05"
	ModeBlackwellCTA2CLC = "blackwell_cta2_clc"
)

// ExtendedArchOptions configures hardware-specific lowering passes on top of
// [ArchConfig] in [CompileExtendedLayer], including Programmatic Dependent
// Launch (PDL), speculative multi-token / tree-verify paged attention (MTP),
// Blackwell sm_100a Tensor Memory (tcgen05) and Cluster Launch Control (CLC)
// kernels, and multi-GPU NVLink TP/EP collectives.
type ExtendedArchOptions struct {
	Device        string   // e.g., "NVIDIA H200", "NVIDIA GB10", "NVIDIA B200", "NVIDIA GB200"
	BatchM        int      // token / sequence batch size M
	SeqLen        int      // maximum KV sequence length
	QTokensPerSeq int      // > 1 selects MTPPagedAttention
	DraftMask     []uint64 // optional bitpacked draft mask for MTP
	TPRanks       int      // > 1 appends AllReduceResidualRMSNorm
	EPRanks       int      // > 1 uses NVLinkMoECombine
	EnablePDL     bool     // when true (or automatically when 8 < BatchM < 64 on Hopper/Blackwell), emits PDLPlanSource
	EnableCLC     bool     // when true on B200/GB200 and BatchM >= 64, emits CLCPersistentTileGEMM
	CTAGroup      int      // 1 or 2 for B200/GB200 tcgen05 (defaults to 2 when BatchM >= 256 on B200/GB200)
}

// ExtendedLayerBundle holds the base [CompiledLayer] together with the
// hardware-specialized kernels, PDL launch chain, MTP attention descriptor,
// verified sm_100a TMEM traces, CLC persistent kernels, and multi-GPU NVLink
// collective sources produced by [CompileExtendedLayer].
type ExtendedLayerBundle struct {
	Base             *CompiledLayer
	Mode             string
	PDL              *PDLPlanSource
	MTPAttention     *MTPPagedAttention
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
	var err error
	cfg, err = cfg.norm()
	if err != nil {
		return nil, err
	}
	dev := strings.TrimSpace(opts.Device)
	if dev == "" {
		dev = "NVIDIA H200"
	}
	if opts.BatchM <= 0 {
		return nil, fmt.Errorf("compile extended layer: batchM %d must be positive", opts.BatchM)
	}
	seqLen := opts.SeqLen
	if seqLen == 0 {
		seqLen = 1024
	}
	if seqLen < 0 {
		return nil, fmt.Errorf("compile extended layer: seqLen %d must be positive", seqLen)
	}
	if opts.CTAGroup != 0 && opts.CTAGroup != 1 && opts.CTAGroup != 2 {
		return nil, fmt.Errorf("compile extended layer: invalid CTAGroup %d (want 1 or 2)", opts.CTAGroup)
	}
	if opts.QTokensPerSeq < 0 {
		return nil, fmt.Errorf("compile extended layer: negative QTokensPerSeq %d", opts.QTokensPerSeq)
	}
	if opts.TPRanks < 0 || opts.EPRanks < 0 {
		return nil, fmt.Errorf("compile extended layer: negative rank count (tp=%d, ep=%d)", opts.TPRanks, opts.EPRanks)
	}

	sm100a := isSM100ADevice(dev)
	baseCfg := cfg
	if baseCfg.WeightDType == E4M3 && baseCfg.KVDType == E4M3 {
		// BuildLayerProgram casts Q to E4M3 when both WeightDType and KVDType are
		// E4M3, whereas TileGEMM requires BF16 or F32 output; lower base plan with
		// BF16 weights and preserve cfg on CompiledLayer.
		baseCfg.WeightDType = BF16
	}
	base, err := CompileLayer(baseCfg, opts.BatchM, seqLen, dev)
	if err != nil && sm100a {
		// Datacenter Blackwell (B200/GB200) uses the Blackwell schedule family;
		// fall back to NVIDIA GB10 for initial lowering when cudaroofline has no
		// measured device entry for B200/GB200, then apply sm_100a cost parameters.
		base, err = CompileLayer(baseCfg, opts.BatchM, seqLen, "NVIDIA GB10")
		if err == nil {
			base.Device = dev
			sm100Cost := CostConfig{
				Device:    dev,
				Residency: cudaroofline.DeviceResident,
				Ceiling: cudaroofline.Ceiling{
					Architecture:      "sm_100a",
					Residency:         cudaroofline.DeviceResident,
					Device:            dev,
					PeakTFLOPS:        90.0,
					PeakMemoryGBps:    8000.0,
					ComputeProvenance: cudaroofline.Unverified,
					MemoryProvenance:  cudaroofline.Unverified,
				},
				TensorTFLOPS: 2250.0,
				SMs:          148,
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

	hopperOrBlackwell := sm100a || isHopperOrBlackwellDevice(dev)

	// 1. Programmatic Dependent Launch (PDL) chain for medium-batch Hopper/Blackwell.
	if opts.EnablePDL || (hopperOrBlackwell && opts.BatchM > 8 && opts.BatchM < 64) {
		pdlName := base.Plan.Name + "_pdl"
		pdlSrc, err := emitLayerPDL(base.Plan, pdlName)
		if err != nil {
			return nil, err
		}
		bundle.PDL = pdlSrc
		bundle.Mode = ModePDLChain
	}

	// 2. Multi-Token Prediction (MTP) / speculative tree-verify paged attention.
	if opts.QTokensPerSeq > 1 {
		if len(opts.DraftMask) > 0 &&
			len(opts.DraftMask) != opts.QTokensPerSeq &&
			len(opts.DraftMask) != opts.BatchM*opts.QTokensPerSeq {
			return nil, fmt.Errorf("compile extended layer: draft mask length %d does not match QTokensPerSeq %d (or BatchM*QTokensPerSeq %d)",
				len(opts.DraftMask), opts.QTokensPerSeq, opts.BatchM*opts.QTokensPerSeq)
		}
		headDim := cfg.HeadDim
		if headDim <= 0 {
			headDim = cfg.HiddenSize / cfg.NumAttentionHeads
		}
		const pageSize = 16
		maxPages := max((seqLen+pageSize-1)/pageSize, 1)
		g := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
		wgmma := cfg.KVDType == E4M3 && headDim%32 == 0 && g*opts.QTokensPerSeq >= 8
		mtp := MTPPagedAttention{
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
		mtp, err = mtp.norm()
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
		ctaGroup := opts.CTAGroup
		if ctaGroup == 0 {
			if opts.BatchM >= 256 {
				ctaGroup = 2
			} else {
				ctaGroup = 1
			}
		}

		// 3a. 2-CTA cooperative tcgen05.mma.cta_group::2 GEMM when CTAGroup == 2 or BatchM >= 256.
		if ctaGroup == 2 || opts.BatchM >= 256 {
			c2 := TCGen05Cluster2GEMM{
				Name: base.Plan.Name + "_tcgen05_cluster2_gemm",
				M:    256,
				N:    128,
				K:    clampMultipleOf16(cfg.HiddenSize, 16, 64),
			}
			src, err := c2.CUDA()
			if err != nil {
				return nil, err
			}
			trace := c2.TraceEvents()
			bundle.TCGen05Kernels = append(bundle.TCGen05Kernels, src)
			bundle.TCGen05Traces = append(bundle.TCGen05Traces, trace)
		}

		// 3b. Block-scaled NVFP4 tcgen05 GEMM with tcgen05.cp scale staging when WeightDType == NVFP4.
		if cfg.WeightDType == NVFP4 {
			bs := TCGen05BlockScaledGEMM{
				Name: base.Plan.Name + "_tcgen05_block_scaled_gemm",
				M:    128,
				N:    128,
				K:    clampMultipleOf16(cfg.HiddenSize, 16, 256),
			}
			src, err := bs.CUDA()
			if err != nil {
				return nil, err
			}
			trace := bs.TraceEvents()
			bundle.TCGen05Kernels = append(bundle.TCGen05Kernels, src)
			bundle.TCGen05Traces = append(bundle.TCGen05Traces, trace)
		}

		// 3c. Standard 1-CTA tcgen05 TMEM GEMM projection.
		nDim := max(128, (cfg.HiddenSize/128)*128)
		kDim := max(64, (cfg.HiddenSize/64)*64)
		g1 := TCGen05GEMM{
			Name:     base.Plan.Name + "_tcgen05_gemm",
			N:        nDim,
			K:        kDim,
			BM:       128,
			BN:       128,
			BK:       64,
			Weight:   cfg.WeightFormat(),
			Out:      BF16,
			Epilogue: TileLinear,
		}
		g1Src, err := g1.CUDA()
		if err != nil {
			return nil, err
		}
		g1Trace := tcgen05GEMMTraceEvents(128, 128)
		bundle.TCGen05Kernels = append(bundle.TCGen05Kernels, g1Src)
		bundle.TCGen05Traces = append(bundle.TCGen05Traces, g1Trace)

		// 3d. TMEM-resident operand A FlashAttention for prefill (BatchM >= 128).
		if opts.BatchM >= 128 {
			fa := TCGen05TMEMFlashAttention{
				Name: base.Plan.Name + "_tcgen05_tmem_flash_attn",
				M:    128,
				KV:   64,
				D:    64,
			}
			faSrc, err := fa.CUDA()
			if err != nil {
				return nil, err
			}
			faTrace := fa.TraceEvents()
			bundle.TCGen05Kernels = append(bundle.TCGen05Kernels, faSrc)
			bundle.TCGen05Traces = append(bundle.TCGen05Traces, faTrace)
		}

		// Validate every emitted tcgen05 hardware trace against cudatmem.SM100A.
		for i, trace := range bundle.TCGen05Traces {
			if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, trace); err != nil {
				return nil, fmt.Errorf("compile extended layer: tcgen05 trace %d failed verification: %w", i, err)
			}
		}

		// 3e. Cluster Launch Control (CLC) persistent work-stealing TileGEMMs.
		if (opts.EnableCLC && opts.BatchM >= 64) || opts.BatchM >= 128 {
			clcKernels, err := emitLayerCLCKernels(base.Plan, cfg)
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
		colSrc, err := emitLayerCollectives(cfg, opts.TPRanks, opts.EPRanks)
		if err != nil {
			return nil, err
		}
		bundle.CollectiveSource = colSrc
	}

	base.Mode = bundle.Mode
	base.ExecutionMode = bundle.Mode
	return bundle, nil
}

func isSM100ADevice(device string) bool {
	d := strings.ToLower(strings.TrimSpace(device))
	return strings.Contains(d, "b200") ||
		strings.Contains(d, "gb200") ||
		strings.Contains(d, "b100") ||
		strings.Contains(d, "sm_100")
}

func isHopperOrBlackwellDevice(device string) bool {
	d := strings.ToLower(strings.TrimSpace(device))
	return strings.Contains(d, "h200") ||
		strings.Contains(d, "h100") ||
		strings.Contains(d, "sm_90") ||
		strings.Contains(d, "gb10") ||
		strings.Contains(d, "b200") ||
		strings.Contains(d, "gb200") ||
		strings.Contains(d, "b100") ||
		strings.Contains(d, "sm_100") ||
		strings.Contains(d, "sm_120") ||
		strings.Contains(d, "sm_121")
}

func clampMultipleOf16(v, lo, hi int) int {
	n := (v / 16) * 16
	if n < lo {
		n = lo
	}
	if n > hi {
		n = hi
	}
	return n
}

func tcgen05GEMMTraceEvents(rows, cols uint32) []cudatmem.TraceEvent {
	events := []cudatmem.TraceEvent{
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: cols, Rows: rows},
		{Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: cols, Rows: rows},
		{Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: cols, Rows: rows},
		{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: cols, Rows: rows},
	}
	for w := range 4 {
		events = append(events, cudatmem.TraceEvent{
			Kind:          cudatmem.EventLd,
			CTA:           0,
			Warp:          w,
			ActiveThreads: 32,
			CTAGroup:      1,
			Offset:        0,
			Columns:       cols,
			RowStart:      uint32(32 * w),
			Rows:          32,
		})
	}
	events = append(events, cudatmem.TraceEvent{
		Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: cols, Rows: rows,
	})
	return events
}

// emitLayerPDL lowers p (or its longest PDL-compatible non-attention segment)
// via Plan.EmitPDL.
func emitLayerPDL(p *Plan, pdlName string) (*PDLPlanSource, error) {
	if p == nil || len(p.Kernels) == 0 {
		return nil, fmt.Errorf("compile extended layer: empty plan for PDL")
	}
	if src, err := p.EmitPDL(pdlName); err == nil {
		return src, nil
	}

	tensorByName := make(map[string]Tensor)
	for _, t := range p.Inputs {
		tensorByName[t.Name] = t
	}
	for _, t := range p.Temps {
		tensorByName[t.Name] = t
	}
	if p.Output.Name != "" {
		tensorByName[p.Output.Name] = p.Output
	}

	var segments [][]PlanKernel
	var cur []PlanKernel
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

func emitLayerCLCKernels(p *Plan, cfg ArchConfig) ([]string, error) {
	var out []string
	seen := make(map[string]bool)
	for i, pk := range p.Kernels {
		if pk.Tile == nil || pk.Tile.Grouped {
			continue
		}
		t := pk.Tile
		nout := t.N
		if t.GLU {
			nout = t.N / 2
		}
		bn := min(64, nout)
		if bn <= 0 || nout%bn != 0 {
			bn = 64
			nout = max(64, (nout/64)*64)
		}
		bk := min(64, t.K)
		if bk <= 0 || t.K%bk != 0 {
			bk = 64
		}
		name := fmt.Sprintf("%s_clc_%d", t.Name, i)
		if seen[name] {
			continue
		}
		seen[name] = true
		clc := CLCPersistentTileGEMM{
			Name:       name,
			N:          nout,
			K:          max(64, (t.K/64)*64),
			BM:         64,
			BN:         bn,
			BK:         bk,
			ActiveCTAs: 4,
			Weight:     t.Weight,
			Out:        BF16,
		}
		src, err := clc.Source()
		if err != nil {
			return nil, err
		}
		out = append(out, src)
	}
	if len(out) == 0 {
		nDim := max(64, (cfg.HiddenSize/64)*64)
		clc := CLCPersistentTileGEMM{
			Name:       p.Name + "_clc_tile_gemm",
			N:          nDim,
			K:          nDim,
			BM:         64,
			BN:         64,
			BK:         64,
			ActiveCTAs: 4,
			Weight:     cfg.WeightFormat(),
			Out:        BF16,
		}
		src, err := clc.Source()
		if err != nil {
			return nil, err
		}
		out = append(out, src)
	}
	return out, nil
}

func emitLayerCollectives(cfg ArchConfig, tpRanks, epRanks int) (string, error) {
	cols := cfg.HiddenSize
	if cols < 2 {
		cols = 2
	}
	if cols%2 != 0 {
		cols++
	}
	threads := 256
	if cols < threads {
		threads = max(32, (cols/32)*32)
	}

	var b strings.Builder
	b.WriteString(Prelude)
	b.WriteString("\n")
	b.WriteString(CollectivePrelude)

	if tpRanks > 1 {
		ar := AllReduceResidualRMSNorm{
			Name:    "allreduce_residual_rmsnorm",
			Cols:    cols,
			Ranks:   tpRanks,
			Eps:     cfg.RMSNormEps,
			DType:   BF16,
			Threads: threads,
		}
		cuda, err := ar.CUDA()
		if err != nil {
			return "", err
		}
		launcher, err := ar.Launcher()
		if err != nil {
			return "", err
		}
		b.WriteString("\n")
		b.WriteString(cuda)
		b.WriteString("\n")
		b.WriteString(launcher)
	}

	if epRanks > 1 {
		topK := cfg.NumExpertsPerTok
		if topK <= 0 {
			topK = 2
		}
		ep := NVLinkMoECombine{
			Name:    "nvlink_moe_combine",
			Cols:    cols,
			TopK:    topK,
			Ranks:   epRanks,
			DType:   BF16,
			Threads: threads,
		}
		cuda, err := ep.CUDA()
		if err != nil {
			return "", err
		}
		launcher, err := ep.Launcher()
		if err != nil {
			return "", err
		}
		b.WriteString("\n")
		b.WriteString(cuda)
		b.WriteString("\n")
		b.WriteString(launcher)
	}

	return b.String(), nil
}
