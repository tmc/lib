package model

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/nvl72"
	"github.com/tmc/lib/nvidia/x/tcgen05"
)

// Extended execution mode names selected by extended layer compilation.
const (
	ModePDLChain         = "pdl_chain"
	ModeBlackwellTCGen05 = "blackwell_tcgen05"
	ModeBlackwellCTA2CLC = "blackwell_cta2_clc"
)

// TargetArch identifies the target GPU architecture family.
type TargetArch string

const (
	ArchHopperSM90     TargetArch = "sm_90a"
	ArchBlackwellSM100 TargetArch = "sm_100a"
	ArchBlackwellSM121 TargetArch = "sm_121a"
)

// AttentionMode identifies the attention lowering path.
type AttentionMode string

const (
	AttentionSingleToken AttentionMode = "single_token"
	AttentionMTP         AttentionMode = "mtp_tree_verify"
	AttentionTMEMFlash   AttentionMode = "tcgen05_tmem_flash"
)

// CollectiveMode identifies the multi-GPU NVLink communication mode.
type CollectiveMode string

const (
	CollectiveNone       CollectiveMode = "none"
	CollectiveNVLinkTP   CollectiveMode = "nvlink_tp"
	CollectiveNVLinkEP   CollectiveMode = "nvlink_ep"
	CollectiveNVLinkTPEP CollectiveMode = "nvlink_tp_ep"
)

// ExtendedArchOptions configures hardware-specific lowering passes on top of
// [ArchConfig], including Programmatic Dependent Launch (PDL), speculative
// multi-token / tree-verify paged attention (MTP), Blackwell sm_100a Tensor
// Memory (tcgen05) and Cluster Launch Control (CLC) kernels, and multi-GPU
// NVLink TP/EP collectives.
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

// ExtendedArchConfig combines [ArchConfig] and [ExtendedArchOptions].
type ExtendedArchConfig struct {
	Arch    ArchConfig
	Options ExtendedArchOptions
}

// ValidateExtendedArchOptions validates cfg and opts, returning normalized
// copies and whether the target device is datacenter Blackwell (sm_100a).
func ValidateExtendedArchOptions(cfg ArchConfig, opts ExtendedArchOptions) (ArchConfig, ExtendedArchOptions, bool, error) {
	var err error
	cfg, err = ValidateArchConfig(cfg)
	if err != nil {
		return cfg, opts, false, err
	}
	opts.Device = strings.TrimSpace(opts.Device)
	if opts.Device == "" {
		opts.Device = "NVIDIA H200"
	}
	if opts.BatchM <= 0 {
		return cfg, opts, false, fmt.Errorf("compile extended layer: batchM %d must be positive", opts.BatchM)
	}
	if opts.SeqLen == 0 {
		opts.SeqLen = 1024
	}
	if opts.SeqLen < 0 {
		return cfg, opts, false, fmt.Errorf("compile extended layer: seqLen %d must be positive", opts.SeqLen)
	}
	if opts.CTAGroup != 0 && opts.CTAGroup != 1 && opts.CTAGroup != 2 {
		return cfg, opts, false, fmt.Errorf("compile extended layer: invalid CTAGroup %d (want 1 or 2)", opts.CTAGroup)
	}
	if opts.QTokensPerSeq < 0 {
		return cfg, opts, false, fmt.Errorf("compile extended layer: negative QTokensPerSeq %d", opts.QTokensPerSeq)
	}
	if opts.TPRanks < 0 || opts.EPRanks < 0 {
		return cfg, opts, false, fmt.Errorf("compile extended layer: negative rank count (tp=%d, ep=%d)", opts.TPRanks, opts.EPRanks)
	}
	if opts.QTokensPerSeq > 1 && len(opts.DraftMask) > 0 &&
		len(opts.DraftMask) != opts.QTokensPerSeq &&
		len(opts.DraftMask) != opts.BatchM*opts.QTokensPerSeq {
		return cfg, opts, false, fmt.Errorf("compile extended layer: draft mask length %d does not match QTokensPerSeq %d (or BatchM*QTokensPerSeq %d)",
			len(opts.DraftMask), opts.QTokensPerSeq, opts.BatchM*opts.QTokensPerSeq)
	}
	return cfg, opts, IsSM100ADevice(opts.Device), nil
}

// ValidateExtendedArchConfig validates an [ExtendedArchConfig].
func ValidateExtendedArchConfig(ec ExtendedArchConfig) (ExtendedArchConfig, error) {
	cfg, opts, _, err := ValidateExtendedArchOptions(ec.Arch, ec.Options)
	if err != nil {
		return ec, err
	}
	return ExtendedArchConfig{Arch: cfg, Options: opts}, nil
}

// IsSM100ADevice reports whether device names a datacenter Blackwell (sm_100a) GPU.
func IsSM100ADevice(device string) bool {
	d := strings.ToLower(strings.TrimSpace(device))
	return strings.Contains(d, "b200") ||
		strings.Contains(d, "gb200") ||
		strings.Contains(d, "b100") ||
		strings.Contains(d, "sm_100")
}

// IsHopperOrBlackwellDevice reports whether device names a Hopper or Blackwell GPU.
func IsHopperOrBlackwellDevice(device string) bool {
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

// SM100ACostCeiling returns the default roofline ceiling for a datacenter
// Blackwell sm_100a device.
func SM100ACostCeiling(dev string) cudaroofline.Ceiling {
	return cudaroofline.Ceiling{
		Architecture:      "sm_100a",
		Residency:         cudaroofline.DeviceResident,
		Device:            dev,
		PeakTFLOPS:        90.0,
		PeakMemoryGBps:    targetMemoryGBps(ArchTargetSpec{ComputeMajor: 10, ComputeMinor: 0}),
		ComputeProvenance: cudaroofline.Unverified,
		MemoryProvenance:  cudaroofline.Unverified,
	}
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

// EmitSM100TCGen05Suite emits and formally verifies the Blackwell sm_100a
// tcgen05 kernel suite and TMEM traces for planName under cfg and opts.
func EmitSM100TCGen05Suite(planName string, cfg ArchConfig, opts ExtendedArchOptions) (kernels []string, traces [][]cudatmem.TraceEvent, ctaGroup int, err error) {
	ctaGroup = opts.CTAGroup
	if ctaGroup == 0 {
		if opts.BatchM >= 256 {
			ctaGroup = 2
		} else {
			ctaGroup = 1
		}
	}

	if ctaGroup == 2 || opts.BatchM >= 256 {
		c2 := tcgen05.TCGen05Cluster2GEMM{
			Name: planName + "_tcgen05_cluster2_gemm",
			M:    256,
			N:    128,
			K:    clampMultipleOf16(cfg.HiddenSize, 16, 64),
		}
		src, err := c2.CUDA()
		if err != nil {
			return nil, nil, 0, err
		}
		kernels = append(kernels, src)
		traces = append(traces, c2.TraceEvents())
	}

	if cfg.WeightDType == NVFP4 {
		bs := tcgen05.TCGen05BlockScaledGEMM{
			Name: planName + "_tcgen05_block_scaled_gemm",
			M:    128,
			N:    128,
			K:    clampMultipleOf16(cfg.HiddenSize, 16, 256),
		}
		src, err := bs.CUDA()
		if err != nil {
			return nil, nil, 0, err
		}
		kernels = append(kernels, src)
		traces = append(traces, bs.TraceEvents())
	}

	nDim := max(128, (cfg.HiddenSize/128)*128)
	kDim := max(64, (cfg.HiddenSize/64)*64)
	g1 := tcgen05.TCGen05GEMM{
		Name:     planName + "_tcgen05_gemm",
		N:        nDim,
		K:        kDim,
		BM:       128,
		BN:       128,
		BK:       64,
		Weight:   tcgen05.WeightFormat(cfg.WeightFormat()),
		Out:      tcgen05.BF16,
		Epilogue: tcgen05.TileLinear,
	}
	g1Src, err := g1.CUDA()
	if err != nil {
		return nil, nil, 0, err
	}
	kernels = append(kernels, g1Src)
	traces = append(traces, tcgen05GEMMTraceEvents(128, 128))

	if opts.BatchM >= 128 {
		fa := tcgen05.TCGen05TMEMFlashAttention{
			Name: planName + "_tcgen05_tmem_flash_attn",
			M:    128,
			KV:   64,
			D:    64,
		}
		faSrc, err := fa.CUDA()
		if err != nil {
			return nil, nil, 0, err
		}
		kernels = append(kernels, faSrc)
		traces = append(traces, fa.TraceEvents())
	}

	for i, trace := range traces {
		if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, trace); err != nil {
			return nil, nil, 0, fmt.Errorf("compile extended layer: tcgen05 trace %d failed verification: %w", i, err)
		}
	}
	return kernels, traces, ctaGroup, nil
}

// CLCTileSpec describes a non-grouped TileGEMM stage to lower to a Blackwell
// Cluster Launch Control (CLC) persistent work-stealing kernel.
type CLCTileSpec struct {
	Index  int
	Name   string
	N      int
	K      int
	GLU    bool
	Weight WeightFormat
}

// EmitLayerCLCKernels emits Blackwell sm_100a Cluster Launch Control (CLC)
// persistent work-stealing TileGEMM kernels for tiles (or a default projection
// kernel when tiles is empty).
func EmitLayerCLCKernels(planName string, tiles []CLCTileSpec, cfg ArchConfig) ([]string, error) {
	var out []string
	seen := make(map[string]bool)
	for _, t := range tiles {
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
		name := fmt.Sprintf("%s_clc_%d", t.Name, t.Index)
		if seen[name] {
			continue
		}
		seen[name] = true
		clc := tcgen05.CLCPersistentTileGEMM{
			Name:       name,
			N:          nout,
			K:          max(64, (t.K/64)*64),
			BM:         64,
			BN:         bn,
			BK:         bk,
			ActiveCTAs: 4,
			Weight:     tcgen05.WeightFormat(t.Weight),
			Out:        tcgen05.BF16,
		}
		src, err := clc.Source()
		if err != nil {
			return nil, err
		}
		out = append(out, src)
	}
	if len(out) == 0 {
		nDim := max(64, (cfg.HiddenSize/64)*64)
		clc := tcgen05.CLCPersistentTileGEMM{
			Name:       planName + "_clc_tile_gemm",
			N:          nDim,
			K:          nDim,
			BM:         64,
			BN:         64,
			BK:         64,
			ActiveCTAs: 4,
			Weight:     tcgen05.WeightFormat(cfg.WeightFormat()),
			Out:        tcgen05.BF16,
		}
		src, err := clc.Source()
		if err != nil {
			return nil, err
		}
		out = append(out, src)
	}
	return out, nil
}

// EmitLayerCollectives emits the multi-GPU NVLink TP AllReduceResidualRMSNorm
// and/or EP NVLinkMoECombine translation unit for cfg.
func EmitLayerCollectives(prelude string, cfg ArchConfig, tpRanks, epRanks int) (string, error) {
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
	b.WriteString(prelude)
	b.WriteString("\n")
	b.WriteString(nvl72.CollectivePrelude)

	if tpRanks > 1 {
		ar := nvl72.AllReduceResidualRMSNorm{
			Name:    "allreduce_residual_rmsnorm",
			Cols:    cols,
			Ranks:   tpRanks,
			Eps:     cfg.RMSNormEps,
			DType:   nvl72.BF16,
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
		ep := nvl72.NVLinkMoECombine{
			Name:    "nvlink_moe_combine",
			Cols:    cols,
			TopK:    topK,
			Ranks:   epRanks,
			DType:   nvl72.BF16,
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
