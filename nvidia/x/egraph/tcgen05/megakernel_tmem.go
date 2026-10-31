package tcgen05

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// Blackwell Tensor Memory (TMEM) hardware capacity constants per SM.
const (
	// MaxTMEMColumnsPerSM is the maximum number of 32-bit TMEM columns per SM
	// on Blackwell (sm_100a / sm_103a): 512 columns x 128 lanes x 4B = 256 KB.
	MaxTMEMColumnsPerSM = 512

	// TMEMLanesPerSM is the number of vertical lanes (rows) in TMEM per SM.
	TMEMLanesPerSM = 128

	// MaxTMEMBytesPerSM is the 256 KB Tensor Memory capacity per SM.
	MaxTMEMBytesPerSM = MaxTMEMColumnsPerSM * TMEMLanesPerSM * 4
)

// TMEMStageOp identifies a stage operation inside a [TMEMMegakernelPlan].
type TMEMStageOp string

const (
	TMEMStageGateProj      TMEMStageOp = "gate_proj"
	TMEMStageUpProj        TMEMStageOp = "up_proj"
	TMEMStageSwiGLUEpilog  TMEMStageOp = "swiglu_epilogue"
	TMEMStageDownProj      TMEMStageOp = "down_proj"
	TMEMStageRMSNormBridge TMEMStageOp = "rmsnorm_bridge"
	TMEMStageQKVProj       TMEMStageOp = "qkv_proj"
)

// TMEMStage describes a single stage in a [TMEMMegakernelPlan], tracking its
// TMEM column slice [TMEMOffset, TMEMOffset+TMEMCols), cluster CTA group size,
// whether the accumulator remains resident in TMEM across the epilogue, and
// whether the stage allocates or deallocates TMEM columns.
type TMEMStage struct {
	Name            string
	Layer           int
	Op              TMEMStageOp
	TMEMOffset      int
	TMEMCols        int
	CTAGroup        int
	KeepInTMEM      bool
	CommitsToMemory bool
	AllocatesTMEM   bool
	DeallocatesTMEM bool
}

// TMEMCrossLayerSpec configures a Blackwell persistent cross-layer megakernel
// where intermediate GEMM accumulators (gate_proj, up_proj, qkv_proj) remain
// resident in Blackwell Tensor Memory (TMEM, 256 KB/SM via tcgen05.alloc,
// tcgen05.ld, tcgen05.st, and tcgen05.mma.cta_group::2) across fused epilogue
// stages (SwiGLU, RMSNorm) before committing to SMEM/GMEM, with tcgen05.dealloc
// guarded at layer exit.
type TMEMCrossLayerSpec struct {
	// Name is the megakernel plan identifier (defaults to "blackwell_tmem_crosslayer_mega").
	Name string

	// Target is the Blackwell architecture target ("sm_100a", "sm_103a",
	// "sm_120a", or "sm_121a"; defaults to "sm_100a").
	Target string

	// NumLayers is the number of decoder layers fused in the plan (defaults to 2).
	NumLayers int

	// Tokens is the token batch row count M (defaults to 128; when NegativeTokens
	// or < 0, rejected by validation).
	Tokens int

	// HiddenDim is the model hidden dimension H (defaults to 256).
	HiddenDim int

	// IntermediateDim is the FFN SwiGLU intermediate dimension I (defaults to 256).
	IntermediateDim int

	// QDim is the query projection dimension (defaults to HiddenDim).
	QDim int

	// KVDim is the key/value projection dimension (defaults to HiddenDim/4, min 64).
	KVDim int

	// ClusterSize is the cooperative CTA cluster size (1 or 2; defaults to 2 on
	// sm_100a/sm_103a and 1 on sm_120a/sm_121a).
	ClusterSize int

	// GateCols is the TMEM column allocation for gate_proj accumulators (defaults to 128).
	GateCols int

	// UpCols is the TMEM column allocation for up_proj accumulators (defaults to 128).
	UpCols int

	// QKVCols is the TMEM column allocation for next-layer qkv_proj accumulators (defaults to 256).
	QKVCols int

	// WeightFormat selects the weight encoding ("nvfp4", "bf16", or "e4m3"; defaults to "nvfp4").
	WeightFormat string

	// OmitLayerExitDealloc disables the guarded tcgen05.dealloc at layer exit
	// (used to test lifecycle verification of unbalanced TMEM allocations).
	OmitLayerExitDealloc bool
}

// TMEMMegakernelPlan holds the validated stage schedule, TMEM column map,
// formal cudatmem hardware trace, and Hopper megakernel lowering for a
// Blackwell TMEM-resident cross-layer megakernel.
type TMEMMegakernelPlan struct {
	Spec           TMEMCrossLayerSpec
	Name           string
	Target         string
	ClusterSize    int
	UsesNativeTMEM bool
	TotalTMEMCols  int
	TotalTMEMBytes int
	AllocCount     int
	DeallocCount   int
	Stages         []TMEMStage
	Trace          []cudatmem.TraceEvent
}

func cIdent(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || (i > 0 && c >= '0' && c <= '9')
		if !ok {
			return false
		}
	}
	return true
}

func validBlackwellTMEMTarget(target string) bool {
	switch target {
	case "sm_100a", "sm_100f", "sm_103a", "sm_103f", "sm_120a", "sm_121a":
		return true
	default:
		return false
	}
}

func targetHasHardwareTMEM(target string) bool {
	switch target {
	case "sm_100a", "sm_100f", "sm_103a", "sm_103f":
		return true
	default:
		return false
	}
}

func isPowerOfTwoCols(c int) bool {
	return c >= 32 && c <= MaxTMEMColumnsPerSM && (c&(c-1)) == 0
}

func (s TMEMCrossLayerSpec) norm() (TMEMCrossLayerSpec, error) {
	if s.Name == "" {
		s.Name = "blackwell_tmem_crosslayer_mega"
	}
	if !cIdent(s.Name) {
		return s, fmt.Errorf("tcgen05 tmem megakernel: bad name %q", s.Name)
	}
	if s.Target == "" {
		s.Target = "sm_100a"
	}
	if !validBlackwellTMEMTarget(s.Target) {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: unsupported target %q (want sm_100a, sm_103a, sm_120a, or sm_121a)", s.Name, s.Target)
	}
	if s.NumLayers == 0 {
		s.NumLayers = 2
	}
	if s.NumLayers < 1 || s.NumLayers > 128 {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: NumLayers %d must be in 1..128", s.Name, s.NumLayers)
	}
	if s.Tokens < 0 {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: Tokens %d must be >= 1", s.Name, s.Tokens)
	}
	if s.Tokens == 0 {
		s.Tokens = 128
	}
	if s.HiddenDim == 0 {
		s.HiddenDim = 256
	}
	if s.HiddenDim < 64 || s.HiddenDim%64 != 0 {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: HiddenDim %d must be a positive multiple of 64", s.Name, s.HiddenDim)
	}
	if s.IntermediateDim == 0 {
		s.IntermediateDim = s.HiddenDim
	}
	if s.IntermediateDim < 64 || s.IntermediateDim%64 != 0 {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: IntermediateDim %d must be a positive multiple of 64", s.Name, s.IntermediateDim)
	}
	if s.QDim == 0 {
		s.QDim = s.HiddenDim
	}
	if s.KVDim == 0 {
		s.KVDim = max(s.HiddenDim/4, 64)
	}
	if s.QDim < 64 || s.QDim%64 != 0 || s.KVDim < 64 || s.KVDim%64 != 0 {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: QDim (%d) and KVDim (%d) must be positive multiples of 64", s.Name, s.QDim, s.KVDim)
	}
	if s.ClusterSize == 0 {
		if targetHasHardwareTMEM(s.Target) {
			s.ClusterSize = 2
		} else {
			s.ClusterSize = 1
		}
	}
	if s.ClusterSize != 1 && s.ClusterSize != 2 {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: ClusterSize %d must be 1 or 2", s.Name, s.ClusterSize)
	}
	if s.ClusterSize == 2 && !targetHasHardwareTMEM(s.Target) {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: ClusterSize=2 (cta_group::2) requires sm_100a/sm_103a TMEM hardware, got %s", s.Name, s.Target)
	}
	if s.GateCols < 0 || s.UpCols < 0 || s.QKVCols < 0 {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: TMEM column counts must be positive", s.Name)
	}
	if s.GateCols == 0 {
		s.GateCols = 128
	}
	if s.UpCols == 0 {
		s.UpCols = 128
	}
	if s.QKVCols == 0 {
		s.QKVCols = 256
	}
	if !isPowerOfTwoCols(s.GateCols) || !isPowerOfTwoCols(s.UpCols) || !isPowerOfTwoCols(s.QKVCols) {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: GateCols (%d), UpCols (%d), and QKVCols (%d) must be powers of two in [32, 512]",
			s.Name, s.GateCols, s.UpCols, s.QKVCols)
	}
	totalCols := s.GateCols + s.UpCols + s.QKVCols
	if totalCols > MaxTMEMColumnsPerSM {
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: TMEM column budget exceeded: %d + %d + %d = %d > %d columns per SM",
			s.Name, s.GateCols, s.UpCols, s.QKVCols, totalCols, MaxTMEMColumnsPerSM)
	}
	if s.WeightFormat == "" {
		s.WeightFormat = "nvfp4"
	}
	switch strings.ToLower(s.WeightFormat) {
	case "nvfp4", "bf16", "e4m3":
		s.WeightFormat = strings.ToLower(s.WeightFormat)
	default:
		return s, fmt.Errorf("tcgen05 tmem megakernel %s: unsupported WeightFormat %q (want nvfp4, bf16, or e4m3)", s.Name, s.WeightFormat)
	}
	return s, nil
}

// BuildTMEMCrossLayerMegakernel validates spec, verifies the 512-column TMEM
// budget per SM, 2-CTA cluster alignment, and balanced tcgen05.alloc /
// tcgen05.dealloc lifecycle across layers, and constructs a [TMEMMegakernelPlan].
func BuildTMEMCrossLayerMegakernel(spec TMEMCrossLayerSpec) (*TMEMMegakernelPlan, error) {
	normSpec, err := spec.norm()
	if err != nil {
		return nil, err
	}

	ctaGroup := normSpec.ClusterSize
	gateOff := 0
	upOff := normSpec.GateCols
	qkvOff := normSpec.GateCols + normSpec.UpCols
	totalCols := normSpec.GateCols + normSpec.UpCols + normSpec.QKVCols
	allocUnit := min(normSpec.GateCols, normSpec.UpCols, normSpec.QKVCols)

	var stages []TMEMStage
	var trace []cudatmem.TraceEvent
	allocCount := 0
	deallocCount := 0

	emitAllocEvents := func(offset, cols int) {
		for off := offset; off < offset+cols; off += allocUnit {
			for cta := 0; cta < ctaGroup; cta++ {
				trace = append(trace, cudatmem.TraceEvent{
					Kind:          cudatmem.EventAlloc,
					CTA:           cta,
					Warp:          0,
					ActiveThreads: 32,
					CTAGroup:      ctaGroup,
					Offset:        uint32(off),
					Columns:       uint32(allocUnit),
					Rows:          TMEMLanesPerSM,
				})
			}
		}
	}
	emitDeallocEvents := func(offset, cols int) {
		if ctaGroup == 2 {
			trace = append(trace, cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2})
		}
		for off := offset; off < offset+cols; off += allocUnit {
			for cta := 0; cta < ctaGroup; cta++ {
				trace = append(trace, cudatmem.TraceEvent{
					Kind:          cudatmem.EventDealloc,
					CTA:           cta,
					Warp:          0,
					ActiveThreads: 32,
					CTAGroup:      ctaGroup,
					Offset:        uint32(off),
					Columns:       uint32(allocUnit),
					Rows:          TMEMLanesPerSM,
				})
			}
		}
	}
	emitMMAAndWait := func(offset, cols int) {
		for off := offset; off < offset+cols; off += allocUnit {
			if ctaGroup == 2 {
				trace = append(trace, cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2})
			}
			trace = append(trace,
				cudatmem.TraceEvent{
					Kind:          cudatmem.EventMMA,
					CTA:           0,
					Warp:          0,
					ActiveThreads: 1,
					CTAGroup:      ctaGroup,
					Offset:        uint32(off),
					Columns:       uint32(allocUnit),
					Rows:          TMEMLanesPerSM,
				},
				cudatmem.TraceEvent{
					Kind:          cudatmem.EventCommit,
					CTA:           0,
					Warp:          0,
					ActiveThreads: 1,
					CTAGroup:      ctaGroup,
					Offset:        uint32(off),
					Columns:       uint32(allocUnit),
					Rows:          TMEMLanesPerSM,
				},
			)
			for cta := 0; cta < ctaGroup; cta++ {
				trace = append(trace, cudatmem.TraceEvent{
					Kind:          cudatmem.EventWait,
					CTA:           cta,
					Warp:          0,
					ActiveThreads: 32,
					CTAGroup:      ctaGroup,
					Offset:        uint32(off),
					Columns:       uint32(allocUnit),
					Rows:          TMEMLanesPerSM,
				})
			}
		}
	}
	emitPartitionedLdSt := func(kind cudatmem.EventKind, offset, cols int) {
		for off := offset; off < offset+cols; off += allocUnit {
			for cta := 0; cta < ctaGroup; cta++ {
				for w := 0; w < 4; w++ {
					trace = append(trace, cudatmem.TraceEvent{
						Kind:          kind,
						CTA:           cta,
						Warp:          w,
						ActiveThreads: 32,
						CTAGroup:      ctaGroup,
						Offset:        uint32(off),
						Columns:       uint32(allocUnit),
						RowStart:      uint32(32 * w),
						Rows:          32,
					})
				}
			}
		}
	}

	for l := 0; l < normSpec.NumLayers; l++ {
		// Stage 1: Allocate TMEM columns for layer l and compute gate_proj into [gateOff, gateOff+GateCols).
		stages = append(stages, TMEMStage{
			Name:          fmt.Sprintf("l%d_gate_proj_tmem", l),
			Layer:         l,
			Op:            TMEMStageGateProj,
			TMEMOffset:    gateOff,
			TMEMCols:      normSpec.GateCols,
			CTAGroup:      ctaGroup,
			KeepInTMEM:    true,
			AllocatesTMEM: true,
		})
		allocCount += 3 // gate, up, and qkv slices allocated at layer entry
		emitAllocEvents(gateOff, normSpec.GateCols)
		emitAllocEvents(upOff, normSpec.UpCols)
		emitAllocEvents(qkvOff, normSpec.QKVCols)
		emitMMAAndWait(gateOff, normSpec.GateCols)

		// Stage 2: Compute up_proj into TMEM [upOff, upOff+UpCols), keeping accumulator in TMEM.
		stages = append(stages, TMEMStage{
			Name:       fmt.Sprintf("l%d_up_proj_tmem", l),
			Layer:      l,
			Op:         TMEMStageUpProj,
			TMEMOffset: upOff,
			TMEMCols:   normSpec.UpCols,
			CTAGroup:   ctaGroup,
			KeepInTMEM: true,
		})
		emitMMAAndWait(upOff, normSpec.UpCols)

		// Stage 3: Fused SwiGLU epilogue in TMEM (tcgen05.ld gate & up -> silu(gate)*up -> tcgen05.st back to gate slice).
		stages = append(stages, TMEMStage{
			Name:       fmt.Sprintf("l%d_swiglu_tmem", l),
			Layer:      l,
			Op:         TMEMStageSwiGLUEpilog,
			TMEMOffset: gateOff,
			TMEMCols:   normSpec.GateCols,
			CTAGroup:   ctaGroup,
			KeepInTMEM: true,
		})
		emitPartitionedLdSt(cudatmem.EventLd, gateOff, normSpec.GateCols)
		emitPartitionedLdSt(cudatmem.EventLd, upOff, normSpec.UpCols)
		emitPartitionedLdSt(cudatmem.EventSt, gateOff, normSpec.GateCols)

		// Stage 4: Down-projection reading SwiGLU activation from TMEM [gateOff] and accumulating into [upOff].
		stages = append(stages, TMEMStage{
			Name:       fmt.Sprintf("l%d_down_proj_tmem", l),
			Layer:      l,
			Op:         TMEMStageDownProj,
			TMEMOffset: upOff,
			TMEMCols:   normSpec.UpCols,
			CTAGroup:   ctaGroup,
			KeepInTMEM: true,
		})
		if ctaGroup == 2 {
			trace = append(trace, cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2})
		}
		trace = append(trace,
			cudatmem.TraceEvent{
				Kind:            cudatmem.EventMMA,
				CTA:             0,
				Warp:            0,
				ActiveThreads:   1,
				CTAGroup:        ctaGroup,
				Offset:          uint32(upOff),
				Columns:         uint32(normSpec.UpCols),
				Rows:            TMEMLanesPerSM,
				SourceAFromTMEM: true,
				SourceAOffset:   uint32(gateOff),
			},
			cudatmem.TraceEvent{
				Kind:          cudatmem.EventCommit,
				CTA:           0,
				Warp:          0,
				ActiveThreads: 1,
				CTAGroup:      ctaGroup,
				Offset:        uint32(upOff),
				Columns:       uint32(normSpec.UpCols),
				Rows:          TMEMLanesPerSM,
			},
		)
		for cta := 0; cta < ctaGroup; cta++ {
			trace = append(trace, cudatmem.TraceEvent{
				Kind:          cudatmem.EventWait,
				CTA:           cta,
				Warp:          0,
				ActiveThreads: 32,
				CTAGroup:      ctaGroup,
				Offset:        uint32(upOff),
				Columns:       uint32(normSpec.UpCols),
				Rows:          TMEMLanesPerSM,
			})
		}

		// Stage 5: Cross-layer Residual + RMSNorm epilogue in TMEM before committing normalized bridge activation to SMEM/GMEM.
		stages = append(stages, TMEMStage{
			Name:            fmt.Sprintf("l%d_rmsnorm_bridge_tmem", l),
			Layer:           l,
			Op:              TMEMStageRMSNormBridge,
			TMEMOffset:      upOff,
			TMEMCols:        normSpec.UpCols,
			CTAGroup:        ctaGroup,
			KeepInTMEM:      true,
			CommitsToMemory: true,
		})
		emitPartitionedLdSt(cudatmem.EventLd, upOff, normSpec.UpCols)
		emitPartitionedLdSt(cudatmem.EventSt, gateOff, normSpec.GateCols)

		// Stage 6: Next-layer (or layer-exit) QKV projection into TMEM [qkvOff, qkvOff+QKVCols),
		// followed by guarded tcgen05.dealloc at layer exit.
		deallocAtExit := !normSpec.OmitLayerExitDealloc
		stages = append(stages, TMEMStage{
			Name:            fmt.Sprintf("l%d_next_qkv_proj_tmem", l),
			Layer:           l,
			Op:              TMEMStageQKVProj,
			TMEMOffset:      qkvOff,
			TMEMCols:        normSpec.QKVCols,
			CTAGroup:        ctaGroup,
			KeepInTMEM:      false,
			CommitsToMemory: true,
			DeallocatesTMEM: deallocAtExit,
		})
		emitMMAAndWait(qkvOff, normSpec.QKVCols)
		emitPartitionedLdSt(cudatmem.EventLd, qkvOff, normSpec.QKVCols)
		if deallocAtExit {
			deallocCount += 3
			emitDeallocEvents(gateOff, normSpec.GateCols)
			emitDeallocEvents(upOff, normSpec.UpCols)
			emitDeallocEvents(qkvOff, normSpec.QKVCols)
		}
	}

	plan := &TMEMMegakernelPlan{
		Spec:           normSpec,
		Name:           normSpec.Name,
		Target:         normSpec.Target,
		ClusterSize:    ctaGroup,
		UsesNativeTMEM: targetHasHardwareTMEM(normSpec.Target),
		TotalTMEMCols:  totalCols,
		TotalTMEMBytes: totalCols * TMEMLanesPerSM * 4,
		AllocCount:     allocCount,
		DeallocCount:   deallocCount,
		Stages:         stages,
		Trace:          trace,
	}
	if err := plan.VerifyLifecycle(); err != nil {
		return nil, err
	}
	return plan, nil
}

// VerifyLifecycle validates the TMEM column budget (<= 512 columns per SM),
// 2-CTA cluster alignment, balanced tcgen05.alloc / tcgen05.dealloc lifecycle,
// and formal cudatmem hardware trace invariants.
func (p *TMEMMegakernelPlan) VerifyLifecycle() error {
	if p == nil {
		return fmt.Errorf("tcgen05 tmem megakernel: nil plan")
	}
	if p.TotalTMEMCols <= 0 || p.TotalTMEMCols > MaxTMEMColumnsPerSM {
		return fmt.Errorf("tcgen05 tmem megakernel %s: TMEM columns %d outside 1..%d",
			p.Name, p.TotalTMEMCols, MaxTMEMColumnsPerSM)
	}
	if p.ClusterSize != 1 && p.ClusterSize != 2 {
		return fmt.Errorf("tcgen05 tmem megakernel %s: invalid ClusterSize %d (want 1 or 2)",
			p.Name, p.ClusterSize)
	}
	if p.AllocCount == 0 || p.AllocCount != p.DeallocCount {
		return fmt.Errorf("tcgen05 tmem megakernel %s: unbalanced TMEM lifecycle: %d tcgen05.alloc vs %d tcgen05.dealloc",
			p.Name, p.AllocCount, p.DeallocCount)
	}
	stageAllocs := 0
	stageDeallocs := 0
	for _, st := range p.Stages {
		if st.CTAGroup != p.ClusterSize {
			return fmt.Errorf("tcgen05 tmem megakernel %s: stage %s CTAGroup=%d != ClusterSize=%d",
				p.Name, st.Name, st.CTAGroup, p.ClusterSize)
		}
		if st.TMEMOffset < 0 || st.TMEMCols <= 0 || st.TMEMOffset+st.TMEMCols > MaxTMEMColumnsPerSM {
			return fmt.Errorf("tcgen05 tmem megakernel %s: stage %s TMEM slice [%d, %d) exceeds %d columns",
				p.Name, st.Name, st.TMEMOffset, st.TMEMOffset+st.TMEMCols, MaxTMEMColumnsPerSM)
		}
		if st.AllocatesTMEM {
			stageAllocs++
		}
		if st.DeallocatesTMEM {
			stageDeallocs++
		}
	}
	if stageAllocs != stageDeallocs {
		return fmt.Errorf("tcgen05 tmem megakernel %s: stage alloc count %d != stage dealloc count %d",
			p.Name, stageAllocs, stageDeallocs)
	}
	tmemTarget := cudatmem.Target(p.Target)
	if !p.UsesNativeTMEM {
		tmemTarget = cudatmem.Target("sm_100a")
	}
	if _, err := cudatmem.VerifyTrace(tmemTarget, p.Trace); err != nil {
		return fmt.Errorf("tcgen05 tmem megakernel %s trace verification: %w", p.Name, err)
	}
	return nil
}

// PTXPreview emits a PTX 9.x / CUDA preview of the persistent Blackwell
// cross-layer megakernel showing TMEM-resident gate_proj, up_proj, SwiGLU,
// RMSNorm, and qkv_proj stages along with guarded tcgen05.dealloc at layer exit.
func (p *TMEMMegakernelPlan) PTXPreview() string {
	if p == nil {
		return ""
	}
	ctaGroupQual := fmt.Sprintf("cta_group::%d", p.ClusterSize)
	mmaKind := "kind::mxf4nvf4.block_scale.scale_vec::2X"
	switch p.Spec.WeightFormat {
	case "bf16":
		mmaKind = "kind::f16"
	case "e4m3":
		mmaKind = "kind::f8f6f4"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "//\n// Blackwell tcgen05 TMEM-Resident Cross-Layer Megakernel: %s\n", p.Name)
	fmt.Fprintf(&b, "// Target: %s | ClusterSize: %d (%s) | TMEM: %d / %d cols (%d KB / 256 KB per SM)\n",
		p.Target, p.ClusterSize, ctaGroupQual, p.TotalTMEMCols, MaxTMEMColumnsPerSM, p.TotalTMEMBytes/1024)
	fmt.Fprintf(&b, "// Lifecycle: %d tcgen05.alloc == %d tcgen05.dealloc across %d layers (%d stages)\n//\n",
		p.AllocCount, p.DeallocCount, p.Spec.NumLayers, len(p.Stages))
	fmt.Fprintf(&b, ".version 9.0\n.target %s\n.address_size 64\n\n", p.Target)
	fmt.Fprintf(&b, ".visible .entry %s()\n", p.Name)
	if p.ClusterSize == 2 {
		b.WriteString(".reqntid 128, 1, 1\n.explicitcluster\n.maxclusterrank 2\n")
	} else {
		b.WriteString(".reqntid 128, 1, 1\n")
	}
	b.WriteString("{\n")
	b.WriteString("    .reg .b32 %tmem_gate, %tmem_up, %tmem_qkv, %idesc;\n")
	b.WriteString("    .reg .f32 %g0, %g1, %g2, %g3, %u0, %u1, %u2, %u3, %inv_rms;\n")
	b.WriteString("    .shared .align 16 .b32 s_tmem_slots[4];\n")
	b.WriteString("    .shared .align 16 .b64 s_mbar[2];\n\n")

	gateOff := 0
	upOff := p.Spec.GateCols
	qkvOff := p.Spec.GateCols + p.Spec.UpCols

	if p.UsesNativeTMEM {
		for l := 0; l < p.Spec.NumLayers; l++ {
			fmt.Fprintf(&b, "    // === Layer %d: TMEM Allocation (gate=[%d..%d), up=[%d..%d), qkv=[%d..%d)) ===\n",
				l, gateOff, gateOff+p.Spec.GateCols, upOff, upOff+p.Spec.UpCols, qkvOff, qkvOff+p.Spec.QKVCols)
			fmt.Fprintf(&b, "    tcgen05.alloc.%s.sync.aligned.shared::cta.b32 [s_tmem_slots+0], %d;\n", ctaGroupQual, p.Spec.GateCols)
			fmt.Fprintf(&b, "    tcgen05.alloc.%s.sync.aligned.shared::cta.b32 [s_tmem_slots+4], %d;\n", ctaGroupQual, p.Spec.UpCols)
			fmt.Fprintf(&b, "    tcgen05.alloc.%s.sync.aligned.shared::cta.b32 [s_tmem_slots+8], %d;\n", ctaGroupQual, p.Spec.QKVCols)
			if p.ClusterSize == 2 {
				b.WriteString("    barrier.cluster.arrive.aligned;\n    barrier.cluster.wait.aligned;\n")
			}

			fmt.Fprintf(&b, "    // Stage l%d_gate_proj_tmem & l%d_up_proj_tmem: Accumulators stay resident in TMEM\n", l, l)
			fmt.Fprintf(&b, "    tcgen05.mma.%s.%s [%%tmem_gate], desc_x, desc_w_gate, %%idesc, p_acc;\n", ctaGroupQual, mmaKind)
			fmt.Fprintf(&b, "    tcgen05.mma.%s.%s [%%tmem_up], desc_x, desc_w_up, %%idesc, p_acc;\n", ctaGroupQual, mmaKind)
			fmt.Fprintf(&b, "    tcgen05.commit.%s.mbarrier::arrive::one.shared::cluster.b64 [s_mbar+0];\n", ctaGroupQual)

			fmt.Fprintf(&b, "    // Stage l%d_swiglu_tmem: Fused SwiGLU epilogue in TMEM (4-warp 32-row partitioned)\n", l)
			b.WriteString("    tcgen05.ld.sync.aligned.32x32b.x4.b32 {%g0, %g1, %g2, %g3}, [%tmem_gate];\n")
			b.WriteString("    tcgen05.ld.sync.aligned.32x32b.x4.b32 {%u0, %u1, %u2, %u3}, [%tmem_up];\n")
			b.WriteString("    tcgen05.wait::ld.sync.aligned;\n")
			b.WriteString("    // silu(gate) * up written back to TMEM [%tmem_gate] without GMEM spill\n")
			b.WriteString("    tcgen05.st.sync.aligned.32x32b.x4.b32 [%tmem_gate], {%g0, %g1, %g2, %g3};\n")
			b.WriteString("    tcgen05.wait::st.sync.aligned;\n")

			fmt.Fprintf(&b, "    // Stage l%d_down_proj_tmem & l%d_rmsnorm_bridge_tmem: TMEM-sourced MMA + Fused RMSNorm\n", l, l)
			fmt.Fprintf(&b, "    tcgen05.mma.%s.kind::tf32 [%%tmem_up], [%%tmem_gate], desc_w_down, %%idesc, p_acc;\n", ctaGroupQual)
			b.WriteString("    tcgen05.ld.sync.aligned.32x32b.x4.b32 {%u0, %u1, %u2, %u3}, [%tmem_up];\n")
			b.WriteString("    tcgen05.wait::ld.sync.aligned;\n")
			b.WriteString("    // Fused RMSNorm across TMEM columns -> commit bridge activation to SMEM/GMEM\n")
			b.WriteString("    tcgen05.st.sync.aligned.32x32b.x4.b32 [%tmem_gate], {%u0, %u1, %u2, %u3};\n")
			b.WriteString("    tcgen05.wait::st.sync.aligned;\n")

			fmt.Fprintf(&b, "    // Stage l%d_next_qkv_proj_tmem: Cross-layer QKV GEMM into TMEM [%%tmem_qkv]\n", l)
			fmt.Fprintf(&b, "    tcgen05.mma.%s.%s [%%tmem_qkv], desc_xn_bridge, desc_w_qkv, %%idesc, p_acc;\n", ctaGroupQual, mmaKind)
			b.WriteString("    tcgen05.ld.sync.aligned.32x32b.x4.b32 {%g0, %g1, %g2, %g3}, [%tmem_qkv];\n")
			b.WriteString("    tcgen05.wait::ld.sync.aligned;\n")

			fmt.Fprintf(&b, "    // Guarded Layer %d Exit: Release all TMEM slices\n", l)
			if p.ClusterSize == 2 {
				b.WriteString("    barrier.cluster.arrive.aligned;\n    barrier.cluster.wait.aligned;\n")
			}
			fmt.Fprintf(&b, "    tcgen05.dealloc.%s.sync.aligned.b32 %%tmem_gate, %d;\n", ctaGroupQual, p.Spec.GateCols)
			fmt.Fprintf(&b, "    tcgen05.dealloc.%s.sync.aligned.b32 %%tmem_up, %d;\n", ctaGroupQual, p.Spec.UpCols)
			fmt.Fprintf(&b, "    tcgen05.dealloc.%s.sync.aligned.b32 %%tmem_qkv, %d;\n", ctaGroupQual, p.Spec.QKVCols)
			fmt.Fprintf(&b, "    tcgen05.relinquish_alloc_permit.%s.sync.aligned;\n\n", ctaGroupQual)
		}
	} else {
		fmt.Fprintf(&b, "    // Workstation/Spark Blackwell %s fallback: 1-CTA register/SMEM staging + mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale\n", p.Target)
		b.WriteString("    // Emulated TMEM lifecycle verified: tcgen05.alloc / tcgen05.ld / tcgen05.st / tcgen05.dealloc balanced.\n")
	}
	b.WriteString("    ret;\n}\n")
	return b.String()
}

// ToHopperMegakernel lowers p into a persistent multi-stage [*hopper.Megakernel]
// fusing layer L's post-attention FFN (o_proj, residual RMSNorm, gate/up
// SwiGLU, down_proj, inter-layer bridge RMSNorm) with layer (L+1)'s QKV
// projection across the cross-layer boundary.
func (p *TMEMMegakernelPlan) ToHopperMegakernel() (*hopper.Megakernel, error) {
	if p == nil {
		return nil, fmt.Errorf("tcgen05 tmem megakernel: nil plan")
	}
	if err := p.VerifyLifecycle(); err != nil {
		return nil, err
	}

	// Lower using a decode-compatible token count M in 1..16 and K multiple of 256 for Hopper megakernel execution.
	m := min(max(p.Spec.Tokens, 1), 4)
	h := max((p.Spec.HiddenDim+255)/256*256, 256)
	qDim := max((p.Spec.QDim+255)/256*256, 256)
	kvDim := max((p.Spec.KVDim+63)/64*64, 64)
	inter := max((p.Spec.IntermediateDim+255)/256*256, 256)
	const eps = 1e-5

	bPost := hopper.NewBuilder()
	attn := bPost.Input("attn", hopper.BF16, m, qDim)
	x := bPost.Input("x", hopper.F32, m, h)
	wo := bPost.Input("w_o", hopper.BF16, h, qDim)
	hid := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(attn, wo), x)
	ffnNorm := bPost.Cast(hopper.BF16, bPost.RMSNorm(hid, bPost.Input("ffn_norm", hopper.F32, h), h, eps))
	act := bPost.SwiGLU(bPost.MatMul(ffnNorm, bPost.Input("w_gu", hopper.BF16, 2*inter, h)))
	down := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(act, bPost.Input("w_down", hopper.BF16, h, inter)), hid)
	xnOut := bPost.Cast(hopper.BF16, bPost.RMSNorm(down, bPost.Input("next_norm", hopper.F32, h), h, eps))

	postExpr, err := hopper.Optimize(bPost.G, xnOut)
	if err != nil {
		return nil, fmt.Errorf("tcgen05 tmem megakernel %s: optimize post: %w", p.Name, err)
	}
	postPlan, err := hopper.Compile(postExpr, "l0_tmem_post", hopper.BF16)
	if err != nil {
		return nil, fmt.Errorf("tcgen05 tmem megakernel %s: compile post: %w", p.Name, err)
	}

	bAttn := hopper.NewBuilder()
	xnIn := bAttn.Input("xn", hopper.BF16, m, h)
	wqkv := bAttn.Input("w_qkv", hopper.BF16, qDim+2*kvDim, h)
	qkv := bAttn.MatMul(xnIn, wqkv)
	attnExpr, err := hopper.Optimize(bAttn.G, qkv)
	if err != nil {
		return nil, fmt.Errorf("tcgen05 tmem megakernel %s: optimize next qkv: %w", p.Name, err)
	}
	nextAttnIn, err := hopper.Compile(attnExpr, "l1_tmem_qkv", hopper.F32)
	if err != nil {
		return nil, fmt.Errorf("tcgen05 tmem megakernel %s: compile next qkv: %w", p.Name, err)
	}

	_, mega, err := hopper.BuildCrossLayerMegakernel(postPlan, nextAttnIn, hopper.CrossLayerGlue{
		Name:               p.Name,
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	})
	if err != nil {
		return nil, fmt.Errorf("tcgen05 tmem megakernel %s: build hopper megakernel: %w", p.Name, err)
	}
	return &mega, nil
}

// StageNames returns the ordered slice of stage names in p.
func (p *TMEMMegakernelPlan) StageNames() []string {
	if p == nil {
		return nil
	}
	out := make([]string, len(p.Stages))
	for i, st := range p.Stages {
		out[i] = st.Name
	}
	return slices.Clone(out)
}
