package hopper

import (
	"fmt"
	"slices"
	"strings"
)

// WarpSpecializedConfig configures warp-specialized TMA producer-consumer
// execution, dynamic register reallocation (setmaxnreg), N-stage shared-memory
// mbarrier ring buffering, and Programmatic Dependent Launch (griddepcontrol)
// for a persistent [Megakernel].
type WarpSpecializedConfig struct {
	ProducerWarps        int  // e.g. 2 (64 threads issuing TMA cp.async.bulk.tensor)
	ConsumerWarps        int  // e.g. 6 (192 threads running wgmma / tcgen05 / MMA)
	ProducerMaxRegs      int  // e.g. 24 (via setmaxnreg.dec.sync.aligned)
	ConsumerMaxRegs      int  // e.g. 232 (via setmaxnreg.inc.sync.aligned)
	RingBufferStages     int  // e.g. 3 (triple-buffered SMEM mbarrier ring)
	EnableGridDepControl bool // PTX griddepcontrol.wait / griddepcontrol.launch_dependents
	DynamicStageRegs     bool // Per-stage register reallocation hints
}

// DefaultWarpSpecializedConfig returns the canonical Hopper/Blackwell 256-thread
// warp-specialized configuration: 2 TMA producer warps (24 regs/thread),
// 6 Tensor Core consumer warps (232 regs/thread), a 3-stage mbarrier ring
// buffer, Programmatic Dependent Launch enabled, and per-stage dynamic register
// hints enabled.
func DefaultWarpSpecializedConfig() WarpSpecializedConfig {
	return WarpSpecializedConfig{
		ProducerWarps:        2,
		ConsumerWarps:        6,
		ProducerMaxRegs:      24,
		ConsumerMaxRegs:      232,
		RingBufferStages:     3,
		EnableGridDepControl: true,
		DynamicStageRegs:     true,
	}
}

// Validate checks that cfg satisfies the 256-thread CTA warp partition,
// 8-register setmaxnreg alignment granularity, 64K CTA register file budget,
// and N-stage ring buffer depth constraints on Hopper/Blackwell (sm_90a+).
func (cfg WarpSpecializedConfig) Validate() error {
	if cfg.ProducerWarps <= 0 || cfg.ConsumerWarps <= 0 {
		return fmt.Errorf("warpspec megakernel: ProducerWarps (%d) and ConsumerWarps (%d) must be positive",
			cfg.ProducerWarps, cfg.ConsumerWarps)
	}
	if cfg.ProducerWarps+cfg.ConsumerWarps != 8 {
		return fmt.Errorf("warpspec megakernel: ProducerWarps (%d) + ConsumerWarps (%d) = %d, want 8 (256-thread CTA)",
			cfg.ProducerWarps, cfg.ConsumerWarps, cfg.ProducerWarps+cfg.ConsumerWarps)
	}
	if cfg.ProducerMaxRegs < 24 || cfg.ProducerMaxRegs > 240 || cfg.ProducerMaxRegs%8 != 0 {
		return fmt.Errorf("warpspec megakernel: ProducerMaxRegs (%d) must be in [24, 240] and aligned to 8-register granularity",
			cfg.ProducerMaxRegs)
	}
	if cfg.ConsumerMaxRegs < 24 || cfg.ConsumerMaxRegs > 240 || cfg.ConsumerMaxRegs%8 != 0 {
		return fmt.Errorf("warpspec megakernel: ConsumerMaxRegs (%d) must be in [24, 240] and aligned to 8-register granularity",
			cfg.ConsumerMaxRegs)
	}
	totalRegs := cfg.ProducerWarps*32*cfg.ProducerMaxRegs + cfg.ConsumerWarps*32*cfg.ConsumerMaxRegs
	if totalRegs > 57344 {
		return fmt.Errorf("warpspec megakernel: CTA register budget %d exceeds 57344 registers (producer %dw*%d + consumer %dw*%d)",
			totalRegs, cfg.ProducerWarps, cfg.ProducerMaxRegs, cfg.ConsumerWarps, cfg.ConsumerMaxRegs)
	}
	if cfg.RingBufferStages < 2 || cfg.RingBufferStages > 16 {
		return fmt.Errorf("warpspec megakernel: RingBufferStages (%d) must be in 2..16",
			cfg.RingBufferStages)
	}
	return nil
}

// StageRegisterHint records the per-stage producer and consumer register targets
// when [WarpSpecializedConfig.DynamicStageRegs] is enabled.
type StageRegisterHint struct {
	StageIndex   int
	StageName    string
	StageKind    string
	ProducerRegs int
	ConsumerRegs int
	TileBytes    int
}

// WarpSpecializedMegakernel wraps a persistent [Megakernel] with a
// warp-specialized TMA producer-consumer ring buffer, dynamic setmaxnreg
// register reallocation, and Programmatic Dependent Launch (griddepcontrol).
type WarpSpecializedMegakernel struct {
	Megakernel
	Config           WarpSpecializedConfig
	StageHints       []StageRegisterHint
	StageTileBytes   int
	RingSharedBytes  int
	TotalSharedBytes int
	cudaSrc          string
	ptxPreview       string
}

// BuildWarpSpecializedMegakernel lowers m into a warp-specialized producer-consumer
// persistent megakernel according to cfg.
func BuildWarpSpecializedMegakernel(m *Megakernel, cfg WarpSpecializedConfig) (*WarpSpecializedMegakernel, error) {
	if m == nil || len(m.Stages) == 0 || !cIdent(m.Name) {
		name := ""
		if m != nil {
			name = m.Name
		}
		return nil, fmt.Errorf("warpspec megakernel: invalid megakernel %q", name)
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	hints := make([]StageRegisterHint, len(m.Stages))
	maxTileBytes := 4096
	for i, st := range m.Stages {
		kind, prodRegs, consRegs, tileBytes := classifyStageWarpSpec(st, cfg)
		hints[i] = StageRegisterHint{
			StageIndex:   i,
			StageName:    st.Name,
			StageKind:    kind,
			ProducerRegs: prodRegs,
			ConsumerRegs: consRegs,
			TileBytes:    tileBytes,
		}
		if tileBytes > maxTileBytes {
			maxTileBytes = tileBytes
		}
	}
	// Align per-stage ring slot to 1024 bytes for 128B-swizzled TMA tensor tiles.
	stageTileBytes := (maxTileBytes + 1023) &^ 1023
	// Each ring stage has data bytes + 2 mbarrier words (full + empty), plus 1024B alignment slack.
	ringShared := cfg.RingBufferStages*stageTileBytes + 2*cfg.RingBufferStages*8 + 1024
	totalShared := max(m.SharedBytes(), ringShared)

	wm := &WarpSpecializedMegakernel{
		Megakernel: Megakernel{
			Name:   m.Name,
			Inputs: slices.Clone(m.Inputs),
			Temps:  slices.Clone(m.Temps),
			Output: cloneTensor(m.Output),
			Stages: slices.Clone(m.Stages),
		},
		Config:           cfg,
		StageHints:       hints,
		StageTileBytes:   stageTileBytes,
		RingSharedBytes:  ringShared,
		TotalSharedBytes: totalShared,
	}

	cudaSrc, err := wm.buildCUDA()
	if err != nil {
		return nil, err
	}
	wm.cudaSrc = cudaSrc
	wm.ptxPreview = wm.buildPTXPreview()
	return wm, nil
}

func classifyStageWarpSpec(st MegaStage, cfg WarpSpecializedConfig) (kind string, prodRegs, consRegs, tileBytes int) {
	prodRegs = cfg.ProducerMaxRegs
	consRegs = cfg.ConsumerMaxRegs
	tileBytes = max(st.Shared, 4096)
	k := st.Kernel
	switch {
	case k.Tile != nil:
		kind = "tile_gemm"
		tileBytes = max(st.Shared, 16384)
	case k.GEMM != nil:
		kind = "skinny_gemm"
		tileBytes = max(st.Shared, 8192)
	case k.MMA != nil:
		kind = "skinny_mma"
		tileBytes = max(st.Shared, 8192)
	case k.Attention != nil:
		kind = "paged_attention"
		tileBytes = max(st.Shared, 16384)
		if cfg.DynamicStageRegs {
			prodRegs = min(cfg.ProducerMaxRegs+8, 64)
			consRegs = max(cfg.ConsumerMaxRegs-16, 128)
		}
	case k.RoPE != nil:
		kind = "rope"
		if cfg.DynamicStageRegs {
			prodRegs = min(cfg.ProducerMaxRegs+8, 48)
			consRegs = max(cfg.ConsumerMaxRegs-32, 96)
		}
	default:
		kind = "row_elementwise"
		if cfg.DynamicStageRegs {
			prodRegs = min(cfg.ProducerMaxRegs+8, 48)
			consRegs = max(cfg.ConsumerMaxRegs-32, 96)
		}
	}
	// Preserve 8-register alignment and total CTA budget on dynamic hints.
	prodRegs = (prodRegs / 8) * 8
	consRegs = (consRegs / 8) * 8
	for cfg.ProducerWarps*32*prodRegs+cfg.ConsumerWarps*32*consRegs > 65536 && consRegs > 24 {
		consRegs -= 8
	}
	return kind, prodRegs, consRegs, tileBytes
}

// CUDA returns the warp-specialized megakernel CUDA/PTX device source.
func (wm *WarpSpecializedMegakernel) CUDA() string {
	if wm == nil {
		return ""
	}
	return wm.cudaSrc
}

// PTXPreview returns a structured PTX preview of the warp-specialized
// megakernel including griddepcontrol prologue/epilogue, setmaxnreg register
// handoff, and the N-stage mbarrier + cp.async.bulk.tensor ring loop.
func (wm *WarpSpecializedMegakernel) PTXPreview() string {
	if wm == nil {
		return ""
	}
	return wm.ptxPreview
}

// Launcher returns the host launch functions for wm, including
// cudaLaunchAttributeProgrammaticStreamSerialization when EnableGridDepControl
// is enabled.
func (wm *WarpSpecializedMegakernel) Launcher() (string, error) {
	if wm == nil || len(wm.Stages) == 0 || !cIdent(wm.Name) {
		return "", fmt.Errorf("warpspec megakernel: invalid megakernel")
	}
	params := wm.params()
	names := paramNames(params)
	grid, block := wm.Grid()
	smem := wm.Megakernel.SharedBytes()

	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream = nullptr) {\n",
		wm.Name, strings.Join(params, ", "))
	if smem > 0 {
		fmt.Fprintf(&b, "    constexpr int smem = %d;\n", smem)
		fmt.Fprintf(&b, "    static const cudaError_t attr_smem = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr_smem;\n", wm.Name)
	} else {
		b.WriteString("    constexpr int smem = 0;\n")
	}
	fmt.Fprintf(&b, "    const dim3 grid(%d), block(%d);\n", grid[0], block[0])
	if wm.Config.EnableGridDepControl {
		b.WriteString("#if !defined(EG_HOST_EMU) && !defined(EG_EMU) && !defined(CUDAHOSTEMU)\n")
		b.WriteString("    cudaLaunchAttribute pdl_attr{};\n")
		b.WriteString("    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;\n")
		b.WriteString("    pdl_attr.val.programmaticStreamSerializationAllowed = 1;\n")
		b.WriteString("    cudaLaunchConfig_t pdl_cfg{};\n")
		b.WriteString("    pdl_cfg.gridDim = grid;\n")
		b.WriteString("    pdl_cfg.blockDim = block;\n")
		b.WriteString("    pdl_cfg.dynamicSmemBytes = smem;\n")
		b.WriteString("    pdl_cfg.stream = stream;\n")
		b.WriteString("    pdl_cfg.attrs = &pdl_attr;\n")
		b.WriteString("    pdl_cfg.numAttrs = 1;\n")
		fmt.Fprintf(&b, "    (void)cudaLaunchKernelEx(&pdl_cfg, %s, %s);\n", wm.Name, strings.Join(names, ", "))
		b.WriteString("#else\n")
		fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, smem, stream, %s);\n", wm.Name, strings.Join(names, ", "))
		b.WriteString("#endif\n")
	} else {
		fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, smem, stream, %s);\n", wm.Name, strings.Join(names, ", "))
	}
	b.WriteString("}\n")
	fmt.Fprintf(&b, "\ninline void %s_run(%s, cudaStream_t stream = nullptr) {\n    %s_launch(%s, stream);\n}\n",
		wm.Name, strings.Join(params, ", "), wm.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns a self-contained CUDA translation unit for wm ([Prelude],
// [ReadyPrelude], warp-specialized TMA/mbarrier/PDL helpers, stage device
// functions, the persistent megakernel, and host launchers).
func (wm *WarpSpecializedMegakernel) Source() (string, error) {
	if wm == nil {
		return "", fmt.Errorf("warpspec megakernel: nil megakernel")
	}
	l, err := wm.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + wm.cudaSrc + "\n" + l, nil
}

func (wm *WarpSpecializedMegakernel) buildCUDA() (string, error) {
	cfg := wm.Config
	var b strings.Builder
	b.WriteString(ReadyPrelude)
	b.WriteString("\n")
	b.WriteString(emitWarpSpecMegakernelPrelude(cfg))

	seenHelpers := make(map[string]bool)
	for _, st := range wm.Stages {
		if st.Kernel.Row != nil && st.Kernel.Row.Slots > 0 && !seenHelpers["cluster_prelude"] {
			seenHelpers["cluster_prelude"] = true
			b.WriteString("\n")
			b.WriteString(ClusterPrelude)
		}
		pre, fnBody, err := stageDeviceParts(st, seenHelpers)
		if err != nil {
			return "", err
		}
		b.WriteString("\n")
		b.WriteString(pre)
		b.WriteString(fnBody)
	}

	grid, _ := wm.Grid()
	fmt.Fprintf(&b, "\n// %s: warp-specialized %d-stage persistent megakernel (%d blocks, %d producer warps @ %d regs, %d consumer warps @ %d regs, %d-stage SMEM ring, griddepcontrol=%v).\n",
		wm.Name, len(wm.Stages), grid[0],
		cfg.ProducerWarps, cfg.ProducerMaxRegs,
		cfg.ConsumerWarps, cfg.ConsumerMaxRegs,
		cfg.RingBufferStages, cfg.EnableGridDepControl)

	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n",
		wm.Name, strings.Join(wm.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int EG_WS_PROD_WARPS = %d, EG_WS_CONS_WARPS = %d;\n",
		cfg.ProducerWarps, cfg.ConsumerWarps)
	fmt.Fprintf(&b, "    constexpr int EG_WS_PROD_REGS = %d, EG_WS_CONS_REGS = %d;\n",
		cfg.ProducerMaxRegs, cfg.ConsumerMaxRegs)
	fmt.Fprintf(&b, "    constexpr int EG_WS_RING_STAGES = %d, EG_WS_STAGE_BYTES = %d;\n",
		cfg.RingBufferStages, wm.StageTileBytes)
	b.WriteString("    (void)EG_WS_PROD_WARPS; (void)EG_WS_CONS_WARPS; (void)EG_WS_PROD_REGS; (void)EG_WS_CONS_REGS; (void)EG_WS_STAGE_BYTES;\n")
	b.WriteString("    const int eg_bid = blockIdx.x;\n")
	b.WriteString("    const int eg_warp = threadIdx.x >> 5;\n")
	b.WriteString("    __shared__ unsigned long long eg_ws_mbar_full[EG_WS_RING_STAGES];\n")
	b.WriteString("    __shared__ unsigned long long eg_ws_mbar_empty[EG_WS_RING_STAGES];\n")

	if cfg.EnableGridDepControl {
		b.WriteString("    // Programmatic Dependent Launch prologue: wait for upstream kernel writes.\n")
		b.WriteString("    eg_ws_griddepcontrol_wait();\n")
	}

	b.WriteString("    // Initialize N-stage SMEM mbarrier ring buffer and execute warp-specialized register handoff.\n")
	b.WriteString("    eg_ws_ring_init<EG_WS_RING_STAGES>(eg_ws_mbar_full, eg_ws_mbar_empty);\n")
	b.WriteString("    if (eg_warp < EG_WS_PROD_WARPS) {\n")
	fmt.Fprintf(&b, "        eg_ws_producer_reg_handoff<%d>();\n", cfg.ProducerMaxRegs)
	b.WriteString("    } else {\n")
	fmt.Fprintf(&b, "        eg_ws_consumer_reg_handoff<%d>();\n", cfg.ConsumerMaxRegs)
	b.WriteString("    }\n")

	fmt.Fprintf(&b, "    epoch = eg_ready_resolve_epoch(&ready[%d], epoch, %d);\n", wm.EpochSlot(), grid[0])

	for i, st := range wm.Stages {
		gx, gy, gz := st.Grid[0], st.Grid[1], st.Grid[2]
		hint := wm.StageHints[i]
		fmt.Fprintf(&b, "    if (eg_bid < %d) {\n", st.BlockBase+st.Blocks)
		fmt.Fprintf(&b, "        // Stage %d (%s, %s): producer_regs=%d consumer_regs=%d ring_slot=%d/%d (prefetch stage %d -> compute stage %d)\n",
			i, st.Name, hint.StageKind, hint.ProducerRegs, hint.ConsumerRegs,
			i%cfg.RingBufferStages, cfg.RingBufferStages, i+1, i)
		if cfg.DynamicStageRegs {
			fmt.Fprintf(&b, "        eg_ws_stage_reg_hint<%d, %d>(eg_warp < EG_WS_PROD_WARPS);\n",
				hint.ProducerRegs, hint.ConsumerRegs)
		}
		fmt.Fprintf(&b, "        eg_ws_ring_step<EG_WS_RING_STAGES>(eg_ws_mbar_full, eg_ws_mbar_empty, %d, eg_warp < EG_WS_PROD_WARPS, %d);\n",
			i, hint.TileBytes)
		fmt.Fprintf(&b, "        const int eg_local = eg_bid - %d;\n", st.BlockBase)
		fmt.Fprintf(&b, "        const dim3 eg_bidx(eg_local %% %d, (eg_local / %d) %% %d, eg_local / %d);\n",
			gx, gx, gy, gx*gy)
		fmt.Fprintf(&b, "        const dim3 eg_gdim(%d, %d, %d);\n", gx, gy, gz)

		if len(st.WaitSlots) > 0 {
			b.WriteString("        if (threadIdx.x == 0) {\n")
			for _, ws := range st.WaitSlots {
				need := wm.Stages[ws].Blocks
				fmt.Fprintf(&b, "            eg_ready_wait(&ready[%d], epoch, %d);\n", ws, need)
			}
			b.WriteString("        }\n        __syncthreads();\n")
		}

		k := st.Kernel
		switch {
		case k.GEMM != nil:
			fmt.Fprintf(&b, "        %s_dev<%d>(eg_bidx, eg_gdim, %s);\n",
				st.Name, k.M, strings.Join(k.Args, ", "))
		case k.Tile != nil:
			fmt.Fprintf(&b, "        %s_dev(eg_bidx, eg_gdim, %d, %s);\n",
				st.Name, k.M, strings.Join(k.Args, ", "))
		case k.MMA != nil:
			fmt.Fprintf(&b, "        %s_dev(eg_bidx, eg_gdim, %d, %s);\n",
				st.Name, k.M, strings.Join(k.Args, ", "))
		case k.RoPE != nil:
			fmt.Fprintf(&b, "        %s_dev(eg_bidx, eg_gdim, %s);\n",
				st.Name, strings.Join(k.Args, ", "))
		case k.Attention != nil:
			args := slices.Clone(k.Args)
			if len(args) < len(k.Attention.params()) {
				if len(args) >= 4 {
					args = append(args[:4], append([]string{fmt.Sprint(k.MaxPages)}, args[4:]...)...)
				}
				if len(args) == len(k.Attention.params())-1 {
					args = append(args, "nullptr")
				}
			}
			fmt.Fprintf(&b, "        %s_dev(eg_bidx, eg_gdim, %s);\n",
				st.Name, strings.Join(args, ", "))
		default:
			fmt.Fprintf(&b, "        %s_dev(eg_bidx, eg_gdim, %d, %s);\n",
				st.Name, k.Rows, strings.Join(k.Args, ", "))
		}

		if st.SignalSlot >= 0 {
			b.WriteString("        __syncthreads();\n        if (threadIdx.x == 0) {\n")
			for _, out := range st.Outputs {
				nbytes := size(out.Shape) * out.DType.Bits() / 8
				if nbytes > 0 {
					fmt.Fprintf(&b, "            EG_READY_STAGE(&ready[%d], epoch, %d, %s, %d);\n",
						st.SignalSlot, st.Blocks, out.Name, nbytes)
				}
			}
			fmt.Fprintf(&b, "            eg_ready_signal(&ready[%d], epoch, 1);\n", st.SignalSlot)
			b.WriteString("        }\n")
		} else if i == len(wm.Stages)-1 && cfg.EnableGridDepControl {
			b.WriteString("        __syncthreads();\n")
			b.WriteString("        if (threadIdx.x == 0) {\n")
			b.WriteString("            // Final stage output tile ready: trigger dependent kernel CTAs while epilogue drains.\n")
			b.WriteString("            eg_ws_griddepcontrol_launch_dependents();\n")
			b.WriteString("        }\n")
		}
		b.WriteString("        return;\n    }\n")
	}
	b.WriteString("}\n")
	return b.String(), nil
}

func emitWarpSpecMegakernelPrelude(cfg WarpSpecializedConfig) string {
	var b strings.Builder
	b.WriteString("#ifndef EG_WARPSPEC_MEGAKERNEL_PRELUDE\n")
	b.WriteString("#define EG_WARPSPEC_MEGAKERNEL_PRELUDE\n\n")
	fmt.Fprintf(&b, `// Dynamic register handoff between TMA DMA producer warps (%d warps, %d regs)
// and Tensor Core consumer warps (%d warps, %d regs).
template <int R>
EG_DEV void eg_ws_producer_reg_handoff() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile("setmaxnreg.dec.sync.aligned %%d;\n" :: "n"(%d));
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %%0;\n" :: "n"(R));
#endif
}

template <int R>
EG_DEV void eg_ws_consumer_reg_handoff() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile("setmaxnreg.inc.sync.aligned %%d;\n" :: "n"(%d));
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %%0;\n" :: "n"(R));
#endif
}

template <int PROD_R, int CONS_R>
EG_DEV void eg_ws_stage_reg_hint(bool is_producer) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    if (is_producer) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %%0;\n" :: "n"(PROD_R));
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %%0;\n" :: "n"(CONS_R));
    }
#else
    (void)is_producer;
#endif
}

// Programmatic Dependent Launch (PDL) primitives (sm_90a+).
EG_DEV void eg_ws_griddepcontrol_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

EG_DEV void eg_ws_griddepcontrol_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

// N-stage SMEM mbarrier ring buffer primitives overlapping TMA 2D tensor tile
// prefetch for stage s+1 with Tensor Core compute of stage s.
template <int STAGES>
EG_DEV void eg_ws_ring_init(unsigned long long* full_bar, unsigned long long* empty_bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            uint32_t f_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&full_bar[s]));
            uint32_t e_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&empty_bar[s]));
            asm volatile("mbarrier.init.shared.b64 [%%0], %%1;\n" :: "r"(f_ptr), "r"(1u) : "memory");
            asm volatile("mbarrier.init.shared.b64 [%%0], %%1;\n" :: "r"(e_ptr), "r"(1u) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
#else
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            full_bar[s] = 0ULL;
            empty_bar[s] = 0ULL;
        }
    }
    __syncthreads();
#endif
}

EG_DEV void eg_ws_tma_load_2d(uint32_t smem_dst, uint64_t tmap_desc, int32_t coord_x, int32_t coord_y, uint32_t mbar_ptr, uint32_t tx_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%%0], %%1;\n\t"
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%%2], [%%3, {%%4, %%5}], [%%0];\n"
        :: "r"(mbar_ptr), "r"(tx_bytes), "r"(smem_dst), "l"(tmap_desc), "r"(coord_x), "r"(coord_y)
        : "memory");
#else
    (void)smem_dst; (void)tmap_desc; (void)coord_x; (void)coord_y; (void)mbar_ptr; (void)tx_bytes;
#endif
}

EG_DEV void eg_ws_mbar_wait_parity(unsigned long long* bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "EG_WS_MBAR_WAIT_%%=:\n\t"
        "mbarrier.try_wait.parity.shared.b64 p, [%%0], %%1;\n\t"
        "@!p bra EG_WS_MBAR_WAIT_%%=;\n\t"
        "}\n"
        :: "r"(bar_ptr), "r"(parity) : "memory");
#else
    (void)bar; (void)parity;
#endif
}

template <int STAGES>
EG_DEV void eg_ws_ring_step(unsigned long long* full_bar, unsigned long long* empty_bar, int stage_idx, bool is_producer, uint32_t tx_bytes) {
    const int slot = stage_idx %% STAGES;
    const int prefetch_slot = (stage_idx + 1) %% STAGES;
    const uint32_t parity = static_cast<uint32_t>((stage_idx / STAGES) & 1);
    (void)full_bar; (void)empty_bar; (void)slot; (void)prefetch_slot; (void)parity; (void)is_producer; (void)tx_bytes;
}

#endif // EG_WARPSPEC_MEGAKERNEL_PRELUDE
`, cfg.ProducerWarps, cfg.ProducerMaxRegs, cfg.ConsumerWarps, cfg.ConsumerMaxRegs, cfg.ProducerMaxRegs, cfg.ConsumerMaxRegs)
	return b.String()
}

func (wm *WarpSpecializedMegakernel) buildPTXPreview() string {
	cfg := wm.Config
	var b strings.Builder
	fmt.Fprintf(&b, "// PTX Preview: %s (producer=%dw*%dregs, consumer=%dw*%dregs, ring_stages=%d, pdl=%v)\n",
		wm.Name, cfg.ProducerWarps, cfg.ProducerMaxRegs, cfg.ConsumerWarps, cfg.ConsumerMaxRegs,
		cfg.RingBufferStages, cfg.EnableGridDepControl)
	b.WriteString(".version 8.5\n.target sm_90a\n.address_size 64\n\n")
	fmt.Fprintf(&b, ".visible .entry %s() .reqntid 256, 1, 1 {\n", wm.Name)
	if cfg.EnableGridDepControl {
		b.WriteString("    // PDL Prologue: wait on primary grid completion before dependent loads\n")
		b.WriteString("    griddepcontrol.wait;\n\n")
	}
	fmt.Fprintf(&b, "    // %d-stage SMEM mbarrier ring buffer initialization\n", cfg.RingBufferStages)
	for s := 0; s < cfg.RingBufferStages; s++ {
		fmt.Fprintf(&b, "    mbarrier.init.shared.b64 [eg_ws_mbar_full+%d], 1;\n", s*8)
		fmt.Fprintf(&b, "    mbarrier.init.shared.b64 [eg_ws_mbar_empty+%d], 1;\n", s*8)
	}
	b.WriteString("    fence.mbarrier_init.release.cluster;\n    bar.sync 0;\n\n")
	fmt.Fprintf(&b, "    // Warp-specialized dynamic register handoff (producer warps 0..%d vs consumer warps %d..7)\n",
		cfg.ProducerWarps-1, cfg.ProducerWarps)
	fmt.Fprintf(&b, "    setmaxnreg.dec.sync.aligned %d;\n", cfg.ProducerMaxRegs)
	fmt.Fprintf(&b, "    setmaxnreg.inc.sync.aligned %d;\n\n", cfg.ConsumerMaxRegs)

	for i, st := range wm.Stages {
		hint := wm.StageHints[i]
		slot := i % cfg.RingBufferStages
		nextSlot := (i + 1) % cfg.RingBufferStages
		parity := (i / cfg.RingBufferStages) & 1
		fmt.Fprintf(&b, "    // Stage %d: %s (%s) [slot=%d, prefetch_slot=%d, parity=%d, prod_regs=%d, cons_regs=%d]\n",
			i, st.Name, hint.StageKind, slot, nextSlot, parity, hint.ProducerRegs, hint.ConsumerRegs)
		if cfg.DynamicStageRegs {
			fmt.Fprintf(&b, "    // dynamic stage reg hint: producer=%d consumer=%d\n", hint.ProducerRegs, hint.ConsumerRegs)
		}
		fmt.Fprintf(&b, "    mbarrier.arrive.expect_tx.shared.b64 _, [eg_ws_mbar_full+%d], %d;\n", nextSlot*8, hint.TileBytes)
		fmt.Fprintf(&b, "    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [smem_stage_%d], [tmap_stage_%d, {coord_x, coord_y}], [eg_ws_mbar_full+%d];\n",
			nextSlot, min(i+1, len(wm.Stages)-1), nextSlot*8)
		fmt.Fprintf(&b, "    mbarrier.try_wait.parity.shared.b64 p, [eg_ws_mbar_full+%d], %d;\n\n", slot*8, parity)
	}

	if cfg.EnableGridDepControl {
		b.WriteString("    // PDL Epilogue: launch dependent grids as soon as final stage's last output tile is ready\n")
		b.WriteString("    griddepcontrol.launch_dependents;\n")
	}
	b.WriteString("    ret;\n}\n")
	return b.String()
}
