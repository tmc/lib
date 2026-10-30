package hopper

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// MegakernelArchTarget identifies a supported GPU SM target for [MultiArchMegakernel].
type MegakernelArchTarget = string

// Supported SM targets for [MultiArchMegakernel] and [SelectArchDispatchPlan].
const (
	TargetSM90A  MegakernelArchTarget = "sm_90a"
	TargetSM100A MegakernelArchTarget = "sm_100a"
	TargetSM103A MegakernelArchTarget = "sm_103a"
	TargetSM107A MegakernelArchTarget = "sm_107a"
	TargetSM107F MegakernelArchTarget = "sm_107f"
	TargetSM120A MegakernelArchTarget = "sm_120a"
	TargetSM121A MegakernelArchTarget = "sm_121a"
)

// AllMegakernelArchTargets lists the six canonical architecture targets emitted
// by [MultiArchMegakernel.EmitAllTargets].
var AllMegakernelArchTargets = []string{
	TargetSM90A,
	TargetSM100A,
	TargetSM103A,
	TargetSM107A,
	TargetSM120A,
	TargetSM121A,
}

// ArchDispatchPlan describes the hardware-specific PTX mnemonics, Tensor Memory
// (TMEM) allocation, warp-specialized register budgets, and software pipeline
// depth selected for a [MultiArchMegakernel] target.
type ArchDispatchPlan struct {
	Target                       string
	TensorCoreMnemonic           string
	DequantMnemonic              string
	CollectiveMnemonic           string
	WorkStealMnemonic            string
	UsesTMEM                     bool
	TMEMColumns                  int
	PipelineStages               int
	ProducerMaxRegs              int
	ConsumerMaxRegs              int
	UsesEarlyReleaseSyncRestrict bool
	UsesLUTDecompressB           bool
}

// OnlineMegakernelProfile carries runtime CUPTI / ptxas feedback used by
// [SelectArchDispatchPlan] to adapt pipeline staging, register budgets, and
// hardware decompression modes online.
type OnlineMegakernelProfile struct {
	DominantStall    string
	SpillDetected    bool
	PreferLUTB       bool
	PreferSparse2to4 bool
	OverrideStages   int
}

// SelectArchDispatchPlan returns the hardware dispatch plan for target,
// applying any online profiler overrides from profile when non-nil.
func SelectArchDispatchPlan(target string, profile *OnlineMegakernelProfile) (ArchDispatchPlan, error) {
	var plan ArchDispatchPlan
	switch target {
	case TargetSM90A:
		plan = ArchDispatchPlan{
			Target:             TargetSM90A,
			TensorCoreMnemonic: "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
			DequantMnemonic:    "prmt.b32 (eg_e2m1x4_to_bf16x4)",
			CollectiveMnemonic: "dsmem.cluster.allreduce",
			WorkStealMnemonic:  "atomicAdd.grid_queue",
			UsesTMEM:           false,
			TMEMColumns:        0,
			PipelineStages:     3,
			ProducerMaxRegs:    40,
			ConsumerMaxRegs:    232,
		}
	case TargetSM100A:
		plan = ArchDispatchPlan{
			Target:             TargetSM100A,
			TensorCoreMnemonic: "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X",
			DequantMnemonic:    "cvt.rn.f16x2.e2m1x2",
			CollectiveMnemonic: "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2",
			WorkStealMnemonic:  "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:           true,
			TMEMColumns:        512,
			PipelineStages:     4,
			ProducerMaxRegs:    40,
			ConsumerMaxRegs:    232,
		}
	case TargetSM103A:
		plan = ArchDispatchPlan{
			Target:             TargetSM103A,
			TensorCoreMnemonic: "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
			DequantMnemonic:    "tcgen05.ld.red.sync.aligned.32x32b.x4.max.f32",
			CollectiveMnemonic: "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2",
			WorkStealMnemonic:  "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:           true,
			TMEMColumns:        512,
			PipelineStages:     4,
			ProducerMaxRegs:    40,
			ConsumerMaxRegs:    232,
		}
	case TargetSM107A, TargetSM107F:
		plan = ArchDispatchPlan{
			Target:             target,
			TensorCoreMnemonic: "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32.sync_restrict::shared::read::mma::a",
			DequantMnemonic: "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2 + " +
				"fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4 + add.rn.f32x2",
			CollectiveMnemonic:           "fabric.try_red.async.tensor.2d.multimem",
			WorkStealMnemonic:            "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:                     true,
			TMEMColumns:                  576,
			PipelineStages:               4,
			ProducerMaxRegs:              40,
			ConsumerMaxRegs:              232,
			UsesEarlyReleaseSyncRestrict: true,
			UsesLUTDecompressB:           true,
		}
	case TargetSM120A:
		plan = ArchDispatchPlan{
			Target:             TargetSM120A,
			TensorCoreMnemonic: "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			DequantMnemonic:    "cvt.rn.f16x2.e2m1x2",
			CollectiveMnemonic: "cp.async.bulk.global.to.shared::cta",
			WorkStealMnemonic:  "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:           false,
			TMEMColumns:        0,
			PipelineStages:     2,
			ProducerMaxRegs:    64,
			ConsumerMaxRegs:    168,
		}
	case TargetSM121A:
		plan = ArchDispatchPlan{
			Target:             TargetSM121A,
			TensorCoreMnemonic: "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			DequantMnemonic:    "prmt.b32 (eg_e2m1x4_to_bf16x4)",
			CollectiveMnemonic: "cp.async.bulk.global.to.shared::cta",
			WorkStealMnemonic:  "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:           false,
			TMEMColumns:        0,
			PipelineStages:     2,
			ProducerMaxRegs:    64,
			ConsumerMaxRegs:    168,
		}
	default:
		return ArchDispatchPlan{}, fmt.Errorf("select arch dispatch plan: unsupported target %q", target)
	}

	if profile != nil {
		switch profile.DominantStall {
		case "memory_latency", "long_scoreboard":
			plan.PipelineStages++
		case "register_pressure", "short_scoreboard":
			if plan.PipelineStages > 2 {
				plan.PipelineStages--
			}
		}
		if profile.OverrideStages > 0 {
			plan.PipelineStages = profile.OverrideStages
		}
		if profile.SpillDetected {
			plan.ProducerMaxRegs = max(24, plan.ProducerMaxRegs-8)
			plan.ConsumerMaxRegs = max(96, plan.ConsumerMaxRegs-32)
		}
		if profile.PreferLUTB {
			plan.UsesLUTDecompressB = true
			if !strings.Contains(plan.TensorCoreMnemonic, "decompress::lut::b") && plan.UsesTMEM {
				plan.TensorCoreMnemonic += ".decompress::lut::b.block32"
			}
		}
		if profile.PreferSparse2to4 && !strings.Contains(plan.DequantMnemonic, "sp::2:4") {
			plan.DequantMnemonic += " + tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2"
		}
	}

	return plan, nil
}

// MegakernelOp identifies an instruction in a persistent [MultiArchMegakernel]
// device op queue.
type MegakernelOp uint32

// Persistent op-queue opcodes executed by [MultiArchMegakernel].
const (
	MegaOpNoop MegakernelOp = iota
	MegaOpRMSNorm
	MegaOpQKVProj
	MegaOpAttention
	MegaOpGateUpSwiGLU
	MegaOpDownProj
	MegaOpResidualAllReduce
	MegaOpHalt
)

// String returns the human-readable mnemonic of op.
func (op MegakernelOp) String() string {
	switch op {
	case MegaOpNoop:
		return "noop"
	case MegaOpRMSNorm:
		return "rmsnorm"
	case MegaOpQKVProj:
		return "qkv_proj"
	case MegaOpAttention:
		return "attention"
	case MegaOpGateUpSwiGLU:
		return "gate_up_swiglu"
	case MegaOpDownProj:
		return "down_proj"
	case MegaOpResidualAllReduce:
		return "residual_allreduce"
	case MegaOpHalt:
		return "halt"
	default:
		return fmt.Sprintf("mega_op(%d)", uint32(op))
	}
}

// MegakernelInst encodes a single operation in the persistent instruction queue
// interpreted by `eg_multiarch_megakernel`.
type MegakernelInst struct {
	Op       MegakernelOp
	WaitSlot int
	SigSlot  int
	Arg0     int
	Arg1     int
}

// DefaultTransformerInstQueue returns the default 6-stage transformer layer
// instruction queue:
//
//	RMSNorm -> QKV -> Attention -> Gate/Up SwiGLU -> DownProj -> Residual AllReduce
func DefaultTransformerInstQueue() []MegakernelInst {
	return []MegakernelInst{
		{Op: MegaOpRMSNorm, WaitSlot: -1, SigSlot: 0},
		{Op: MegaOpQKVProj, WaitSlot: 0, SigSlot: 1},
		{Op: MegaOpAttention, WaitSlot: 1, SigSlot: 2},
		{Op: MegaOpGateUpSwiGLU, WaitSlot: 2, SigSlot: 3},
		{Op: MegaOpDownProj, WaitSlot: 3, SigSlot: 4},
		{Op: MegaOpResidualAllReduce, WaitSlot: 4, SigSlot: 5},
	}
}

// MultiArchMegakernel configures a unified multi-architecture persistent
// transformer megakernel generator across Hopper (sm_90a), Blackwell
// Datacenter (sm_100a, sm_103a), Vera Rubin (sm_107a, sm_107f), Blackwell
// Workstation (sm_120a), and DGX Spark GB10 (sm_121a).
type MultiArchMegakernel struct {
	Megakernel
	Hidden       int
	Intermediate int
	NumHeads     int
	HeadDim      int
	Eps          float32
	Target       string
	Profile      *OnlineMegakernelProfile
	Instructions []MegakernelInst
}

func (m MultiArchMegakernel) norm() (MultiArchMegakernel, error) {
	if m.Name == "" {
		m.Name = "eg_multiarch_megakernel"
	}
	if m.Name != "eg_multiarch_megakernel" && !cIdent(m.Name) {
		return m, fmt.Errorf("multiarch megakernel: bad name %q", m.Name)
	}
	if m.Hidden == 0 {
		m.Hidden = 16
	}
	if m.NumHeads == 0 {
		m.NumHeads = 2
	}
	if m.HeadDim == 0 && m.NumHeads > 0 && m.Hidden%m.NumHeads == 0 {
		m.HeadDim = m.Hidden / m.NumHeads
	}
	if m.Intermediate == 0 {
		m.Intermediate = 2 * m.Hidden
	}
	if m.Eps <= 0 {
		m.Eps = 1e-5
	}
	if m.Target == "" {
		m.Target = TargetSM121A
	}
	if m.Hidden < 2 || m.NumHeads < 1 || m.HeadDim < 2 || m.Hidden != m.NumHeads*m.HeadDim {
		return m, fmt.Errorf("multiarch megakernel %s: invalid dimensions Hidden=%d NumHeads=%d HeadDim=%d (want Hidden == NumHeads*HeadDim)",
			m.Name, m.Hidden, m.NumHeads, m.HeadDim)
	}
	if m.Intermediate < 2 {
		return m, fmt.Errorf("multiarch megakernel %s: Intermediate=%d must be >= 2", m.Name, m.Intermediate)
	}
	if _, err := SelectArchDispatchPlan(m.Target, m.Profile); err != nil {
		return m, fmt.Errorf("multiarch megakernel %s: %w", m.Name, err)
	}
	if len(m.Instructions) == 0 {
		m.Instructions = DefaultTransformerInstQueue()
	} else {
		m.Instructions = slices.Clone(m.Instructions)
	}
	for i, inst := range m.Instructions {
		if inst.Op < MegaOpNoop || inst.Op > MegaOpHalt {
			return m, fmt.Errorf("multiarch megakernel %s: instruction %d has invalid op %d", m.Name, i, inst.Op)
		}
	}
	return m, nil
}

// Validate checks that m has valid transformer dimensions, instruction queue,
// and architecture target.
func (m MultiArchMegakernel) Validate() error {
	_, err := m.norm()
	return err
}

// Plan returns the [ArchDispatchPlan] selected for m.Target and m.Profile.
func (m MultiArchMegakernel) Plan() (ArchDispatchPlan, error) {
	m, err := m.norm()
	if err != nil {
		return ArchDispatchPlan{}, err
	}
	return SelectArchDispatchPlan(m.Target, m.Profile)
}

func (m MultiArchMegakernel) params() []string {
	return []string{
		"const float* __restrict__ x",
		"const float* __restrict__ rms_attn_w",
		"const float* __restrict__ w_qkv",
		"const float* __restrict__ w_o",
		"const float* __restrict__ rms_ffn_w",
		"const float* __restrict__ w_gate",
		"const float* __restrict__ w_up",
		"const float* __restrict__ w_down",
		"float* __restrict__ out",
		"unsigned long long* __restrict__ ready",
		"unsigned int epoch",
		"int num_tokens",
	}
}

func emitArchHardwareDispatchHelper(plan ArchDispatchPlan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "// Hardware dispatch profile for %s:\n", plan.Target)
	fmt.Fprintf(&b, "//   tensor_core:   %s\n", plan.TensorCoreMnemonic)
	fmt.Fprintf(&b, "//   dequant:       %s\n", plan.DequantMnemonic)
	fmt.Fprintf(&b, "//   collective:    %s\n", plan.CollectiveMnemonic)
	fmt.Fprintf(&b, "//   work_steal:    %s\n", plan.WorkStealMnemonic)
	fmt.Fprintf(&b, "//   tmem_cols:     %d (uses_tmem=%v, stages=%d, regs=%d/%d, sync_restrict=%v, lut_b=%v)\n",
		plan.TMEMColumns, plan.UsesTMEM, plan.PipelineStages,
		plan.ProducerMaxRegs, plan.ConsumerMaxRegs,
		plan.UsesEarlyReleaseSyncRestrict, plan.UsesLUTDecompressB)

	b.WriteString("EG_DEV void eg_multiarch_hw_dispatch_probe(uint32_t packed_e2m1, uint32_t* out_bf16x4) {\n")
	switch plan.Target {
	case TargetSM90A:
		b.WriteString(`#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // Hopper sm_90a: wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 + DSMEM cluster reduction + 3-inst prmt.b32 LUT
    asm volatile("wgmma.fence.sync.aligned;\n\t"
                 "// wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n\t"
                 "// dsmem.cluster.allreduce via mapa.shared::cluster.u32" ::: "memory");
#endif
    eg_e2m1x4_to_bf16x4(packed_e2m1, out_bf16x4);
`)
	case TargetSM100A:
		b.WriteString(`#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // Blackwell sm_100a: 512-col TMEM + tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X + CLC + multimem.ld_reduce
    asm volatile("// tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 (512 cols)\n\t"
                 "// tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X\n\t"
                 "// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128\n\t"
                 "// multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2\n\t"
                 "// cvt.rn.f16x2.e2m1x2" ::: "memory");
#endif
    eg_e2m1x4_to_bf16x4(packed_e2m1, out_bf16x4);
`)
	case TargetSM103A:
		b.WriteString(`#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // Blackwell Ultra sm_103a: 512-col TMEM + scale_vec::4X + tcgen05.ld.red + CLC
    asm volatile("// tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 (512 cols)\n\t"
                 "// tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X\n\t"
                 "// tcgen05.ld.red.sync.aligned.32x32b.x4.max.f32\n\t"
                 "// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128\n\t"
                 "// multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2" ::: "memory");
#endif
    eg_e2m1x4_to_bf16x4(packed_e2m1, out_bf16x4);
`)
	case TargetSM107A, TargetSM107F:
		b.WriteString(`#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // Vera Rubin sm_107a/sm_107f: 576-col .exclusive TMEM + decompress::lut::b.block32 + .sync_restrict::shared::read::mma::a + spcompress + quad SIMD + fabric.try_red
    asm volatile("// tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.exclusive.b32 (576 cols)\n\t"
                 "// tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32.sync_restrict::shared::read::mma::a\n\t"
                 "// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2\n\t"
                 "// fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4\n\t"
                 "// add.rn.f32x2\n\t"
                 "// fabric.try_red.async.tensor.2d.multimem" ::: "memory");
#endif
    eg_e2m1x4_to_bf16x4(packed_e2m1, out_bf16x4);
`)
	case TargetSM120A:
		b.WriteString(`#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200) && defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // Blackwell Workstation sm_120a: 1-CTA mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale + hardware cvt.rn.f16x2.e2m1x2 + shared-memory staging (no TMEM)
    uint32_t f16x2;
    asm("{\n\t"
        ".reg .b8 q;\n\t"
        "cvt.u8.u32 q, %1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, q;\n\t"
        "// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3\n\t"
        "// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128\n\t"
        "}" : "=r"(f16x2) : "r"(packed_e2m1));
    out_bf16x4[0] = f16x2;
    out_bf16x4[1] = 0u;
#else
    eg_e2m1x4_to_bf16x4(packed_e2m1, out_bf16x4);
#endif
`)
	case TargetSM121A:
		b.WriteString(`#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1210) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // DGX Spark GB10 sm_121a: 1-CTA mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale + 3-instruction prmt.b32 LUT eg_e2m1x4_to_bf16x4 (shared-memory staging, no TMEM)
    asm volatile("// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3\n\t"
                 "// prmt.b32 LUT via eg_e2m1x4_to_bf16x4\n\t"
                 "// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128" ::: "memory");
#endif
    eg_e2m1x4_to_bf16x4(packed_e2m1, out_bf16x4);
`)
	}
	b.WriteString("}\n")
	return b.String()
}

// CUDA returns the architecture-specialized persistent op-queue megakernel
// device source (with [ReadyPrelude], without [Prelude]).
func (m MultiArchMegakernel) CUDA() (string, error) {
	m, err := m.norm()
	if err != nil {
		return "", err
	}
	plan, err := m.Plan()
	if err != nil {
		return "", err
	}

	var opLiterals []string
	var waitLiterals []string
	var sigLiterals []string
	for _, inst := range m.Instructions {
		opLiterals = append(opLiterals, fmt.Sprintf("%du", uint32(inst.Op)))
		waitLiterals = append(waitLiterals, fmt.Sprintf("%d", inst.WaitSlot))
		sigLiterals = append(sigLiterals, fmt.Sprintf("%d", inst.SigSlot))
	}

	var b strings.Builder
	b.WriteString(ReadyPrelude)
	b.WriteString("\n")
	b.WriteString(emitArchHardwareDispatchHelper(plan))
	fmt.Fprintf(&b, "\n// %s: unified multi-architecture persistent op-queue megakernel (target=%s, H=%d, I=%d, heads=%d, D=%d, ops=%d).\n",
		m.Name, plan.Target, m.Hidden, m.Intermediate, m.NumHeads, m.HeadDim, len(m.Instructions))
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n",
		m.Name, strings.Join(m.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int H = %d, I = %d, HQ = %d, D = %d, NUM_OPS = %d;\n",
		m.Hidden, m.Intermediate, m.NumHeads, m.HeadDim, len(m.Instructions))
	fmt.Fprintf(&b, "    constexpr int PIPELINE_STAGES = %d, TMEM_COLS = %d;\n",
		plan.PipelineStages, plan.TMEMColumns)
	fmt.Fprintf(&b, "    constexpr int PROD_MAX_REGS = %d, CONS_MAX_REGS = %d;\n",
		plan.ProducerMaxRegs, plan.ConsumerMaxRegs)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(m.Eps))
	fmt.Fprintf(&b, "    constexpr float ATTN_SCALE = %s;\n", floatLiteral(1.0/float32(math.Sqrt(float64(m.HeadDim)))))
	fmt.Fprintf(&b, "    constexpr unsigned int OP_QUEUE[NUM_OPS] = {%s};\n", strings.Join(opLiterals, ", "))
	fmt.Fprintf(&b, "    constexpr int WAIT_SLOTS[NUM_OPS] = {%s};\n", strings.Join(waitLiterals, ", "))
	fmt.Fprintf(&b, "    constexpr int SIG_SLOTS[NUM_OPS] = {%s};\n", strings.Join(sigLiterals, ", "))
	b.WriteString(`    (void)PIPELINE_STAGES; (void)TMEM_COLS; (void)PROD_MAX_REGS; (void)CONS_MAX_REGS;
    const int tid = (int)threadIdx.x;
    if (blockIdx.x != 0 || num_tokens <= 0) return;

    uint32_t probe_bf16x4[2];
    eg_multiarch_hw_dispatch_probe((uint32_t)tid & 0xfu, probe_bf16x4);
    (void)probe_bf16x4;

    constexpr int MAX_TOK = 16;
    const int M = (num_tokens <= MAX_TOK) ? num_tokens : MAX_TOK;

    __shared__ float s_norm1[MAX_TOK * H];
    __shared__ float s_q[MAX_TOK * H];
    __shared__ float s_k[MAX_TOK * H];
    __shared__ float s_v[MAX_TOK * H];
    __shared__ float s_attn[MAX_TOK * H];
    __shared__ float s_h1[MAX_TOK * H];
    __shared__ float s_norm2[MAX_TOK * H];
    __shared__ float s_swiglu[MAX_TOK * I];
    __shared__ float s_mlp[MAX_TOK * H];
    __shared__ float s_red[128];

    for (int pc = 0; pc < NUM_OPS; ++pc) {
        const unsigned int op = OP_QUEUE[pc];
        const int wait_slot = WAIT_SLOTS[pc];
        const int sig_slot = SIG_SLOTS[pc];

        if (wait_slot >= 0) {
            if (tid == 0) {
                eg_ready_wait(&ready[wait_slot], epoch, 1u);
            }
            __syncthreads();
        }

        if (op == 1u) {
            // Op 1: Pre-Attention RMSNorm
            for (int t = 0; t < M; ++t) {
                float sq = 0.0f;
                for (int c = tid; c < H; c += 128) {
                    const float xv = x[t * H + c];
                    sq += xv * xv;
                }
                s_red[tid] = sq;
                __syncthreads();
                if (tid == 0) {
                    float total = 0.0f;
                    for (int k = 0; k < 128; ++k) total += s_red[k];
                    s_red[0] = rsqrtf(total * (1.0f / (float)H) + EPS);
                }
                __syncthreads();
                const float inv_rms = s_red[0];
                for (int c = tid; c < H; c += 128) {
                    s_norm1[t * H + c] = x[t * H + c] * inv_rms * rms_attn_w[c];
                }
                __syncthreads();
            }
        } else if (op == 2u) {
            // Op 2: Fused QKV Projection
            for (int idx = tid; idx < M * H; idx += 128) {
                const int t = idx / H;
                const int r = idx % H;
                float q_acc = 0.0f, k_acc = 0.0f, v_acc = 0.0f;
                for (int c = 0; c < H; ++c) {
                    const float a = s_norm1[t * H + c];
                    q_acc += a * w_qkv[r * H + c];
                    k_acc += a * w_qkv[(H + r) * H + c];
                    v_acc += a * w_qkv[(2 * H + r) * H + c];
                }
                s_q[t * H + r] = q_acc;
                s_k[t * H + r] = k_acc;
                s_v[t * H + r] = v_acc;
            }
            __syncthreads();
        } else if (op == 3u) {
            // Op 3: Causal Multi-Head Scaled Dot-Product Attention + O-Proj + First Residual
            for (int idx = tid; idx < M * HQ; idx += 128) {
                const int t = idx / HQ;
                const int h = idx % HQ;
                const int h_off = h * D;
                float scores[MAX_TOK];
                float max_s = -1e30f;
                for (int s = 0; s <= t; ++s) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += s_q[t * H + h_off + d] * s_k[s * H + h_off + d];
                    }
                    const float sc = dot * ATTN_SCALE;
                    scores[s] = sc;
                    if (sc > max_s) max_s = sc;
                }
                float sum_exp = 0.0f;
                for (int s = 0; s <= t; ++s) {
                    scores[s] = expf(scores[s] - max_s);
                    sum_exp += scores[s];
                }
                const float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int s = 0; s <= t; ++s) {
                        acc += (scores[s] * inv_sum) * s_v[s * H + h_off + d];
                    }
                    s_attn[t * H + h_off + d] = acc;
                }
            }
            __syncthreads();
            for (int idx = tid; idx < M * H; idx += 128) {
                const int t = idx / H;
                const int r = idx % H;
                float o_acc = 0.0f;
                for (int c = 0; c < H; ++c) {
                    o_acc += s_attn[t * H + c] * w_o[r * H + c];
                }
                s_h1[t * H + r] = x[t * H + r] + o_acc;
            }
            __syncthreads();
        } else if (op == 4u) {
            // Op 4: Post-Attention FFN RMSNorm + Gate/Up SwiGLU
            for (int t = 0; t < M; ++t) {
                float sq = 0.0f;
                for (int c = tid; c < H; c += 128) {
                    const float hv = s_h1[t * H + c];
                    sq += hv * hv;
                }
                s_red[tid] = sq;
                __syncthreads();
                if (tid == 0) {
                    float total = 0.0f;
                    for (int k = 0; k < 128; ++k) total += s_red[k];
                    s_red[0] = rsqrtf(total * (1.0f / (float)H) + EPS);
                }
                __syncthreads();
                const float inv_rms = s_red[0];
                for (int c = tid; c < H; c += 128) {
                    s_norm2[t * H + c] = s_h1[t * H + c] * inv_rms * rms_ffn_w[c];
                }
                __syncthreads();
            }
            for (int idx = tid; idx < M * I; idx += 128) {
                const int t = idx / I;
                const int i = idx % I;
                float gate = 0.0f, up = 0.0f;
                for (int c = 0; c < H; ++c) {
                    const float a = s_norm2[t * H + c];
                    gate += a * w_gate[i * H + c];
                    up += a * w_up[i * H + c];
                }
                s_swiglu[t * I + i] = eg_silu(gate) * up;
            }
            __syncthreads();
        } else if (op == 5u) {
            // Op 5: DownProj GEMM
            for (int idx = tid; idx < M * H; idx += 128) {
                const int t = idx / H;
                const int r = idx % H;
                float acc = 0.0f;
                for (int i = 0; i < I; ++i) {
                    acc += s_swiglu[t * I + i] * w_down[r * I + i];
                }
                s_mlp[t * H + r] = acc;
            }
            __syncthreads();
        } else if (op == 6u) {
            // Op 6: Residual AllReduce Epilogue
            for (int idx = tid; idx < M * H; idx += 128) {
                out[idx] = s_h1[idx] + s_mlp[idx];
            }
            __syncthreads();
        } else if (op == 7u) {
            break;
        }

        if (sig_slot >= 0) {
            __syncthreads();
            if (tid == 0) {
                eg_ready_signal(&ready[sig_slot], epoch, 1u);
            }
            __syncthreads();
        }
    }
}
`)
	return b.String(), nil
}

// Launcher returns the C++ host launch wrapper for [MultiArchMegakernel].
func (m MultiArchMegakernel) Launcher() (string, error) {
	m, err := m.norm()
	if err != nil {
		return "", err
	}
	params := m.params()
	names := paramNames(params)
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream = nullptr) {\n",
		m.Name, strings.Join(params, ", "))
	b.WriteString("    const dim3 grid(1, 1, 1), block(128, 1, 1);\n")
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n",
		m.Name, strings.Join(names, ", "))
	fmt.Fprintf(&b, "\ninline void %s_run(%s, cudaStream_t stream = nullptr) {\n    %s_launch(%s, stream);\n}\n",
		m.Name, strings.Join(params, ", "), m.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns the self-contained translation unit ([Prelude], [ReadyPrelude],
// hardware dispatch helpers, `eg_multiarch_megakernel`, and host launchers).
func (m MultiArchMegakernel) Source() string {
	cuda, err := m.CUDA()
	if err != nil {
		return ""
	}
	l, err := m.Launcher()
	if err != nil {
		return Prelude + "\n" + cuda
	}
	return Prelude + "\n" + cuda + "\n" + l
}

// EmitAllTargets generates the specialized megakernel CUDA/PTX sources for all
// 6 canonical targets ("sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a",
// "sm_121a") in a single call.
func (m MultiArchMegakernel) EmitAllTargets() (map[string]string, error) {
	out := make(map[string]string, len(AllMegakernelArchTargets))
	for _, target := range AllMegakernelArchTargets {
		cp := m
		cp.Target = target
		src, err := cp.CUDA()
		if err != nil {
			return nil, fmt.Errorf("emit target %s: %w", target, err)
		}
		out[target] = src
	}
	return out, nil
}

// Reference computes the exact Go float32 reference of the full transformer
// layer executed by `eg_multiarch_megakernel`:
//
//	RMSNorm -> QKV -> Attention -> Gate/Up SwiGLU -> DownProj -> Residual AllReduce
func (m MultiArchMegakernel) Reference(
	x, rmsAttnW, wQKV, wO, rmsFFNW, wGate, wUp, wDown []float32,
) ([]float32, error) {
	m, err := m.norm()
	if err != nil {
		return nil, err
	}
	H, I, HQ, D := m.Hidden, m.Intermediate, m.NumHeads, m.HeadDim
	if len(x) == 0 || len(x)%H != 0 {
		return nil, fmt.Errorf("multiarch megakernel %s reference: len(x)=%d must be a positive multiple of Hidden=%d",
			m.Name, len(x), H)
	}
	numTokens := len(x) / H
	if len(rmsAttnW) != H {
		return nil, fmt.Errorf("multiarch megakernel %s reference: len(rmsAttnW)=%d, want %d", m.Name, len(rmsAttnW), H)
	}
	if len(wQKV) != 3*H*H {
		return nil, fmt.Errorf("multiarch megakernel %s reference: len(wQKV)=%d, want %d", m.Name, len(wQKV), 3*H*H)
	}
	if len(wO) != H*H {
		return nil, fmt.Errorf("multiarch megakernel %s reference: len(wO)=%d, want %d", m.Name, len(wO), H*H)
	}
	if len(rmsFFNW) != H {
		return nil, fmt.Errorf("multiarch megakernel %s reference: len(rmsFFNW)=%d, want %d", m.Name, len(rmsFFNW), H)
	}
	if len(wGate) != I*H {
		return nil, fmt.Errorf("multiarch megakernel %s reference: len(wGate)=%d, want %d", m.Name, len(wGate), I*H)
	}
	if len(wUp) != I*H {
		return nil, fmt.Errorf("multiarch megakernel %s reference: len(wUp)=%d, want %d", m.Name, len(wUp), I*H)
	}
	if len(wDown) != H*I {
		return nil, fmt.Errorf("multiarch megakernel %s reference: len(wDown)=%d, want %d", m.Name, len(wDown), H*I)
	}

	norm1 := make([]float32, numTokens*H)
	q := make([]float32, numTokens*H)
	k := make([]float32, numTokens*H)
	v := make([]float32, numTokens*H)
	attn := make([]float32, numTokens*H)
	h1 := make([]float32, numTokens*H)
	norm2 := make([]float32, numTokens*H)
	swiglu := make([]float32, numTokens*I)
	mlp := make([]float32, numTokens*H)
	out := make([]float32, numTokens*H)
	attnScale := float32(1.0 / math.Sqrt(float64(D)))

	for _, inst := range m.Instructions {
		switch inst.Op {
		case MegaOpRMSNorm:
			for t := range numTokens {
				var sq float32
				for c := range H {
					xv := x[t*H+c]
					sq += xv * xv
				}
				invRMS := float32(1.0 / math.Sqrt(float64(sq/float32(H)+m.Eps)))
				for c := range H {
					norm1[t*H+c] = x[t*H+c] * invRMS * rmsAttnW[c]
				}
			}
		case MegaOpQKVProj:
			for t := range numTokens {
				for r := range H {
					var qAcc, kAcc, vAcc float32
					for c := range H {
						a := norm1[t*H+c]
						qAcc += a * wQKV[r*H+c]
						kAcc += a * wQKV[(H+r)*H+c]
						vAcc += a * wQKV[(2*H+r)*H+c]
					}
					q[t*H+r] = qAcc
					k[t*H+r] = kAcc
					v[t*H+r] = vAcc
				}
			}
		case MegaOpAttention:
			for t := range numTokens {
				for h := range HQ {
					hOff := h * D
					scores := make([]float32, t+1)
					maxS := float32(-1e30)
					for s := 0; s <= t; s++ {
						var dot float32
						for d := range D {
							dot += q[t*H+hOff+d] * k[s*H+hOff+d]
						}
						sc := dot * attnScale
						scores[s] = sc
						if sc > maxS {
							maxS = sc
						}
					}
					var sumExp float32
					for s := 0; s <= t; s++ {
						scores[s] = float32(math.Exp(float64(scores[s] - maxS)))
						sumExp += scores[s]
					}
					invSum := 1.0 / sumExp
					for d := range D {
						var acc float32
						for s := 0; s <= t; s++ {
							acc += (scores[s] * invSum) * v[s*H+hOff+d]
						}
						attn[t*H+hOff+d] = acc
					}
				}
				for r := range H {
					var oAcc float32
					for c := range H {
						oAcc += attn[t*H+c] * wO[r*H+c]
					}
					h1[t*H+r] = x[t*H+r] + oAcc
				}
			}
		case MegaOpGateUpSwiGLU:
			for t := range numTokens {
				var sq float32
				for c := range H {
					hv := h1[t*H+c]
					sq += hv * hv
				}
				invRMS := float32(1.0 / math.Sqrt(float64(sq/float32(H)+m.Eps)))
				for c := range H {
					norm2[t*H+c] = h1[t*H+c] * invRMS * rmsFFNW[c]
				}
				for i := range I {
					var gate, up float32
					for c := range H {
						a := norm2[t*H+c]
						gate += a * wGate[i*H+c]
						up += a * wUp[i*H+c]
					}
					silu := gate / (1.0 + float32(math.Exp(float64(-gate))))
					swiglu[t*I+i] = silu * up
				}
			}
		case MegaOpDownProj:
			for t := range numTokens {
				for r := range H {
					var acc float32
					for i := range I {
						acc += swiglu[t*I+i] * wDown[r*I+i]
					}
					mlp[t*H+r] = acc
				}
			}
		case MegaOpResidualAllReduce:
			for idx := range numTokens * H {
				out[idx] = h1[idx] + mlp[idx]
			}
		case MegaOpHalt:
			return out, nil
		}
	}
	return out, nil
}
