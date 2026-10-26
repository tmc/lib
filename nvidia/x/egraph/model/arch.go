package model

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// A DType is the storage element type of a model tensor.
type DType uint8

const (
	F32  DType = iota + 1
	BF16       // bfloat16, round to nearest even
	E4M3       // float8_e4m3fn, saturating to ±448
	E2M1       // float4_e2m1fn, two codes per byte
	U8
	I32
	I64
)

// NVFP4 is the DType alias for 4-bit E2M1 block-scaled weights.
const NVFP4 = E2M1

var dtypeNames = [...]string{F32: "f32", BF16: "bf16", E4M3: "e4m3", E2M1: "e2m1", U8: "u8", I32: "i32", I64: "i64"}

func (d DType) String() string {
	if int(d) < len(dtypeNames) && dtypeNames[d] != "" {
		return dtypeNames[d]
	}
	return fmt.Sprintf("DType(%d)", uint8(d))
}

// Bits reports the storage size of one element in bits.
func (d DType) Bits() int {
	switch d {
	case F32, I32:
		return 32
	case BF16:
		return 16
	case E4M3, U8:
		return 8
	case E2M1:
		return 4
	case I64:
		return 64
	}
	return 0
}

// CType returns the C element type used in generated kernel signatures.
func (d DType) CType() string {
	switch d {
	case F32:
		return "float"
	case BF16:
		return "uint16_t"
	case E4M3, E2M1, U8:
		return "uint8_t"
	case I32:
		return "int32_t"
	case I64:
		return "int64_t"
	}
	return ""
}

// A WeightFormat says how a weight matrix W [N, K] is stored in memory.
type WeightFormat uint8

const (
	WeightBF16 WeightFormat = iota + 1
	WeightBF16HiLo
	WeightNVFP4
	WeightE4M3
)

func (w WeightFormat) String() string {
	switch w {
	case WeightBF16:
		return "bf16"
	case WeightBF16HiLo:
		return "bf16hilo"
	case WeightNVFP4:
		return "nvfp4"
	case WeightE4M3:
		return "e4m3"
	}
	return fmt.Sprintf("WeightFormat(%d)", uint8(w))
}

// Execution mode names selected for layer compilation.
const (
	ModePersistentMegakernel = "persistent_megakernel"
	ModeStreamKWarpSpec      = "streamk_warpspec"
	ModeMultiLaunch          = "multi_launch"
)

// SelectLayerExecutionMode returns the execution mode for a layer at batchM.
func SelectLayerExecutionMode(batchM int) string {
	switch {
	case batchM <= 8:
		return ModePersistentMegakernel
	case batchM >= 64:
		return ModeStreamKWarpSpec
	default:
		return ModeMultiLaunch
	}
}

// AttentionType classifies the attention block structure of a layer.
type AttentionType string

const (
	AttentionGQA      AttentionType = "gqa"
	AttentionGatedGQA AttentionType = "gated_gqa"
	AttentionMLA      AttentionType = "mla"
)

// FFNType classifies the feed-forward network structure of a layer.
type FFNType string

const (
	FFNSwiGLU    FFNType = "swiglu"
	FFNRoutedMoE FFNType = "routed_moe"
)

// QuantConfig describes weight and KV-cache quantization settings in a model config.
type QuantConfig struct {
	QuantMethod   string `json:"quant_method,omitempty"`
	QuantAlgo     string `json:"quant_algo,omitempty"`
	Format        string `json:"format,omitempty"`
	KVCacheScheme string `json:"kv_cache_scheme,omitempty"`
}

// RoPEConfig describes rotary position embedding parameters.
type RoPEConfig struct {
	Theta             float64 `json:"rope_theta,omitempty"`
	RotaryDim         int     `json:"rotary_dim,omitempty"`
	PartialRotaryFact float64 `json:"partial_rotary_factor,omitempty"`
}

// MoEConfig summarizes routed Mixture-of-Experts parameters.
type MoEConfig struct {
	NumExperts          int `json:"num_experts"`
	NumLocalExperts     int `json:"num_local_experts"`
	NumExpertsPerTok    int `json:"num_experts_per_tok"`
	MoEIntermediateSize int `json:"moe_intermediate_size"`
}

// ArchConfig describes a transformer decoder architecture (dense GQA or
// routed MoE, optional gated attention and QK norm, BF16/E4M3/NVFP4 weights).
// It unmarshals directly from standard HuggingFace config.json bytes via
// [ParseArchConfig].
type ArchConfig struct {
	Architectures       []string `json:"architectures,omitempty"`
	HiddenSize          int      `json:"hidden_size"`
	IntermediateSize    int      `json:"intermediate_size"`
	MoEIntermediateSize int      `json:"moe_intermediate_size,omitempty"`
	NumHiddenLayers     int      `json:"num_hidden_layers"`
	NumAttentionHeads   int      `json:"num_attention_heads"`
	NumKeyValueHeads    int      `json:"num_key_value_heads"`
	HeadDim             int      `json:"head_dim,omitempty"`
	NumLocalExperts     int      `json:"num_local_experts,omitempty"`
	NumExperts          int      `json:"num_experts,omitempty"`
	NumExpertsPerTok    int      `json:"num_experts_per_tok,omitempty"`
	RMSNormEps          float32  `json:"rms_norm_eps,omitempty"`
	QKNorm              bool     `json:"qk_norm,omitempty"`
	AttentionGate       bool     `json:"attention_gate,omitempty"`
	WeightDType         DType    `json:"-"` // BF16, E4M3, NVFP4 (inferred from quantization_config or explicit)
	KVDType             DType    `json:"-"` // BF16 or E4M3
}

type rawArchConfig struct {
	ArchConfig
	NRoutedExperts     int          `json:"n_routed_experts,omitempty"`
	NumExpertsPerToken int          `json:"num_experts_per_token,omitempty"`
	KVCacheDType       string       `json:"kv_cache_dtype,omitempty"`
	WeightDTypeName    string       `json:"weight_dtype,omitempty"`
	QuantizationConfig *QuantConfig `json:"quantization_config,omitempty"`
}

// ParseArchConfig unmarshals a HuggingFace-style config.json payload into an
// [ArchConfig], inferring WeightDType and KVDType from quantization_config when
// present and filling defaults.
func ParseArchConfig(data []byte) (ArchConfig, error) {
	var raw rawArchConfig
	if err := json.Unmarshal(data, &raw); err != nil {
		return ArchConfig{}, fmt.Errorf("parse arch config: %w", err)
	}
	cfg := raw.ArchConfig
	if cfg.NumExperts == 0 && raw.NRoutedExperts > 0 {
		cfg.NumExperts = raw.NRoutedExperts
	}
	if cfg.NumExpertsPerTok == 0 && raw.NumExpertsPerToken > 0 {
		cfg.NumExpertsPerTok = raw.NumExpertsPerToken
	}
	cfg.WeightDType = BF16
	cfg.KVDType = BF16

	if raw.QuantizationConfig != nil {
		qm := strings.ToLower(strings.TrimSpace(raw.QuantizationConfig.QuantMethod))
		qa := strings.ToLower(strings.TrimSpace(raw.QuantizationConfig.QuantAlgo + " " + raw.QuantizationConfig.Format))
		switch {
		case qm == "nvfp4" || qm == "fp4" || qm == "e2m1" || strings.Contains(qm, "nvfp4") || strings.Contains(qa, "nvfp4"):
			cfg.WeightDType = NVFP4
			cfg.KVDType = E4M3
		case qm == "fp8" || qm == "e4m3" || strings.Contains(qm, "fp8") || strings.Contains(qa, "fp8") || strings.Contains(qa, "e4m3"):
			cfg.WeightDType = E4M3
			cfg.KVDType = E4M3
		case qm == "modelopt":
			cfg.WeightDType = NVFP4
			cfg.KVDType = E4M3
		}
		if kvs := strings.ToLower(strings.TrimSpace(raw.QuantizationConfig.KVCacheScheme)); kvs != "" {
			if strings.Contains(kvs, "fp8") || strings.Contains(kvs, "e4m3") {
				cfg.KVDType = E4M3
			} else if strings.Contains(kvs, "bf16") {
				cfg.KVDType = BF16
			}
		}
	}
	if raw.WeightDTypeName != "" {
		switch strings.ToLower(strings.TrimSpace(raw.WeightDTypeName)) {
		case "nvfp4", "fp4", "e2m1":
			cfg.WeightDType = NVFP4
		case "fp8", "e4m3", "fp8_e4m3":
			cfg.WeightDType = E4M3
		case "bf16", "bfloat16":
			cfg.WeightDType = BF16
		}
	}
	if raw.KVCacheDType != "" {
		switch strings.ToLower(strings.TrimSpace(raw.KVCacheDType)) {
		case "fp8", "e4m3", "fp8_e4m3":
			cfg.KVDType = E4M3
		case "bf16", "bfloat16":
			cfg.KVDType = BF16
		}
	}
	return ValidateArchConfig(cfg)
}

// IsMoE reports whether c configures a routed Mixture-of-Experts layer.
func (c ArchConfig) IsMoE() bool {
	return c.NumExperts > 0 || c.NumLocalExperts > 0
}

// AttentionKind returns the [AttentionType] of c.
func (c ArchConfig) AttentionKind() AttentionType {
	if c.AttentionGate {
		return AttentionGatedGQA
	}
	return AttentionGQA
}

// FFNKind returns the [FFNType] of c.
func (c ArchConfig) FFNKind() FFNType {
	if c.IsMoE() {
		return FFNRoutedMoE
	}
	return FFNSwiGLU
}

// WeightFormat returns the [WeightFormat] corresponding to c.WeightDType.
func (c ArchConfig) WeightFormat() WeightFormat {
	switch c.WeightDType {
	case NVFP4:
		return WeightNVFP4
	case E4M3:
		return WeightE4M3
	default:
		return WeightBF16
	}
}

// ValidateArchConfig validates and normalizes c, filling defaults for
// NumKeyValueHeads, HeadDim, NumHiddenLayers, RMSNormEps, MoE expert counts,
// WeightDType, and KVDType.
func ValidateArchConfig(c ArchConfig) (ArchConfig, error) {
	if c.HiddenSize <= 0 {
		return c, fmt.Errorf("arch config: hidden_size %d must be positive", c.HiddenSize)
	}
	if c.NumAttentionHeads <= 0 {
		return c, fmt.Errorf("arch config: num_attention_heads %d must be positive", c.NumAttentionHeads)
	}
	if c.NumKeyValueHeads <= 0 {
		c.NumKeyValueHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return c, fmt.Errorf("arch config: num_attention_heads %d not divisible by num_key_value_heads %d",
			c.NumAttentionHeads, c.NumKeyValueHeads)
	}
	if c.HeadDim <= 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.NumHiddenLayers <= 0 {
		c.NumHiddenLayers = 1
	}
	if c.RMSNormEps <= 0 {
		c.RMSNormEps = 1e-5
	}
	if c.NumLocalExperts == 0 && c.NumExperts > 0 {
		c.NumLocalExperts = c.NumExperts
	}
	if c.NumExperts == 0 && c.NumLocalExperts > 0 {
		c.NumExperts = c.NumLocalExperts
	}
	if c.IsMoE() {
		if c.NumExpertsPerTok <= 0 {
			c.NumExpertsPerTok = min(2, c.NumExperts)
		}
		if c.MoEIntermediateSize <= 0 {
			c.MoEIntermediateSize = c.IntermediateSize
		}
		if c.MoEIntermediateSize <= 0 {
			return c, fmt.Errorf("arch config: moe_intermediate_size must be positive")
		}
	} else if c.IntermediateSize <= 0 {
		return c, fmt.Errorf("arch config: intermediate_size %d must be positive", c.IntermediateSize)
	}
	if c.WeightDType == 0 {
		c.WeightDType = BF16
	}
	if c.KVDType == 0 {
		if c.WeightDType == E4M3 || c.WeightDType == NVFP4 {
			c.KVDType = E4M3
		} else {
			c.KVDType = BF16
		}
	}
	return c, nil
}

// LayerOpKind identifies a logical operator stage in a [LayerProgram].
type LayerOpKind string

const (
	OpRMSNorm             LayerOpKind = "rmsnorm"
	OpQProj               LayerOpKind = "q_proj"
	OpQKNorm              LayerOpKind = "qk_norm"
	OpGateProj            LayerOpKind = "attn_gate_proj"
	OpPagedAttention      LayerOpKind = "paged_attention"
	OpGatedPagedAttention LayerOpKind = "gated_paged_attention"
	OpOProj               LayerOpKind = "o_proj"
	OpResidualAdd         LayerOpKind = "residual_add"
	OpGateUpProj          LayerOpKind = "gate_up_proj"
	OpSwiGLU              LayerOpKind = "swiglu"
	OpDownProj            LayerOpKind = "down_proj"
	OpMoERouteTopK        LayerOpKind = "moe_route_topk"
	OpMoEExpertAlign      LayerOpKind = "moe_expert_align"
	OpMoEGroupedGateUp    LayerOpKind = "moe_grouped_gate_up"
	OpMoEGroupedDown      LayerOpKind = "moe_grouped_down"
	OpMoEExpertSum        LayerOpKind = "moe_expert_sum"
	OpResidualRMSNorm     LayerOpKind = "residual_rmsnorm"
)

// A LayerOp describes one operator node in a [LayerProgram].
type LayerOp struct {
	Name    string      `json:"name"`
	Kind    LayerOpKind `json:"kind"`
	Inputs  []string    `json:"inputs"`
	Outputs []string    `json:"outputs"`
	M       int         `json:"m,omitempty"`
	N       int         `json:"n,omitempty"`
	K       int         `json:"k,omitempty"`
}

// A LayerProgram holds the normalized architecture configuration and ordered
// operator DAG for one transformer decoder layer.
type LayerProgram struct {
	Config  ArchConfig `json:"config"`
	BatchM  int        `json:"batch_m"`
	SeqLen  int        `json:"seq_len"`
	OutType DType      `json:"out_type"`
	Ops     []LayerOp  `json:"ops"`
}

// BuildLayerProgram constructs the structural [LayerProgram] for one
// transformer decoder layer under cfg at batchM and seqLen.
func BuildLayerProgram(cfg ArchConfig, batchM, seqLen int) (*LayerProgram, error) {
	var err error
	cfg, err = ValidateArchConfig(cfg)
	if err != nil {
		return nil, err
	}
	if batchM <= 0 {
		return nil, fmt.Errorf("build layer program: batchM %d must be positive", batchM)
	}
	if seqLen <= 0 {
		return nil, fmt.Errorf("build layer program: seqLen %d must be positive", seqLen)
	}

	H := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * 64
	var ops []LayerOp
	ops = append(ops,
		LayerOp{Name: "attn_rmsnorm", Kind: OpRMSNorm, Inputs: []string{"x", "attn_norm_w"}, Outputs: []string{"norm1"}, M: batchM, K: H},
		LayerOp{Name: "q_proj", Kind: OpQProj, Inputs: []string{"norm1", "wq"}, Outputs: []string{"q"}, M: batchM, N: qDim, K: H},
	)
	if cfg.QKNorm {
		ops = append(ops, LayerOp{Name: "qk_norm", Kind: OpQKNorm, Inputs: []string{"q", "qk_norm_w"}, Outputs: []string{"q_normed"}, M: batchM, K: qDim})
	}
	if cfg.AttentionGate {
		ops = append(ops,
			LayerOp{Name: "attn_gate_proj", Kind: OpGateProj, Inputs: []string{"norm1", "w_attn_gate"}, Outputs: []string{"attn_gate"}, M: batchM, N: qDim, K: H},
			LayerOp{Name: "paged_attn", Kind: OpGatedPagedAttention, Inputs: []string{"q", "k_cache", "v_cache", "page_table", "seqlens", "attn_gate"}, Outputs: []string{"attn_out"}, M: batchM, N: qDim, K: seqLen},
		)
	} else {
		ops = append(ops,
			LayerOp{Name: "paged_attn", Kind: OpPagedAttention, Inputs: []string{"q", "k_cache", "v_cache", "page_table", "seqlens"}, Outputs: []string{"attn_out"}, M: batchM, N: qDim, K: seqLen},
		)
	}
	ops = append(ops,
		LayerOp{Name: "o_proj", Kind: OpOProj, Inputs: []string{"attn_out", "wo"}, Outputs: []string{"o_out"}, M: batchM, N: H, K: qDim},
		LayerOp{Name: "attn_residual", Kind: OpResidualAdd, Inputs: []string{"o_out", "x"}, Outputs: []string{"h1"}, M: batchM, K: H},
		LayerOp{Name: "ffn_rmsnorm", Kind: OpRMSNorm, Inputs: []string{"h1", "ffn_norm_w"}, Outputs: []string{"norm2"}, M: batchM, K: H},
	)
	if cfg.IsMoE() {
		I := cfg.MoEIntermediateSize
		ops = append(ops,
			LayerOp{Name: "moe_topk", Kind: OpMoERouteTopK, Inputs: []string{"router_logits", "router_bias"}, Outputs: []string{"expert_ids", "expert_weights"}, M: batchM, K: cfg.NumExperts},
			LayerOp{Name: "moe_align", Kind: OpMoEExpertAlign, Inputs: []string{"expert_ids", "local_start"}, Outputs: []string{"offsets", "rows", "slots"}, M: batchM},
			LayerOp{Name: "moe_w13", Kind: OpMoEGroupedGateUp, Inputs: []string{"norm2", "w13", "offsets", "rows"}, Outputs: []string{"moe_inter"}, M: batchM, N: 2 * I, K: H},
			LayerOp{Name: "moe_w2", Kind: OpMoEGroupedDown, Inputs: []string{"moe_inter", "w2", "offsets"}, Outputs: []string{"moe_down"}, M: batchM, N: H, K: I},
			LayerOp{Name: "moe_sum", Kind: OpMoEExpertSum, Inputs: []string{"moe_down", "post_expert_w", "slots", "expert_weights"}, Outputs: []string{"mlp_out"}, M: batchM, K: H},
		)
	} else {
		I := cfg.IntermediateSize
		ops = append(ops,
			LayerOp{Name: "gate_up_proj", Kind: OpGateUpProj, Inputs: []string{"norm2", "w_gate_up"}, Outputs: []string{"gate_up"}, M: batchM, N: 2 * I, K: H},
			LayerOp{Name: "swiglu", Kind: OpSwiGLU, Inputs: []string{"gate_up"}, Outputs: []string{"mlp_act"}, M: batchM, K: I},
			LayerOp{Name: "down_proj", Kind: OpDownProj, Inputs: []string{"mlp_act", "w_down"}, Outputs: []string{"mlp_out"}, M: batchM, N: H, K: I},
		)
	}
	ops = append(ops,
		LayerOp{Name: "post_residual_rmsnorm", Kind: OpResidualRMSNorm, Inputs: []string{"mlp_out", "h1", "post_norm_w"}, Outputs: []string{"out"}, M: batchM, K: H},
	)

	return &LayerProgram{
		Config:  cfg,
		BatchM:  batchM,
		SeqLen:  seqLen,
		OutType: BF16,
		Ops:     ops,
	}, nil
}

// A CostEstimate records the roofline traffic, FLOPs, estimated latency, and
// selected schedule of one kernel step in a compiled plan.
type CostEstimate struct {
	Name        string  `json:"name"`
	Op          string  `json:"op"`
	Schedule    string  `json:"schedule,omitempty"`
	Bytes       uint64  `json:"bytes"`
	FLOPs       uint64  `json:"flops"`
	EstimatedNs float64 `json:"estimated_ns"`
}

// LayerCostBreakdown aggregates per-step [CostEstimate] metrics for a layer.
type LayerCostBreakdown struct {
	Device      string         `json:"device"`
	Mode        string         `json:"mode"`
	Steps       []CostEstimate `json:"steps"`
	TotalBytes  uint64         `json:"total_bytes"`
	TotalFLOPs  uint64         `json:"total_flops"`
	TotalNs     float64        `json:"total_ns"`
	ArithmeticI float64        `json:"arithmetic_intensity"`
}

// EstimateLayerCost computes an analytical roofline breakdown for prog on device.
func EstimateLayerCost(cfg ArchConfig, batchM, seqLen int, device string) (*LayerCostBreakdown, error) {
	prog, err := BuildLayerProgram(cfg, batchM, seqLen)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(device) == "" {
		device = "NVIDIA H200"
	}
	bw := 4800.0
	tflops := 778.0
	if ceil, err := cudaroofline.LookupDeviceCeiling(device, cudaroofline.DeviceResident); err == nil && ceil.PeakMemoryGBps > 0 {
		bw = ceil.PeakMemoryGBps
		if ceil.PeakTFLOPS > tflops {
			tflops = ceil.PeakTFLOPS
		}
	}
	const launchNs = 2000.0
	wBits := max(prog.Config.WeightDType.Bits(), 4)
	var (
		steps      []CostEstimate
		totalBytes uint64
		totalFLOPs uint64
		totalNs    float64
	)
	for _, op := range prog.Ops {
		var b, f uint64
		switch op.Kind {
		case OpQProj, OpGateProj, OpOProj, OpGateUpProj, OpDownProj, OpMoEGroupedGateUp, OpMoEGroupedDown:
			wBytes := uint64(op.N * op.K * wBits / 8)
			xBytes := uint64(op.M * op.K * 2)
			outBytes := uint64(op.M * op.N * 2)
			b = wBytes + xBytes + outBytes
			f = uint64(2 * op.M * op.N * op.K)
		case OpPagedAttention, OpGatedPagedAttention:
			b = uint64(op.M * op.K * prog.Config.NumKeyValueHeads * 64 * 2)
			f = uint64(op.M * prog.Config.NumAttentionHeads * op.K * 4 * 64)
		default:
			b = uint64(op.M * max(op.K, 64) * 4)
			f = uint64(op.M * max(op.K, 64) * 4)
		}
		ns := math.Max(float64(b)/bw, float64(f)/(tflops*1000.0)) + launchNs
		steps = append(steps, CostEstimate{
			Name:        op.Name,
			Op:          string(op.Kind),
			Bytes:       b,
			FLOPs:       f,
			EstimatedNs: ns,
		})
		totalBytes += b
		totalFLOPs += f
		totalNs += ns
	}
	var ai float64
	if totalBytes > 0 {
		ai = float64(totalFLOPs) / float64(totalBytes)
	}
	return &LayerCostBreakdown{
		Device:      device,
		Mode:        SelectLayerExecutionMode(batchM),
		Steps:       steps,
		TotalBytes:  totalBytes,
		TotalFLOPs:  totalFLOPs,
		TotalNs:     totalNs,
		ArithmeticI: ai,
	}, nil
}

// MultiBucketConfig configures multi-bucket M compilation.
type MultiBucketConfig struct {
	Config  ArchConfig
	Buckets []int
	SeqLen  int
	Device  string
}

// ValidateBuckets deduplicates and sorts positive M buckets in ascending order.
func ValidateBuckets(buckets []int) ([]int, error) {
	if len(buckets) == 0 {
		return nil, fmt.Errorf("compile multi-bucket layer: empty buckets")
	}
	seen := make(map[int]bool, len(buckets))
	var sorted []int
	for _, m := range buckets {
		if m <= 0 {
			return nil, fmt.Errorf("compile multi-bucket layer: bucket M %d must be positive", m)
		}
		if !seen[m] {
			seen[m] = true
			sorted = append(sorted, m)
		}
	}
	slices.Sort(sorted)
	return sorted, nil
}

// BucketPlanSpec holds the emitted source and metadata of one M bucket for
// [BuildMultiBucketDispatcherSource].
type BucketPlanSpec struct {
	M          int
	PlanName   string
	Mode       string
	TotalNs    float64
	PlanSource string
}

// BuildMultiBucketDispatcherSource emits a combined C++ translation unit
// containing all bucket plans and the dispatch_layer_m / dispatch_layer_m_c
// entry points.
func BuildMultiBucketDispatcherSource(prelude string, buckets []BucketPlanSpec, params, callArgs []string) string {
	var b strings.Builder
	b.WriteString(prelude)

	for _, bk := range buckets {
		pSrc := strings.TrimPrefix(bk.PlanSource, prelude)
		fmt.Fprintf(&b, "\n// === Bucket M <= %d (mode: %s, est: %.1f ns) ===\n", bk.M, bk.Mode, bk.TotalNs)
		b.WriteString(pSrc)
	}

	paramSig := strings.Join(append([]string{"int M"}, params...), ", ")
	argList := strings.Join(callArgs, ", ")

	fmt.Fprintf(&b, "\ninline void dispatch_layer_m(%s, cudaStream_t stream) {\n", paramSig)
	for i, bk := range buckets {
		runFn := fmt.Sprintf("%s_run", bk.PlanName)
		if i < len(buckets)-1 {
			fmt.Fprintf(&b, "    if (M <= %d) {\n        %s(%s, stream);\n        return;\n    }\n", bk.M, runFn, argList)
		} else {
			fmt.Fprintf(&b, "    %s(%s, stream);\n", runFn, argList)
		}
	}
	b.WriteString("}\n")
	fmt.Fprintf(&b, "\nextern \"C\" void dispatch_layer_m_c(%s, cudaStream_t stream) {\n    dispatch_layer_m(M, %s, stream);\n}\n",
		paramSig, argList)
	return b.String()
}
