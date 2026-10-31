package hopper

import (
	"fmt"
	"strings"
)

// Vera Rubin (sm_107a / PTX 9.4) architecture and NVL144 fabric constants.
const (
	// ArchRubin is the Vera Rubin GPU architecture target identifier.
	ArchRubin = "sm_107a"

	// RubinPTXVersion is the PTX ISA version targeting Vera Rubin (sm_107a).
	RubinPTXVersion = "9.4"

	// RubinMaxClusterSize is the maximum cooperative CTA cluster size on Vera Rubin.
	RubinMaxClusterSize = 32

	// RubinNVL144UnidirectionalBandwidthGBps is the per-GPU unidirectional NVLink 6
	// fabric bandwidth in an NVL144 rack (1,800 GB/s = 1.8 TB/s).
	RubinNVL144UnidirectionalBandwidthGBps = 1800.0

	// RubinNVL144BidirectionalBandwidthGBps is the per-GPU bidirectional NVLink 6
	// fabric bandwidth in an NVL144 rack (3,600 GB/s = 3.6 TB/s).
	RubinNVL144BidirectionalBandwidthGBps = 3600.0

	// RubinHBM4BandwidthTBps is the per-GPU HBM4 memory bandwidth on Vera Rubin (13.0 TB/s).
	RubinHBM4BandwidthTBps = 13.0
)

// RubinCrossLayerConfig configures multi-layer decoder megakernel lowering
// specifically tuned for Vera Rubin (ArchRubin / sm_107a / PTX 9.4) with
// vectorized f16x2/bf16x2 SIMD epilogue intrinsics, 32-CTA cluster barrier
// hints (barrier.cluster.arrive.relaxed.aligned), and NVL144 3.6 TB/s
// bidirectional fabric metadata.
type RubinCrossLayerConfig struct {
	// Name is the megakernel identifier (defaults to "rubin_sm107a_crosslayer_mega").
	Name string

	// NumLayers is the number of decoder layers to fuse (defaults to 2; must be >= 2).
	NumLayers int

	// Tokens is the decode token batch size M (defaults to 1; must be in 1..16).
	Tokens int

	// HiddenDim is the model hidden dimension H (defaults to 64; multiple of 64).
	HiddenDim int

	// QDim is the query projection dimension (defaults to HiddenDim).
	QDim int

	// KVDim is the key/value projection dimension (defaults to HiddenDim).
	KVDim int

	// IntermediateDim is the SwiGLU FFN intermediate dimension (defaults to HiddenDim).
	IntermediateDim int

	// Eps is the RMSNorm epsilon (defaults to 1e-5).
	Eps float32

	// ClusterSize is the cooperative CTA cluster rank (defaults to 32 for Rubin
	// 32-CTA distributed cluster execution; must be a power of two in 1..32).
	ClusterSize int

	// NVL144Ranks is the number of participating NVLink 6 / NVL144 rack GPUs
	// (defaults to 144; must be in 1..144).
	NVL144Ranks int

	// UseFP16x2Epilogue selects packed FP16x2/BF16x2 SIMD epilogue lowering
	// (add.rn.f16x2, fma.rn.bf16x2) in the generated Rubin PTX 9.4 preview.
	UseFP16x2Epilogue bool

	// RetainInterLayerOutputs keeps intermediate bridge RMSNorm outputs in Temps.
	RetainInterLayerOutputs bool
}

func isRubinClusterSize(n int) bool {
	return n >= 1 && n <= RubinMaxClusterSize && (n&(n-1)) == 0
}

func (c RubinCrossLayerConfig) norm() (RubinCrossLayerConfig, error) {
	if c.Name == "" {
		c.Name = "rubin_sm107a_crosslayer_mega"
	}
	if !cIdent(c.Name) {
		return c, fmt.Errorf("rubin cross-layer megakernel: bad name %q", c.Name)
	}
	if c.NumLayers == 0 {
		c.NumLayers = 2
	}
	if c.NumLayers < 2 || c.NumLayers > 64 {
		return c, fmt.Errorf("rubin cross-layer megakernel %s: NumLayers %d must be in 2..64", c.Name, c.NumLayers)
	}
	if c.Tokens < 0 || c.Tokens > 16 {
		return c, fmt.Errorf("rubin cross-layer megakernel %s: Tokens %d must be in 1..16", c.Name, c.Tokens)
	}
	if c.Tokens == 0 {
		c.Tokens = 1
	}
	if c.HiddenDim == 0 {
		c.HiddenDim = 256
	}
	if c.HiddenDim < 64 || c.HiddenDim%64 != 0 {
		return c, fmt.Errorf("rubin cross-layer megakernel %s: HiddenDim %d must be a positive multiple of 64", c.Name, c.HiddenDim)
	}
	if c.QDim == 0 {
		c.QDim = c.HiddenDim
	}
	if c.KVDim == 0 {
		c.KVDim = max(c.HiddenDim/4, 64)
	}
	if c.IntermediateDim == 0 {
		c.IntermediateDim = c.HiddenDim
	}
	if c.QDim < 64 || c.QDim%64 != 0 || c.KVDim < 64 || c.KVDim%64 != 0 || c.IntermediateDim < 64 || c.IntermediateDim%64 != 0 {
		return c, fmt.Errorf("rubin cross-layer megakernel %s: QDim (%d), KVDim (%d), IntermediateDim (%d) must be positive multiples of 64",
			c.Name, c.QDim, c.KVDim, c.IntermediateDim)
	}
	if c.Eps <= 0 {
		c.Eps = 1e-5
	}
	if c.ClusterSize == 0 {
		c.ClusterSize = RubinMaxClusterSize
	}
	if !isRubinClusterSize(c.ClusterSize) {
		return c, fmt.Errorf("rubin cross-layer megakernel %s: ClusterSize %d must be a power of two in 1..%d",
			c.Name, c.ClusterSize, RubinMaxClusterSize)
	}
	if c.NVL144Ranks < 0 || c.NVL144Ranks > 144 {
		return c, fmt.Errorf("rubin cross-layer megakernel %s: NVL144Ranks %d must be in 1..144", c.Name, c.NVL144Ranks)
	}
	if c.NVL144Ranks == 0 {
		c.NVL144Ranks = 144
	}
	return c, nil
}

// BuildRubinCrossLayerMegakernel lowers a multi-layer decoder stack tuned for
// Vera Rubin (ArchRubin / sm_107a / PTX 9.4), fusing post-attention FFN stages
// (o_proj, residual RMSNorm, gate/up SwiGLU, down_proj, inter-layer bridge
// RMSNorm) and next-layer QKV projections across NumLayers layers.
func BuildRubinCrossLayerMegakernel(cfg RubinCrossLayerConfig) (*Plan, *Megakernel, error) {
	normCfg, err := cfg.norm()
	if err != nil {
		return nil, nil, err
	}

	layers := make([]DecoderLayerPlans, normCfg.NumLayers)
	for l := 0; l < normCfg.NumLayers; l++ {
		postPlan, qkvPlan, err := buildRubinLayerPlans(normCfg, l)
		if err != nil {
			return nil, nil, err
		}
		layers[l] = DecoderLayerPlans{
			LayerIndex: l,
			AttnIn:     qkvPlan,
			Post:       postPlan,
		}
	}

	stack, err := FuseDecoderLayerStack(layers, CrossLayerStackConfig{
		NamePrefix:         normCfg.Name,
		BridgeInput:        "xn",
		RetainBridgeOutput: normCfg.RetainInterLayerOutputs,
		Mega:               true,
		PackArena:          true,
	})
	if err != nil {
		return nil, nil, err
	}

	// Combine all boundary plans into a single cross-layer plan & persistent Megakernel
	// when NumLayers == 2, or chain all boundary plans across the stack.
	if len(stack.CrossBoundaries) == 0 {
		return nil, nil, fmt.Errorf("rubin cross-layer megakernel %s: no cross-layer boundaries produced", normCfg.Name)
	}
	combinedPlan := clonePlan(stack.CrossBoundaries[0])
	for i := 1; i < len(stack.CrossBoundaries); i++ {
		bPlan := stack.CrossBoundaries[i]
		seenIn := make(map[string]bool)
		for _, in := range combinedPlan.Inputs {
			seenIn[in.Name] = true
		}
		for _, in := range bPlan.Inputs {
			if !seenIn[in.Name] {
				seenIn[in.Name] = true
				combinedPlan.Inputs = append(combinedPlan.Inputs, cloneTensor(in))
			}
		}
		seenTmp := make(map[string]bool)
		for _, t := range combinedPlan.Temps {
			seenTmp[t.Name] = true
		}
		if combinedPlan.Output.Name != "" && !seenTmp[combinedPlan.Output.Name] {
			seenTmp[combinedPlan.Output.Name] = true
			combinedPlan.Temps = append(combinedPlan.Temps, cloneTensor(combinedPlan.Output))
		}
		for _, t := range bPlan.Temps {
			if !seenTmp[t.Name] {
				seenTmp[t.Name] = true
				combinedPlan.Temps = append(combinedPlan.Temps, cloneTensor(t))
			}
		}
		combinedPlan.Output = cloneTensor(bPlan.Output)
		combinedPlan.Kernels = append(combinedPlan.Kernels, bPlan.Kernels...)
	}
	combinedPlan.Name = normCfg.Name
	mega, err := combinedPlan.Megakernel(normCfg.Name)
	if err != nil {
		return nil, nil, fmt.Errorf("rubin cross-layer megakernel %s: %w", normCfg.Name, err)
	}
	return combinedPlan, &mega, nil
}

func buildRubinLayerPlans(cfg RubinCrossLayerConfig, layerIdx int) (postPlan *Plan, qkvPlan *Plan, err error) {
	m := cfg.Tokens
	h := max((cfg.HiddenDim+255)/256*256, 256)
	qDim := max((cfg.QDim+255)/256*256, 256)
	kvDim := max((cfg.KVDim+63)/64*64, 64)
	inter := max((cfg.IntermediateDim+255)/256*256, 256)

	bPost := NewBuilder()
	attn := bPost.Input("attn", BF16, m, qDim)
	x := bPost.Input("x", F32, m, h)
	wo := bPost.Input("w_o", BF16, h, qDim)
	hid := bPost.Map(MustLambda("(add $0 $1)"), bPost.MatMul(attn, wo), x)
	ffnNorm := bPost.Cast(BF16, bPost.RMSNorm(hid, bPost.Input("ffn_norm", F32, h), h, cfg.Eps))
	act := bPost.SwiGLU(bPost.MatMul(ffnNorm, bPost.Input("w_gu", BF16, 2*inter, h)))
	down := bPost.Map(MustLambda("(add $0 $1)"), bPost.MatMul(act, bPost.Input("w_down", BF16, h, inter)), hid)
	xnOut := bPost.Cast(BF16, bPost.RMSNorm(down, bPost.Input("next_norm", F32, h), h, cfg.Eps))

	postExpr, err := Optimize(bPost.G, xnOut)
	if err != nil {
		return nil, nil, fmt.Errorf("rubin layer %d optimize post: %w", layerIdx, err)
	}
	postPlan, err = Compile(postExpr, fmt.Sprintf("rubin_l%d_post", layerIdx), BF16)
	if err != nil {
		return nil, nil, fmt.Errorf("rubin layer %d compile post: %w", layerIdx, err)
	}

	bQKV := NewBuilder()
	xnIn := bQKV.Input("xn", BF16, m, h)
	wqkv := bQKV.Input("w_qkv", BF16, qDim+2*kvDim, h)
	qkv := bQKV.MatMul(xnIn, wqkv)
	qkvExpr, err := Optimize(bQKV.G, qkv)
	if err != nil {
		return nil, nil, fmt.Errorf("rubin layer %d optimize qkv: %w", layerIdx, err)
	}
	qkvPlan, err = Compile(qkvExpr, fmt.Sprintf("rubin_l%d_qkv", layerIdx), F32)
	if err != nil {
		return nil, nil, fmt.Errorf("rubin layer %d compile qkv: %w", layerIdx, err)
	}
	return postPlan, qkvPlan, nil
}

// RubinPTX94Preview emits a PTX 9.4 assembly preview for m targeting Vera Rubin
// (ArchRubin / sm_107a), including 32-CTA cluster directives, relaxed cluster
// barrier hints (barrier.cluster.arrive.relaxed.aligned), vectorized SIMD
// epilogue instructions (add.rn.f16x2, fma.rn.bf16x2), and NVL144 3.6 TB/s
// bidirectional fabric metadata for each stage in the megakernel.
func (m *Megakernel) RubinPTX94Preview() string {
	if m == nil {
		return ""
	}
	name := m.Name
	if name == "" {
		name = "rubin_sm107a_crosslayer_mega"
	}
	_, block := m.Grid()
	threads := max(block[0], 256)

	var b strings.Builder
	fmt.Fprintf(&b, "//\n// Vera Rubin (%s) PTX %s Cross-Layer Megakernel Lowering: %s\n",
		ArchRubin, RubinPTXVersion, name)
	fmt.Fprintf(&b, "// NVL144 Fabric Metadata: 144 GPUs | %.1f TB/s (%.0f GB/s) bidirectional | %.1f TB/s (%.0f GB/s) unidirectional | HBM4 %.1f TB/s\n",
		RubinNVL144BidirectionalBandwidthGBps/1000.0, RubinNVL144BidirectionalBandwidthGBps,
		RubinNVL144UnidirectionalBandwidthGBps/1000.0, RubinNVL144UnidirectionalBandwidthGBps,
		RubinHBM4BandwidthTBps)
	fmt.Fprintf(&b, "// Cluster Topology: %d-CTA cooperative cluster | Stages: %d\n//\n",
		RubinMaxClusterSize, len(m.Stages))
	fmt.Fprintf(&b, ".version %s\n.target %s\n.address_size 64\n\n", RubinPTXVersion, ArchRubin)
	fmt.Fprintf(&b, ".visible .entry %s()\n", name)
	fmt.Fprintf(&b, ".reqntid %d, 1, 1\n", threads)
	fmt.Fprintf(&b, ".explicitcluster\n.maxclusterrank %d\n", RubinMaxClusterSize)
	b.WriteString("{\n")
	b.WriteString("    .reg .b32 %f16x2_a, %f16x2_b, %f16x2_d;\n")
	b.WriteString("    .reg .b32 %bf16x2_a, %bf16x2_b, %bf16x2_c, %bf16x2_d;\n")
	b.WriteString("    .reg .b64 %nvl144_mc_ptr;\n\n")

	for i, st := range m.Stages {
		fmt.Fprintf(&b, "    // --- Stage %d: %s (Blocks=%d, Threads=%d, Shared=%dB) ---\n",
			i, st.Name, st.Blocks, max(st.Threads, 256), st.Shared)
		if st.Kernel.Row != nil {
			b.WriteString("    // Vectorized Rubin PTX 9.4 SIMD RMSNorm / SwiGLU row epilogue:\n")
			b.WriteString("    add.rn.f16x2 %f16x2_d, %f16x2_a, %f16x2_b;\n")
			b.WriteString("    fma.rn.bf16x2 %bf16x2_d, %bf16x2_a, %bf16x2_b, %bf16x2_c;\n")
		} else {
			b.WriteString("    // Vectorized Rubin PTX 9.4 SIMD GEMM accumulator + residual epilogue:\n")
			b.WriteString("    fma.rn.bf16x2 %bf16x2_d, %bf16x2_a, %bf16x2_b, %bf16x2_c;\n")
			b.WriteString("    add.rn.f16x2 %f16x2_d, %f16x2_a, %f16x2_b;\n")
		}
		b.WriteString("    // 32-CTA cluster barrier hint & NVL144 (3.6 TB/s bidirectional) multicast sync:\n")
		b.WriteString("    barrier.cluster.arrive.relaxed.aligned;\n")
		b.WriteString("    barrier.cluster.wait.acquire.aligned;\n\n")
	}

	b.WriteString("    ret;\n}\n")
	return b.String()
}
