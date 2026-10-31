package hopper

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// QuantFormat identifies a sub-byte or low-bit quantization format for dense
// or 2:4 structured sparse weights.
type QuantFormat uint8

const (
	// FormatNVFP4 is 4-bit float4_e2m1fn with per-group scales.
	FormatNVFP4 QuantFormat = iota + 1

	// FormatFP8E4M3 is 8-bit float8_e4m3fn with per-group scales.
	FormatFP8E4M3

	// FormatMXFP4 is 4-bit float4_e2m1fn with UE8M0 power-of-two block scales.
	FormatMXFP4
)

func (f QuantFormat) String() string {
	switch f {
	case FormatNVFP4:
		return "nvfp4"
	case FormatFP8E4M3:
		return "fp8_e4m3"
	case FormatMXFP4:
		return "mxfp4"
	default:
		return fmt.Sprintf("QuantFormat(%d)", uint8(f))
	}
}

// Bits returns the bit-width per quantized non-zero weight element.
func (f QuantFormat) Bits() int {
	switch f {
	case FormatNVFP4, FormatMXFP4:
		return 4
	case FormatFP8E4M3:
		return 8
	default:
		return 0
	}
}

// Arch identifies an NVIDIA GPU SM target architecture.
type Arch = string

// Canonical SM target architecture constants.
const (
	ArchSM80   Arch = "sm_80"
	ArchSM89   Arch = "sm_89"
	ArchSM90   Arch = "sm_90"
	ArchSM90A  Arch = "sm_90a"
	ArchSM100  Arch = "sm_100"
	ArchSM100A Arch = "sm_100a"
	ArchSM103A Arch = "sm_103a"
	ArchSM107A Arch = "sm_107a"
	ArchSM120A Arch = "sm_120a"
	ArchSM121A Arch = "sm_121a"
)

// FastWeightRuleKind selects the fast-weight matrix state recurrence rule for
// [FastWeightLinearAttn].
type FastWeightRuleKind string

const (
	// RuleDeltaNet is the Householder delta-rule fast-weight recurrence:
	//   S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
	//   o_t = S_t q_t
	RuleDeltaNet FastWeightRuleKind = "deltanet"

	// RuleGatedDeltaNet adds scalar/channel forget-gate decay alpha_t:
	//   S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
	//   o_t = S_t q_t
	RuleGatedDeltaNet FastWeightRuleKind = "gated_deltanet"

	// RuleRWKV7 is the RWKV-7 ("Goose") generalized vector-gated delta rule
	// with diagonal decay and rank-1 state erasure/replacement:
	//   S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
	//   o_t = S_t q_t
	RuleRWKV7 FastWeightRuleKind = "rwkv7"

	// RuleTTTLinear is the Test-Time Training (TTT-Linear) online gradient
	// descent update on L2 reconstruction loss 1/2 ||S_{t-1} k_t - v_t||^2:
	//   S_t = alpha_t S_{t-1} - beta_t (alpha_t S_{t-1} k_t - v_t) k_t^T
	//       = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
	//   o_t = S_t q_t
	RuleTTTLinear FastWeightRuleKind = "ttt_linear"
)

func (r FastWeightRuleKind) String() string {
	if r == "" {
		return string(RuleGatedDeltaNet)
	}
	return string(r)
}

func (r FastWeightRuleKind) valid() bool {
	switch r {
	case RuleDeltaNet, RuleGatedDeltaNet, RuleRWKV7, RuleTTTLinear:
		return true
	default:
		return false
	}
}

// FastWeightLinearAttn configures a fused fast-weight linear attention
// megakernel stage (DeltaNet, Gated-DeltaNet, RWKV-7, or TTT-Linear).
//
// Each CTA processes one (batch, head) pair and keeps rows of the d_h x d_h
// fast-weight state matrix S_t in registers and shared memory across recurrent
// steps so that the Householder state update and query readout execute in a
// single fused stage without intermediate global memory round-trips:
//
//	S_t = alpha_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
//	o_t = S_t q_t
type FastWeightLinearAttn struct {
	Name     string
	Rule     FastWeightRuleKind
	Batch    int
	NumHeads int
	HeadDim  int // d_k == d_v (e.g. 64 or 128), state S_t is [Batch, NumHeads, HeadDim, HeadDim]
	UseDecay bool
	Arch     Arch
}

func (fw FastWeightLinearAttn) norm() (FastWeightLinearAttn, error) {
	if strings.TrimSpace(fw.Name) == "" {
		fw.Name = "fastweight_linear_attn"
	}
	if !cIdent(fw.Name) {
		return fw, fmt.Errorf("fastweight linear attn: invalid kernel name %q", fw.Name)
	}
	if fw.Rule == "" {
		fw.Rule = RuleGatedDeltaNet
	}
	if !fw.Rule.valid() {
		return fw, fmt.Errorf("fastweight linear attn %s: unsupported rule %q", fw.Name, fw.Rule)
	}
	if fw.Batch <= 0 {
		return fw, fmt.Errorf("fastweight linear attn %s: Batch %d must be >= 1", fw.Name, fw.Batch)
	}
	if fw.NumHeads <= 0 {
		return fw, fmt.Errorf("fastweight linear attn %s: NumHeads %d must be >= 1", fw.Name, fw.NumHeads)
	}
	if fw.HeadDim <= 0 || fw.HeadDim > 256 {
		return fw, fmt.Errorf("fastweight linear attn %s: HeadDim %d must be in 1..256", fw.Name, fw.HeadDim)
	}
	if strings.TrimSpace(fw.Arch) == "" {
		fw.Arch = ArchSM90A
	}
	if fw.Rule == RuleGatedDeltaNet || fw.Rule == RuleRWKV7 {
		fw.UseDecay = true
	}
	return fw, nil
}

// CUDA emits the fused device stage and __global__ kernel for fw. Each CTA is
// assigned a (batch, head) state tile S_t of shape [HeadDim, HeadDim] and keeps
// rows of S_t in registers/SMEM across the Householder delta-rule update and
// query readout o_t = S_t q_t.
func (fw FastWeightLinearAttn) CUDA() string {
	normed, err := fw.norm()
	if err != nil {
		if fw.Name == "" {
			fw.Name = "fastweight_linear_attn"
		}
		if fw.Rule == "" {
			fw.Rule = RuleGatedDeltaNet
		}
		if fw.Batch <= 0 {
			fw.Batch = 1
		}
		if fw.NumHeads <= 0 {
			fw.NumHeads = 1
		}
		if fw.HeadDim <= 0 {
			fw.HeadDim = 64
		}
		if fw.Arch == "" {
			fw.Arch = ArchSM90A
		}
		normed = fw
	}

	useDecayInt := 0
	if normed.UseDecay {
		useDecayInt = 1
	}

	var b strings.Builder
	fmt.Fprintf(&b, "// %s: Fused Fast-Weight Linear Attention Stage (rule=%s, arch=%s, B=%d, H=%d, D=%d, decay=%v)\n",
		normed.Name, normed.Rule, normed.Arch, normed.Batch, normed.NumHeads, normed.HeadDim, normed.UseDecay)
	b.WriteString("// State update: S_t = alpha_t * S_{t-1} * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T\n")
	b.WriteString("// Query readout: o_t = S_t * q_t (fused in registers/SMEM without global state spill)\n")
	fmt.Fprintf(&b, "EG_DEV void %s_stage_dev(\n", normed.Name)
	b.WriteString("    int bh_idx,\n")
	b.WriteString("    int num_steps,\n")
	b.WriteString("    const float* __restrict__ q,\n")
	b.WriteString("    const float* __restrict__ k,\n")
	b.WriteString("    const float* __restrict__ v,\n")
	b.WriteString("    const float* __restrict__ beta,\n")
	b.WriteString("    const float* __restrict__ alpha,\n")
	b.WriteString("    const float* __restrict__ state_in,\n")
	b.WriteString("    float* __restrict__ out,\n")
	b.WriteString("    float* __restrict__ state_out) {\n")
	fmt.Fprintf(&b, "    constexpr int B = %d, H = %d, D = %d, USE_DECAY = %d;\n",
		normed.Batch, normed.NumHeads, normed.HeadDim, useDecayInt)
	b.WriteString(`    const int tid = (int)threadIdx.x;
    if (bh_idx >= B * H) return;

    __shared__ float s_q[D];
    __shared__ float s_k[D];
    __shared__ float s_v[D];

    const size_t state_base = (size_t)bh_idx * (size_t)(D * D);
    float s_row[D];
    if (tid < D) {
        #pragma unroll 4
        for (int j = 0; j < D; ++j) {
            s_row[j] = state_in ? state_in[state_base + (size_t)tid * D + j] : 0.0f;
        }
    }
    __syncthreads();

    for (int step = 0; step < num_steps; ++step) {
        const size_t vec_base = ((size_t)step * (size_t)(B * H) + (size_t)bh_idx) * (size_t)D;
        const size_t gate_idx = (size_t)step * (size_t)(B * H) + (size_t)bh_idx;
        if (tid < D) {
            s_q[tid] = q[vec_base + tid];
            s_k[tid] = k[vec_base + tid];
            s_v[tid] = v[vec_base + tid];
        }
        __syncthreads();

        const float beta_t = beta ? beta[gate_idx] : 1.0f;
        const float alpha_t = (USE_DECAY && alpha) ? alpha[gate_idx] : 1.0f;

        if (tid < D) {
            // 1. Project state row onto k_t: p_i = sum_j S_{t-1}[i, j] * k_t[j]
            float proj_k = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < D; ++j) {
                proj_k = fmaf(s_row[j], s_k[j], proj_k);
            }

            // 2. Householder / gated delta-rule fast-weight update + fused query readout o_t = S_t * q_t
            const float v_i = s_v[tid];
            float o_i = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < D; ++j) {
                const float k_j = s_k[j];
                const float s_prev = s_row[j];
                const float s_new = alpha_t * (s_prev - beta_t * proj_k * k_j) + beta_t * v_i * k_j;
                s_row[j] = s_new;
                o_i = fmaf(s_new, s_q[j], o_i);
            }
            out[vec_base + tid] = o_i;
        }
        __syncthreads();
    }

    if (tid < D && state_out) {
        #pragma unroll 4
        for (int j = 0; j < D; ++j) {
            state_out[state_base + (size_t)tid * D + j] = s_row[j];
        }
    }
}

`)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n", normed.Name)
	b.WriteString("    int num_steps,\n")
	b.WriteString("    const float* __restrict__ q,\n")
	b.WriteString("    const float* __restrict__ k,\n")
	b.WriteString("    const float* __restrict__ v,\n")
	b.WriteString("    const float* __restrict__ beta,\n")
	b.WriteString("    const float* __restrict__ alpha,\n")
	b.WriteString("    const float* __restrict__ state_in,\n")
	b.WriteString("    float* __restrict__ out,\n")
	b.WriteString("    float* __restrict__ state_out) {\n")
	fmt.Fprintf(&b, "    %s_stage_dev((int)blockIdx.x, num_steps, q, k, v, beta, alpha, state_in, out, state_out);\n", normed.Name)
	b.WriteString("}\n")
	return b.String()
}

// EvalFastWeightLinearAttnHost evaluates the exact fast-weight matrix state
// update and query readout on the host for [FastWeightLinearAttn]:
//
//	S_t = alpha_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
//	o_t = S_t q_t
//
// The input vectors q, k, v have shape [Batch, NumHeads, HeadDim] for a single
// recurrent step or [Steps, Batch, NumHeads, HeadDim] for multiple steps.
// The state matrix stateIn (and returned stateOut) has shape
// [Batch, NumHeads, HeadDim, HeadDim]. When stateIn is empty, S_0 is
// initialized to zero.
func EvalFastWeightLinearAttnHost(
	fw FastWeightLinearAttn,
	q, k, v, beta, alpha, stateIn []float32,
) (out, stateOut []float32, err error) {
	normed, err := fw.norm()
	if err != nil {
		return nil, nil, err
	}
	b := normed.Batch
	h := normed.NumHeads
	d := normed.HeadDim
	bh := b * h
	stepElems := bh * d
	stateElems := bh * d * d

	if len(q) == 0 || len(q)%stepElems != 0 {
		return nil, nil, fmt.Errorf("eval fastweight linear attn %s: len(q)=%d must be a positive multiple of Batch*NumHeads*HeadDim=%d",
			normed.Name, len(q), stepElems)
	}
	if len(k) != len(q) || len(v) != len(q) {
		return nil, nil, fmt.Errorf("eval fastweight linear attn %s: q/k/v length mismatch (%d, %d, %d)",
			normed.Name, len(q), len(k), len(v))
	}
	steps := len(q) / stepElems

	if len(stateIn) != 0 && len(stateIn) != stateElems {
		return nil, nil, fmt.Errorf("eval fastweight linear attn %s: len(stateIn)=%d, want 0 or %d",
			normed.Name, len(stateIn), stateElems)
	}
	if len(beta) != 0 && len(beta) != steps*bh && len(beta) != bh && len(beta) != steps*stepElems && len(beta) != stepElems {
		return nil, nil, fmt.Errorf("eval fastweight linear attn %s: len(beta)=%d, want %d (or %d)",
			normed.Name, len(beta), steps*bh, bh)
	}
	if len(alpha) != 0 && len(alpha) != steps*bh && len(alpha) != bh && len(alpha) != steps*stepElems && len(alpha) != stepElems {
		return nil, nil, fmt.Errorf("eval fastweight linear attn %s: len(alpha)=%d, want %d (or %d)",
			normed.Name, len(alpha), steps*bh, bh)
	}

	state := make([]float64, stateElems)
	if len(stateIn) == stateElems {
		for i, sv := range stateIn {
			state[i] = float64(sv)
		}
	}
	out = make([]float32, len(q))

	applyDecay := fw.UseDecay || fw.Rule == RuleGatedDeltaNet || fw.Rule == RuleRWKV7 || (fw.Rule == RuleTTTLinear && fw.UseDecay)

	betaAt := func(step, bhIdx, row int) float64 {
		switch len(beta) {
		case 0:
			return 1.0
		case steps * bh:
			return float64(beta[step*bh+bhIdx])
		case bh:
			return float64(beta[bhIdx])
		case steps * stepElems:
			return float64(beta[(step*bh+bhIdx)*d+row])
		case stepElems:
			return float64(beta[bhIdx*d+row])
		default:
			return 1.0
		}
	}

	alphaAt := func(step, bhIdx, col int) float64 {
		if !applyDecay || len(alpha) == 0 {
			return 1.0
		}
		switch len(alpha) {
		case steps * bh:
			return float64(alpha[step*bh+bhIdx])
		case bh:
			return float64(alpha[bhIdx])
		case steps * stepElems:
			return float64(alpha[(step*bh+bhIdx)*d+col])
		case stepElems:
			return float64(alpha[bhIdx*d+col])
		default:
			return 1.0
		}
	}

	projK := make([]float64, d)
	for step := range steps {
		for bhIdx := range bh {
			vecOff := (step*bh + bhIdx) * d
			stOff := bhIdx * d * d
			qVec := q[vecOff : vecOff+d]
			kVec := k[vecOff : vecOff+d]
			vVec := v[vecOff : vecOff+d]

			// 1. Compute projection p = S_{t-1} k_t for each row i in 0..d-1.
			for i := range d {
				var sum float64
				rowOff := stOff + i*d
				for j := range d {
					sum += state[rowOff+j] * float64(kVec[j])
				}
				projK[i] = sum
			}

			// 2. Update S_t = alpha_t * S_{t-1} * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T
			// and compute fused readout o_t = S_t * q_t.
			for i := range d {
				bVal := betaAt(step, bhIdx, i)
				vVal := float64(vVec[i])
				pk := projK[i]
				rowOff := stOff + i*d
				var oAcc float64
				for j := range d {
					aVal := alphaAt(step, bhIdx, j)
					kVal := float64(kVec[j])
					sPrev := state[rowOff+j]
					sNew := aVal*(sPrev-bVal*pk*kVal) + bVal*vVal*kVal
					state[rowOff+j] = sNew
					oAcc += sNew * float64(qVec[j])
				}
				out[vecOff+i] = float32(oAcc)
			}
		}
	}

	stateOut = make([]float32, stateElems)
	for i, sv := range state {
		if math.IsNaN(sv) {
			return nil, nil, fmt.Errorf("eval fastweight linear attn %s: NaN in state at index %d", normed.Name, i)
		}
		stateOut[i] = float32(sv)
	}
	return out, stateOut, nil
}

// BuildHybridDeltaNetAttentionMegakernel builds a multi-layer hybrid
// FastWeightLinearAttn (DeltaNet / Gated-DeltaNet / RWKV-7 / TTT) + GQA
// softmax attention decoder stack (such as 3:1 Gated-DeltaNet : GQA in
// Qwen3-Next and Kimi-Linear) and lowers the entire stack into a single
// persistent cross-layer [Megakernel].
//
// Layers with (layerIdx+1)%gqaInterval == 0 are lowered as GQA attention +
// SwiGLU FFN layers; all other layers are lowered as FastWeightLinearAttn
// delta-rule + SwiGLU FFN layers.
func BuildHybridDeltaNetAttentionMegakernel(
	name string,
	numLayers int,
	gqaInterval int,
	fw FastWeightLinearAttn,
	hiddenDim, intermDim int,
) (*Plan, *Megakernel, error) {
	if strings.TrimSpace(name) == "" {
		name = "hybrid_deltanet_attn"
	}
	if !cIdent(name) {
		return nil, nil, fmt.Errorf("hybrid deltanet megakernel: invalid name %q", name)
	}
	if numLayers <= 0 {
		return nil, nil, fmt.Errorf("hybrid deltanet megakernel %s: numLayers %d must be >= 1", name, numLayers)
	}
	if gqaInterval <= 0 {
		gqaInterval = 4
	}
	normFW, err := fw.norm()
	if err != nil {
		return nil, nil, fmt.Errorf("hybrid deltanet megakernel %s: %w", name, err)
	}
	if hiddenDim <= 0 {
		hiddenDim = normFW.NumHeads * normFW.HeadDim
	}
	if hiddenDim < 16 {
		return nil, nil, fmt.Errorf("hybrid deltanet megakernel %s: hiddenDim %d must be >= 16", name, hiddenDim)
	}
	if intermDim <= 0 {
		intermDim = 2 * hiddenDim
	}
	if intermDim < 16 {
		return nil, nil, fmt.Errorf("hybrid deltanet megakernel %s: intermDim %d must be >= 16", name, intermDim)
	}

	m := min(max(normFW.Batch, 1), 16)
	hPlan := max(((hiddenDim+255)/256)*256, 256)
	interPlan := max(((intermDim+255)/256)*256, 256)
	stateDim := max(((normFW.NumHeads*normFW.HeadDim+255)/256)*256, hPlan)
	const eps = 1e-5

	buildLayerPlan := func(layerIdx int) (*Plan, error) {
		isGQA := (layerIdx+1)%gqaInterval == 0
		prefix := fmt.Sprintf("l%d", layerIdx)

		b := NewBuilder()
		xn := b.Input("xn", BF16, m, hPlan)
		xRes := b.Input(prefix+"_res_in", F32, m, hPlan)

		var attnHid egraph.Id
		var layerKind string
		if isGQA {
			layerKind = "gqa"
			wQKV := b.Input(prefix+"_gqa_qkv_w", BF16, hPlan, hPlan)
			qkv := b.MatMul(xn, wQKV)
			ctxScale := b.Input(prefix+"_gqa_ctx_scale", F32, m, hPlan)
			ctx := b.Cast(BF16, b.Map(MustLambda("(mul $0 $1)"), qkv, ctxScale))
			wO := b.Input(prefix+"_gqa_o_w", BF16, hPlan, hPlan)
			attnOut := b.MatMul(ctx, wO)
			attnHid = b.Map(MustLambda("(add $0 $1)"), attnOut, xRes)
		} else {
			layerKind = string(normFW.Rule)
			wIn := b.Input(prefix+"_dn_in_w", BF16, stateDim, hPlan)
			xProj := b.MatMul(xn, wIn)
			sPrev := b.Input(prefix+"_dn_state", F32, m, stateDim)
			betaGate := b.Input(prefix+"_dn_beta", F32, m, stateDim)
			var sNext egraph.Id
			if normFW.UseDecay {
				alphaDecay := b.Input(prefix+"_dn_alpha", F32, m, stateDim)
				sNext = b.Map(MustLambda("(add (mul $0 $1) (mul $2 $3))"), alphaDecay, sPrev, betaGate, xProj)
			} else {
				sNext = b.Map(MustLambda("(add $0 (mul $1 $2))"), sPrev, betaGate, xProj)
			}
			sNextBF16 := b.Cast(BF16, sNext)
			wOut := b.Input(prefix+"_dn_out_w", BF16, hPlan, stateDim)
			dnOut := b.MatMul(sNextBF16, wOut)
			attnHid = b.Map(MustLambda("(add $0 $1)"), dnOut, xRes)
		}

		ffnNormW := b.Input(prefix+"_ffn_norm_w", F32, hPlan)
		ffnIn := b.Cast(BF16, b.RMSNorm(attnHid, ffnNormW, hPlan, eps))
		wGU := b.Input(prefix+"_ffn_gate_up_w", BF16, 2*interPlan, hPlan)
		act := b.SwiGLU(b.MatMul(ffnIn, wGU))
		wDown := b.Input(prefix+"_ffn_down_w", BF16, hPlan, interPlan)
		down := b.MatMul(act, wDown)
		postHid := b.Map(MustLambda("(add $0 $1)"), down, attnHid)
		nextNormW := b.Input(prefix+"_next_norm_w", F32, hPlan)
		outNode := b.Cast(BF16, b.RMSNorm(postHid, nextNormW, hPlan, eps))

		planName := fmt.Sprintf("%s_%s", prefix, layerKind)
		expr, err := Optimize(b.G, outNode)
		if err != nil {
			return nil, fmt.Errorf("optimize %s: %w", planName, err)
		}
		pl, err := Compile(expr, planName, BF16)
		if err != nil {
			return nil, fmt.Errorf("compile %s: %w", planName, err)
		}
		normalizeHybridGEMMRows(pl, m)
		return pl, nil
	}

	var fused *Plan
	for l := range numLayers {
		lp, err := buildLayerPlan(l)
		if err != nil {
			return nil, nil, fmt.Errorf("hybrid deltanet megakernel %s layer %d: %w", name, l, err)
		}
		if fused == nil {
			fused = preparePlanForMegakernel(lp)
			normalizeHybridGEMMRows(fused, m)
			continue
		}
		glue := CrossLayerGlue{
			Name:               fmt.Sprintf("%s_upto_l%d", name, l),
			LayerLPrefix:       fmt.Sprintf("s%d_", l-1),
			LayerNextPrefix:    fmt.Sprintf("l%d_", l),
			BridgeInput:        "xn",
			RetainBridgeOutput: true,
		}
		fused, err = FuseCrossLayerPlans(fused, lp, glue)
		if err != nil {
			return nil, nil, fmt.Errorf("hybrid deltanet megakernel %s fuse layer %d: %w", name, l, err)
		}
		normalizeHybridGEMMRows(fused, m)
	}

	fused.Name = name
	mega, err := fused.Megakernel(name + "_mega")
	if err != nil {
		return nil, nil, fmt.Errorf("hybrid deltanet megakernel %s lower: %w", name, err)
	}
	return fused, &mega, nil
}

func normalizeHybridGEMMRows(pl *Plan, m int) {
	if pl == nil {
		return
	}
	for i := range pl.Kernels {
		if g := pl.Kernels[i].GEMM; g != nil {
			if g.MaxM < m {
				g.MaxM = max(m, 1)
			}
			for _, r := range []int{32, 16, 8, 4, 2, 1} {
				if g.GLU && r%2 != 0 {
					continue
				}
				if g.N%r == 0 && r*max(m, g.MaxM, 1) <= 256 {
					g.RowsPerBlock = r
					break
				}
			}
		}
		if mm := pl.Kernels[i].MMA; mm != nil {
			if mm.MaxM < m {
				mm.MaxM = max(m, 1)
			}
			if len(pl.Kernels[i].Args) == 3+len(mm.Extras) {
				mm.SplitK = 1
			}
		}
	}
	_ = slices.Clone(pl.Kernels)
}
