package specdecode

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

const defaultVerifyStageName = "eg_spec_grammar_verify_sample"

const verifyNegInfPrelude = `#ifndef INFINITY
#define INFINITY (__uint_as_float(0x7f800000u))
#endif
`

// FusedVerifySampleConfig configures [BuildFinalHeadVerifySampleMegakernel].
type FusedVerifySampleConfig struct {
	// Name is the fused megakernel identifier. When empty, it defaults to
	// "<finalPost.Name>_<head.Name>_verify_sample_mega".
	Name string

	// Glue configures the cross-layer boundary fusion between finalPost and
	// head. When empty, BridgeInput defaults to "xn", LayerLPrefix to
	// "final_", and LayerNextPrefix to "head_".
	Glue hopper.CrossLayerGlue

	// BridgeInput overrides Glue.BridgeInput when non-empty.
	BridgeInput string

	// VerifyStageName names the speculative verification and sampling
	// epilogue stage. When empty, it defaults to
	// "eg_spec_grammar_verify_sample", which emits device function
	// eg_spec_grammar_verify_sample_dev.
	VerifyStageName string

	// DraftNodes is the number of speculative tree positions M (1..64).
	// When 0, it is inferred from head.Output.Shape[0].
	DraftNodes int

	// VocabSize is the vocabulary dimension V (2..1<<24).
	// When 0, it is inferred from head.Output.Shape[len(Shape)-1].
	VocabSize int

	// PerPositionBitmask indicates that vocab_bitmask holds M * ((V+31)/32)
	// uint32 words (one bitmask per draft position m in [0, M-1]) rather
	// than a single ((V+31)/32)-word mask shared across draft positions.
	PerPositionBitmask bool

	// DefaultTemperature is the default sampling temperature (>= 0; 0 means
	// greedy). The generated megakernel also takes runtime temperature as a
	// launch parameter.
	DefaultTemperature float32
}

// VerifySampleResult holds the deterministic output of
// [EvalFinalHeadVerifySample] and the in-megakernel epilogue stage
// eg_spec_grammar_verify_sample_dev.
type VerifySampleResult struct {
	// AcceptedTokens holds the accepted draft token IDs along the verified
	// tree path (length NumAccepted).
	AcceptedTokens []int32

	// NumAccepted is the number of accepted draft tokens (0 <= NumAccepted <= M).
	NumAccepted int32

	// NextBonusToken is the bonus or replacement token sampled from the
	// terminal or residual distribution (-1 when the grammar allows no token).
	NextBonusToken int32

	// AcceptedNodes holds the draft tree node indices (in [0, M-1]) of the
	// accepted tokens along the verified path.
	AcceptedNodes []int32

	// TargetProbs holds the [M][V] grammar-masked, temperature-scaled target
	// softmax distributions computed via online softmax.
	TargetProbs [][]float32
}

// FusedVerifySampleMegakernel fuses the final decoder layer's post-attention
// plan (o_proj -> rmsnorm -> gate_up -> swiglu -> down_proj -> final_rmsnorm)
// and lm_head projection with an in-megakernel grammar bitmask,
// Philox-4x32-10, and speculative tree verification epilogue stage
// (eg_spec_grammar_verify_sample_dev) synchronized via [hopper.ReadyPrelude].
type FusedVerifySampleMegakernel struct {
	Name               string
	Config             FusedVerifySampleConfig
	FusedPlan          *hopper.Plan
	BaseMegakernel     hopper.Megakernel
	Stages             []hopper.MegaStage
	LogitsTensor       hopper.Tensor
	DraftNodes         int
	VocabSize          int
	BitmaskWords       int
	HeadStageIndex     int
	HeadReadySlot      int
	HeadBlocks         int
	VerifyStageIndex   int
	VerifyStageName    string
	PerPositionBitmask bool
}

// BuildFinalHeadVerifySampleMegakernel fuses finalPost and head via
// [hopper.BuildCrossLayerMegakernel] and appends an in-megakernel grammar
// bitmask + Philox-4x32-10 + speculative tree verification epilogue stage
// (eg_spec_grammar_verify_sample_dev) that waits on the lm_head stage via
// eg_ready_wait and writes accepted_tokens, num_accepted, and next_bonus_token
// directly to device memory.
func BuildFinalHeadVerifySampleMegakernel(finalPost, head *hopper.Plan, cfg FusedVerifySampleConfig) (*FusedVerifySampleMegakernel, error) {
	if finalPost == nil || len(finalPost.Kernels) == 0 {
		return nil, fmt.Errorf("fused verify sample megakernel: empty finalPost plan")
	}
	if head == nil || len(head.Kernels) == 0 {
		return nil, fmt.Errorf("fused verify sample megakernel: empty head plan")
	}
	if len(head.Output.Shape) < 2 {
		return nil, fmt.Errorf("fused verify sample megakernel: head output %q shape %v needs at least 2 dims [M, V]",
			head.Output.Name, head.Output.Shape)
	}
	if head.Output.DType != hopper.F32 && head.Output.DType != hopper.BF16 {
		return nil, fmt.Errorf("fused verify sample megakernel: unsupported head output dtype %v (want f32 or bf16)",
			head.Output.DType)
	}
	if math.IsNaN(float64(cfg.DefaultTemperature)) || math.IsInf(float64(cfg.DefaultTemperature), 0) || cfg.DefaultTemperature < 0 {
		return nil, fmt.Errorf("fused verify sample megakernel: invalid DefaultTemperature %v", cfg.DefaultTemperature)
	}

	headM := head.Output.Shape[0]
	for _, dim := range head.Output.Shape[1 : len(head.Output.Shape)-1] {
		headM *= dim
	}
	headV := head.Output.Shape[len(head.Output.Shape)-1]

	mNodes := cfg.DraftNodes
	if mNodes == 0 {
		mNodes = headM
	}
	if mNodes != headM {
		return nil, fmt.Errorf("fused verify sample megakernel: DraftNodes %d != head output rows %d", mNodes, headM)
	}
	if mNodes < 1 || mNodes > 64 {
		return nil, fmt.Errorf("fused verify sample megakernel: DraftNodes %d outside 1..64", mNodes)
	}

	vocabSize := cfg.VocabSize
	if vocabSize == 0 {
		vocabSize = headV
	}
	if vocabSize != headV {
		return nil, fmt.Errorf("fused verify sample megakernel: VocabSize %d != head output cols %d", vocabSize, headV)
	}
	if vocabSize < 2 || vocabSize > 1<<24 {
		return nil, fmt.Errorf("fused verify sample megakernel: VocabSize %d outside 2..%d", vocabSize, 1<<24)
	}

	glue := cfg.Glue
	if cfg.BridgeInput != "" {
		glue.BridgeInput = cfg.BridgeInput
	}
	if glue.BridgeInput == "" {
		glue.BridgeInput = "xn"
	}
	if glue.LayerLPrefix == "" {
		glue.LayerLPrefix = "final_"
	}
	if glue.LayerNextPrefix == "" {
		glue.LayerNextPrefix = "head_"
	}
	if glue.Name == "" {
		switch {
		case cfg.Name != "":
			glue.Name = cfg.Name
		case finalPost.Name != "" && head.Name != "":
			glue.Name = finalPost.Name + "_" + head.Name + "_verify_sample"
		default:
			glue.Name = "final_head_verify_sample"
		}
	}

	fusedPlan, baseMega, err := hopper.BuildCrossLayerMegakernel(finalPost, head, glue)
	if err != nil {
		return nil, fmt.Errorf("fused verify sample megakernel: %w", err)
	}

	megaName := cfg.Name
	if megaName == "" {
		megaName = baseMega.Name
	}
	if !cIdent(megaName) {
		return nil, fmt.Errorf("fused verify sample megakernel: bad name %q", megaName)
	}
	baseMega.Name = megaName

	verifyStageName := cfg.VerifyStageName
	if verifyStageName == "" || verifyStageName == defaultVerifyStageName+"_dev" {
		verifyStageName = defaultVerifyStageName
	}
	if verifyStageName != defaultVerifyStageName && !cIdent(verifyStageName) {
		return nil, fmt.Errorf("fused verify sample megakernel: bad verify stage name %q", verifyStageName)
	}

	headIdx := len(baseMega.Stages) - 1
	headSlot := headIdx
	stages := make([]hopper.MegaStage, len(baseMega.Stages)+1)
	copy(stages, baseMega.Stages)

	// Wire lm_head stage to publish its logits tensor on ready[headSlot].
	stages[headIdx].SignalSlot = headSlot
	stages[headIdx].Outputs = []hopper.Tensor{fusedPlan.Output}

	headEndBlock := stages[headIdx].BlockBase + stages[headIdx].Blocks
	verifyIdx := len(baseMega.Stages)
	stages[verifyIdx] = hopper.MegaStage{
		Name:       verifyStageName,
		Grid:       [3]int{1, 1, 1},
		Threads:    256,
		BlockBase:  headEndBlock,
		Blocks:     1,
		Shared:     0,
		WaitSlots:  []int{headSlot},
		SignalSlot: -1,
	}

	normCfg := cfg
	normCfg.Name = megaName
	normCfg.Glue = glue
	normCfg.BridgeInput = glue.BridgeInput
	normCfg.VerifyStageName = verifyStageName
	normCfg.DraftNodes = mNodes
	normCfg.VocabSize = vocabSize

	return &FusedVerifySampleMegakernel{
		Name:               megaName,
		Config:             normCfg,
		FusedPlan:          fusedPlan,
		BaseMegakernel:     baseMega,
		Stages:             stages,
		LogitsTensor:       fusedPlan.Output,
		DraftNodes:         mNodes,
		VocabSize:          vocabSize,
		BitmaskWords:       (vocabSize + 31) / 32,
		HeadStageIndex:     headIdx,
		HeadReadySlot:      headSlot,
		HeadBlocks:         stages[headIdx].Blocks,
		VerifyStageIndex:   verifyIdx,
		VerifyStageName:    verifyStageName,
		PerPositionBitmask: cfg.PerPositionBitmask,
	}, nil
}

// Grid returns the 1D persistent launch grid and 256-thread block size of m.
func (m *FusedVerifySampleMegakernel) Grid() (grid, block [3]int) {
	if m == nil {
		return [3]int{}, [3]int{}
	}
	total := 0
	for _, st := range m.Stages {
		total += st.Blocks
	}
	return [3]int{total, 1, 1}, [3]int{256, 1, 1}
}

// SharedBytes returns the dynamic shared memory bytes per CTA needed by m.
func (m *FusedVerifySampleMegakernel) SharedBytes() int {
	if m == nil {
		return 0
	}
	return m.BaseMegakernel.SharedBytes()
}

// EpochSlot returns the index in the ready counter array reserved for the
// device-resident launch epoch counter when epoch == 0.
func (m *FusedVerifySampleMegakernel) EpochSlot() int {
	if m == nil {
		return 0
	}
	return max(len(m.Stages)-1, 1)
}

// ReadyWords returns the number of uint64 words required in the ready counter
// buffer (inter-stage ready slots plus the device-resident epoch slot).
func (m *FusedVerifySampleMegakernel) ReadyWords() int {
	if m == nil {
		return 0
	}
	return m.EpochSlot() + 1
}

// ReadyBytes returns the byte size of the ready counter buffer.
func (m *FusedVerifySampleMegakernel) ReadyBytes() int {
	return 8 * m.ReadyWords()
}

// StageFileLabels returns a map from synthetic #line file names to stage names
// and the megakernel entry name.
func (m *FusedVerifySampleMegakernel) StageFileLabels() map[string]string {
	if m == nil {
		return nil
	}
	labels := make(map[string]string, len(m.Stages)+1)
	if m.Name != "" {
		labels[m.Name+".cu"] = m.Name
	}
	for _, st := range m.Stages {
		if st.Name != "" {
			labels[st.Name+".cu"] = st.Name
		}
	}
	return labels
}

// StageAttributionMap returns the 1-based #line span ([hopper.StageLineSpan])
// for each stage of m in the translation unit emitted by [FusedVerifySampleMegakernel.LineInfoSource],
// keyed by stage name.
func (m *FusedVerifySampleMegakernel) StageAttributionMap() map[string]hopper.StageLineSpan {
	if m == nil {
		return nil
	}
	_, spans, err := m.LineInfoSource()
	if err != nil {
		return nil
	}
	out := make(map[string]hopper.StageLineSpan, len(spans))
	for _, sp := range spans {
		out[sp.StageName] = sp
	}
	return out
}

func (m *FusedVerifySampleMegakernel) params() []string {
	mutated := make(map[string]bool)
	for _, st := range m.BaseMegakernel.Stages {
		for _, out := range st.Outputs {
			mutated[out.Name] = true
		}
		if st.Kernel.RoPE != nil && len(st.Kernel.Args) >= 3 {
			for _, a := range st.Kernel.Args[len(st.Kernel.Args)-3:] {
				mutated[a] = true
			}
		}
	}
	var params []string
	for _, t := range m.BaseMegakernel.Inputs {
		qual := "const "
		if mutated[t.Name] {
			qual = ""
		}
		params = append(params, qual+t.DType.CType()+"* __restrict__ "+t.Name)
	}
	for _, t := range m.BaseMegakernel.Temps {
		params = append(params, t.DType.CType()+"* __restrict__ "+t.Name)
	}
	if m.LogitsTensor.Name != "" {
		params = append(params, m.LogitsTensor.DType.CType()+"* __restrict__ "+m.LogitsTensor.Name)
	}
	params = append(params,
		"const uint32_t* __restrict__ vocab_bitmask",
		"const int32_t* __restrict__ tree_parent_idx",
		"const int32_t* __restrict__ draft_token_ids",
		"const float* __restrict__ draft_probs",
		"int32_t* __restrict__ accepted_tokens",
		"int32_t* __restrict__ num_accepted",
		"int32_t* __restrict__ next_bonus_token",
		"float temperature",
		"uint64_t philox_seed",
		"uint64_t philox_offset",
		"unsigned long long* __restrict__ ready",
		"unsigned int epoch",
	)
	return params
}

func (m *FusedVerifySampleMegakernel) verifyStageDeviceFunc() string {
	logitsCType := m.LogitsTensor.DType.CType()
	readLogit := "logits[(size_t)m * V + v]"
	if m.LogitsTensor.DType == hopper.BF16 {
		readLogit = "eg_bf16_to_f32(logits[(size_t)m * V + v])"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "// %s: grammar bitmask + Philox-4x32-10 + speculative tree verification epilogue (M=%d, V=%d).\n",
		m.VerifyStageName, m.DraftNodes, m.VocabSize)
	fmt.Fprintf(&b, "EG_DEV void eg_spec_grammar_verify_sample_dev(\n")
	fmt.Fprintf(&b, "    dim3 eg_bidx,\n")
	fmt.Fprintf(&b, "    dim3 eg_gdim,\n")
	fmt.Fprintf(&b, "    const %s* __restrict__ logits,\n", logitsCType)
	fmt.Fprintf(&b, "    const uint32_t* __restrict__ vocab_bitmask,\n")
	fmt.Fprintf(&b, "    const int32_t* __restrict__ tree_parent_idx,\n")
	fmt.Fprintf(&b, "    const int32_t* __restrict__ draft_token_ids,\n")
	fmt.Fprintf(&b, "    const float* __restrict__ draft_probs,\n")
	fmt.Fprintf(&b, "    int32_t* __restrict__ accepted_tokens,\n")
	fmt.Fprintf(&b, "    int32_t* __restrict__ num_accepted,\n")
	fmt.Fprintf(&b, "    int32_t* __restrict__ next_bonus_token,\n")
	fmt.Fprintf(&b, "    float temperature,\n")
	fmt.Fprintf(&b, "    uint64_t philox_seed,\n")
	fmt.Fprintf(&b, "    uint64_t philox_offset,\n")
	fmt.Fprintf(&b, "    unsigned long long* __restrict__ ready,\n")
	fmt.Fprintf(&b, "    unsigned int epoch) {\n")
	fmt.Fprintf(&b, "    (void)eg_bidx;\n")
	fmt.Fprintf(&b, "    (void)eg_gdim;\n")
	fmt.Fprintf(&b, "    constexpr int M = %d, V = %d, W = %d;\n", m.DraftNodes, m.VocabSize, m.BitmaskWords)
	fmt.Fprintf(&b, "    constexpr bool PER_POS_MASK = %v;\n", m.PerPositionBitmask)
	fmt.Fprintf(&b, "    const int tid = threadIdx.x;\n")
	fmt.Fprintf(&b, "    if (tid == 0) {\n")
	fmt.Fprintf(&b, "        eg_ready_wait(&ready[%d], epoch, %d);\n", m.HeadReadySlot, m.HeadBlocks)
	fmt.Fprintf(&b, "    }\n")
	fmt.Fprintf(&b, "    __syncthreads();\n\n")
	b.WriteString(`    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    __shared__ int32_t s_argmax[256];
    __shared__ float row_max[M];
    __shared__ float row_sum[M];
    __shared__ int32_t row_argmax[M];

    const bool greedy = !(temperature > 0.0f);
    const float inv_temp = greedy ? 1.0f : (1.0f / temperature);

    // Per draft position m in [0, M-1]: apply optional vocab_bitmask and online softmax.
    for (int m = 0; m < M; ++m) {
        const uint32_t* mask_row = (vocab_bitmask != nullptr)
            ? (vocab_bitmask + (PER_POS_MASK ? (size_t)m * W : 0u))
            : nullptr;
        float local_max = -INFINITY;
        float local_sum = 0.0f;
        int32_t local_argmax = -1;
        for (int v = tid; v < V; v += 256) {
`)
	fmt.Fprintf(&b, "            float val = %s;\n", readLogit)
	b.WriteString(`            if (vocab_bitmask != nullptr) {
                if (!PER_POS_MASK && (((vocab_bitmask[v >> 5] >> (v & 31)) & 1u) == 0u)) {
                    val = -INFINITY;
                } else if (PER_POS_MASK && (((mask_row[v >> 5] >> (v & 31)) & 1u) == 0u)) {
                    val = -INFINITY;
                }
            }
            if (val > -INFINITY) {
                const float scaled = greedy ? val : (val * inv_temp);
                if (local_argmax < 0) {
                    local_max = scaled;
                    local_sum = 1.0f;
                    local_argmax = v;
                } else if (scaled > local_max) {
                    local_sum = local_sum * expf(local_max - scaled) + 1.0f;
                    local_max = scaled;
                    local_argmax = v;
                } else {
                    local_sum += expf(scaled - local_max);
                }
            }
        }
        s_max[tid] = local_max;
        s_sum[tid] = local_sum;
        s_argmax[tid] = local_argmax;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float ma = s_max[tid], mb = s_max[tid + stride];
                const float da = s_sum[tid], db = s_sum[tid + stride];
                const int32_t ida = s_argmax[tid], idb = s_argmax[tid + stride];
                if (idb >= 0) {
                    if (ida < 0) {
                        s_max[tid] = mb;
                        s_sum[tid] = db;
                        s_argmax[tid] = idb;
                    } else {
                        const float m_new = fmaxf(ma, mb);
                        s_max[tid] = m_new;
                        s_sum[tid] = da * expf(ma - m_new) + db * expf(mb - m_new);
                        if (mb > ma || (mb == ma && idb < ida)) {
                            s_argmax[tid] = idb;
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            row_max[m] = s_max[0];
            row_sum[m] = s_sum[0];
            row_argmax[m] = s_argmax[0];
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int m = 0; m < M; ++m) {
            accepted_tokens[m] = -1;
        }
        const uint32_t k0 = (uint32_t)philox_seed;
        const uint32_t k1 = (uint32_t)(philox_seed >> 32);
        const uint32_t c0 = (uint32_t)philox_offset;
        const uint32_t c1 = (uint32_t)(philox_offset >> 32);
        uint32_t draws = 0u;

        auto token_prob = [&](int m, int v) -> float {
            if (v < 0 || v >= V || row_argmax[m] < 0) return 0.0f;
            if (vocab_bitmask != nullptr) {
                const uint32_t* mask_row = vocab_bitmask + (PER_POS_MASK ? (size_t)m * W : 0u);
                if (((mask_row[v >> 5] >> (v & 31)) & 1u) == 0u) return 0.0f;
            }
`)
	fmt.Fprintf(&b, "            const float raw = %s;\n", readLogit)
	b.WriteString(`            if (!(raw > -INFINITY)) return 0.0f;
            if (greedy) return (v == row_argmax[m]) ? 1.0f : 0.0f;
            const float d = row_sum[m];
            if (!(d > 0.0f)) return 0.0f;
            return expf((raw * inv_temp) - row_max[m]) / d;
        };

        int32_t cur_parent = -1;
        int32_t search_start = 0;
        int32_t acc_count = 0;
        int32_t last_node = 0;
        int32_t rej_tokens[M];
        float rej_probs[M];
        int32_t num_rej = 0;

        for (int step = 0; step < M; ++step) {
            int32_t accepted_node = -1;
            num_rej = 0;
            for (int m = search_start; m < M; ++m) {
                const int32_t par = (tree_parent_idx != nullptr) ? tree_parent_idx[m] : (m - 1);
                if (par != cur_parent) continue;
                last_node = m;
                const int32_t draft_id = (draft_token_ids != nullptr) ? draft_token_ids[m] : -1;
                const float q_prob = (draft_probs != nullptr) ? draft_probs[m] : 1.0f;
                float p_prob = token_prob(m, draft_id);
                for (int r = 0; r < num_rej; ++r) {
                    if (rej_tokens[r] == draft_id) {
                        p_prob = fmaxf(0.0f, p_prob - rej_probs[r]);
                    }
                }
                bool accept = false;
                if (greedy) {
                    accept = (draft_id >= 0 && draft_id < V && draft_id == row_argmax[m]);
                } else {
                    uint32_t w[4];
                    eg_philox4x32(c0, c1, draws++, (uint32_t)m, k0, k1, w);
                    const float u = eg_philox_uniform(w[0]);
                    if (draft_id >= 0 && draft_id < V && p_prob > 0.0f) {
                        accept = (q_prob <= 0.0f) ? true : (u * q_prob <= p_prob);
                    }
                }
                if (accept) {
                    accepted_tokens[acc_count++] = draft_id;
                    accepted_node = m;
                    break;
                }
                rej_tokens[num_rej] = draft_id;
                rej_probs[num_rej] = fmaxf(0.0f, q_prob);
                ++num_rej;
            }
            if (accepted_node < 0) {
                break;
            }
            cur_parent = accepted_node;
            search_start = accepted_node + 1;
            num_rej = 0;
        }

        const int32_t bonus_node = (num_rej > 0) ? last_node : ((cur_parent >= 0) ? cur_parent : 0);
        int32_t bonus_token = row_argmax[bonus_node];
        if (bonus_token >= 0 && !greedy) {
            uint32_t w[4];
            eg_philox4x32(c0, c1, draws++, (uint32_t)bonus_node, k0, k1, w);
            const float u = eg_philox_uniform(w[0]);
            float total_resid = 0.0f;
            for (int v = 0; v < V; ++v) {
                float rv = token_prob(bonus_node, v);
                for (int r = 0; r < num_rej; ++r) {
                    if (rej_tokens[r] == v) {
                        rv = fmaxf(0.0f, rv - rej_probs[r]);
                    }
                }
                total_resid += rv;
            }
            const bool use_resid = (total_resid > 0.0f);
            const float norm_total = use_resid ? total_resid : 1.0f;
            const float target = u * norm_total;
            float cum = 0.0f;
            int32_t last_valid = bonus_token;
            for (int v = 0; v < V; ++v) {
                float rv = token_prob(bonus_node, v);
                if (use_resid) {
                    for (int r = 0; r < num_rej; ++r) {
                        if (rej_tokens[r] == v) {
                            rv = fmaxf(0.0f, rv - rej_probs[r]);
                        }
                    }
                }
                if (rv <= 0.0f) continue;
                last_valid = v;
                cum += rv;
                if (cum > target) {
                    bonus_token = v;
                    last_valid = -1;
                    break;
                }
            }
            if (last_valid >= 0) {
                bonus_token = last_valid;
            }
        }
        *num_accepted = acc_count;
        *next_bonus_token = bonus_token;
    }
}
`)
	if m.VerifyStageName != defaultVerifyStageName {
		fmt.Fprintf(&b, "\nEG_DEV void %s_dev(\n", m.VerifyStageName)
		fmt.Fprintf(&b, "    dim3 eg_bidx,\n    dim3 eg_gdim,\n")
		fmt.Fprintf(&b, "    const %s* __restrict__ logits,\n", logitsCType)
		fmt.Fprintf(&b, "    const uint32_t* __restrict__ vocab_bitmask,\n")
		fmt.Fprintf(&b, "    const int32_t* __restrict__ tree_parent_idx,\n")
		fmt.Fprintf(&b, "    const int32_t* __restrict__ draft_token_ids,\n")
		fmt.Fprintf(&b, "    const float* __restrict__ draft_probs,\n")
		fmt.Fprintf(&b, "    int32_t* __restrict__ accepted_tokens,\n")
		fmt.Fprintf(&b, "    int32_t* __restrict__ num_accepted,\n")
		fmt.Fprintf(&b, "    int32_t* __restrict__ next_bonus_token,\n")
		fmt.Fprintf(&b, "    float temperature,\n    uint64_t philox_seed,\n    uint64_t philox_offset,\n")
		fmt.Fprintf(&b, "    unsigned long long* __restrict__ ready,\n    unsigned int epoch) {\n")
		fmt.Fprintf(&b, "    eg_spec_grammar_verify_sample_dev(eg_bidx, eg_gdim, logits, vocab_bitmask, tree_parent_idx, draft_token_ids, draft_probs, accepted_tokens, num_accepted, next_bonus_token, temperature, philox_seed, philox_offset, ready, epoch);\n}\n")
	}
	return b.String()
}

func (m *FusedVerifySampleMegakernel) cudaSource(lineInfo bool) (string, []hopper.StageLineSpan, error) {
	if m == nil || len(m.Stages) < 2 {
		return "", nil, fmt.Errorf("fused verify sample megakernel: uninitialized megakernel")
	}
	var (
		baseSrc   string
		baseSpans []hopper.StageLineSpan
		err       error
	)
	if lineInfo {
		baseSrc, baseSpans, err = m.BaseMegakernel.LineInfoCUDA()
	} else {
		baseSrc, err = m.BaseMegakernel.CUDA()
	}
	if err != nil {
		return "", nil, err
	}

	baseGrid, _ := m.BaseMegakernel.Grid()
	entryComment := fmt.Sprintf("\n// %s: persistent %d-stage megakernel (%d blocks, %d ready counter words).\n",
		m.BaseMegakernel.Name, len(m.BaseMegakernel.Stages), baseGrid[0], m.BaseMegakernel.ReadyWords())
	splitIdx := strings.Index(baseSrc, entryComment)
	if splitIdx < 0 {
		return "", nil, fmt.Errorf("fused verify sample megakernel: missing base megakernel entry marker in generated CUDA")
	}

	stageDefs := baseSrc[:splitIdx]
	baseEntry := baseSrc[splitIdx+len(entryComment):]

	var b strings.Builder
	b.WriteString(stageDefs)
	b.WriteString("\n")
	b.WriteString(verifyNegInfPrelude)
	b.WriteString(philoxDevice)
	b.WriteString("\n")

	verifyFn := m.verifyStageDeviceFunc()
	commentIdx := strings.Index(verifyFn, "\n")
	verifyComment := verifyFn[:commentIdx+1]
	verifyBody := verifyFn[commentIdx+1:]

	b.WriteString(verifyComment)
	var spans []hopper.StageLineSpan
	if lineInfo {
		spans = make([]hopper.StageLineSpan, len(baseSpans)+1)
		copy(spans, baseSpans)
		file := m.VerifyStageName + ".cu"
		fmt.Fprintf(&b, "#line 1 %q\n", file)
		startLine := 1 + strings.Count(b.String(), "\n")
		b.WriteString(verifyBody)
		endLine := startLine + strings.Count(strings.TrimSuffix(verifyBody, "\n"), "\n")
		spans[len(baseSpans)] = hopper.StageLineSpan{
			StageIndex: m.VerifyStageIndex,
			StageName:  m.VerifyStageName,
			File:       file,
			StartLine:  startLine,
			EndLine:    endLine,
		}
	} else {
		b.WriteString(verifyBody)
	}

	// Build the fused __global__ entry kernel by adapting the base entry body.
	openBrace := strings.Index(baseEntry, ") {\n")
	if openBrace < 0 {
		return "", nil, fmt.Errorf("fused verify sample megakernel: malformed base entry header")
	}
	entryBody := baseEntry[openBrace+len(") {\n"):]

	totalGrid, _ := m.Grid()
	oldResolve := fmt.Sprintf("    epoch = eg_ready_resolve_epoch(&ready[%d], epoch, %d);\n",
		m.BaseMegakernel.EpochSlot(), baseGrid[0])
	newResolve := fmt.Sprintf("    epoch = eg_ready_resolve_epoch(&ready[%d], epoch, %d);\n",
		m.EpochSlot(), totalGrid[0])
	if !strings.Contains(entryBody, oldResolve) {
		return "", nil, fmt.Errorf("fused verify sample megakernel: missing epoch resolution line in base entry")
	}
	entryBody = strings.Replace(entryBody, oldResolve, newResolve, 1)

	// In the lm_head stage (the last stage of BaseMegakernel), signal ready[headSlot] before returning.
	headTailOld := "        return;\n    }\n}\n"
	if !strings.HasSuffix(entryBody, headTailOld) {
		return "", nil, fmt.Errorf("fused verify sample megakernel: unexpected base entry tail")
	}
	logitsElems := 1
	for _, d := range m.LogitsTensor.Shape {
		logitsElems *= d
	}
	logitsBytes := logitsElems * m.LogitsTensor.DType.Bits() / 8

	var tail strings.Builder
	tail.WriteString("        __syncthreads();\n")
	tail.WriteString("        if (threadIdx.x == 0) {\n")
	if logitsBytes > 0 {
		fmt.Fprintf(&tail, "            EG_READY_STAGE(&ready[%d], epoch, %d, %s, %d);\n",
			m.HeadReadySlot, m.HeadBlocks, m.LogitsTensor.Name, logitsBytes)
	}
	fmt.Fprintf(&tail, "            eg_ready_signal(&ready[%d], epoch, 1);\n", m.HeadReadySlot)
	tail.WriteString("        }\n")
	tail.WriteString("        return;\n")
	tail.WriteString("    }\n")

	verifySt := m.Stages[m.VerifyStageIndex]
	fmt.Fprintf(&tail, "    if (eg_bid < %d) {\n", verifySt.BlockBase+verifySt.Blocks)
	fmt.Fprintf(&tail, "        const int eg_local = eg_bid - %d;\n", verifySt.BlockBase)
	tail.WriteString("        const dim3 eg_bidx(eg_local % 1, (eg_local / 1) % 1, eg_local / 1);\n")
	tail.WriteString("        const dim3 eg_gdim(1, 1, 1);\n")
	fmt.Fprintf(&tail, "        eg_spec_grammar_verify_sample_dev(eg_bidx, eg_gdim, %s, vocab_bitmask, tree_parent_idx, draft_token_ids, draft_probs, accepted_tokens, num_accepted, next_bonus_token, temperature, philox_seed, philox_offset, ready, epoch);\n",
		m.LogitsTensor.Name)
	tail.WriteString("        return;\n")
	tail.WriteString("    }\n")
	tail.WriteString("}\n")

	entryBody = entryBody[:len(entryBody)-len(headTailOld)] + tail.String()

	fmt.Fprintf(&b, "\n// %s: persistent %d-stage megakernel (%d blocks, %d ready counter words).\n",
		m.Name, len(m.Stages), totalGrid[0], m.ReadyWords())
	if lineInfo {
		fmt.Fprintf(&b, "#line 1 %q\n", m.Name+".cu")
	}
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n",
		m.Name, strings.Join(m.params(), ",\n    "))
	b.WriteString(entryBody)

	return b.String(), spans, nil
}

// CUDA returns the fused megakernel device source (with [hopper.ReadyPrelude],
// without [hopper.Prelude]).
func (m *FusedVerifySampleMegakernel) CUDA() (string, error) {
	src, _, err := m.cudaSource(false)
	return src, err
}

// LineInfoCUDA returns the fused megakernel device source annotated with
// synthetic #line directives for every stage (including
// eg_spec_grammar_verify_sample_dev) and the megakernel entry, along with the
// 1-based [hopper.StageLineSpan] slice.
func (m *FusedVerifySampleMegakernel) LineInfoCUDA() (string, []hopper.StageLineSpan, error) {
	return m.cudaSource(true)
}

// Launcher returns the host launch functions <Name>_launch and <Name>_run.
func (m *FusedVerifySampleMegakernel) Launcher() (string, error) {
	if m == nil || len(m.Stages) < 2 {
		return "", fmt.Errorf("fused verify sample megakernel: uninitialized megakernel")
	}
	params := m.params()
	names := paramNames(params)
	grid, block := m.Grid()
	smem := m.SharedBytes()

	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n",
		m.Name, strings.Join(params, ", "))
	if smem > 0 {
		fmt.Fprintf(&b, "    constexpr int smem = %d;\n", smem)
		fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", m.Name)
		fmt.Fprintf(&b, "    const dim3 grid(%d), block(%d);\n", grid[0], block[0])
		fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n",
			m.Name, strings.Join(names, ", "))
	} else {
		fmt.Fprintf(&b, "    const dim3 grid(%d), block(%d);\n", grid[0], block[0])
		fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n",
			m.Name, strings.Join(names, ", "))
	}
	fmt.Fprintf(&b, "\ninline void %s_run(%s, cudaStream_t stream) {\n    %s_launch(%s, stream);\n}\n",
		m.Name, strings.Join(params, ", "), m.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns a complete translation unit with [hopper.Prelude],
// [hopper.ReadyPrelude], all stage device functions including
// eg_spec_grammar_verify_sample_dev, the persistent megakernel entry, and its
// host launchers.
func (m *FusedVerifySampleMegakernel) Source() (string, error) {
	src, err := m.CUDA()
	if err != nil {
		return "", err
	}
	l, err := m.Launcher()
	if err != nil {
		return "", err
	}
	return hopper.Prelude + "\n" + src + "\n" + l, nil
}

// LineInfoSource returns a complete translation unit with [hopper.Prelude],
// [hopper.ReadyPrelude], #line-annotated stage device functions (including
// eg_spec_grammar_verify_sample_dev) and megakernel entry, and host launchers,
// along with the shifted 1-based [hopper.StageLineSpan] slice.
func (m *FusedVerifySampleMegakernel) LineInfoSource() (string, []hopper.StageLineSpan, error) {
	src, spans, err := m.LineInfoCUDA()
	if err != nil {
		return "", nil, err
	}
	l, err := m.Launcher()
	if err != nil {
		return "", nil, err
	}
	shift := strings.Count(hopper.Prelude, "\n") + 1
	for i := range spans {
		spans[i].StartLine += shift
		spans[i].EndLine += shift
	}
	return hopper.Prelude + "\n" + src + "\n" + l, spans, nil
}

// EvalFinalHeadVerifySample is the deterministic CPU reference evaluator for
// the in-megakernel grammar bitmask, online softmax / temperature scaling, and
// Philox-4x32-10 speculative tree rejection sampling epilogue stage
// (eg_spec_grammar_verify_sample_dev).
//
// It applies the same 256-lane online softmax tree reduction and Philox-4x32-10
// counter schedule as the device stage so the CPU reference and device
// execution agree bit-for-bit.
func EvalFinalHeadVerifySample(
	logits [][]float32,
	vocabMask []uint32,
	draftIDs []int32,
	draftProbs []float32,
	parentIdx []int32,
	temp float32,
	seed, offset uint64,
) (VerifySampleResult, error) {
	mNodes := len(logits)
	if mNodes < 1 || mNodes > 64 {
		return VerifySampleResult{}, fmt.Errorf("eval verify sample: logits rows M=%d outside 1..64", mNodes)
	}
	vocabSize := len(logits[0])
	if vocabSize < 1 || vocabSize > 1<<24 {
		return VerifySampleResult{}, fmt.Errorf("eval verify sample: vocab size V=%d outside 1..%d", vocabSize, 1<<24)
	}
	for m, row := range logits {
		if len(row) != vocabSize {
			return VerifySampleResult{}, fmt.Errorf("eval verify sample: logits[%d] len=%d, want %d", m, len(row), vocabSize)
		}
	}
	if math.IsNaN(float64(temp)) || math.IsInf(float64(temp), 0) || temp < 0 {
		return VerifySampleResult{}, fmt.Errorf("eval verify sample: invalid temperature %v", temp)
	}

	wordsPerRow := (vocabSize + 31) / 32
	perPosMask := false
	if len(vocabMask) != 0 {
		switch len(vocabMask) {
		case wordsPerRow:
			perPosMask = false
		case mNodes * wordsPerRow:
			perPosMask = true
		default:
			return VerifySampleResult{}, fmt.Errorf("eval verify sample: vocabMask len=%d, want 0, %d, or %d",
				len(vocabMask), wordsPerRow, mNodes*wordsPerRow)
		}
	}

	if len(draftIDs) != mNodes {
		return VerifySampleResult{}, fmt.Errorf("eval verify sample: draftIDs len=%d, want %d", len(draftIDs), mNodes)
	}
	if len(draftProbs) != 0 && len(draftProbs) != mNodes {
		return VerifySampleResult{}, fmt.Errorf("eval verify sample: draftProbs len=%d, want 0 or %d", len(draftProbs), mNodes)
	}
	if len(parentIdx) != 0 && len(parentIdx) != mNodes {
		return VerifySampleResult{}, fmt.Errorf("eval verify sample: parentIdx len=%d, want 0 or %d", len(parentIdx), mNodes)
	}
	for m := range mNodes {
		par := int32(m - 1)
		if len(parentIdx) == mNodes {
			par = parentIdx[m]
		}
		if m == 0 && par != -1 {
			return VerifySampleResult{}, fmt.Errorf("eval verify sample: parentIdx[0]=%d, want -1", par)
		}
		if m > 0 && (par < -1 || int(par) >= m) {
			return VerifySampleResult{}, fmt.Errorf("eval verify sample: parentIdx[%d]=%d outside -1..%d", m, par, m-1)
		}
	}

	negInf := float32(math.Inf(-1))
	greedy := !(temp > 0)
	invTemp := float32(1.0)
	if !greedy {
		invTemp = 1.0 / temp
	}

	allowedAt := func(m, v int) bool {
		if v < 0 || v >= vocabSize {
			return false
		}
		if len(vocabMask) == 0 {
			return true
		}
		base := 0
		if perPosMask {
			base = m * wordsPerRow
		}
		w := vocabMask[base+(v>>5)]
		return ((w >> uint(v&31)) & 1) != 0
	}

	rowMax := make([]float32, mNodes)
	rowSum := make([]float32, mNodes)
	rowArgmax := make([]int32, mNodes)
	targetProbs := make([][]float32, mNodes)

	// Step 1 & 2: Per-position grammar bitmasking and 256-lane online softmax.
	var sMax [256]float32
	var sSum [256]float32
	var sArgmax [256]int32
	for m := range mNodes {
		for tid := range 256 {
			localMax := negInf
			localSum := float32(0)
			localArgmax := int32(-1)
			for v := tid; v < vocabSize; v += 256 {
				val := logits[m][v]
				if !allowedAt(m, v) {
					val = negInf
				}
				if val > negInf {
					scaled := val
					if !greedy {
						scaled = val * invTemp
					}
					if localArgmax < 0 {
						localMax = scaled
						localSum = 1.0
						localArgmax = int32(v)
					} else if scaled > localMax {
						localSum = localSum*float32(math.Exp(float64(localMax-scaled))) + 1.0
						localMax = scaled
						localArgmax = int32(v)
					} else {
						localSum += float32(math.Exp(float64(scaled - localMax)))
					}
				}
			}
			sMax[tid] = localMax
			sSum[tid] = localSum
			sArgmax[tid] = localArgmax
		}
		for stride := 128; stride > 0; stride >>= 1 {
			for tid := range stride {
				ma, mb := sMax[tid], sMax[tid+stride]
				da, db := sSum[tid], sSum[tid+stride]
				ida, idb := sArgmax[tid], sArgmax[tid+stride]
				if idb >= 0 {
					if ida < 0 {
						sMax[tid] = mb
						sSum[tid] = db
						sArgmax[tid] = idb
					} else {
						mNew := max(ma, mb)
						sMax[tid] = mNew
						sSum[tid] = da*float32(math.Exp(float64(ma-mNew))) + db*float32(math.Exp(float64(mb-mNew)))
						if mb > ma || (mb == ma && idb < ida) {
							sArgmax[tid] = idb
						}
					}
				}
			}
		}
		rowMax[m] = sMax[0]
		rowSum[m] = sSum[0]
		rowArgmax[m] = sArgmax[0]

		probs := make([]float32, vocabSize)
		for v := range vocabSize {
			if rowArgmax[m] < 0 || !allowedAt(m, v) {
				continue
			}
			raw := logits[m][v]
			if !(raw > negInf) {
				continue
			}
			if greedy {
				if int32(v) == rowArgmax[m] {
					probs[v] = 1.0
				}
				continue
			}
			if rowSum[m] > 0 {
				probs[v] = float32(math.Exp(float64((raw*invTemp)-rowMax[m]))) / rowSum[m]
			}
		}
		targetProbs[m] = probs
	}

	tokenProb := func(m, v int) float32 {
		if v < 0 || v >= vocabSize {
			return 0
		}
		return targetProbs[m][v]
	}

	// Step 3: Speculative tree rejection sampling with Philox-4x32-10.
	key := [2]uint32{uint32(seed), uint32(seed >> 32)}
	c0, c1 := uint32(offset), uint32(offset>>32)
	draws := uint32(0)

	curParent := int32(-1)
	searchStart := 0
	var acceptedTokens []int32
	var acceptedNodes []int32
	lastNode := int32(0)
	var rejTokens []int32
	var rejProbs []float32

	for range mNodes {
		acceptedNode := int32(-1)
		rejTokens = rejTokens[:0]
		rejProbs = rejProbs[:0]
		for m := searchStart; m < mNodes; m++ {
			par := int32(m - 1)
			if len(parentIdx) == mNodes {
				par = parentIdx[m]
			}
			if par != curParent {
				continue
			}
			lastNode = int32(m)
			draftID := draftIDs[m]
			qProb := float32(1.0)
			if len(draftProbs) == mNodes {
				qProb = draftProbs[m]
			}
			pProb := tokenProb(m, int(draftID))
			for r := range rejTokens {
				if rejTokens[r] == draftID {
					pProb = max(float32(0), pProb-rejProbs[r])
				}
			}
			accept := false
			if greedy {
				accept = draftID >= 0 && int(draftID) < vocabSize && draftID == rowArgmax[m]
			} else {
				w := Philox4x32([4]uint32{c0, c1, draws, uint32(m)}, key)
				draws++
				u := philoxUniform(w[0])
				if draftID >= 0 && int(draftID) < vocabSize && pProb > 0 {
					if qProb <= 0 {
						accept = true
					} else {
						accept = u*qProb <= pProb
					}
				}
			}
			if accept {
				acceptedTokens = append(acceptedTokens, draftID)
				acceptedNodes = append(acceptedNodes, int32(m))
				acceptedNode = int32(m)
				break
			}
			rejTokens = append(rejTokens, draftID)
			rejProbs = append(rejProbs, max(float32(0), qProb))
		}
		if acceptedNode < 0 {
			break
		}
		curParent = acceptedNode
		searchStart = int(acceptedNode) + 1
		rejTokens = rejTokens[:0]
		rejProbs = rejProbs[:0]
	}

	bonusNode := curParent
	if bonusNode < 0 {
		bonusNode = 0
	}
	if len(rejTokens) > 0 {
		bonusNode = lastNode
	}

	bonusToken := rowArgmax[bonusNode]
	if bonusToken >= 0 && !greedy {
		w := Philox4x32([4]uint32{c0, c1, draws, uint32(bonusNode)}, key)
		draws++
		u := philoxUniform(w[0])
		var totalResid float32
		for v := range vocabSize {
			rv := tokenProb(int(bonusNode), v)
			for r := range rejTokens {
				if int(rejTokens[r]) == v {
					rv = max(float32(0), rv-rejProbs[r])
				}
			}
			totalResid += rv
		}
		useResid := totalResid > 0
		normTotal := float32(1.0)
		if useResid {
			normTotal = totalResid
		}
		target := u * normTotal
		var cum float32
		lastValid := bonusToken
		for v := range vocabSize {
			rv := tokenProb(int(bonusNode), v)
			if useResid {
				for r := range rejTokens {
					if int(rejTokens[r]) == v {
						rv = max(float32(0), rv-rejProbs[r])
					}
				}
			}
			if rv <= 0 {
				continue
			}
			lastValid = int32(v)
			cum += rv
			if cum > target {
				bonusToken = int32(v)
				lastValid = -1
				break
			}
		}
		if lastValid >= 0 {
			bonusToken = lastValid
		}
	}

	return VerifySampleResult{
		AcceptedTokens: slices.Clone(acceptedTokens),
		NumAccepted:    int32(len(acceptedTokens)),
		NextBonusToken: bonusToken,
		AcceptedNodes:  slices.Clone(acceptedNodes),
		TargetProbs:    targetProbs,
	}, nil
}
