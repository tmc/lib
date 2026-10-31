package specdecode

import (
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

const (
	nativeVerifyDraftNodes   = 4
	nativeVerifyHiddenDim    = 64
	nativeVerifyIntermDim    = 128
	nativeVerifyVocabSize    = 128
	nativeVerifyTemperature  = float32(0.7)
	nativeVerifyPhiloxSeed   = uint64(0xfeedfacecafebeef)
	nativeVerifyPhiloxOffset = uint64(77)
	nativeVerifyTimingBlocks = 12
)

var (
	nativeVerifyTreeParents = []int32{-1, 0, 0, 2}
	nativeVerifyDraftIDs    = []int32{13, 28, 41, 97}
	nativeVerifyDraftProbs  = []float32{0.25, 0.90, 0.20, 0.25}
)

type nativeSpecVerifyGateClosedError struct{}

func (nativeSpecVerifyGateClosedError) Error() string {
	return "native qualification pending: set CUDATEST_NATIVE=1 under gpu-lock"
}

func (nativeSpecVerifyGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func checkNativeSpecVerifyGate(getenv func(string) string) error {
	if getenv("CUDATEST_NATIVE") != "1" {
		return nativeSpecVerifyGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("specdecode: native verify gate open without inherited gpu lock: %w", err)
	}
	return nil
}

func requireNativeSpecVerifyGate(t *testing.T) (nvcc, cudaVer, arch string) {
	t.Helper()
	if os.Getenv("CUDATEST_NATIVE") != "1" {
		t.Skip(nativeSpecVerifyGateClosedError{}.Error())
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native qualification requires gpu-lock: %v", err)
	}
	nvccPath, ver := cudatest.NVCC(t)
	arch = strings.TrimSpace(os.Getenv("CUDATEST_SM_ARCH"))
	if arch == "" {
		arch = string(hopper.TargetSM121A)
	}
	cudatest.SkipUnlessTarget(t, nvccPath, arch)
	return nvccPath, ver.String(), arch
}

func nativeSpecVerifyTempDir(t *testing.T, prefix string) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		t.Fatalf("resolve home dir: %v", err)
	}
	base := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", base, err)
	}
	dir, err := os.MkdirTemp(base, prefix+"-*")
	if err != nil {
		t.Fatalf("mktemp in %s: %v", base, err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return dir
}

func buildNativeSpecVerifyGrammarMask() GrammarBitmask {
	bm := NewGrammarBitmask(nativeVerifyVocabSize)
	for v := range nativeVerifyVocabSize {
		// Allow tokens where v%4 == 1 (includes 13, 41, 97; disallows 28).
		if v%4 == 1 {
			bm.Allow(v)
		}
	}
	return bm
}

func buildNativeSpecVerifySampleMegakernel(t *testing.T) *FusedVerifySampleMegakernel {
	t.Helper()
	const (
		m     = nativeVerifyDraftNodes
		d     = nativeVerifyHiddenDim
		inter = nativeVerifyIntermDim
		vocab = nativeVerifyVocabSize
		eps   = 1e-5
	)

	rmsBody := hopper.MustLambda(fmt.Sprintf("(mul (mul $0 (rsqrt (add (div $1 %d) %g))) $2)", d, eps))

	oProj := hopper.SkinnyMMA{
		Name:     "final_o_proj",
		N:        d,
		K:        d,
		MaxM:     m,
		Out:      hopper.F32,
		Epilogue: hopper.MustLambda("(add $0 $1)"),
		Extras: []hopper.Operand{
			{Name: "res_in", DType: hopper.F32, Broadcast: hopper.Full},
		},
		Warps:  8,
		Tiles:  1,
		Chunks: 1,
		Steps:  2,
		SplitK: 1,
	}

	resRMSNorm := hopper.RowKernel{
		Name: "final_residual_rmsnorm",
		D:    d,
		Inputs: []hopper.RowTensor{
			{Name: "in0", Buffer: "hid", DType: hopper.F32, Kind: hopper.RowData},
			{Name: "in1", Buffer: "ffn_norm", DType: hopper.F32, Kind: hopper.ColVector},
		},
		Values: []hopper.RowValue{
			{Name: "v_sq", Body: hopper.MustLambda("(mul $0 $0)"), Args: []string{"in0"}},
			{Name: "r_ss", Reduce: "sum", Body: hopper.Identity, Args: []string{"v_sq"}},
			{Name: "v_out", Body: rmsBody, Args: []string{"in0", "r_ss", "in1"}},
		},
		Outputs: []hopper.RowTensor{
			{Name: "ffn_norm_out", DType: hopper.BF16, Kind: hopper.RowData, Value: "v_out"},
		},
	}

	gateUp := hopper.SkinnyMMA{
		Name:     "final_gate_up",
		N:        2 * inter,
		K:        d,
		MaxM:     m,
		Out:      hopper.BF16,
		GLU:      true,
		Epilogue: hopper.MustLambda("(mul (silu $0) $1)"),
		Warps:    8,
		Tiles:    2,
		Chunks:   1,
		Steps:    2,
		SplitK:   1,
	}

	downProj := hopper.SkinnyMMA{
		Name:     "final_down_proj",
		N:        d,
		K:        inter,
		MaxM:     m,
		Out:      hopper.F32,
		Epilogue: hopper.MustLambda("(add $0 $1)"),
		Extras: []hopper.Operand{
			{Name: "res_in", DType: hopper.F32, Broadcast: hopper.Full},
		},
		Warps:  8,
		Tiles:  1,
		Chunks: 1,
		Steps:  2,
		SplitK: 1,
	}

	finalRMSNorm := hopper.RowKernel{
		Name: "final_rmsnorm",
		D:    d,
		Inputs: []hopper.RowTensor{
			{Name: "in0", Buffer: "down_out", DType: hopper.F32, Kind: hopper.RowData},
			{Name: "in1", Buffer: "final_norm", DType: hopper.F32, Kind: hopper.ColVector},
		},
		Values: []hopper.RowValue{
			{Name: "v_sq", Body: hopper.MustLambda("(mul $0 $0)"), Args: []string{"in0"}},
			{Name: "r_ss", Reduce: "sum", Body: hopper.Identity, Args: []string{"v_sq"}},
			{Name: "v_out", Body: rmsBody, Args: []string{"in0", "r_ss", "in1"}},
		},
		Outputs: []hopper.RowTensor{
			{Name: "xn", DType: hopper.BF16, Kind: hopper.RowData, Value: "v_out"},
		},
	}

	finalPost := &hopper.Plan{
		Name: "final_post",
		Inputs: []hopper.Tensor{
			{Name: "attn", DType: hopper.BF16, Shape: []int{m, d}},
			{Name: "w_o", DType: hopper.BF16, Shape: []int{d, d}},
			{Name: "x", DType: hopper.F32, Shape: []int{m, d}},
			{Name: "ffn_norm", DType: hopper.F32, Shape: []int{d}},
			{Name: "w_gu", DType: hopper.BF16, Shape: []int{2 * inter, d}},
			{Name: "w_down", DType: hopper.BF16, Shape: []int{d, inter}},
			{Name: "final_norm", DType: hopper.F32, Shape: []int{d}},
		},
		Temps: []hopper.Tensor{
			{Name: "hid", DType: hopper.F32, Shape: []int{m, d}},
			{Name: "ffn_norm_out", DType: hopper.BF16, Shape: []int{m, d}},
			{Name: "swiglu_out", DType: hopper.BF16, Shape: []int{m, inter}},
			{Name: "down_out", DType: hopper.F32, Shape: []int{m, d}},
		},
		Output: hopper.Tensor{Name: "xn", DType: hopper.BF16, Shape: []int{m, d}},
		Kernels: []hopper.PlanKernel{
			{MMA: &oProj, M: m, Args: []string{"attn", "w_o", "x", "hid"}},
			{Row: &resRMSNorm, Rows: m, Args: []string{"hid", "ffn_norm", "ffn_norm_out"}},
			{MMA: &gateUp, M: m, Args: []string{"ffn_norm_out", "w_gu", "swiglu_out"}},
			{MMA: &downProj, M: m, Args: []string{"swiglu_out", "w_down", "hid", "down_out"}},
			{Row: &finalRMSNorm, Rows: m, Args: []string{"down_out", "final_norm", "xn"}},
		},
	}

	lmHead := hopper.SkinnyMMA{
		Name:   "lm_head",
		N:      vocab,
		K:      d,
		MaxM:   m,
		Out:    hopper.F32,
		Warps:  8,
		Tiles:  1,
		Chunks: 1,
		Steps:  2,
		SplitK: 1,
	}

	head := &hopper.Plan{
		Name: "lm_head",
		Inputs: []hopper.Tensor{
			{Name: "xn", DType: hopper.BF16, Shape: []int{m, d}},
			{Name: "w_head", DType: hopper.BF16, Shape: []int{vocab, d}},
		},
		Output: hopper.Tensor{Name: "logits", DType: hopper.F32, Shape: []int{m, vocab}},
		Kernels: []hopper.PlanKernel{
			{MMA: &lmHead, M: m, Args: []string{"xn", "w_head", "logits"}},
		},
	}

	mega, err := BuildFinalHeadVerifySampleMegakernel(finalPost, head, FusedVerifySampleConfig{
		Name:               "spec_verify_sample_mega",
		BridgeInput:        "xn",
		DraftNodes:         m,
		VocabSize:          vocab,
		DefaultTemperature: nativeVerifyTemperature,
	})
	if err != nil {
		t.Fatalf("BuildFinalHeadVerifySampleMegakernel: %v", err)
	}
	if mega.DraftNodes != m || mega.VocabSize != vocab || mega.BitmaskWords != 4 {
		t.Fatalf("mega dims = (M=%d, V=%d, W=%d), want (4, 128, 4)",
			mega.DraftNodes, mega.VocabSize, mega.BitmaskWords)
	}
	if len(mega.Stages) != 7 {
		t.Fatalf("len(mega.Stages) = %d, want 7", len(mega.Stages))
	}
	return mega
}

func generateNativeSpecVerifyBenchmarkCUDA(t *testing.T, mega *FusedVerifySampleMegakernel) string {
	t.Helper()

	unfusedPlan := *mega.FusedPlan
	unfusedPlan.Name = "spec_verify_unfused"
	unfusedSrc, err := unfusedPlan.Source()
	if err != nil {
		t.Fatalf("unfusedPlan.Source(): %v", err)
	}
	megaCUDA, err := mega.CUDA()
	if err != nil {
		t.Fatalf("mega.CUDA(): %v", err)
	}
	megaLauncher, err := mega.Launcher()
	if err != nil {
		t.Fatalf("mega.Launcher(): %v", err)
	}

	bm := buildNativeSpecVerifyGrammarMask()

	var b strings.Builder
	b.WriteString("#include <cuda_runtime.h>\n")
	b.WriteString("#include <cmath>\n")
	b.WriteString("#include <cstdint>\n")
	b.WriteString("#include <cstdio>\n")
	b.WriteString("#include <cstdlib>\n")
	b.WriteString("#include <cstring>\n")
	b.WriteString("#include <vector>\n\n")
	b.WriteString(unfusedSrc)
	b.WriteString("\n")
	b.WriteString(megaCUDA)
	b.WriteString("\n")
	b.WriteString(megaLauncher)
	b.WriteString("\n")

	b.WriteString(`static inline uint16_t host_f32_to_bf16(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(2); \
    } \
} while (0)

// Host-side Philox-4x32-10 and rejection sampler used by Reference (with host round-trip).
static inline void host_philox4x32(
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t k0, uint32_t k1, uint32_t out[4]) {
    constexpr uint32_t M0 = 0xD2511F53u;
    constexpr uint32_t M1 = 0xCD9E8D57u;
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;
    for (int r = 0; r < 10; ++r) {
        uint64_t p0 = (uint64_t)M0 * (uint64_t)c0;
        uint64_t p1 = (uint64_t)M1 * (uint64_t)c2;
        uint32_t lo0 = (uint32_t)p0, hi0 = (uint32_t)(p0 >> 32);
        uint32_t lo1 = (uint32_t)p1, hi1 = (uint32_t)(p1 >> 32);
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 += W0;
        k1 += W1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

static inline float host_philox_uniform(uint32_t x) {
    return (float)(x >> 8) * 5.9604644775390625e-8f + 2.98023223876953125e-8f;
}

static void host_verify_sample(
    const float* logits,
    const uint32_t* vocab_bitmask,
    const int32_t* tree_parent_idx,
    const int32_t* draft_token_ids,
    const float* draft_probs,
    int32_t* accepted_tokens,
    int32_t* num_accepted,
    int32_t* next_bonus_token,
    float temperature,
    uint64_t philox_seed,
    uint64_t philox_offset) {
    constexpr int M = 4, V = 128;
    const bool greedy = !(temperature > 0.0f);
    const float inv_temp = greedy ? 1.0f : (1.0f / temperature);

    float row_max[M];
    float row_sum[M];
    int32_t row_argmax[M];
    float s_max[256];
    float s_sum[256];
    int32_t s_argmax[256];

    for (int m = 0; m < M; ++m) {
        for (int tid = 0; tid < 256; ++tid) {
            float local_max = -INFINITY;
            float local_sum = 0.0f;
            int32_t local_argmax = -1;
            for (int v = tid; v < V; v += 256) {
                float val = logits[(size_t)m * V + v];
                if (vocab_bitmask != nullptr && (((vocab_bitmask[v >> 5] >> (v & 31)) & 1u) == 0u)) {
                    val = -INFINITY;
                }
                if (val > -INFINITY) {
                    const float scaled = greedy ? val : (val * inv_temp);
                    if (local_argmax < 0) {
                        local_max = scaled;
                        local_sum = 1.0f;
                        local_argmax = v;
                    } else if (scaled > local_max) {
                        local_sum = local_sum * std::exp(local_max - scaled) + 1.0f;
                        local_max = scaled;
                        local_argmax = v;
                    } else {
                        local_sum += std::exp(scaled - local_max);
                    }
                }
            }
            s_max[tid] = local_max;
            s_sum[tid] = local_sum;
            s_argmax[tid] = local_argmax;
        }
        for (int stride = 128; stride > 0; stride >>= 1) {
            for (int tid = 0; tid < stride; ++tid) {
                const float ma = s_max[tid], mb = s_max[tid + stride];
                const float da = s_sum[tid], db = s_sum[tid + stride];
                const int32_t ida = s_argmax[tid], idb = s_argmax[tid + stride];
                if (idb >= 0) {
                    if (ida < 0) {
                        s_max[tid] = mb;
                        s_sum[tid] = db;
                        s_argmax[tid] = idb;
                    } else {
                        const float m_new = std::fmax(ma, mb);
                        s_max[tid] = m_new;
                        s_sum[tid] = da * std::exp(ma - m_new) + db * std::exp(mb - m_new);
                        if (mb > ma || (mb == ma && idb < ida)) {
                            s_argmax[tid] = idb;
                        }
                    }
                }
            }
        }
        row_max[m] = s_max[0];
        row_sum[m] = s_sum[0];
        row_argmax[m] = s_argmax[0];
    }

    for (int m = 0; m < M; ++m) accepted_tokens[m] = -1;
    const uint32_t k0 = (uint32_t)philox_seed;
    const uint32_t k1 = (uint32_t)(philox_seed >> 32);
    const uint32_t c0 = (uint32_t)philox_offset;
    const uint32_t c1 = (uint32_t)(philox_offset >> 32);
    uint32_t draws = 0u;

    auto token_prob = [&](int m, int v) -> float {
        if (v < 0 || v >= V || row_argmax[m] < 0) return 0.0f;
        if (vocab_bitmask != nullptr && (((vocab_bitmask[v >> 5] >> (v & 31)) & 1u) == 0u)) return 0.0f;
        const float raw = logits[(size_t)m * V + v];
        if (!(raw > -INFINITY)) return 0.0f;
        if (greedy) return (v == row_argmax[m]) ? 1.0f : 0.0f;
        const float d = row_sum[m];
        if (!(d > 0.0f)) return 0.0f;
        return std::exp((raw * inv_temp) - row_max[m]) / d;
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
            const int32_t par = tree_parent_idx[m];
            if (par != cur_parent) continue;
            last_node = m;
            const int32_t draft_id = draft_token_ids[m];
            const float q_prob = draft_probs[m];
            float p_prob = token_prob(m, draft_id);
            for (int r = 0; r < num_rej; ++r) {
                if (rej_tokens[r] == draft_id) {
                    p_prob = std::fmax(0.0f, p_prob - rej_probs[r]);
                }
            }
            bool accept = false;
            if (greedy) {
                accept = (draft_id >= 0 && draft_id < V && draft_id == row_argmax[m]);
            } else {
                uint32_t w[4];
                host_philox4x32(c0, c1, draws++, (uint32_t)m, k0, k1, w);
                const float u = host_philox_uniform(w[0]);
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
            rej_probs[num_rej] = std::fmax(0.0f, q_prob);
            ++num_rej;
        }
        if (accepted_node < 0) break;
        cur_parent = accepted_node;
        search_start = accepted_node + 1;
        num_rej = 0;
    }

    const int32_t bonus_node = (num_rej > 0) ? last_node : ((cur_parent >= 0) ? cur_parent : 0);
    int32_t bonus_token = row_argmax[bonus_node];
    if (bonus_token >= 0 && !greedy) {
        uint32_t w[4];
        host_philox4x32(c0, c1, draws++, (uint32_t)bonus_node, k0, k1, w);
        const float u = host_philox_uniform(w[0]);
        float total_resid = 0.0f;
        for (int v = 0; v < V; ++v) {
            float rv = token_prob(bonus_node, v);
            for (int r = 0; r < num_rej; ++r) {
                if (rej_tokens[r] == v) rv = std::fmax(0.0f, rv - rej_probs[r]);
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
                    if (rej_tokens[r] == v) rv = std::fmax(0.0f, rv - rej_probs[r]);
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
        if (last_valid >= 0) bonus_token = last_valid;
    }
    *num_accepted = acc_count;
    *next_bonus_token = bonus_token;
}

int main() {
    constexpr int M = 4;
    constexpr int V = 128;
    constexpr int W = 4;
    constexpr int BLOCKS = 12;
    constexpr int WARMUP = 4;
`)
	fmt.Fprintf(&b, "    constexpr int READY_WORDS = %d;\n", mega.ReadyWords())
	fmt.Fprintf(&b, "    const float temp = %s;\n", floatLiteral(nativeVerifyTemperature))
	fmt.Fprintf(&b, "    const uint64_t seed = %#xULL;\n", nativeVerifyPhiloxSeed)
	fmt.Fprintf(&b, "    const uint64_t offset = %#xULL;\n\n", nativeVerifyPhiloxOffset)

	// Allocate and deterministically initialize BaseMegakernel inputs.
	for idx, in := range mega.BaseMegakernel.Inputs {
		elems := tensorElemCount(in.Shape)
		ctype := in.DType.CType()
		fmt.Fprintf(&b, "    %s* d_%s = nullptr;\n", ctype, in.Name)
		fmt.Fprintf(&b, "    CHECK_CUDA(cudaMalloc(&d_%s, sizeof(%s) * %d));\n", in.Name, ctype, elems)
		fmt.Fprintf(&b, "    {\n        std::vector<%s> h_buf(%d);\n", ctype, elems)
		switch {
		case strings.HasSuffix(in.Name, "_norm") && in.DType == hopper.F32:
			fmt.Fprintf(&b, "        for (int i = 0; i < %d; ++i) h_buf[i] = 1.0f + 0.05f * std::sin((float)(i + %d));\n", elems, idx)
		case in.Name == "head_w_head":
			fmt.Fprintf(&b, "        for (int r = 0; r < %d; ++r) {\n", in.Shape[0])
			fmt.Fprintf(&b, "            for (int c = 0; c < %d; ++c) {\n", in.Shape[1])
			b.WriteString("                float v = 0.06f * std::sin((float)r * 0.31f + (float)c * 0.17f);\n")
			b.WriteString("                if (r == 13 || r == 41 || r == 97) v += 0.45f * std::cos((float)c * 0.11f);\n")
			b.WriteString("                h_buf[r * 64 + c] = host_f32_to_bf16(v);\n")
			b.WriteString("            }\n        }\n")
		case in.Name == "final_x":
			fmt.Fprintf(&b, "        for (int i = 0; i < %d; ++i) h_buf[i] = 0.8f * std::cos((float)i * 0.11f);\n", elems)
		case in.DType == hopper.BF16:
			fmt.Fprintf(&b, "        for (int i = 0; i < %d; ++i) {\n", elems)
			fmt.Fprintf(&b, "            float v = 0.05f * std::sin((float)i * 0.043f + (float)%d * 0.37f);\n", idx+1)
			b.WriteString("            h_buf[i] = host_f32_to_bf16(v);\n        }\n")
		default:
			fmt.Fprintf(&b, "        for (int i = 0; i < %d; ++i) h_buf[i] = 0.05f * std::sin((float)i * 0.043f + (float)%d * 0.37f);\n", elems, idx+1)
		}
		fmt.Fprintf(&b, "        CHECK_CUDA(cudaMemcpy(d_%s, h_buf.data(), sizeof(%s) * %d, cudaMemcpyHostToDevice));\n    }\n",
			in.Name, ctype, elems)
	}
	b.WriteString("\n")

	for _, tmp := range mega.BaseMegakernel.Temps {
		elems := tensorElemCount(tmp.Shape)
		ctype := tmp.DType.CType()
		fmt.Fprintf(&b, "    %s* d_ref_%s = nullptr;\n", ctype, tmp.Name)
		fmt.Fprintf(&b, "    CHECK_CUDA(cudaMalloc(&d_ref_%s, sizeof(%s) * %d));\n", tmp.Name, ctype, elems)
		fmt.Fprintf(&b, "    %s* d_cand_%s = nullptr;\n", ctype, tmp.Name)
		fmt.Fprintf(&b, "    CHECK_CUDA(cudaMalloc(&d_cand_%s, sizeof(%s) * %d));\n", tmp.Name, ctype, elems)
	}
	b.WriteString("\n")

	var maskLits, parentLits, draftIDLits, draftProbLits []string
	for _, w := range bm.Words {
		maskLits = append(maskLits, fmt.Sprintf("%#xu", w))
	}
	for _, p := range nativeVerifyTreeParents {
		parentLits = append(parentLits, strconv.Itoa(int(p)))
	}
	for _, id := range nativeVerifyDraftIDs {
		draftIDLits = append(draftIDLits, strconv.Itoa(int(id)))
	}
	for _, pr := range nativeVerifyDraftProbs {
		draftProbLits = append(draftProbLits, floatLiteral(pr))
	}
	fmt.Fprintf(&b, "    const uint32_t h_mask[W] = {%s};\n", strings.Join(maskLits, ", "))
	fmt.Fprintf(&b, "    const int32_t h_parents[M] = {%s};\n", strings.Join(parentLits, ", "))
	fmt.Fprintf(&b, "    const int32_t h_draft_ids[M] = {%s};\n", strings.Join(draftIDLits, ", "))
	fmt.Fprintf(&b, "    const float h_draft_probs[M] = {%s};\n\n", strings.Join(draftProbLits, ", "))

	b.WriteString(`    uint32_t* d_mask = nullptr;
    int32_t* d_parents = nullptr;
    int32_t* d_draft_ids = nullptr;
    float* d_draft_probs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_mask, sizeof(h_mask)));
    CHECK_CUDA(cudaMalloc(&d_parents, sizeof(h_parents)));
    CHECK_CUDA(cudaMalloc(&d_draft_ids, sizeof(h_draft_ids)));
    CHECK_CUDA(cudaMalloc(&d_draft_probs, sizeof(h_draft_probs)));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask, sizeof(h_mask), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_parents, h_parents, sizeof(h_parents), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_draft_ids, h_draft_ids, sizeof(h_draft_ids), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_draft_probs, h_draft_probs, sizeof(h_draft_probs), cudaMemcpyHostToDevice));

    float* d_ref_logits = nullptr;
    float* d_cand_logits = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ref_logits, sizeof(float) * M * V));
    CHECK_CUDA(cudaMalloc(&d_cand_logits, sizeof(float) * M * V));

    int32_t* d_ref_accepted = nullptr;
    int32_t* d_ref_num_acc = nullptr;
    int32_t* d_ref_bonus = nullptr;
    int32_t* d_cand_accepted = nullptr;
    int32_t* d_cand_num_acc = nullptr;
    int32_t* d_cand_bonus = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ref_accepted, sizeof(int32_t) * M));
    CHECK_CUDA(cudaMalloc(&d_ref_num_acc, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_ref_bonus, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_cand_accepted, sizeof(int32_t) * M));
    CHECK_CUDA(cudaMalloc(&d_cand_num_acc, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_cand_bonus, sizeof(int32_t)));

    // Pinned host buffers for Reference's D2H logits copy + host rejection sampler + H2D token copy.
    float* h_pinned_logits = nullptr;
    int32_t* h_pinned_accepted = nullptr;
    int32_t* h_pinned_num_acc = nullptr;
    int32_t* h_pinned_bonus = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_pinned_logits, sizeof(float) * M * V));
    CHECK_CUDA(cudaMallocHost(&h_pinned_accepted, sizeof(int32_t) * M));
    CHECK_CUDA(cudaMallocHost(&h_pinned_num_acc, sizeof(int32_t)));
    CHECK_CUDA(cudaMallocHost(&h_pinned_bonus, sizeof(int32_t)));

    unsigned long long* d_ready = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ready, sizeof(unsigned long long) * READY_WORDS));
    CHECK_CUDA(cudaMemset(d_ready, 0, sizeof(unsigned long long) * READY_WORDS));
`)

	var refCallArgs, candCallArgs []string
	for _, in := range mega.BaseMegakernel.Inputs {
		refCallArgs = append(refCallArgs, "d_"+in.Name)
		candCallArgs = append(candCallArgs, "d_"+in.Name)
	}
	for _, tmp := range mega.BaseMegakernel.Temps {
		refCallArgs = append(refCallArgs, "d_ref_"+tmp.Name)
		candCallArgs = append(candCallArgs, "d_cand_"+tmp.Name)
	}
	refCallArgs = append(refCallArgs, "d_ref_logits", "0")
	candCallArgs = append(candCallArgs,
		"d_cand_logits",
		"d_mask",
		"d_parents",
		"d_draft_ids",
		"d_draft_probs",
		"d_cand_accepted",
		"d_cand_num_acc",
		"d_cand_bonus",
		"temp",
		"seed",
		"offset",
		"d_ready",
		"0u",
		"0",
	)

	b.WriteString("\n    auto run_ref = [&]() {\n")
	fmt.Fprintf(&b, "        spec_verify_unfused_run(%s);\n", strings.Join(refCallArgs, ", "))
	b.WriteString(`        CHECK_CUDA(cudaMemcpyAsync(h_pinned_logits, d_ref_logits, sizeof(float) * M * V, cudaMemcpyDeviceToHost, 0));
        CHECK_CUDA(cudaStreamSynchronize(0));
        host_verify_sample(
            h_pinned_logits,
            h_mask,
            h_parents,
            h_draft_ids,
            h_draft_probs,
            h_pinned_accepted,
            h_pinned_num_acc,
            h_pinned_bonus,
            temp,
            seed,
            offset);
        CHECK_CUDA(cudaMemcpyAsync(d_ref_accepted, h_pinned_accepted, sizeof(int32_t) * M, cudaMemcpyHostToDevice, 0));
        CHECK_CUDA(cudaMemcpyAsync(d_ref_num_acc, h_pinned_num_acc, sizeof(int32_t), cudaMemcpyHostToDevice, 0));
        CHECK_CUDA(cudaMemcpyAsync(d_ref_bonus, h_pinned_bonus, sizeof(int32_t), cudaMemcpyHostToDevice, 0));
    };

    auto run_cand = [&]() {
`)
	fmt.Fprintf(&b, "        spec_verify_sample_mega_run(%s);\n", strings.Join(candCallArgs, ", "))
	b.WriteString(`    };

    run_ref();
    run_cand();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_ref_logits(M * V);
    std::vector<float> h_cand_logits(M * V);
    CHECK_CUDA(cudaMemcpy(h_ref_logits.data(), d_ref_logits, sizeof(float) * M * V, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cand_logits.data(), d_cand_logits, sizeof(float) * M * V, cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    for (int i = 0; i < M * V; ++i) {
        float diff = std::fabs(h_ref_logits[i] - h_cand_logits[i]);
        if (diff > max_abs) max_abs = diff;
    }
    std::printf("LOGIT_MAX_ABS=%.8f\n", max_abs);
    for (int m = 0; m < M; ++m) {
        std::printf("CAND_LOGITS_ROW_%d=", m);
        for (int v = 0; v < V; ++v) {
            std::printf("%s%.8g", v ? "," : "", h_cand_logits[m * V + v]);
        }
        std::printf("\n");
    }

    int32_t h_ref_num = 0, h_cand_num = 0;
    int32_t h_ref_bon = -1, h_cand_bon = -1;
    int32_t h_ref_acc[M] = {-1, -1, -1, -1};
    int32_t h_cand_acc[M] = {-1, -1, -1, -1};
    CHECK_CUDA(cudaMemcpy(&h_ref_num, d_ref_num_acc, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_cand_num, d_cand_num_acc, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_ref_bon, d_ref_bonus, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_cand_bon, d_cand_bonus, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ref_acc, d_ref_accepted, sizeof(int32_t) * M, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cand_acc, d_cand_accepted, sizeof(int32_t) * M, cudaMemcpyDeviceToHost));

    std::printf("REF_NUM_ACCEPTED=%d\n", h_ref_num);
    std::printf("REF_BONUS_TOKEN=%d\n", h_ref_bon);
    std::printf("REF_ACCEPTED_TOKENS=");
    for (int i = 0; i < h_ref_num && i < M; ++i) std::printf("%s%d", i ? "," : "", h_ref_acc[i]);
    std::printf("\n");

    std::printf("CAND_NUM_ACCEPTED=%d\n", h_cand_num);
    std::printf("CAND_BONUS_TOKEN=%d\n", h_cand_bon);
    std::printf("CAND_ACCEPTED_TOKENS=");
    for (int i = 0; i < h_cand_num && i < M; ++i) std::printf("%s%d", i ? "," : "", h_cand_acc[i]);
    std::printf("\n");

    // Device mutant negative control: replace grammar bitmask so only token 2 (v % 4 == 2) is allowed.
    {
        const uint32_t h_mut_mask[W] = {0x4u, 0x0u, 0x0u, 0x0u};
        CHECK_CUDA(cudaMemcpy(d_mask, h_mut_mask, sizeof(h_mut_mask), cudaMemcpyHostToDevice));
        run_cand();
        CHECK_CUDA(cudaDeviceSynchronize());
        int32_t h_mut_num = 0, h_mut_bon = -1;
        CHECK_CUDA(cudaMemcpy(&h_mut_num, d_cand_num_acc, sizeof(int32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_mut_bon, d_cand_bonus, sizeof(int32_t), cudaMemcpyDeviceToHost));
        bool rejected = (h_mut_num != h_ref_num) || (h_mut_bon != h_ref_bon);
        std::printf("MUTANT_REJECTED=%d\n", rejected ? 1 : 0);
        // Restore clean grammar bitmask before timing.
        CHECK_CUDA(cudaMemcpy(d_mask, h_mask, sizeof(h_mask), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    for (int w = 0; w < WARMUP; ++w) {
        run_ref();
        run_cand();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    auto time_one = [&](auto&& fn) -> float {
        CHECK_CUDA(cudaEventRecord(ev0, 0));
        fn();
        CHECK_CUDA(cudaEventRecord(ev1, 0));
        CHECK_CUDA(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
        return ms * 1000.0f;
    };

    auto measure_abba = [&](const char* tag, int b, auto&& fnA, auto&& fnB) {
        float a1 = time_one(fnA);
        float b1 = time_one(fnB);
        float b2 = time_one(fnB);
        float a2 = time_one(fnA);
        std::printf("%s %d a1=%.3f b1=%.3f b2=%.3f a2=%.3f\n", tag, b, a1, b1, b2, a2);
    };

    for (int b = 0; b < BLOCKS; ++b) {
        measure_abba("AA_BLOCK", b, run_ref, run_ref);
    }
    for (int b = 0; b < BLOCKS; ++b) {
        measure_abba("SCREEN_BLOCK", b, run_ref, run_cand);
    }
    for (int b = 0; b < BLOCKS; ++b) {
        measure_abba("HOLDOUT_BLOCK", b, run_ref, run_cand);
    }

    std::printf("SPEC_VERIFY_NATIVE_OK\n");
    return 0;
}
`)
	return b.String()
}

type nativeSpecVerifyRunMetrics struct {
	logitMaxAbs    float64
	candLogits     [][]float32
	refNumAcc      int32
	refBonus       int32
	refAccepted    []int32
	candNumAcc     int32
	candBonus      int32
	candAccepted   []int32
	mutantRejected bool
	aaBlocks       []pairedtiming.Block
	screenBlocks   []pairedtiming.Block
	holdoutBlocks  []pairedtiming.Block
}

func parseNativeSpecVerifyInt32CSV(s string) ([]int32, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]int32, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, err
		}
		out = append(out, int32(v))
	}
	return out, nil
}

func parseNativeSpecVerifyFloat32CSV(s string) ([]float32, error) {
	parts := strings.Split(strings.TrimSpace(s), ",")
	out := make([]float32, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, err
		}
		out = append(out, float32(v))
	}
	return out, nil
}

func parseNativeSpecVerifyTimingBlock(line string) (pairedtiming.Block, error) {
	var b pairedtiming.Block
	for _, f := range strings.Fields(line) {
		k, v, ok := strings.Cut(f, "=")
		if !ok {
			continue
		}
		val, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return b, fmt.Errorf("parse %s in %q: %w", f, line, err)
		}
		switch k {
		case "a1":
			b.A1 = val
		case "b1":
			b.B1 = val
		case "b2":
			b.B2 = val
		case "a2":
			b.A2 = val
		}
	}
	if b.A1 <= 0 || b.B1 <= 0 || b.B2 <= 0 || b.A2 <= 0 {
		return b, fmt.Errorf("nonpositive timing block in %q: %+v", line, b)
	}
	return b, nil
}

func parseNativeSpecVerifyOutput(out string) (nativeSpecVerifyRunMetrics, error) {
	var m nativeSpecVerifyRunMetrics
	m.candLogits = make([][]float32, nativeVerifyDraftNodes)
	var sawLogit, sawRef, sawCand, sawMut, sawOK bool

	for _, rawLine := range strings.Split(out, "\n") {
		line := strings.TrimSpace(rawLine)
		switch {
		case strings.HasPrefix(line, "LOGIT_MAX_ABS="):
			v, err := strconv.ParseFloat(strings.TrimPrefix(line, "LOGIT_MAX_ABS="), 64)
			if err != nil {
				return m, fmt.Errorf("parse LOGIT_MAX_ABS: %w", err)
			}
			m.logitMaxAbs = v
			sawLogit = true
		case strings.HasPrefix(line, "CAND_LOGITS_ROW_"):
			rest := strings.TrimPrefix(line, "CAND_LOGITS_ROW_")
			idxStr, csv, ok := strings.Cut(rest, "=")
			if !ok {
				return m, fmt.Errorf("malformed %q", line)
			}
			rowIdx, err := strconv.Atoi(idxStr)
			if err != nil || rowIdx < 0 || rowIdx >= nativeVerifyDraftNodes {
				return m, fmt.Errorf("bad row index in %q", line)
			}
			vals, err := parseNativeSpecVerifyFloat32CSV(csv)
			if err != nil {
				return m, fmt.Errorf("parse %q: %w", line, err)
			}
			m.candLogits[rowIdx] = vals
		case strings.HasPrefix(line, "REF_NUM_ACCEPTED="):
			v, err := strconv.Atoi(strings.TrimPrefix(line, "REF_NUM_ACCEPTED="))
			if err != nil {
				return m, err
			}
			m.refNumAcc = int32(v)
		case strings.HasPrefix(line, "REF_BONUS_TOKEN="):
			v, err := strconv.Atoi(strings.TrimPrefix(line, "REF_BONUS_TOKEN="))
			if err != nil {
				return m, err
			}
			m.refBonus = int32(v)
		case strings.HasPrefix(line, "REF_ACCEPTED_TOKENS="):
			toks, err := parseNativeSpecVerifyInt32CSV(strings.TrimPrefix(line, "REF_ACCEPTED_TOKENS="))
			if err != nil {
				return m, err
			}
			m.refAccepted = toks
			sawRef = true
		case strings.HasPrefix(line, "CAND_NUM_ACCEPTED="):
			v, err := strconv.Atoi(strings.TrimPrefix(line, "CAND_NUM_ACCEPTED="))
			if err != nil {
				return m, err
			}
			m.candNumAcc = int32(v)
		case strings.HasPrefix(line, "CAND_BONUS_TOKEN="):
			v, err := strconv.Atoi(strings.TrimPrefix(line, "CAND_BONUS_TOKEN="))
			if err != nil {
				return m, err
			}
			m.candBonus = int32(v)
		case strings.HasPrefix(line, "CAND_ACCEPTED_TOKENS="):
			toks, err := parseNativeSpecVerifyInt32CSV(strings.TrimPrefix(line, "CAND_ACCEPTED_TOKENS="))
			if err != nil {
				return m, err
			}
			m.candAccepted = toks
			sawCand = true
		case strings.HasPrefix(line, "MUTANT_REJECTED="):
			v, err := strconv.Atoi(strings.TrimPrefix(line, "MUTANT_REJECTED="))
			if err != nil {
				return m, err
			}
			m.mutantRejected = v == 1
			sawMut = true
		case strings.HasPrefix(line, "AA_BLOCK "):
			b, err := parseNativeSpecVerifyTimingBlock(line)
			if err != nil {
				return m, err
			}
			m.aaBlocks = append(m.aaBlocks, b)
		case strings.HasPrefix(line, "SCREEN_BLOCK "):
			b, err := parseNativeSpecVerifyTimingBlock(line)
			if err != nil {
				return m, err
			}
			m.screenBlocks = append(m.screenBlocks, b)
		case strings.HasPrefix(line, "HOLDOUT_BLOCK "):
			b, err := parseNativeSpecVerifyTimingBlock(line)
			if err != nil {
				return m, err
			}
			m.holdoutBlocks = append(m.holdoutBlocks, b)
		case line == "SPEC_VERIFY_NATIVE_OK":
			sawOK = true
		}
	}
	if !sawLogit || !sawRef || !sawCand || !sawMut || !sawOK {
		return m, fmt.Errorf("incomplete benchmark output (logit=%v ref=%v cand=%v mut=%v ok=%v)",
			sawLogit, sawRef, sawCand, sawMut, sawOK)
	}
	for r, row := range m.candLogits {
		if len(row) != nativeVerifyVocabSize {
			return m, fmt.Errorf("candLogits[%d] len = %d, want %d", r, len(row), nativeVerifyVocabSize)
		}
	}
	return m, nil
}

func queryNativeSpecVerifyGPUAndDriver() (gpuName, driverVer string) {
	gpuName = "NVIDIA GB10"
	driverVer = "unknown"
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader").Output()
	if err != nil {
		return gpuName, driverVer
	}
	line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	parts := strings.Split(line, ",")
	if len(parts) >= 2 {
		if n := strings.TrimSpace(parts[0]); n != "" {
			gpuName = n
		}
		if d := strings.TrimSpace(parts[1]); d != "" {
			driverVer = d
		}
	}
	return gpuName, driverVer
}

func TestSpecVerifySampleNativeClosedGateControl(t *testing.T) {
	t.Setenv("CUDATEST_NATIVE", "")
	err := checkNativeSpecVerifyGate(os.Getenv)
	if err == nil {
		t.Fatal("expected non-nil gate error when CUDATEST_NATIVE is unset")
	}
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("gate error %v does not unwrap to cudaerrors.ErrHardwarePending", err)
	}
	if errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("gate error %v must not match cudaerrors.ErrUnsupported", err)
	}
	const wantPrefix = "native qualification pending: set CUDATEST_NATIVE=1 under gpu-lock"
	if !strings.HasPrefix(err.Error(), wantPrefix) {
		t.Fatalf("gate skip reason = %q, want prefix %q", err.Error(), wantPrefix)
	}

	// Presence-only GPU_LOCK_HELD=1 must be rejected by cudalock.Inherited.
	badEnv := func(k string) string {
		switch k {
		case "CUDATEST_NATIVE", "GPU_LOCK_HELD":
			return "1"
		default:
			return ""
		}
	}
	if lockErr := checkNativeSpecVerifyGate(badEnv); lockErr == nil || errors.Is(lockErr, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkNativeSpecVerifyGate with GPU_LOCK_HELD=1 = %v, want non-pending lock error", lockErr)
	}

	// Genuine inherited lock must pass.
	lockDir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("specverify-gate-test"))
	if err != nil {
		t.Fatalf("cudalock.Acquire: %v", err)
	}
	t.Cleanup(func() { _ = h.Release() })
	goodEnv := func(k string) string {
		switch k {
		case "CUDATEST_NATIVE":
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return lockDir
		default:
			return ""
		}
	}
	if err := checkNativeSpecVerifyGate(goodEnv); err != nil {
		t.Fatalf("checkNativeSpecVerifyGate with inherited lock: %v", err)
	}

	mega := buildNativeSpecVerifySampleMegakernel(t)
	harnessSrc := generateNativeSpecVerifyBenchmarkCUDA(t, mega)
	if strings.Contains(harnessSrc, "%!") {
		t.Fatal("generated speculative verify benchmark CUDA contains %! format errors")
	}

	wantSubstrings := []string{
		"EG_DEV void eg_spec_grammar_verify_sample_dev(",
		"eg_philox4x32(",
		"eg_philox_uniform(",
		"eg_ready_resolve_epoch",
		"spec_verify_unfused_run(",
		"spec_verify_sample_mega_run(",
		"cudaMallocHost(",
		"cudaMemcpyAsync(",
		"AA_BLOCK",
		"SCREEN_BLOCK",
		"HOLDOUT_BLOCK",
		"SPEC_VERIFY_NATIVE_OK",
	}
	for _, sub := range wantSubstrings {
		if !strings.Contains(harnessSrc, sub) {
			t.Errorf("generated speculative verify benchmark CUDA missing %q", sub)
		}
	}
}

func TestSpecVerifySampleNativeSM121A(t *testing.T) {
	nvcc, cudaVer, arch := requireNativeSpecVerifyGate(t)

	mega := buildNativeSpecVerifySampleMegakernel(t)
	harnessSrc := generateNativeSpecVerifyBenchmarkCUDA(t, mega)
	if strings.Contains(harnessSrc, "%!") {
		t.Fatal("generated speculative verify benchmark CUDA contains %! format errors")
	}

	workDir := nativeSpecVerifyTempDir(t, "specverify-sm121a")
	cuPath := filepath.Join(workDir, "specverify_bench.cu")
	binPath := filepath.Join(workDir, "specverify_bench")
	if err := os.WriteFile(cuPath, []byte(harnessSrc), 0o644); err != nil {
		t.Fatalf("write %s: %v", cuPath, err)
	}

	compileCmd := exec.Command(nvcc, "-std=c++17", "-O3", "-arch="+arch, "-o", binPath, cuPath)
	if out, err := compileCmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc compile failed: %v\n%s", err, out)
	}

	runCmd := exec.Command(binPath)
	rawOut, err := runCmd.CombinedOutput()
	if err != nil {
		t.Fatalf("benchmark execution failed: %v\n%s", err, rawOut)
	}

	metrics, err := parseNativeSpecVerifyOutput(string(rawOut))
	if err != nil {
		t.Fatalf("parse benchmark output: %v\n%s", err, rawOut)
	}

	if math.IsNaN(metrics.logitMaxAbs) || metrics.logitMaxAbs >= 1e-2 {
		t.Fatalf("logit_max_abs = %g, want < 1e-2", metrics.logitMaxAbs)
	}

	// Verify exact accepted token trajectory match against both Reference and EvalFinalHeadVerifySample.
	bm := buildNativeSpecVerifyGrammarMask()
	wantEval, err := EvalFinalHeadVerifySample(
		metrics.candLogits,
		bm.Words,
		nativeVerifyDraftIDs,
		nativeVerifyDraftProbs,
		nativeVerifyTreeParents,
		nativeVerifyTemperature,
		nativeVerifyPhiloxSeed,
		nativeVerifyPhiloxOffset,
	)
	if err != nil {
		t.Fatalf("EvalFinalHeadVerifySample: %v", err)
	}

	if metrics.candNumAcc != wantEval.NumAccepted ||
		metrics.candBonus != wantEval.NextBonusToken ||
		!slices.Equal(metrics.candAccepted, wantEval.AcceptedTokens) {
		t.Fatalf("GPU candidate vs EvalFinalHeadVerifySample mismatch: got (num=%d, acc=%v, bonus=%d), want (num=%d, acc=%v, bonus=%d)",
			metrics.candNumAcc, metrics.candAccepted, metrics.candBonus,
			wantEval.NumAccepted, wantEval.AcceptedTokens, wantEval.NextBonusToken)
	}
	if metrics.candNumAcc != metrics.refNumAcc ||
		metrics.candBonus != metrics.refBonus ||
		!slices.Equal(metrics.candAccepted, metrics.refAccepted) {
		t.Fatalf("GPU candidate vs Reference mismatch: cand=(num=%d, acc=%v, bonus=%d), ref=(num=%d, acc=%v, bonus=%d)",
			metrics.candNumAcc, metrics.candAccepted, metrics.candBonus,
			metrics.refNumAcc, metrics.refAccepted, metrics.refBonus)
	}
	for _, tok := range metrics.candAccepted {
		if !bm.Allowed(int(tok)) {
			t.Fatalf("accepted token %d disallowed by grammar bitmask", tok)
		}
	}
	if !bm.Allowed(int(metrics.candBonus)) {
		t.Fatalf("bonus token %d disallowed by grammar bitmask", metrics.candBonus)
	}
	if !metrics.mutantRejected {
		t.Fatal("device mutant negative control was not rejected")
	}

	if len(metrics.aaBlocks) != nativeVerifyTimingBlocks {
		t.Fatalf("AA blocks = %d, want %d", len(metrics.aaBlocks), nativeVerifyTimingBlocks)
	}
	if len(metrics.screenBlocks) != nativeVerifyTimingBlocks {
		t.Fatalf("screen blocks = %d, want %d", len(metrics.screenBlocks), nativeVerifyTimingBlocks)
	}
	if len(metrics.holdoutBlocks) != nativeVerifyTimingBlocks {
		t.Fatalf("holdout blocks = %d, want %d", len(metrics.holdoutBlocks), nativeVerifyTimingBlocks)
	}

	aaSummary, err := pairedtiming.Analyze(metrics.aaBlocks, 0)
	if err != nil {
		t.Fatalf("analyze AA blocks: %v", err)
	}
	screenSummary, err := pairedtiming.Analyze(metrics.screenBlocks, 0)
	if err != nil {
		t.Fatalf("analyze screen blocks: %v", err)
	}
	holdoutSummary, err := pairedtiming.Analyze(metrics.holdoutBlocks, 0)
	if err != nil {
		t.Fatalf("analyze holdout blocks: %v", err)
	}

	gpuName, driverVer := queryNativeSpecVerifyGPUAndDriver()
	t.Logf("AA_NOISE_FLOOR: n=%d ref1_median_us=%.2f ref2_median_us=%.2f ratio=%.4fx",
		len(metrics.aaBlocks), aaSummary.Baseline, aaSummary.Candidate, aaSummary.Ratio)
	t.Logf("SCREEN_TIMING: n=%d ref_median_us=%.2f cand_median_us=%.2f speedup=%.3fx sign_p=%.4f",
		len(metrics.screenBlocks), screenSummary.Baseline, screenSummary.Candidate,
		screenSummary.Ratio, screenSummary.SignP)
	t.Logf("HOLDOUT_TIMING: n=%d ref_median_us=%.2f cand_median_us=%.2f speedup=%.3fx sign_p=%.4f",
		len(metrics.holdoutBlocks), holdoutSummary.Baseline, holdoutSummary.Candidate,
		holdoutSummary.Ratio, holdoutSummary.SignP)
	t.Logf("CLAIM: native-qualified on %s (%s), driver %s, CUDA %s, workload=fused_final_head_spec_verify_sample_megakernel(M=4,D=64,I=128,V=128,stages=7), metric=holdout_median_us=%.2f(ref_unfused_host_roundtrip=%.2f,speedup=%.3fx,aa_ratio=%.4fx,sign_p=%.4f,logit_max_abs=%.6f,num_accepted=%d,bonus_token=%d,mutant_rejected=true)",
		gpuName, arch, driverVer, cudaVer,
		holdoutSummary.Candidate, holdoutSummary.Baseline,
		holdoutSummary.Ratio, aaSummary.Ratio, holdoutSummary.SignP,
		metrics.logitMaxAbs, metrics.candNumAcc, metrics.candBonus)
}
