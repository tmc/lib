package hopper

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
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

const nativeFullLayerTimingBlocks = 12

type nativeFullLayerGateClosedError struct{}

func (nativeFullLayerGateClosedError) Error() string {
	return "native qualification pending: set CUDATEST_NATIVE=1 under gpu-lock"
}

func (nativeFullLayerGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func checkNativeFullLayerGate(getenv func(string) string) error {
	if getenv("CUDATEST_NATIVE") != "1" {
		return nativeFullLayerGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("hopper: native full-layer gate open without inherited gpu lock: %w", err)
	}
	return nil
}

func requireNativeFullLayerGate(t *testing.T) (nvcc, cudaVer, arch string) {
	t.Helper()
	if os.Getenv("CUDATEST_NATIVE") != "1" {
		t.Skip(nativeFullLayerGateClosedError{}.Error())
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native qualification requires gpu-lock: %v", err)
	}
	nvccPath, ver := cudatest.NVCC(t)
	arch = strings.TrimSpace(os.Getenv("CUDATEST_SM_ARCH"))
	if arch == "" {
		arch = string(TargetSM121A)
	}
	cudatest.SkipUnlessTarget(t, nvccPath, arch)
	return nvccPath, ver.String(), arch
}

func nativeFullLayerTempDir(t *testing.T, prefix string) string {
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

func buildNativeFullLayerMegakernel(t *testing.T) (*Plan, *Megakernel) {
	t.Helper()
	specs := []FullDecoderLayerSpec{
		{
			LayerIndex: 0,
			BatchM:     1,
			HiddenDim:  64,
			IntermDim:  128,
			NumQHeads:  4,
			NumKVHeads: 2,
			HeadDim:    16,
			PageSize:   16,
			MaxPages:   4,
			RoPETheta:  10000.0,
			RMSEps:     1e-5,
			Arch:       TargetSM121A,
		},
		{
			LayerIndex: 1,
			BatchM:     1,
			HiddenDim:  64,
			IntermDim:  128,
			NumQHeads:  4,
			NumKVHeads: 2,
			HeadDim:    16,
			PageSize:   16,
			MaxPages:   4,
			RoPETheta:  10000.0,
			RMSEps:     1e-5,
			Arch:       TargetSM121A,
		},
	}
	fusedPlan, mega, err := BuildFullLayerMegakernel("full_layer_2l_mega", specs)
	if err != nil {
		t.Fatalf("BuildFullLayerMegakernel(2L): %v", err)
	}
	if len(fusedPlan.Kernels) != 18 {
		t.Fatalf("len(fusedPlan.Kernels) = %d, want 18", len(fusedPlan.Kernels))
	}
	assertFullLayerStageKinds(t, mega.Stages, 2)
	return fusedPlan, mega
}

func fullLayerTensorElemCount(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func generateNativeFullLayerBenchmarkCUDA(t *testing.T, fusedPlan *Plan, mega *Megakernel) string {
	t.Helper()

	unfusedPlan := clonePlan(fusedPlan)
	unfusedPlan.Name = "full_layer_2l_unfused"
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

	outElems := fullLayerTensorElemCount(fusedPlan.Output.Shape)

	var b strings.Builder
	b.WriteString("#include <cuda_runtime.h>\n")
	b.WriteString("#include <cmath>\n")
	b.WriteString("#include <cstdint>\n")
	b.WriteString("#include <cstdio>\n")
	b.WriteString("#include <cstdlib>\n")
	b.WriteString("#include <vector>\n\n")
	b.WriteString(unfusedSrc)
	b.WriteString("\n")
	b.WriteString(megaCUDA)
	b.WriteString("\n")
	b.WriteString(megaLauncher)
	b.WriteString("\n")

	b.WriteString(`#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(2); \
    } \
} while (0)

int main() {
    constexpr int STEPS = 8;
    constexpr int BLOCKS = 12;
    constexpr int WARMUP = 4;
`)
	fmt.Fprintf(&b, "    constexpr int OUT_ELEMS = %d;\n", outElems)
	fmt.Fprintf(&b, "    constexpr int READY_WORDS = %d;\n\n", mega.ReadyWords())

	b.WriteString("    // Per-step decode inputs and metadata.\n")
	b.WriteString("    float* d_x_step[STEPS];\n")
	b.WriteString("    int32_t* d_pos_step[STEPS];\n")
	b.WriteString("    int32_t* d_seqlen_step[STEPS];\n")
	b.WriteString("    float* d_out_ref[STEPS];\n")
	b.WriteString("    float* d_out_cand[STEPS];\n")
	b.WriteString("    for (int s = 0; s < STEPS; ++s) {\n")
	b.WriteString("        std::vector<float> hx(OUT_ELEMS);\n")
	b.WriteString("        for (int i = 0; i < OUT_ELEMS; ++i) {\n")
	b.WriteString("            hx[i] = 0.35f * std::cos((float)(s + 1) * 0.71f + (float)i * 0.13f);\n")
	b.WriteString("        }\n")
	b.WriteString("        hx[(s * 29 + 7) % OUT_ELEMS] += 3.5f;\n")
	b.WriteString("        CHECK_CUDA(cudaMalloc(&d_x_step[s], sizeof(float) * OUT_ELEMS));\n")
	b.WriteString("        CHECK_CUDA(cudaMemcpy(d_x_step[s], hx.data(), sizeof(float) * OUT_ELEMS, cudaMemcpyHostToDevice));\n")
	b.WriteString("        int32_t h_pos = s;\n")
	b.WriteString("        int32_t h_seqlen = s + 1;\n")
	b.WriteString("        CHECK_CUDA(cudaMalloc(&d_pos_step[s], sizeof(int32_t)));\n")
	b.WriteString("        CHECK_CUDA(cudaMemcpy(d_pos_step[s], &h_pos, sizeof(int32_t), cudaMemcpyHostToDevice));\n")
	b.WriteString("        CHECK_CUDA(cudaMalloc(&d_seqlen_step[s], sizeof(int32_t)));\n")
	b.WriteString("        CHECK_CUDA(cudaMemcpy(d_seqlen_step[s], &h_seqlen, sizeof(int32_t), cudaMemcpyHostToDevice));\n")
	b.WriteString("        CHECK_CUDA(cudaMalloc(&d_out_ref[s], sizeof(float) * OUT_ELEMS));\n")
	b.WriteString("        CHECK_CUDA(cudaMemset(d_out_ref[s], 0, sizeof(float) * OUT_ELEMS));\n")
	b.WriteString("        CHECK_CUDA(cudaMalloc(&d_out_cand[s], sizeof(float) * OUT_ELEMS));\n")
	b.WriteString("        CHECK_CUDA(cudaMemset(d_out_cand[s], 0, sizeof(float) * OUT_ELEMS));\n")
	b.WriteString("    }\n\n")

	// Emit device allocations and deterministic initialization for all static inputs.
	for idx, in := range fusedPlan.Inputs {
		if in.Name == "l0_x" || strings.HasSuffix(in.Name, "_positions") || strings.HasSuffix(in.Name, "_seqlens") {
			continue
		}
		elems := fullLayerTensorElemCount(in.Shape)
		ctype := in.DType.CType()
		fmt.Fprintf(&b, "    %s* d_%s = nullptr;\n", ctype, in.Name)
		fmt.Fprintf(&b, "    CHECK_CUDA(cudaMalloc(&d_%s, sizeof(%s) * %d));\n", in.Name, ctype, elems)
		fmt.Fprintf(&b, "    {\n        std::vector<%s> h_buf(%d);\n", ctype, elems)
		switch {
		case strings.HasSuffix(in.Name, "_seq_ids"):
			b.WriteString("        for (int i = 0; i < (int)h_buf.size(); ++i) h_buf[i] = i;\n")
		case strings.HasSuffix(in.Name, "_page_table"):
			b.WriteString("        for (int i = 0; i < (int)h_buf.size(); ++i) h_buf[i] = i;\n")
		case strings.HasSuffix(in.Name, "_scale"):
			b.WriteString("        for (int i = 0; i < (int)h_buf.size(); ++i) h_buf[i] = 1.0f;\n")
		case strings.HasSuffix(in.Name, "_cos_sin"):
			hd := in.Shape[len(in.Shape)-1]
			half := hd / 2
			fmt.Fprintf(&b, "        for (int r = 0; r < %d; ++r) {\n", in.Shape[0])
			fmt.Fprintf(&b, "            for (int c = 0; c < %d; ++c) {\n", half)
			fmt.Fprintf(&b, "                float freq = std::pow(10000.0f, -2.0f * (float)c / (float)%d);\n", hd)
			fmt.Fprintf(&b, "                h_buf[r * %d + c] = std::cos((float)r * freq);\n", hd)
			fmt.Fprintf(&b, "                h_buf[r * %d + %d + c] = std::sin((float)r * freq);\n", hd, half)
			b.WriteString("            }\n        }\n")
		case strings.HasSuffix(in.Name, "_norm_w") && in.DType == F32:
			fmt.Fprintf(&b, "        for (int i = 0; i < %d; ++i) h_buf[i] = 1.0f + 0.05f * std::sin((float)(i + %d));\n", elems, idx)
		case strings.HasSuffix(in.Name, "_norm_w") && in.DType == BF16:
			fmt.Fprintf(&b, "        for (int i = 0; i < %d; ++i) h_buf[i] = eg_f32_to_bf16(1.0f + 0.05f * std::cos((float)(i + %d)));\n", elems, idx)
		case in.DType == BF16:
			fmt.Fprintf(&b, "        for (int i = 0; i < %d; ++i) {\n", elems)
			fmt.Fprintf(&b, "            float v = 0.04f * std::sin((float)i * 0.037f + (float)%d * 0.51f);\n", idx+1)
			b.WriteString("            h_buf[i] = eg_f32_to_bf16(v);\n        }\n")
		case in.DType == F32:
			fmt.Fprintf(&b, "        for (int i = 0; i < %d; ++i) h_buf[i] = 0.04f * std::sin((float)i * 0.037f + (float)%d * 0.51f);\n", elems, idx+1)
		default:
			b.WriteString("        for (int i = 0; i < (int)h_buf.size(); ++i) h_buf[i] = 0;\n")
		}
		fmt.Fprintf(&b, "        CHECK_CUDA(cudaMemcpy(d_%s, h_buf.data(), sizeof(%s) * %d, cudaMemcpyHostToDevice));\n    }\n",
			in.Name, ctype, elems)
	}
	b.WriteString("\n")

	// Allocate separate intermediate & KV-cache buffers for Reference and Candidate.
	for _, tmp := range fusedPlan.Temps {
		elems := fullLayerTensorElemCount(tmp.Shape)
		ctype := tmp.DType.CType()
		fmt.Fprintf(&b, "    %s* d_ref_%s = nullptr;\n", ctype, tmp.Name)
		fmt.Fprintf(&b, "    CHECK_CUDA(cudaMalloc(&d_ref_%s, sizeof(%s) * %d));\n", tmp.Name, ctype, elems)
		fmt.Fprintf(&b, "    CHECK_CUDA(cudaMemset(d_ref_%s, 0, sizeof(%s) * %d));\n", tmp.Name, ctype, elems)
		fmt.Fprintf(&b, "    %s* d_cand_%s = nullptr;\n", ctype, tmp.Name)
		fmt.Fprintf(&b, "    CHECK_CUDA(cudaMalloc(&d_cand_%s, sizeof(%s) * %d));\n", tmp.Name, ctype, elems)
		fmt.Fprintf(&b, "    CHECK_CUDA(cudaMemset(d_cand_%s, 0, sizeof(%s) * %d));\n", tmp.Name, ctype, elems)
	}
	b.WriteString("\n    unsigned long long* d_ready = nullptr;\n")
	b.WriteString("    CHECK_CUDA(cudaMalloc(&d_ready, sizeof(unsigned long long) * READY_WORDS));\n")
	b.WriteString("    CHECK_CUDA(cudaMemset(d_ready, 0, sizeof(unsigned long long) * READY_WORDS));\n\n")

	// Build argument lists for step s in Reference and Candidate.
	var refArgs, candArgs []string
	for _, in := range fusedPlan.Inputs {
		switch {
		case in.Name == "l0_x":
			refArgs = append(refArgs, "d_x_step[s]")
			candArgs = append(candArgs, "d_x_step[s]")
		case strings.HasSuffix(in.Name, "_positions"):
			refArgs = append(refArgs, "d_pos_step[s]")
			candArgs = append(candArgs, "d_pos_step[s]")
		case strings.HasSuffix(in.Name, "_seqlens"):
			refArgs = append(refArgs, "d_seqlen_step[s]")
			candArgs = append(candArgs, "d_seqlen_step[s]")
		default:
			refArgs = append(refArgs, "d_"+in.Name)
			candArgs = append(candArgs, "d_"+in.Name)
		}
	}
	for _, tmp := range fusedPlan.Temps {
		refArgs = append(refArgs, "d_ref_"+tmp.Name)
		candArgs = append(candArgs, "d_cand_"+tmp.Name)
	}
	refArgs = append(refArgs, "d_out_ref[s]", "0")
	candArgs = append(candArgs, "d_out_cand[s]", "d_ready", "0u", "0")

	b.WriteString("    auto run_ref = [&]() {\n")
	b.WriteString("        for (int s = 0; s < STEPS; ++s) {\n")
	fmt.Fprintf(&b, "            full_layer_2l_unfused_run(%s);\n", strings.Join(refArgs, ", "))
	b.WriteString("        }\n    };\n\n")

	b.WriteString("    auto run_cand = [&]() {\n")
	b.WriteString("        for (int s = 0; s < STEPS; ++s) {\n")
	fmt.Fprintf(&b, "            full_layer_2l_mega_run(%s);\n", strings.Join(candArgs, ", "))
	b.WriteString("        }\n    };\n\n")

	b.WriteString(`    run_ref();
    run_cand();
    CHECK_CUDA(cudaDeviceSynchronize());

    float max_abs_diff = 0.0f;
    std::vector<int> ref_tokens(STEPS, 0);
    std::vector<int> cand_tokens(STEPS, 0);
    for (int s = 0; s < STEPS; ++s) {
        std::vector<float> h_ref(OUT_ELEMS);
        std::vector<float> h_cand(OUT_ELEMS);
        CHECK_CUDA(cudaMemcpy(h_ref.data(), d_out_ref[s], sizeof(float) * OUT_ELEMS, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_cand.data(), d_out_cand[s], sizeof(float) * OUT_ELEMS, cudaMemcpyDeviceToHost));
        int best_r = 0, best_c = 0;
        for (int i = 0; i < OUT_ELEMS; ++i) {
            float diff = std::fabs(h_ref[i] - h_cand[i]);
            if (diff > max_abs_diff) max_abs_diff = diff;
            if (h_ref[i] > h_ref[best_r]) best_r = i;
            if (h_cand[i] > h_cand[best_c]) best_c = i;
        }
        ref_tokens[s] = best_r;
        cand_tokens[s] = best_c;
    }
    std::printf("LOGIT_MAX_ABS=%.8f\n", max_abs_diff);
    std::printf("REF_TOKENS=");
    for (int s = 0; s < STEPS; ++s) std::printf("%s%d", s ? "," : "", ref_tokens[s]);
    std::printf("\nCAND_TOKENS=");
    for (int s = 0; s < STEPS; ++s) std::printf("%s%d", s ? "," : "", cand_tokens[s]);
    std::printf("\n");

    // Device mutant negative control: perturb d_l1_ffn_norm_w and d_l1_w_down to shift argmax.
    {
        std::vector<float> orig_norm(OUT_ELEMS);
        CHECK_CUDA(cudaMemcpy(orig_norm.data(), d_l1_ffn_norm_w, sizeof(float) * OUT_ELEMS, cudaMemcpyDeviceToHost));
        std::vector<float> mut_norm(OUT_ELEMS, 40.0f);
        CHECK_CUDA(cudaMemcpy(d_l1_ffn_norm_w, mut_norm.data(), sizeof(float) * OUT_ELEMS, cudaMemcpyHostToDevice));

        int w_down_elems = OUT_ELEMS * 256;
        std::vector<uint16_t> orig_w_down(w_down_elems);
        CHECK_CUDA(cudaMemcpy(orig_w_down.data(), d_l1_w_down, sizeof(uint16_t) * w_down_elems, cudaMemcpyDeviceToHost));
        std::vector<uint16_t> mut_w_down(w_down_elems);
        for (int r = 0; r < OUT_ELEMS; ++r) {
            float val = (r == 0) ? 12.0f : -12.0f;
            uint16_t bits = eg_f32_to_bf16(val);
            for (int c = 0; c < 256; ++c) {
                mut_w_down[r * 256 + c] = bits;
            }
        }
        CHECK_CUDA(cudaMemcpy(d_l1_w_down, mut_w_down.data(), sizeof(uint16_t) * w_down_elems, cudaMemcpyHostToDevice));

        run_cand();
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<int> mut_tokens(STEPS, 0);
        bool diff_any = false;
        for (int s = 0; s < STEPS; ++s) {
            std::vector<float> h_mut(OUT_ELEMS);
            CHECK_CUDA(cudaMemcpy(h_mut.data(), d_out_cand[s], sizeof(float) * OUT_ELEMS, cudaMemcpyDeviceToHost));
            int best_m = 0;
            for (int i = 0; i < OUT_ELEMS; ++i) {
                if (h_mut[i] > h_mut[best_m]) best_m = i;
            }
            mut_tokens[s] = best_m;
            if (mut_tokens[s] != ref_tokens[s]) diff_any = true;
        }
        std::printf("MUT_TOKENS=");
        for (int s = 0; s < STEPS; ++s) std::printf("%s%d", s ? "," : "", mut_tokens[s]);
        std::printf("\nMUTANT_REJECTED=%d\n", diff_any ? 1 : 0);

        // Restore clean weights before timing.
        CHECK_CUDA(cudaMemcpy(d_l1_ffn_norm_w, orig_norm.data(), sizeof(float) * OUT_ELEMS, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_l1_w_down, orig_w_down.data(), sizeof(uint16_t) * w_down_elems, cudaMemcpyHostToDevice));
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

    std::printf("FULL_LAYER_NATIVE_OK\n");
    return 0;
}
`)
	return b.String()
}

type nativeFullLayerRunMetrics struct {
	logitMaxAbs    float64
	refTokens      []int
	candTokens     []int
	mutTokens      []int
	mutantRejected bool
	aaBlocks       []pairedtiming.Block
	screenBlocks   []pairedtiming.Block
	holdoutBlocks  []pairedtiming.Block
}

func parseNativeFullLayerTokenCSV(s string) ([]int, error) {
	parts := strings.Split(strings.TrimSpace(s), ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

func parseNativeFullLayerTimingBlock(line string) (pairedtiming.Block, error) {
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

func parseNativeFullLayerOutput(out string) (nativeFullLayerRunMetrics, error) {
	var m nativeFullLayerRunMetrics
	var sawLogit, sawRefTok, sawCandTok, sawMutRej, sawOK bool
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
		case strings.HasPrefix(line, "REF_TOKENS="):
			toks, err := parseNativeFullLayerTokenCSV(strings.TrimPrefix(line, "REF_TOKENS="))
			if err != nil {
				return m, fmt.Errorf("parse REF_TOKENS: %w", err)
			}
			m.refTokens = toks
			sawRefTok = true
		case strings.HasPrefix(line, "CAND_TOKENS="):
			toks, err := parseNativeFullLayerTokenCSV(strings.TrimPrefix(line, "CAND_TOKENS="))
			if err != nil {
				return m, fmt.Errorf("parse CAND_TOKENS: %w", err)
			}
			m.candTokens = toks
			sawCandTok = true
		case strings.HasPrefix(line, "MUT_TOKENS="):
			toks, err := parseNativeFullLayerTokenCSV(strings.TrimPrefix(line, "MUT_TOKENS="))
			if err != nil {
				return m, fmt.Errorf("parse MUT_TOKENS: %w", err)
			}
			m.mutTokens = toks
		case strings.HasPrefix(line, "MUTANT_REJECTED="):
			v, err := strconv.Atoi(strings.TrimPrefix(line, "MUTANT_REJECTED="))
			if err != nil {
				return m, fmt.Errorf("parse MUTANT_REJECTED: %w", err)
			}
			m.mutantRejected = v == 1
			sawMutRej = true
		case strings.HasPrefix(line, "AA_BLOCK "):
			b, err := parseNativeFullLayerTimingBlock(line)
			if err != nil {
				return m, err
			}
			m.aaBlocks = append(m.aaBlocks, b)
		case strings.HasPrefix(line, "SCREEN_BLOCK "):
			b, err := parseNativeFullLayerTimingBlock(line)
			if err != nil {
				return m, err
			}
			m.screenBlocks = append(m.screenBlocks, b)
		case strings.HasPrefix(line, "HOLDOUT_BLOCK "):
			b, err := parseNativeFullLayerTimingBlock(line)
			if err != nil {
				return m, err
			}
			m.holdoutBlocks = append(m.holdoutBlocks, b)
		case line == "FULL_LAYER_NATIVE_OK":
			sawOK = true
		}
	}
	if !sawLogit || !sawRefTok || !sawCandTok || !sawMutRej || !sawOK {
		return m, fmt.Errorf("incomplete benchmark output (logit=%v ref=%v cand=%v mut=%v ok=%v)",
			sawLogit, sawRefTok, sawCandTok, sawMutRej, sawOK)
	}
	return m, nil
}

func queryNativeFullLayerGPUAndDriver() (gpuName, driverVer string) {
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

func TestFullLayerMegakernelNativeClosedGateControl(t *testing.T) {
	t.Setenv("CUDATEST_NATIVE", "")
	err := checkNativeFullLayerGate(os.Getenv)
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
	if lockErr := checkNativeFullLayerGate(badEnv); lockErr == nil || errors.Is(lockErr, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkNativeFullLayerGate with GPU_LOCK_HELD=1 = %v, want non-pending lock error", lockErr)
	}

	// Genuine inherited lock must pass.
	lockDir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("fulllayer-gate-test"))
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
	if err := checkNativeFullLayerGate(goodEnv); err != nil {
		t.Fatalf("checkNativeFullLayerGate with inherited lock: %v", err)
	}

	fusedPlan, mega := buildNativeFullLayerMegakernel(t)
	harnessSrc := generateNativeFullLayerBenchmarkCUDA(t, fusedPlan, mega)
	if strings.Contains(harnessSrc, "%!") {
		t.Fatal("generated full-layer benchmark CUDA contains %! format errors")
	}

	wantSubstrings := []string{
		`asm volatile("bar.sync 1, 128;")`,
		`if (threadIdx.x >= 128) return;`,
		`eg_ready_resolve_epoch`,
		`full_layer_2l_unfused_run(`,
		`full_layer_2l_mega_run(`,
		`AA_BLOCK`,
		`SCREEN_BLOCK`,
		`HOLDOUT_BLOCK`,
		`LOGIT_MAX_ABS=`,
		`REF_TOKENS=`,
		`CAND_TOKENS=`,
		`MUTANT_REJECTED=`,
		`FULL_LAYER_NATIVE_OK`,
	}
	for _, sub := range wantSubstrings {
		if !strings.Contains(harnessSrc, sub) {
			t.Errorf("generated full-layer benchmark CUDA missing %q", sub)
		}
	}
	if got := strings.Count(harnessSrc, "if (threadIdx.x >= 128) return;"); got < 4 {
		t.Errorf("threadIdx.x >= 128 sub-CTA guard count = %d, want >= 4 (RoPE + PagedAttention in layers 0 and 1)", got)
	}
}

func TestFullLayerMegakernelNativeSM121A(t *testing.T) {
	nvcc, cudaVer, arch := requireNativeFullLayerGate(t)

	fusedPlan, mega := buildNativeFullLayerMegakernel(t)
	harnessSrc := generateNativeFullLayerBenchmarkCUDA(t, fusedPlan, mega)
	if strings.Contains(harnessSrc, "%!") {
		t.Fatal("generated full-layer benchmark CUDA contains %! format errors")
	}

	workDir := nativeFullLayerTempDir(t, "fulllayer-sm121a")
	cuPath := filepath.Join(workDir, "fulllayer_bench.cu")
	binPath := filepath.Join(workDir, "fulllayer_bench")
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

	metrics, err := parseNativeFullLayerOutput(string(rawOut))
	if err != nil {
		t.Fatalf("parse benchmark output: %v\n%s", err, rawOut)
	}

	if math.IsNaN(metrics.logitMaxAbs) || metrics.logitMaxAbs >= 1e-2 {
		t.Fatalf("logit_max_abs = %g, want < 1e-2", metrics.logitMaxAbs)
	}
	if len(metrics.refTokens) != 8 || !slices.Equal(metrics.refTokens, metrics.candTokens) {
		t.Fatalf("token parity mismatch: ref=%v cand=%v", metrics.refTokens, metrics.candTokens)
	}
	if !metrics.mutantRejected || slices.Equal(metrics.mutTokens, metrics.refTokens) {
		t.Fatalf("device mutant control not rejected: mutantRejected=%v ref=%v mut=%v",
			metrics.mutantRejected, metrics.refTokens, metrics.mutTokens)
	}

	if len(metrics.aaBlocks) != nativeFullLayerTimingBlocks {
		t.Fatalf("AA blocks = %d, want %d", len(metrics.aaBlocks), nativeFullLayerTimingBlocks)
	}
	if len(metrics.screenBlocks) != nativeFullLayerTimingBlocks {
		t.Fatalf("screen blocks = %d, want %d", len(metrics.screenBlocks), nativeFullLayerTimingBlocks)
	}
	if len(metrics.holdoutBlocks) != nativeFullLayerTimingBlocks {
		t.Fatalf("holdout blocks = %d, want %d", len(metrics.holdoutBlocks), nativeFullLayerTimingBlocks)
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

	gpuName, driverVer := queryNativeFullLayerGPUAndDriver()
	t.Logf("AA_NOISE_FLOOR: n=%d ref1_median_us=%.2f ref2_median_us=%.2f ratio=%.4fx",
		len(metrics.aaBlocks), aaSummary.Baseline, aaSummary.Candidate, aaSummary.Ratio)
	t.Logf("SCREEN_TIMING: n=%d ref_median_us=%.2f cand_median_us=%.2f speedup=%.3fx sign_p=%.4f",
		len(metrics.screenBlocks), screenSummary.Baseline, screenSummary.Candidate,
		screenSummary.Ratio, screenSummary.SignP)
	t.Logf("HOLDOUT_TIMING: n=%d ref_median_us=%.2f cand_median_us=%.2f speedup=%.3fx sign_p=%.4f",
		len(metrics.holdoutBlocks), holdoutSummary.Baseline, holdoutSummary.Candidate,
		holdoutSummary.Ratio, holdoutSummary.SignP)
	t.Logf("CLAIM: native-qualified on %s (%s), driver %s, CUDA %s, workload=full_layer_2l_monolithic_megakernel_8step_decode(M=1,D=64,I=128,Hq=4,Hkv=2,Hd=16,stages=18), metric=holdout_median_us=%.2f(ref_18launch=%.2f,speedup=%.3fx,aa_ratio=%.4fx,sign_p=%.4f,logit_max_abs=%.6f,token_parity=8/8,mutant_rejected=true)",
		gpuName, arch, driverVer, cudaVer,
		holdoutSummary.Candidate, holdoutSummary.Baseline,
		holdoutSummary.Ratio, aaSummary.Ratio, holdoutSummary.SignP,
		metrics.logitMaxAbs)
}
