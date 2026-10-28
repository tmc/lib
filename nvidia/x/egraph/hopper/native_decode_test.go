package hopper

import (
	"context"
	"fmt"
	"maps"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// A decodeNativeCase is one kernel's part of the decode native harness:
// device code, and a block of host code inside main that returns nonzero
// on failure and prints a PASS line on success.
type decodeNativeCase struct {
	name string
	src  string
	main string
}

// TestHopperNativeDecodeKernels compiles and runs the decode-loop kernels
// of buildDecodeNativeHarness on the GPU behind the same gate as
// [TestHopperNativeQualification], driving the device-side loops through
// real CUDA graph conditional WHILE nodes. It is a separate binary so a
// toolchain that rejects one of these kernels does not hide the results of
// the existing qualification suite.
func TestHopperNativeDecodeKernels(t *testing.T) {
	if os.Getenv(hopperNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_EGRAPH_HOPPER_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	target := hopperNativeArch()
	cudatest.SkipUnlessTarget(t, nvcc, target)

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "hopper-decode-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	src := buildDecodeNativeHarness(t)
	cu := filepath.Join(dir, "decode_native.cu")
	bin := filepath.Join(dir, "decode_native")
	if err := os.WriteFile(cu, []byte(src), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O2", "-arch="+target, cu, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("nvcc (%s, CUDA %s): %v\n%s", target, cudaVer, err, out)
	}
	ctx, cancel = context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	out, err := exec.CommandContext(ctx, bin).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "DECODE_NATIVE_OK") {
		t.Fatalf("decode native harness (%s, CUDA %s): %v\n%s", target, cudaVer, err, out)
	}
	t.Logf("decode native harness passed (%s, CUDA %s):\n%s", target, cudaVer, strings.TrimSpace(string(out)))
}

func TestDecodeNativeHarnessBuilds(t *testing.T) {
	src := buildDecodeNativeHarness(t)
	for _, want := range []string{
		"cudaGraphCondTypeWhile",
		"tree_verify_native_reserve",
		"PASS TreeSpecVerify",
		"DECODE_NATIVE_OK",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("decode native harness missing %q", want)
		}
	}
}

func buildDecodeNativeHarness(t *testing.T) string {
	t.Helper()
	cases := []decodeNativeCase{
		treeSpecNativeCase(t),
	}
	var b strings.Builder
	b.WriteString("#include <cuda_runtime.h>\n#include <cstdio>\n#include <cstring>\n#include <cmath>\n#include <vector>\n")
	b.WriteString("#if CUDART_VERSION < 12040\n#error \"conditional WHILE graph nodes need CUDA 12.4 or later\"\n#endif\n")
	b.WriteString(Prelude)
	b.WriteString(decodeNativeHelpers)
	for _, c := range cases {
		fmt.Fprintf(&b, "\n// ---- %s ----\n%s\n", c.name, c.src)
	}
	b.WriteString("\nint main() {\n")
	for _, c := range cases {
		fmt.Fprintf(&b, "    {\n%s    }\n", c.main)
	}
	b.WriteString("    std::printf(\"DECODE_NATIVE_OK\\n\");\n    return 0;\n}\n")
	return b.String()
}

const decodeNativeHelpers = `
#define DN_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); return 2; } } while (0)

template <typename T>
static T* dn_dup(const T* h, size_t n) {
    T* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return d;
}

template <typename T>
static std::vector<T> dn_get(const T* d, size_t n) {
    std::vector<T> h(n);
    if (cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) h.clear();
    return h;
}

// dn_add_while adds a conditional WHILE node to graph and returns its body.
static cudaError_t dn_add_while(cudaGraph_t graph, cudaGraphConditionalHandle handle, cudaGraph_t* body) {
    cudaGraphNodeParams p = {};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = handle;
    p.conditional.type = cudaGraphCondTypeWhile;
    p.conditional.size = 1;
    cudaGraphNode_t node;
#if CUDART_VERSION >= 13000
    cudaError_t err = cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &p);
#else
    cudaError_t err = cudaGraphAddNode(&node, graph, nullptr, 0, &p);
#endif
    if (err == cudaSuccess) *body = p.conditional.phGraph_out[0];
    return err;
}

// dn_add_kernel appends a kernel node to body after *prev.
static cudaError_t dn_add_kernel(cudaGraph_t body, cudaGraphNode_t* prev, void* fn, void** args,
                                 dim3 grid = dim3(1), dim3 block = dim3(32)) {
    cudaKernelNodeParams k = {};
    k.func = fn;
    k.gridDim = grid;
    k.blockDim = block;
    k.kernelParams = args;
    cudaGraphNode_t node;
    cudaError_t err = cudaGraphAddKernelNode(&node, body, *prev ? prev : nullptr, *prev ? 1 : 0, &k);
    if (err == cudaSuccess) *prev = node;
    return err;
}
`

func cWords(b []byte) string {
	w := le32u(b)
	var s strings.Builder
	for i, x := range w {
		if i > 0 {
			s.WriteString(",")
		}
		fmt.Fprintf(&s, "0x%xu", x)
	}
	return s.String()
}

// treeSpecNativeCase runs the 32-step greedy tree verification loop of
// TestTreeSpecVerifyEmulatedLoop as one CUDA graph launch whose WHILE node
// runs reserve, the fixture stage and verify, and compares the final device
// state with [TreeSpecVerify.Reference].
func treeSpecNativeCase(t *testing.T) decodeNativeCase {
	t.Helper()
	const steps = 32
	s := TreeSpecVerify{Name: "tree_verify_native", Nodes: 8, Batch: 3, KVWords: 3, PageSize: 4, MaxPages: 24, MaxSteps: steps}
	r := rand.New(rand.NewPCG(5, 8))
	in := TreeSpecInput{
		Prefix:     [][]int32{{1, 2, 3}, {4, 5, 6, 7, 8}, {9}},
		Roots:      []int32{10, 11, 12},
		Steps:      treeSpecGreedyFixture(r, s, steps, 999),
		TotalPages: 48,
		EOSTokenID: 999,
	}
	want, err := s.Reference(in)
	if err != nil {
		t.Fatal(err)
	}
	k, err := s.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	bufs := s.loopBuffers(in, steps)
	names := slices.Sorted(maps.Keys(bufs))

	var src, m strings.Builder
	src.WriteString(k)
	src.WriteString(s.stageCUDA("tree_verify_native_stage"))
	for _, n := range names {
		fmt.Fprintf(&src, "static const uint32_t tv_h_%s[] = {%s};\n", n, cWords(bufs[n]))
	}

	// Expected final state in device layout.
	capTokens := s.MaxPages * s.PageSize
	wantTokens := make([]int32, s.Batch*capTokens)
	wantPT := slices.Repeat([]int32{-1}, s.Batch*s.MaxPages)
	var wantLens []int32
	var wantKV []uint32
	for b := range s.Batch {
		copy(wantTokens[b*capTokens:], want.Tokens[b])
		copy(wantPT[b*s.MaxPages:], want.PageTable[b])
		wantLens = append(wantLens, int32(len(want.Tokens[b])))
		wantKV = append(wantKV, want.KV[b]...)
	}
	fmt.Fprintf(&src, "static const int32_t tv_want_tokens[] = {%s};\n", formatI32List(wantTokens))
	fmt.Fprintf(&src, "static const int32_t tv_want_pt[] = {%s};\n", formatI32List(wantPT))
	fmt.Fprintf(&src, "static const int32_t tv_want_lens[] = {%s};\n", formatI32List(wantLens))
	fmt.Fprintf(&src, "static const int32_t tv_want_roots[] = {%s};\n", formatI32List(want.Roots))
	fmt.Fprintf(&src, "static const int32_t tv_want_free[] = {%s};\n", formatI32List(want.FreePages))
	fmt.Fprintf(&src, "static const uint32_t tv_want_kv[] = {%s};\n", cWords(u32le(wantKV)))

	for _, n := range names {
		fmt.Fprintf(&m, "        uint32_t* d_%s = dn_dup(tv_h_%s, sizeof(tv_h_%s) / 4);\n        if (!d_%s) return 2;\n", n, n, n, n)
	}
	ptr := func(n, ctype string) string { return fmt.Sprintf("reinterpret_cast<%s*>(d_%s)", ctype, n) }
	fmt.Fprintf(&m, `        int32_t *page_table = %s, *num_pages = %s, *free_pages = %s, *free_top = %s;
        int32_t *seqlens = %s, *active = %s, *out_tokens = %s, *root_tokens = %s, *accepted = %s, *step_idx = %s;
        int32_t *all_parent = %s, *all_tokens = %s, *all_argmax = %s, *parent = %s, *tree_tokens = %s, *target_argmax = %s;
        float *all_dprobs = %s, *all_tprobs = %s, *draft_probs = %s, *target_probs = %s;
        uint32_t* kv = d_kv;
        int32_t eos_token_id = %d;
        cudaGraph_t graph, body;
        cudaGraphConditionalHandle cond_handle;
        DN_CHECK(cudaGraphCreate(&graph, 0));
        DN_CHECK(cudaGraphConditionalHandleCreate(&cond_handle, graph, 1, cudaGraphCondAssignDefault));
        DN_CHECK(dn_add_while(graph, cond_handle, &body));
        void* reserve_args[] = {&page_table, &num_pages, &free_pages, &free_top, &seqlens, &active};
        void* stage_args[] = {&all_parent, &all_tokens, &all_argmax, &all_dprobs, &all_tprobs, &step_idx, &root_tokens,
                              &seqlens, &page_table, &active, &parent, &tree_tokens, &target_argmax, &draft_probs, &target_probs, &kv};
        void* verify_args[] = {&parent, &tree_tokens, &target_argmax, &kv, &page_table, &num_pages, &free_pages, &free_top,
                               &seqlens, &active, &out_tokens, &root_tokens, &accepted, &step_idx, &eos_token_id, &cond_handle};
        cudaGraphNode_t prev = nullptr;
        DN_CHECK(dn_add_kernel(body, &prev, (void*)tree_verify_native_reserve, reserve_args));
        DN_CHECK(dn_add_kernel(body, &prev, (void*)tree_verify_native_stage, stage_args));
        DN_CHECK(dn_add_kernel(body, &prev, (void*)tree_verify_native, verify_args));
        cudaGraphExec_t exec;
        DN_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        DN_CHECK(cudaGraphLaunch(exec, 0));
        DN_CHECK(cudaDeviceSynchronize());

        auto steps = dn_get(step_idx, 1);
        auto lens = dn_get(seqlens, %d);
        auto toks = dn_get(out_tokens, sizeof(tv_want_tokens) / 4);
        auto pt = dn_get(page_table, sizeof(tv_want_pt) / 4);
        auto roots = dn_get(root_tokens, %d);
        auto top = dn_get(free_top, 1);
        auto freep = dn_get(free_pages, sizeof(tv_h_free_pages) / 4);
        auto kvh = dn_get(kv, sizeof(tv_h_kv) / 4);
        if (steps.empty() || steps[0] != %d) { std::printf("FAIL TreeSpecVerify steps %%d, want %d\n", steps.empty() ? -1 : steps[0]); return 3; }
        if (std::memcmp(lens.data(), tv_want_lens, sizeof(tv_want_lens)) || std::memcmp(toks.data(), tv_want_tokens, sizeof(tv_want_tokens)) ||
            std::memcmp(roots.data(), tv_want_roots, sizeof(tv_want_roots))) { std::printf("FAIL TreeSpecVerify tokens\n"); return 3; }
        if (std::memcmp(pt.data(), tv_want_pt, sizeof(tv_want_pt)) || top[0] != %d ||
            std::memcmp(freep.data(), tv_want_free, sizeof(tv_want_free))) { std::printf("FAIL TreeSpecVerify pages\n"); return 3; }
        for (int b = 0, i = 0; b < %d; ++b) {
            for (int pos = 0; pos < lens[b]; ++pos) {
                const int slot = pt[b * %d + pos / %d] * %d + pos %% %d;
                for (int w = 0; w < %d; ++w, ++i) {
                    if (kvh[slot * %d + w] != tv_want_kv[i]) { std::printf("FAIL TreeSpecVerify KV seq %%d pos %%d\n", b, pos); return 3; }
                }
            }
        }
        std::printf("PASS TreeSpecVerify graph WHILE iterations=%%d host_round_trips=0\n", steps[0]);
`,
		ptr("page_table", "int32_t"), ptr("num_pages", "int32_t"), ptr("free_pages", "int32_t"), ptr("free_top", "int32_t"),
		ptr("seqlens", "int32_t"), ptr("active", "int32_t"), ptr("out_tokens", "int32_t"), ptr("root_tokens", "int32_t"), ptr("accepted", "int32_t"), ptr("step_idx", "int32_t"),
		ptr("all_parent", "int32_t"), ptr("all_tokens", "int32_t"), ptr("all_argmax", "int32_t"), ptr("parent", "int32_t"), ptr("tree_tokens", "int32_t"), ptr("target_argmax", "int32_t"),
		ptr("all_dprobs", "float"), ptr("all_tprobs", "float"), ptr("draft_probs", "float"), ptr("target_probs", "float"),
		in.EOSTokenID,
		s.Batch, s.Batch,
		want.Steps, want.Steps,
		len(want.FreePages),
		s.Batch, s.MaxPages, s.PageSize, s.PageSize, s.PageSize, s.KVWords, s.KVWords,
	)
	return decodeNativeCase{name: "TreeSpecVerify", src: src.String(), main: m.String()}
}
