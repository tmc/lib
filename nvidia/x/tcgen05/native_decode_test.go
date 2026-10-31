package tcgen05

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// tcgen05DecodeFixtures holds the decode-path kernels (runtime 2:4 SwiGLU,
// learned-LUT GEMM, KV subspace projection, flash logits) as standalone
// translation units and as one harness function, tcgen05_decode_harness,
// that runs each kernel's EG_EMU path against the host reference and a
// mutant and returns 0 on success.
type tcgen05DecodeFixtures struct {
	tuFiles   map[string]string
	tuTargets map[string][]string // tcgen05 targets whose device path each TU emits
	harness   string              // kernels and tcgen05_decode_harness, without includes or main
}

// deviceSource returns the harness as an nvcc translation unit with main.
func (fx tcgen05DecodeFixtures) deviceSource() string {
	return "#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdint>\n#include <vector>\n" +
		fx.harness + "\nint main() {\n    if (cudaFree(0) != cudaSuccess) return 2;\n    return tcgen05_decode_harness();\n}\n"
}

// hostSource returns the harness rewritten for cudahostemu: each
// kernel<<<grid, block>>>(args) launch becomes eg_emu_launch.
func (fx tcgen05DecodeFixtures) hostSource() string {
	return regexp.MustCompile(`(\w+)<<<(\d+), (\d+)>>>\(`).ReplaceAllString(fx.harness, "eg_emu_launch(dim3($2), dim3($3), 0, $1, ")
}

func formatTCGen05I32List(v []int32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprint(x)
	}
	return strings.Join(parts, ", ")
}

func buildTCGen05DecodeFixtures(t *testing.T) tcgen05DecodeFixtures {
	t.Helper()
	tu := make(map[string]string)
	targets := make(map[string][]string)
	var kernels strings.Builder
	var body strings.Builder
	r := rand.New(rand.NewPCG(2026, 1027))
	add := func(name string, archs []string, cuda func() (string, error), file func() (string, error)) {
		targets[name] = archs
		src, err := cuda()
		if err != nil {
			t.Fatalf("%s.CUDA: %v", name, err)
		}
		f, err := file()
		if err != nil {
			t.Fatalf("%s.CUDAFile: %v", name, err)
		}
		tu[name] = f
		kernels.WriteString(src + "\n")
	}

	// 1. TCGen05DynSparseSwiGLU; the mutant is the first-two selector.
	ds := TCGen05DynSparseSwiGLU{Name: "decode_dynsparse_swiglu", I: 32, N: 32, Target: "sm_107a"}
	add("TCGen05DynSparseSwiGLU", []string{"sm_100a", "sm_107a"}, ds.CUDA, ds.CUDAFile)
	gate, up, h := randSwiGLU(r, 128*32)
	wd := make([]float32, 32*32)
	for i := range wd {
		wd[i] = float32(r.NormFloat64() * 0.1)
	}
	dsWant, err := ds.Reference(gate, up, wd)
	if err != nil {
		t.Fatal(err)
	}
	first2 := DynSparseTile{Rows: 128, DenseCols: 32, Values: make([]float32, 128*16), Meta: make([]uint32, 128)}
	for g := range 128 * 8 {
		first2.Values[2*g], first2.Values[2*g+1] = h[4*g], h[4*g+1]
		first2.Meta[g/8] |= 0x4 << (4 * (g % 8))
	}
	dsMut, err := SparseMMAActivationDenseWeight(first2, wd, 32)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(&body, `
    {   // 1. TCGen05DynSparseSwiGLU (M=128, I=32, N=32)
        const float h_gate[] = {%s};
        const float h_up[] = {%s};
        const float h_w[] = {%s};
        const float want_out[] = {%s};
        const uint32_t want_meta[] = {%s};
        const float mut_out[] = {%s};
        float *d_gate = dev_dup(h_gate, 4096), *d_up = dev_dup(h_up, 4096), *d_w = dev_dup(h_w, 1024);
        float *d_hc = dev_alloc<float>(2048), *d_absmax = dev_alloc<float>(128), *d_out = dev_alloc<float>(4096);
        uint32_t* d_meta = dev_alloc<uint32_t>(128);
        decode_dynsparse_swiglu<<<1, 128>>>(d_gate, d_up, d_w, d_hc, d_meta, d_absmax, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
        std::vector<float> out(4096);
        std::vector<uint32_t> meta(128);
        if (!host_copy(out.data(), d_out, 4096) || !host_copy(meta.data(), d_meta, 128)) return 5;
        for (int i = 0; i < 128; ++i) if (meta[i] != want_meta[i]) { std::printf("FAIL dynsparse meta[%%d]=%%08x want %%08x\n", i, meta[i], want_meta[i]); return 6; }
        if (!close_all("dynsparse out", out.data(), want_out, 4096, 1e-3f)) return 6;
        if (max_diff(out.data(), mut_out, 4096) < 1e-2f) return 7;
        std::printf("PASS TCGen05DynSparseSwiGLU M=128 I=32 N=32 mutant_rejected=1\n");
    }
`, formatTCGen05F32List(gate), formatTCGen05F32List(up), formatTCGen05F32List(wd),
		formatTCGen05F32List(dsWant.Out), formatTCGen05U32List(dsWant.H.Meta), formatTCGen05F32List(dsMut))

	// 2. TCGen05LUTDecompressGEMM with learned Hessian-weighted LUTs; the
	// mutant swaps in Quantize3BitLUTB's LUTs.
	lk := TCGen05LUTDecompressGEMM{Name: "decode_learned_lut_gemm", K: 64, Target: "sm_107a"}
	add("TCGen05LUTDecompressGEMM(learned)", []string{"sm_107a"}, lk.CUDA, lk.CUDAFile)
	bw, hd := lutTestMatrix(64, 128, 9)
	idx, lut, sc, _, err := QuantizeLearnedLUTB(bw, hd, 64, 128, 100)
	if err != nil {
		t.Fatal(err)
	}
	_, baseLUT, _, _, err := Quantize3BitLUTB(bw, 64, 128)
	if err != nil {
		t.Fatal(err)
	}
	a := make([]float32, 128*64)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	aE4M3, aSc := QuantizeMXFP8E4M3(a)
	lutWant, err := lk.Reference(aE4M3, aSc, idx, lut, sc)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(&body, `
    {   // 2. TCGen05LUTDecompressGEMM with learned LUTs (M=128, N=128, K=64)
        const uint8_t h_a[] = {%s};
        const uint8_t h_as[] = {%s};
        const uint8_t h_idx[] = {%s};
        const uint8_t h_lut[] = {%s};
        const uint8_t h_lut_mut[] = {%s};
        const uint8_t h_bs[] = {%s};
        const float want_out[] = {%s};
        uint8_t *d_a = dev_dup(h_a, sizeof(h_a)), *d_as = dev_dup(h_as, sizeof(h_as)), *d_idx = dev_dup(h_idx, sizeof(h_idx));
        uint8_t *d_lut = dev_dup(h_lut, sizeof(h_lut)), *d_bs = dev_dup(h_bs, sizeof(h_bs));
        float* d_out = dev_alloc<float>(128 * 128);
        decode_learned_lut_gemm<<<1, 128>>>(d_a, d_as, d_idx, d_lut, d_bs, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
        std::vector<float> out(128 * 128);
        if (!host_copy(out.data(), d_out, 128 * 128)) return 5;
        if (!close_all("learned lut out", out.data(), want_out, 128 * 128, 1e-3f)) return 6;
        if (cudaMemcpy(d_lut, h_lut_mut, sizeof(h_lut_mut), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
        decode_learned_lut_gemm<<<1, 128>>>(d_a, d_as, d_idx, d_lut, d_bs, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
        if (!host_copy(out.data(), d_out, 128 * 128)) return 5;
        if (max_diff(out.data(), want_out, 128 * 128) < 1e-2f) return 7;
        std::printf("PASS TCGen05LUTDecompressGEMM(learned) M=128 N=128 K=64 mutant_rejected=1\n");
    }
`, formatTCGen05U8List(aE4M3), formatTCGen05U8List(aSc), formatTCGen05U8List(idx), formatTCGen05U8List(lut),
		formatTCGen05U8List(baseLUT), formatTCGen05U8List(sc), formatTCGen05F32List(lutWant))

	// 3. TCGen05KVSubspaceProject; the mutant perturbs the basis.
	kp := TCGen05KVSubspaceProject{Name: "decode_kv_subspace_project", Dim: 64, Rank: 32, Target: "sm_100a"}
	add("TCGen05KVSubspaceProject", []string{"sm_100a", "sm_107a"}, kp.CUDA, kp.CUDAFile)
	proj, _, ks, _ := testProjector(t, 11, 128, 2, 32, 32)
	x := ks.rows(r, 128, 0.02)
	kpWant, err := kp.Reference(x, proj.BasisK)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(&body, `
    {   // 3. TCGen05KVSubspaceProject (T=128, D=64, r=32)
        const float h_x[] = {%s};
        float h_u[] = {%s};
        const float want_out[] = {%s};
        float *d_x = dev_dup(h_x, 128 * 64), *d_u = dev_dup(h_u, 64 * 32), *d_out = dev_alloc<float>(128 * 32);
        decode_kv_subspace_project<<<1, 128>>>(d_x, d_u, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
        std::vector<float> out(128 * 32);
        if (!host_copy(out.data(), d_out, 128 * 32)) return 5;
        if (!close_all("kv subspace out", out.data(), want_out, 128 * 32, 1e-3f)) return 6;
        for (int i = 0; i < 64 * 32; i += 7) h_u[i] += 0.5f;
        if (cudaMemcpy(d_u, h_u, sizeof(h_u), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
        decode_kv_subspace_project<<<1, 128>>>(d_x, d_u, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
        if (!host_copy(out.data(), d_out, 128 * 32)) return 5;
        if (max_diff(out.data(), want_out, 128 * 32) < 1e-2f) return 7;
        std::printf("PASS TCGen05KVSubspaceProject T=128 D=64 R=32 mutant_rejected=1\n");
    }
`, formatTCGen05F32List(x), formatTCGen05F32List(proj.BasisK), formatTCGen05F32List(kpWant))

	// 4. FlashLogitsSchedule; the mutant perturbs the hidden states.
	fl := FlashLogitsSchedule{Name: "decode_flash_logits", TileV: 32, VocabSize: 97, HiddenDim: 32, Temperature: 0.8, Seed: 20261027, DoubleBuffer: true}
	add("FlashLogitsSchedule", []string{"sm_100a", "sm_107a"}, fl.CUDA, fl.CUDAFile)
	fc := newFlashCase(r, 128, 97, 32)
	flWant, err := fl.ExecuteReference(fc.h, fc.w, fc.draft)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(&body, `
    {   // 4. FlashLogitsSchedule (M=128, V=97, D=32, TileV=32, double buffer)
        float h_h[] = {%s};
        const float h_w[] = {%s};
        const int h_draft[] = {%s};
        const int want_tok[] = {%s};
        const float want_lse[] = {%s};
        const float want_lp[] = {%s};
        const float want_dp[] = {%s};
        float *d_h = dev_dup(h_h, 128 * 32), *d_w = dev_dup(h_w, 97 * 32);
        int* d_draft = dev_dup(h_draft, 128);
        int* d_tok = dev_alloc<int>(128);
        float *d_lp = dev_alloc<float>(128), *d_lse = dev_alloc<float>(128), *d_dp = dev_alloc<float>(128);
        decode_flash_logits<<<1, 128>>>(d_h, d_w, d_draft, d_tok, d_lp, d_lse, d_dp);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
        int tok[128];
        float lse[128], lp[128], dp[128];
        if (!host_copy(tok, d_tok, 128) || !host_copy(lse, d_lse, 128) || !host_copy(lp, d_lp, 128) || !host_copy(dp, d_dp, 128)) return 5;
        for (int i = 0; i < 128; ++i) if (tok[i] != want_tok[i]) { std::printf("FAIL flash token[%%d]=%%d want %%d\n", i, tok[i], want_tok[i]); return 6; }
        if (!close_all("flash lse", lse, want_lse, 128, 1e-4f) || !close_all("flash logprob", lp, want_lp, 128, 1e-3f) ||
            !close_all("flash draft prob", dp, want_dp, 128, 1e-3f)) return 6;
        for (int i = 0; i < 128 * 32; i += 3) h_h[i] *= -1.5f;
        if (cudaMemcpy(d_h, h_h, sizeof(h_h), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
        decode_flash_logits<<<1, 128>>>(d_h, d_w, d_draft, d_tok, d_lp, d_lse, d_dp);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
        if (!host_copy(lse, d_lse, 128)) return 5;
        if (max_diff(lse, want_lse, 128) < 1e-2f) return 7;
        std::printf("PASS FlashLogitsSchedule M=128 V=97 D=32 mutant_rejected=1\n");
    }
`, formatTCGen05F32List(fc.h), formatTCGen05F32List(fc.w), formatTCGen05I32List(fc.draft),
		formatTCGen05I32List(flWant.SampledTokenID), formatTCGen05F32List(flWant.LSE),
		formatTCGen05F32List(flWant.SampledLogProb), formatTCGen05F32List(flWant.DraftTokenProb))

	var b strings.Builder
	// The device paths of these kernels use structural descriptors and are
	// checked only by ptxas; the harness always runs the EG_EMU path.
	b.WriteString("#define EG_EMU 1\n")
	b.WriteString(Prelude + "\n")
	b.WriteString(kernels.String())
	b.WriteString(`
template <typename T>
static T* dev_alloc(size_t n) {
    T* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess) return nullptr;
    return d;
}

template <typename T>
static T* dev_dup(const T* h, size_t n) {
    T* d = dev_alloc<T>(n);
    if (d != nullptr && cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return d;
}

template <typename T>
static bool host_copy(T* h, const T* d, size_t n) {
    return cudaMemcpy(h, d, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
}

static float max_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

static bool close_all(const char* name, const float* got, const float* want, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (std::isnan(got[i]) || std::fabs(got[i] - want[i]) > tol * std::fmax(1.0f, std::fabs(want[i]))) {
            std::printf("FAIL %s[%d] = %g, want %g\n", name, i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

static int tcgen05_decode_harness() {
`)
	b.WriteString(body.String())
	b.WriteString(`
    std::printf("TCGEN05_DECODE_NATIVE_OK decode_mutant_rejected=1\n");
    return 0;
}
`)
	return tcgen05DecodeFixtures{tuFiles: tu, tuTargets: targets, harness: b.String()}
}

// TestTCGen05DecodeNativeSuiteStructure checks on the host that the decode
// kernels emit complete translation units and that the device harness
// carries every kernel, its instructions, and the pass markers.
func TestTCGen05DecodeNativeSuiteStructure(t *testing.T) {
	fx := buildTCGen05DecodeFixtures(t)
	if len(fx.tuFiles) != 4 {
		t.Fatalf("len(tuFiles) = %d, want 4", len(fx.tuFiles))
	}
	for name, tu := range fx.tuFiles {
		if !strings.Contains(tu, "EG_PRELUDE") || !strings.Contains(tu, "__global__ void") {
			t.Fatalf("%s CUDAFile() is not a complete translation unit", name)
		}
	}
	for _, want := range []string{
		"tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2",
		"tcgen05.mma.sp.cta_group::1.kind::tf32",
		"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32",
		"tcgen05.mma.cta_group::1.kind::tf32",
		"mbarrier.try_wait.parity.shared::cta.b64",
		"eg_philox4x32_10_x0",
		"TCGEN05_DECODE_NATIVE_OK",
		"decode_mutant_rejected=1",
	} {
		if !strings.Contains(fx.harness, want) {
			t.Errorf("decode harness missing %q", want)
		}
	}
	if n := strings.Count(fx.harness, "mutant_rejected=1\\n"); n != 5 {
		t.Errorf("decode harness has %d mutant markers, want 5", n)
	}
	if n := strings.Count(fx.hostSource(), "dim3(128), 0, decode_"); n != 7 {
		t.Errorf("host harness has %d eg_emu_launch calls, want 7", n)
	}
}

// TestTCGen05DecodeHarnessEmu runs the whole device harness, every kernel
// and mutant, under cudahostemu.
func TestTCGen05DecodeHarnessEmu(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: runs seven emulated launches")
	}
	fx := buildTCGen05DecodeFixtures(t)
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  fx.hostSource(),
		Launch:  "status[0] = tcgen05_decode_harness()",
		Buffers: []cudahostemu.Buffer{{Name: "status", CType: "int", Data: poison(4)}},
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler")
	}
	if err != nil {
		t.Fatal(err)
	}
	if st := int32(bytesU32(res["status"])[0]); st != 0 {
		t.Fatalf("tcgen05_decode_harness returned %d", st)
	}
}

// TestTCGen05DecodeNativeQualification compiles each decode kernel for the
// device architecture and, when the toolkit supports them, for the tcgen05
// targets whose device path it emits (sm_100a and sm_107a; sm_107a only for
// the Rubin LUT GEMM), which checks ptxas acceptance and is inspected with
// nvdisasm. It then builds and runs the harness, which executes the EG_EMU
// path of every kernel on the GPU against the host reference and a mutant.
// Run with CUDA_TCGEN05_NATIVE=1 under an inherited GPU lock.
func TestTCGen05DecodeNativeQualification(t *testing.T) {
	if os.Getenv(tcgen05NativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_TCGEN05_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "tcgen05-decode-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	fx := buildTCGen05DecodeFixtures(t)
	devArch := os.Getenv("CUDA_TCGEN05_NATIVE_ARCH")
	if devArch == "" {
		devArch = detectDeviceArchFlag(t, nvcc)
	}
	nvccRun := func(args ...string) ([]byte, error) {
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
		defer cancel()
		return exec.CommandContext(ctx, nvcc, args...).CombinedOutput()
	}
	for name, src := range fx.tuFiles {
		file := strings.NewReplacer("(", "_", ")", "").Replace(name)
		cu := filepath.Join(dir, file+".cu")
		if err := os.WriteFile(cu, []byte(src), 0o600); err != nil {
			t.Fatal(err)
		}
		for _, arch := range append([]string{devArch}, fx.tuTargets[name]...) {
			if arch != devArch && !nvccSupportsTarget(t, nvcc, arch) {
				t.Logf("%s: nvcc (CUDA %s) lacks %s; skipped", name, cudaVer, arch)
				continue
			}
			cubin := filepath.Join(dir, file+"."+arch+".cubin")
			if out, err := nvccRun("-std=c++17", "-arch="+arch, "-cubin", cu, "-o", cubin); err != nil {
				t.Errorf("nvcc -arch=%s -cubin %s (CUDA %s): %v\n%s", arch, name, cudaVer, err, out)
				continue
			}
			inspectCubinWithNVDisasm(t, cubin, arch)
		}
	}

	harnessCU := filepath.Join(dir, "tcgen05_decode_qualify.cu")
	harnessBin := filepath.Join(dir, "tcgen05_decode_qualify")
	if err := os.WriteFile(harnessCU, []byte(fx.deviceSource()), 0o600); err != nil {
		t.Fatal(err)
	}
	if out, err := nvccRun("-std=c++17", "-O2", "-arch="+devArch, harnessCU, "-o", harnessBin); err != nil {
		t.Fatalf("nvcc decode harness (%s, CUDA %s): %v\n%s", devArch, cudaVer, err, out)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, harnessBin)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("decode harness failed (arch=%s, CUDA=%s): %v\n%s", devArch, cudaVer, err, out)
	}
	if !strings.Contains(string(out), "TCGEN05_DECODE_NATIVE_OK") || !strings.Contains(string(out), "decode_mutant_rejected=1") {
		t.Fatalf("decode harness output missing markers:\n%s", out)
	}
	t.Logf("decode kernels (EG_EMU path) passed on %s (CUDA %s):\n%s", devArch, cudaVer, strings.TrimSpace(string(out)))
}
