package tcgen05

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// A tmNativeArg is one kernel argument of a tmNativeCase: a scalar int
// (data nil) or a device buffer of float or int.
type tmNativeArg struct {
	name   string
	scalar int
	f32    []float32
	i32    []int32
	out    int // output buffer length in floats (poisoned with NaN)
}

// A tmNativeCheck compares output buffer name with want.
type tmNativeCheck struct {
	name string
	want []float32
}

// A tmNativeCase is one kernel run of the device harness, plus a mutant of
// the same kernel whose first check must miss by more than 1e-2.
type tmNativeCase struct {
	name, src, mutSrc string
	grid, block       [3]int
	smem              int
	args              []tmNativeArg
	checks            []tmNativeCheck
	tol               float64
}

// newTMNativeCase emits src with name and its mutant (old replaced by new)
// with name_mut.
func newTMNativeCase(t *testing.T, name string, emit func(name string) (string, error), old, new string) tmNativeCase {
	t.Helper()
	src, err := emit(name)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	mut, err := emit(name + "_mut")
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if !strings.Contains(mut, old) {
		t.Fatalf("%s: source lacks %q", name, old)
	}
	return tmNativeCase{name: name, src: src, mutSrc: strings.Replace(mut, old, new, 1), tol: 1e-4}
}

// tmNativeCases builds the SSM, TTT, LoRA, tubelet and jagged attention
// cases at sizes whose shared memory fits a 99 KiB sm_121a CTA. Expected
// outputs come from the Go references.
func tmNativeCases(t *testing.T) []tmNativeCase {
	t.Helper()
	var cases []tmNativeCase

	for _, v := range []SSMVariant{SSMMamba2SSD, SSMGatedDeltaNet} {
		s := SSMChunkSchedule{Variant: v, HeadDimK: 32, HeadDimV: 32, ChunkLen: 32, NumChunks: 4, ConvKernel: 4}
		c := newTMNativeCase(t, "native_ssm_"+v.String(), func(n string) (string, error) { s.Name = n; return s.CUDA() },
			"eg_tm_mma_ss(tm, base + C_S, ut, qs, DK, L, c > 0);", "eg_tm_mma_ss(tm, base + C_S, ut, qs, DK, L, 0);")
		const heads = 3
		r := rand.New(rand.NewPCG(31, uint64(v)))
		var q, k, vv, a, b, cq, ck, cv, want, wantS []float32
		for range heads {
			in := randSSMInput(r, s)
			o, st, err := s.Reference(in)
			if err != nil {
				t.Fatal(err)
			}
			q, k, vv = append(q, flat(in.Q)...), append(k, flat(in.K)...), append(vv, flat(in.V)...)
			a, b = append(a, in.Alpha...), append(b, in.Beta...)
			cq, ck, cv = append(cq, flat(in.ConvQ)...), append(ck, flat(in.ConvK)...), append(cv, flat(in.ConvV)...)
			want, wantS = append(want, flat(o)...), append(wantS, flat(st)...)
		}
		c.grid, c.block = s.Grid(heads, 0)
		c.smem = s.SharedBytes()
		c.args = []tmNativeArg{{name: "heads", scalar: heads}, {name: "q", f32: q}, {name: "k", f32: k}, {name: "v", f32: vv},
			{name: "alpha", f32: a}, {name: "beta", f32: b}, {name: "conv_q", f32: cq}, {name: "conv_k", f32: ck}, {name: "conv_v", f32: cv},
			{name: "out", out: len(want)}, {name: "state", out: len(wantS)}}
		c.checks = []tmNativeCheck{{"out", want}, {"state", wantS}}
		cases = append(cases, c)
	}

	for _, v := range []TTTVariant{TTTLinearSGD, TTTTitansMomentum} {
		s := TTTSchedule{Variant: v, HeadDim: 32, ChunkLen: 32, NumChunks: 3, LR: 0.003, WeightDecay: 0.01, Momentum: 0.5}
		c := newTMNativeCase(t, "native_ttt_"+v.String(), func(n string) (string, error) { s.Name = n; return s.CUDA() },
			"- mr - zh * mrz) * rsig", "- mr) * rsig")
		tc := newTTTCase(t, s, 2, uint64(40+v))
		var w0, q, k, vv []float32
		for h := range tc.w0s {
			w0, q, k, vv = append(w0, flat(tc.w0s[h])...), append(q, flat(tc.qs[h])...), append(k, flat(tc.ks[h])...), append(vv, flat(tc.vs[h])...)
		}
		c.grid, c.block = s.Grid(2, 0)
		c.smem = s.SharedBytes()
		c.args = []tmNativeArg{{name: "heads", scalar: 2}, {name: "w0", f32: w0}, {name: "q", f32: q}, {name: "k", f32: k}, {name: "v", f32: vv},
			{name: "out", out: len(tc.out)}, {name: "w", out: len(tc.w)}, {name: "loss", out: 2 * s.SeqLen()}}
		c.checks = []tmNativeCheck{{"out", tc.out}, {"w", tc.w}}
		cases = append(cases, c)
	}

	{
		s := SegmentedLoRASchedule{DimIn: 64, DimOut: 128, TileN: 64, BK: 32, MaxRank: 64}
		c := newTMNativeCase(t, "native_lora", func(n string) (string, error) { s.Name = n; return s.CUDA() },
			"eg_tm_mma_ss(tm, base + C_B, us, bs, TN, R, 1);", "eg_tm_mma_ss(tm, base + C_B, us, bs, TN, R, 0);")
		b := newLoRABatch(rand.New(rand.NewPCG(32, 200)), 200, s.DimIn, s.DimOut)
		want, err := ReferenceFusedLoRA(b.x, b.w, b.ids, b.aPool, b.bPool, b.adapters)
		if err != nil {
			t.Fatal(err)
		}
		plan, err := PlanLoRAGather(b.ids, b.adapters)
		if err != nil {
			t.Fatal(err)
		}
		order, start, end, rank, aoff, boff, scale := LoRAKernelArgs(plan)
		c.grid, c.block = s.Grid(len(b.x))
		c.smem = s.SharedBytes()
		c.args = []tmNativeArg{{name: "tokens", scalar: len(b.x)}, {name: "nsegs", scalar: len(start)},
			{name: "order", i32: order}, {name: "seg_start", i32: start}, {name: "seg_end", i32: end}, {name: "seg_rank", i32: rank},
			{name: "seg_aoff", i32: aoff}, {name: "seg_boff", i32: boff}, {name: "seg_scale", f32: scale},
			{name: "x", f32: flat(b.x)}, {name: "w", f32: flat(b.w)}, {name: "a_pool", f32: flat(b.aPool)}, {name: "b_pool", f32: flat(b.bPool)},
			{name: "y", out: len(b.x) * s.DimOut}}
		c.checks = []tmNativeCheck{{"y", flat(want)}}
		cases = append(cases, c)
	}

	{
		s := TubeletPatchSpec{Frames: 3, Height: 9, Width: 10, Channels: 4, PatchT: 2, PatchH: 4, PatchW: 4, EmbedDim: 256}
		c := newTMNativeCase(t, "native_tubelet", func(n string) (string, error) { s.Name = n; return s.CUDA() },
			"p < P && t < T && h < H && x < W ?", "p < P && t < T && h < H ?")
		r := rand.New(rand.NewPCG(33, 1))
		video := flat(randMat(r, s.Frames*s.Height*s.Width, s.Channels, 1))
		w := randMat(r, s.EmbedDim, s.PatchLen(), 0.1)
		want, err := s.Reference(video, w)
		if err != nil {
			t.Fatal(err)
		}
		c.grid, c.block = s.Grid()
		c.smem = s.SharedBytes()
		c.args = []tmNativeArg{{name: "video", f32: video}, {name: "w", f32: flat(w)}, {name: "out", out: s.Patches() * s.EmbedDim}}
		c.checks = []tmNativeCheck{{"out", flat(want)}}
		cases = append(cases, c)
	}

	for _, causal := range []bool{false, true} {
		a := JaggedAttention{Causal: causal, MRoPE: MRoPESpec{HeadDim: 32, PairsT: 4, PairsH: 6, PairsW: 6, BaseTheta: 1e4}}
		name := "native_jagged"
		if causal {
			name += "_causal"
		}
		c := newTMNativeCase(t, name, func(n string) (string, error) { a.Name = n; return a.CUDA() },
			"const int kv_end = CAUSAL ? (q0 + qrows < seq1 ? q0 + qrows : seq1) : seq1;", "const int kv_end = tokens;")
		cu, q, k, v, pos := jaggedInputs(rand.New(rand.NewPCG(34, 3)), []int{73, 300, 190}, a.MRoPE.HeadDim)
		sched, err := BuildJaggedBatchSchedule(cu, 128)
		if err != nil {
			t.Fatal(err)
		}
		want, err := a.Reference(sched, q, k, v, pos)
		if err != nil {
			t.Fatal(err)
		}
		var tseq, tidx, p []int32
		for _, m := range sched.TileMap {
			tseq, tidx = append(tseq, int32(m[0])), append(tidx, int32(m[1]))
		}
		for _, x := range pos {
			p = append(p, x[:]...)
		}
		c.grid, c.block = a.Grid(sched, 1)
		c.smem = a.SharedBytes()
		c.args = []tmNativeArg{{name: "tokens", scalar: len(q)}, {name: "tile_seq", i32: tseq}, {name: "tile_idx", i32: tidx},
			{name: "cu_seqlens", i32: cu}, {name: "inv_freq", f32: a.MRoPE.InvFreq()}, {name: "pos", i32: p},
			{name: "q", f32: flat(q)}, {name: "k", f32: flat(k)}, {name: "v", f32: flat(v)}, {name: "out", out: len(q) * a.MRoPE.HeadDim}}
		c.checks = []tmNativeCheck{{"out", flat(want)}}
		cases = append(cases, c)
	}
	return cases
}

// tmNativeKernels returns Prelude followed by every case's kernel and
// mutant. It is the translation unit compiled for sm_100a.
func tmNativeKernels(cases []tmNativeCase) string {
	var b strings.Builder
	b.WriteString(Prelude)
	for _, c := range cases {
		b.WriteString(c.src)
		b.WriteString(c.mutSrc)
	}
	return b.String()
}

// buildTMNativeHarness returns a CUDA program that runs every case and its
// mutant on the current device and prints RECURRENT_NATIVE_OK
// recurrent_mutant_rejected=1 when all cases match and all mutants miss.
func buildTMNativeHarness(cases []tmNativeCase) string {
	var b strings.Builder
	b.WriteString("#define EG_TM_EMU_CTAS 64\n#include <cstdio>\n#include <cmath>\n#include <cstring>\n")
	b.WriteString(tmNativeKernels(cases))
	b.WriteString(`
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return -1.0; } } while (0)

static double rel_l2(const float* got, const float* want, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * (double)want[i];
    }
    return std::isfinite(num) ? std::sqrt(num / (den > 0 ? den : 1)) : INFINITY;
}

template <class T> static T* upload(const T* h, size_t n) {
    T* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

static float* poisoned(size_t n) {
    float* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return nullptr;
    cudaMemset(d, 0xff, n * sizeof(float));
    return d;
}
`)
	for _, c := range cases {
		for _, a := range c.args {
			switch {
			case a.f32 != nil:
				fmt.Fprintf(&b, "static const float h_%s_%s[] = {%s};\n", c.name, a.name, cFloatList(a.f32))
			case a.i32 != nil:
				fmt.Fprintf(&b, "static const int h_%s_%s[] = {%s};\n", c.name, a.name, cIntList(a.i32))
			}
		}
		for _, k := range c.checks {
			fmt.Fprintf(&b, "static const float want_%s_%s[] = {%s};\n", c.name, k.name, cFloatList(k.want))
		}
		// run_<case> returns the largest relative L2 error over the checks
		// (the first check only for the mutant), or -1 on a CUDA error.
		fmt.Fprintf(&b, "\nstatic double run_%s(bool mutant) {\n", c.name)
		var call []string
		for _, a := range c.args {
			switch {
			case a.f32 != nil:
				fmt.Fprintf(&b, "    float* d_%[2]s = upload(h_%[1]s_%[2]s, %[3]d);\n", c.name, a.name, len(a.f32))
			case a.i32 != nil:
				fmt.Fprintf(&b, "    int* d_%[2]s = upload(h_%[1]s_%[2]s, %[3]d);\n", c.name, a.name, len(a.i32))
			case a.out > 0:
				fmt.Fprintf(&b, "    float* d_%s = poisoned(%d);\n", a.name, a.out)
			default:
				call = append(call, fmt.Sprint(a.scalar))
				continue
			}
			call = append(call, "d_"+a.name)
		}
		fmt.Fprintf(&b, `    auto kern = mutant ? %[1]s_mut : %[1]s;
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, %[2]d));
    kern<<<dim3(%[3]d, %[4]d, %[5]d), dim3(%[6]d, %[7]d, %[8]d), %[2]d>>>(%[9]s);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    double worst = 0;
`, c.name, c.smem, c.grid[0], c.grid[1], c.grid[2], c.block[0], c.block[1], c.block[2], strings.Join(call, ", "))
		for i, k := range c.checks {
			fmt.Fprintf(&b, `    {
        static float got[%[3]d];
        CK(cudaMemcpy(got, d_%[2]s, sizeof got, cudaMemcpyDeviceToHost));
        const double e = rel_l2(got, want_%[1]s_%[2]s, %[3]d);
        std::printf("  %[1]s%%s %[2]s rel_l2=%%.3g\n", mutant ? "_mut" : "", e);
        if (%[4]d == 0 || !mutant) worst = e > worst ? e : worst;
    }
`, c.name, k.name, len(k.want), i)
		}
		for _, a := range c.args {
			if a.f32 != nil || a.i32 != nil || a.out > 0 {
				fmt.Fprintf(&b, "    cudaFree(d_%s);\n", a.name)
			}
		}
		b.WriteString("    return worst;\n}\n")
	}
	b.WriteString("\nint main() {\n    int bad = 0, mutants_missed = 0;\n")
	for _, c := range cases {
		fmt.Fprintf(&b, `    {
        const double e = run_%[1]s(false), m = run_%[1]s(true);
        const bool ok = e >= 0 && e <= %[2]g, rejected = m < 0 || m > 1e-2;
        std::printf("%%s %[1]s rel_l2=%%.3g mutant_rel_l2=%%.3g mutant_rejected=%%d\n", ok ? "PASS" : "FAIL", e, m, rejected ? 1 : 0);
        bad += !ok;
        mutants_missed += !rejected;
    }
`, c.name, c.tol)
	}
	b.WriteString(`    if (bad || mutants_missed) {
        std::printf("RECURRENT_NATIVE_FAIL bad=%d mutants_missed=%d\n", bad, mutants_missed);
        return 1;
    }
    std::printf("RECURRENT_NATIVE_OK recurrent_mutant_rejected=1\n");
    return 0;
}
`)
	return b.String()
}

func cFloatList(v []float32) string {
	var b strings.Builder
	for i, x := range v {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(floatLiteral(x))
	}
	return b.String()
}

func cIntList(v []int32) string {
	var b strings.Builder
	for i, x := range v {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprint(&b, x)
	}
	return b.String()
}

// TestTMNativeHarnessStructure checks on the host that the device harness
// covers every recurrent and multimodal kernel with a mutant, and that each
// case fits the sm_121a shared memory limit.
func TestTMNativeHarnessStructure(t *testing.T) {
	cases := tmNativeCases(t)
	src := buildTMNativeHarness(cases)
	for _, c := range cases {
		if c.smem > 99*1024 {
			t.Errorf("%s: %d bytes of shared memory exceed the sm_121a limit", c.name, c.smem)
		}
		if c.src == c.mutSrc || !strings.Contains(src, "run_"+c.name+"(true)") {
			t.Errorf("%s: no mutant run", c.name)
		}
	}
	for _, want := range []string{"native_ssm_mamba2", "native_ttt_titans", "native_lora", "native_tubelet", "native_jagged_causal", "RECURRENT_NATIVE_OK recurrent_mutant_rejected=1"} {
		if !strings.Contains(src, want) {
			t.Errorf("harness lacks %q", want)
		}
	}
}

// TestTMNativeQualification compiles the SSM, TTT, LoRA, tubelet and jagged
// attention kernels for sm_100a (PTX and cubin, when nvcc supports it) and
// runs the device harness on the current GPU. On sm_121a the kernels take
// the EG_EMU path (TMEM emulated in global memory), so a pass there
// qualifies the kernel logic, not tcgen05 hardware.
//
//	CUDA_TCGEN05_NATIVE=1 gpu-lock go test ./x/tcgen05 -run TestTMNativeQualification -v
func TestTMNativeQualification(t *testing.T) {
	if os.Getenv(tcgen05NativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_TCGEN05_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("user home dir: %v", err)
	}
	tmpBase := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(tmpBase, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", tmpBase, err)
	}
	dir, err := os.MkdirTemp(tmpBase, "tcgen05-recurrent-")
	if err != nil {
		t.Fatalf("mkdirtemp: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	cases := tmNativeCases(t)
	kernCU := filepath.Join(dir, "recurrent_kernels.cu")
	if err := os.WriteFile(kernCU, []byte(tmNativeKernels(cases)), 0o600); err != nil {
		t.Fatal(err)
	}
	if nvccSupportsTarget(t, nvcc, "sm_100a") {
		cubin := filepath.Join(dir, "recurrent_kernels_sm100a.cubin")
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
		defer cancel()
		if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_100a", "-cubin", kernCU, "-o", cubin).CombinedOutput(); err != nil {
			t.Fatalf("nvcc sm_100a cubin (CUDA %s): %v\n%s", cudaVer, err, out)
		}
		inspectCubinWithNVDisasm(t, cubin, "sm_100a")
	} else {
		t.Logf("nvcc (CUDA %s) lacks sm_100a; hardware tcgen05 path not compiled", cudaVer)
	}

	devArch := os.Getenv("CUDA_TCGEN05_NATIVE_ARCH")
	if devArch == "" {
		devArch = detectDeviceArchFlag(t, nvcc)
	}
	harnessCU := filepath.Join(dir, "recurrent_qualify.cu")
	harnessBin := filepath.Join(dir, "recurrent_qualify")
	if err := os.WriteFile(harnessCU, []byte(buildTMNativeHarness(cases)), 0o600); err != nil {
		t.Fatal(err)
	}
	buildCtx, buildCancel := context.WithTimeout(t.Context(), 10*time.Minute)
	defer buildCancel()
	if out, err := exec.CommandContext(buildCtx, nvcc, "-std=c++17", "-O2", "-arch="+devArch, harnessCU, "-o", harnessBin).CombinedOutput(); err != nil {
		t.Fatalf("nvcc device harness (%s, CUDA %s): %v\n%s", devArch, cudaVer, err, out)
	}
	runCtx, runCancel := context.WithTimeout(t.Context(), 10*time.Minute)
	defer runCancel()
	out, err := exec.CommandContext(runCtx, harnessBin).CombinedOutput()
	if err != nil {
		t.Fatalf("recurrent device harness (%s, CUDA %s): %v\n%s", devArch, cudaVer, err, out)
	}
	if !strings.Contains(string(out), "RECURRENT_NATIVE_OK recurrent_mutant_rejected=1") {
		t.Fatalf("recurrent device harness output missing markers:\n%s", out)
	}
	if err := classifyTCGen05DeviceArch(devArch); errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Logf("device arch %s has no TMEM (%v); ran the EG_EMU path:\n%s", devArch, err, strings.TrimSpace(string(out)))
	} else {
		t.Logf("recurrent native qualification passed on %s (CUDA %s):\n%s", devArch, cudaVer, strings.TrimSpace(string(out)))
	}
}

// passGuardedDeviceGlobals returns the __device__ variable declarations in
// src that sit inside a preprocessor conditional naming __CUDA_ARCH__ or
// EG_EMU. nvcc's host pass registers every device global, so such a
// declaration compiles in the device pass and fails in the host pass. A
// branch of a conditional counts as guarded by every condition of that
// conditional before it.
func passGuardedDeviceGlobals(src string) []string {
	var stack []string
	var bad []string
	for _, line := range strings.Split(src, "\n") {
		l := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(l, "#if"):
			stack = append(stack, l)
		case strings.HasPrefix(l, "#el") && len(stack) > 0:
			stack[len(stack)-1] += " " + l
		case strings.HasPrefix(l, "#endif") && len(stack) > 0:
			stack = stack[:len(stack)-1]
		case strings.Contains(l, "__device__") && !strings.HasPrefix(l, "#") && !strings.HasPrefix(l, "//") &&
			!strings.Contains(l, "(") && strings.HasSuffix(l, ";"):
			for _, cond := range stack {
				c := strings.NewReplacer("EG_HOST_EMU", "", "EG_EMU_", "").Replace(cond)
				if strings.Contains(c, "__CUDA_ARCH__") || strings.Contains(c, "EG_EMU") {
					bad = append(bad, l+"  (under "+cond+")")
					break
				}
			}
		}
	}
	return bad
}

func TestTMNativeDeviceGlobalsUnguarded(t *testing.T) {
	for _, tt := range []struct {
		name, src string
		bad       int
	}{
		{"harness", buildTMNativeHarness(tmNativeCases(t)), 0},
		{"guarded", "#if defined(EG_EMU)\nstatic __device__ float x[4];\n#endif\n", 1},
		{"else of arch", "#ifdef __CUDA_ARCH__\n#else\n__device__ int y;\n#endif\n", 1},
		{"host emu else", "#if defined(EG_HOST_EMU)\n#else\nstatic __device__ float z[4];\n#endif\n", 0},
		{"elif emu", "#if defined(EG_HOST_EMU)\n#elif defined(EG_EMU)\nstatic __device__ float w[4];\n#endif\n", 1},
	} {
		if got := passGuardedDeviceGlobals(tt.src); len(got) != tt.bad {
			t.Errorf("%s: pass-guarded device globals %q, want %d", tt.name, got, tt.bad)
		}
	}
}
