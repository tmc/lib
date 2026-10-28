package expertparallel

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

const nativeEnv = "CUDA_EGRAPH_EP_NATIVE"

func nativeArch() string {
	if arch := os.Getenv("CUDA_EGRAPH_EP_NATIVE_ARCH"); arch != "" {
		return arch
	}
	return "sm_121a"
}

// nativeRun is one plan the native binary runs: a host statement over
// device buffers named after the inputs, writing the BF16 buffer out.
type nativeRun struct {
	name, call string
	outBytes   int
	wsBytes    int
	temps      []Tensor
}

// nativeProgram returns a CUDA program that loads each input from
// dir/<name>.bin into device memory, runs each nativeRun into a fresh
// 0xff-filled output and workspace, and writes dir/<run>.out.
func nativeProgram(src, dir string, inputs []Tensor, runs []nativeRun) string {
	var b strings.Builder
	b.WriteString("#include <cstdio>\n#include <cstdlib>\n#include <vector>\n" + src)
	b.WriteString(`
static void* ep_load(const char* path, size_t n) {
    std::vector<char> h(n);
    FILE* f = fopen(path, "rb");
    if (!f || fread(h.data(), 1, n, f) != n) { fprintf(stderr, "read %s\n", path); exit(1); }
    fclose(f);
    void* d;
    cudaMalloc(&d, n);
    cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice);
    return d;
}
static void* ep_poison(size_t n) {
    void* d;
    cudaMalloc(&d, n ? n : 1);
    cudaMemset(d, 0xff, n);
    return d;
}
static void ep_save(const char* path, const void* d, size_t n) {
    std::vector<char> h(n);
    cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost);
    FILE* f = fopen(path, "wb");
    fwrite(h.data(), 1, n, f);
    fclose(f);
}
static void ep_check(const char* what) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}
int main() {
`)
	for _, t := range inputs {
		n := size(t.Shape) * t.DType.Bits() / 8
		fmt.Fprintf(&b, "    auto* %s = (%s*)ep_load(%q, %d);\n", t.Name, ctype(t.DType), filepath.Join(dir, t.Name+".bin"), n)
	}
	for _, r := range runs {
		fmt.Fprintf(&b, "    {\n        auto* out = (uint16_t*)ep_poison(%d);\n        void* ws = ep_poison(%d);\n", r.outBytes, r.wsBytes)
		for _, t := range r.temps {
			fmt.Fprintf(&b, "        auto* %s = (%s*)ep_poison(%d);\n", t.Name, ctype(t.DType), size(t.Shape)*t.DType.Bits()/8)
		}
		fmt.Fprintf(&b, "        %s;\n        ep_check(%q);\n        ep_save(%q, out, %d);\n    }\n",
			r.call, r.name, filepath.Join(dir, r.name+".out"), r.outBytes)
	}
	b.WriteString("    printf(\"EP_NATIVE_OK\\n\");\n    return 0;\n}\n")
	return b.String()
}

// TestNative runs EP=1, 2 and 4 on one GPU, with and without a shared
// expert, plus the single-rank hopper plan of [Config.Dense], and checks
// that every EP=N output equals EP=1 bit for bit, that the routed-only EP
// plans equal the dense plan, and that all agree with the CPU reference.
func TestNative(t *testing.T) {
	if os.Getenv(nativeEnv) != "1" {
		t.Skip("native EP run pending: " + nativeEnv + "=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	arch := nativeArch()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "ep-native-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })

	const tokens, hidden, inter, shared = 64, 256, 128, 128
	full := testConfig(1, tokens, hidden, inter, shared)
	env := testEnv(full, 11)
	inputs := full.Inputs()
	inputs = append(inputs, Tensor{Name: "local_start", DType: hopper.I32, Shape: []int{1}})
	env["local_start"] = hopper.Value{Shape: []int{1}, Data: []float64{0}}
	for _, in := range inputs {
		if err := os.WriteFile(filepath.Join(dir, in.Name+".bin"), encode(in.DType, env[in.Name].Data), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	var src strings.Builder
	var runs []nativeRun
	refs := make(map[string][]float32)
	outBytes := tokens * hidden * 2
	for _, sh := range []int{0, shared} {
		for _, ranks := range []int{1, 2, 4} {
			cfg := testConfig(ranks, tokens, hidden, inter, sh)
			name := fmt.Sprintf("ep%d_s%d", ranks, sh)
			p, err := New(cfg, name)
			if err != nil {
				t.Fatal(err)
			}
			s, err := p.Source()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			var args []string
			for _, in := range cfg.Inputs() {
				args = append(args, in.Name)
			}
			runs = append(runs, nativeRun{
				name:     name,
				call:     fmt.Sprintf("%s_run(%s, out, ws)", name, strings.Join(args, ", ")),
				outBytes: outBytes,
				wsBytes:  p.WorkspaceBytes(),
			})
			if refs[name], err = p.Reference(env); err != nil {
				t.Fatal(err)
			}
		}
	}
	b, root, err := testConfig(1, tokens, hidden, inter, 0).Dense()
	if err != nil {
		t.Fatal(err)
	}
	e, err := hopper.Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	dense, err := hopper.Compile(e, "dense", hopper.BF16)
	if err != nil {
		t.Fatal(err)
	}
	ds, err := dense.Source()
	if err != nil {
		t.Fatal(err)
	}
	src.WriteString(ds)
	var dargs []string
	for _, in := range dense.Inputs {
		dargs = append(dargs, in.Name)
	}
	for _, tmp := range dense.Temps {
		dargs = append(dargs, tmp.Name)
	}
	runs = append(runs, nativeRun{
		name:     "dense",
		call:     fmt.Sprintf("dense_run(%s, out, nullptr)", strings.Join(dargs, ", ")),
		outBytes: outBytes,
		temps:    dense.Temps,
	})

	cu := filepath.Join(dir, "ep.cu")
	bin := filepath.Join(dir, "ep")
	if err := os.WriteFile(cu, []byte(nativeProgram(src.String(), dir, inputs, runs)), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O2", "-arch="+arch, cu, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("nvcc %s (CUDA %s): %v\n%s", arch, cudaVer, err, out)
	}
	out, err := exec.CommandContext(ctx, bin).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "EP_NATIVE_OK") {
		t.Fatalf("native EP binary: %v\n%s", err, out)
	}

	got := make(map[string][]float32)
	for _, r := range runs {
		raw, err := os.ReadFile(filepath.Join(dir, r.name+".out"))
		if err != nil {
			t.Fatal(err)
		}
		got[r.name] = decodeBF16(raw)
	}
	for _, sh := range []int{0, shared} {
		one := got[fmt.Sprintf("ep1_s%d", sh)]
		for _, ranks := range []int{1, 2, 4} {
			name := fmt.Sprintf("ep%d_s%d", ranks, sh)
			g := got[name]
			if i := firstDiff(g, one); i >= 0 {
				t.Errorf("%s differs from EP=1 at %d: %g vs %g (%d of %d differ)", name, i, g[i], one[i], countDiff(g, one), len(g))
			}
			e := relErr(g, refs[name])
			if e > 5e-3 || math.IsNaN(e) {
				t.Errorf("%s vs CPU reference: relative L2 error %.2e > 5e-3", name, e)
			}
			t.Logf("%s: %d/%d BF16 outputs differ from EP=1; relative L2 error vs CPU reference %.2e", name, countDiff(g, one), len(g), e)
		}
	}
	d, one := got["dense"], got["ep1_s0"]
	t.Logf("dense hopper plan vs EP=1 routed-only: %d/%d BF16 outputs differ, max |diff| %.3g", countDiff(d, one), len(d), maxAbsDiff(d, one))
	if i := firstDiff(d, one); i >= 0 {
		t.Errorf("dense plan differs from EP=1 at %d: %g vs %g", i, d[i], one[i])
	}
	t.Logf("arch %s, CUDA %s", arch, cudaVer)
}

func countDiff(a, b []float32) int {
	n := 0
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			n++
		}
	}
	return n
}

func maxAbsDiff(a, b []float32) float64 {
	m := 0.0
	for i := range a {
		m = max(m, math.Abs(float64(a[i])-float64(b[i])))
	}
	return m
}

// TestCompileTargets compiles the EP=4 plan with nvcc for GB10 and the
// Blackwell datacenter targets (sm_100a, sm_103a). It needs nvcc but no GPU.
func TestCompileTargets(t *testing.T) {
	if os.Getenv(nativeEnv) != "1" {
		t.Skip(nativeEnv + "=1 required")
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	p, err := New(testConfig(4, 64, 256, 128, 128), "ep")
	if err != nil {
		t.Fatal(err)
	}
	src, err := p.Source()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	cu := filepath.Join(dir, "ep.cu")
	// A kernel-only TU: the host run function is inline and unused.
	if err := os.WriteFile(cu, []byte(src), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, arch := range []string{"sm_121a", "sm_100a", "sm_103a"} {
		out, err := exec.CommandContext(t.Context(), nvcc, "-std=c++17", "-O2", "-cubin", "-arch="+arch,
			"-Xptxas=-v", cu, "-o", filepath.Join(dir, arch+".cubin")).CombinedOutput()
		if err != nil {
			t.Errorf("nvcc %s (CUDA %s): %v\n%s", arch, cudaVer, err, out)
			continue
		}
		var spills []string
		for _, line := range strings.Split(string(out), "\n") {
			if strings.Contains(line, "spill") && !strings.Contains(line, " 0 bytes spill stores, 0 bytes spill loads") {
				spills = append(spills, strings.TrimSpace(line))
			}
		}
		t.Logf("%s: compiled; nonzero spill lines: %d", arch, len(spills))
		for _, s := range spills {
			t.Log(s)
		}
	}
}
