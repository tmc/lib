package main

import (
	"os"
	"os/exec"
	"regexp"
	"sort"
	"testing"
)

// weightKernelBases is the set of kernels that read model weights, and so must
// exist in one entry point per weight dtype.
//
// It is written out rather than derived from the Kernels struct because that
// struct is behind a build tag, and this check must run wherever the source
// does. The linkage between the two is asserted by
// TestKernelNamesMatchSource in the CUDA build.
var weightKernelBases = []string{
	"EmbeddingLookup",
	"RMSNorm",
	"LinearGEMV",
	"FusedRMSNormQKV",
	"FusedSwiGLUDown",
	"BatchEmbeddingLookup",
	"BatchRMSNorm",
	"BatchLinearGEMV",
	"BatchAddBias",
	"BatchFusedRMSNormQKV",
	"BatchFusedSwiGLUDown",
}

// plainKernelNames read no weights and have exactly one entry point each.
//
// They are listed so that a kernel accidentally gaining or losing weight
// variants is caught in both directions.
var plainKernelNames = []string{
	"ApplyRoPE",
	"StoreKVCache",
	"SwiGLU",
	"AddResidual",
	"BatchSwiGLU",
	"BatchAddResidual",
	"BatchArgmax",
}

var externCKernel = regexp.MustCompile(`extern\s+"C"\s+__global__\s+void\s+(\w+)`)

// kernelSourceNames returns the entry points forward.cu defines.
//
// The source is preprocessed first, because the weight kernels are emitted by
// the WEIGHT_KERNELS macro: reading the raw text would find the macro body and
// none of the names it expands to. forward.cu includes no headers, by
// construction, since NVRTC compiles it without an include path, so the
// preprocessor needs nothing beyond the file itself and leaves the CUDA
// keywords in place for the pattern to match.
func kernelSourceNames(t *testing.T) map[string]bool {
	t.Helper()
	cxx, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("clang++ not found; kernel enumeration skipped")
	}
	src, err := exec.Command(cxx, "-E", "-x", "c++", "forward.cu").Output()
	if err != nil {
		t.Fatalf("preprocess forward.cu: %v", err)
	}
	names := make(map[string]bool)
	for _, m := range externCKernel.FindAllStringSubmatch(string(src), -1) {
		if names[m[1]] {
			t.Errorf("kernel %s is defined twice", m[1])
		}
		names[m[1]] = true
	}
	if len(names) == 0 {
		t.Fatal("no extern \"C\" kernels found; the pattern no longer matches the source")
	}
	return names
}

// TestForwardSourceDefinesEveryWeightVariant checks that each weight-consuming
// kernel has an f32, f16 and bf16 entry point, and that the kernels which read
// no weights have exactly one.
//
// This is what makes a dispatch failure a load-time error rather than a
// silent fallback: bindWeightKernels appends the dtype suffix and refuses a
// missing name, so a variant absent from the source fails the module load
// instead of quietly running in f32.
func TestForwardSourceDefinesEveryWeightVariant(t *testing.T) {
	names := kernelSourceNames(t)
	want := make(map[string]bool)
	for _, base := range weightKernelBases {
		for _, suffix := range []string{"", "F16", "BF16"} {
			n := base + suffix
			want[n] = true
			if !names[n] {
				t.Errorf("forward.cu does not define %s", n)
			}
		}
	}
	for _, n := range plainKernelNames {
		want[n] = true
		if !names[n] {
			t.Errorf("forward.cu does not define %s", n)
		}
		for _, suffix := range []string{"F16", "BF16"} {
			if names[n+suffix] {
				t.Errorf("%s reads no weights but defines %s%s", n, n, suffix)
			}
		}
	}
	var extra []string
	for n := range names {
		if !want[n] {
			extra = append(extra, n)
		}
	}
	sort.Strings(extra)
	if len(extra) > 0 {
		t.Errorf("forward.cu defines kernels this test does not account for: %v", extra)
	}
	t.Logf("%d entry points: %d weight kernels x 3 dtypes + %d without weights",
		len(names), len(weightKernelBases), len(plainKernelNames))
}

// TestForwardSourceCompilesOnHost parses forward.cu with a host C++ compiler.
//
// It catches a syntax or type error in the templated kernels without a GPU, on
// the machine writing the code rather than on the machine running it. The shim
// supplies the CUDA builtins the source uses; the inline PTX is stubbed out,
// since a host compiler cannot assemble it. A GPU is still required to prove
// the kernels compute correctly, and this check does not claim otherwise.
func TestForwardSourceCompilesOnHost(t *testing.T) {
	cxx, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("clang++ not found; host syntax check skipped")
	}
	cmd := exec.Command(cxx,
		"-std=c++14", "-fsyntax-only", "-Wno-unused-value",
		"-include", "testdata/cudashim.h",
		"-D", "asm(x)=(void)0",
		"-x", "c++", "forward.cu")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("clang++ -fsyntax-only forward.cu failed: %v\n%s", err, out)
	}

	// Negative control: the same command must fail on deliberately broken
	// source, or a passing result above would mean nothing.
	src, err := os.ReadFile("forward.cu")
	if err != nil {
		t.Fatal(err)
	}
	broken := append([]byte(nil), src...)
	broken = append(broken, []byte("\nextern \"C\" __global__ void BrokenControl() { undefined_symbol_xyz(1); }\n")...)
	dir := t.TempDir()
	path := dir + "/broken.cu"
	if err := os.WriteFile(path, broken, 0o644); err != nil {
		t.Fatal(err)
	}
	shim, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctl := exec.Command(cxx,
		"-std=c++14", "-fsyntax-only", "-Wno-unused-value",
		"-include", shim+"/testdata/cudashim.h",
		"-D", "asm(x)=(void)0",
		"-x", "c++", path)
	if out, err := ctl.CombinedOutput(); err == nil {
		t.Fatalf("negative control compiled; the syntax check proves nothing\n%s", out)
	}
}
