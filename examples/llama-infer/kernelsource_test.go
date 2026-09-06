package main

import (
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"
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

	// Not bound by Kernels: a conversion probe used only by the native tests,
	// which reach it through the module directly. It is listed here because it
	// is a weight-reading kernel and must have all three variants.
	"WeightConvertProbe",
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

// vectorLoop matches a strided loop and captures the identifier bounding it.
//
// The initializer is deliberately left open: block-strided loops start at tid,
// grid-strided ones at blockIdx.x * blockDim.x + threadIdx.x, and an earlier
// version of this pattern that only matched the first form missed the
// vectorized load in EmbeddingLookupImpl entirely.
var vectorLoop = regexp.MustCompile(`for \(int i = [^;]+; i < (\w+); i \+= [^)]+\) \{`)

// declStart marks the beginning of a kernel template or a plain kernel.
//
// Everything between one of these and the next is the body of a single kernel,
// which is the scope a declaration and the loops that use it share.
var declStart = regexp.MustCompile(`(?m)^(?:template <class W>|extern "C" __global__)`)

// A kernelRegion is one kernel's source text and the name it is declared with.
type kernelRegion struct {
	name string
	body string
}

// kernelRegions splits forward.cu into one region per kernel.
//
// Scoping matters: an earlier version of the checks below built a single
// file-wide map keyed on bare identifier names, so two kernels declaring the
// same name were indistinguishable. forward.cu declares n4 twice and in_dim4
// twice, once guarded and once not, and a file-wide map recorded each as
// guarded on the strength of the other kernel's declaration. Rewriting one of
// the two into an unguarded vector load then left the check green. Regions
// remove that: a declaration only speaks for the kernel it is written in.
func kernelRegions(t *testing.T, text string) []kernelRegion {
	t.Helper()
	starts := declStart.FindAllStringIndex(text, -1)
	if len(starts) == 0 {
		t.Fatal("no kernel declarations found; the pattern no longer matches the source")
	}
	name := regexp.MustCompile(`(?:void|__forceinline__ void)\s+(\w+)`)
	regions := make([]kernelRegion, 0, len(starts))
	for i, loc := range starts {
		end := len(text)
		if i+1 < len(starts) {
			end = starts[i+1][0]
		}
		body := text[loc[0]:end]
		n := "?"
		if m := name.FindStringSubmatch(body); m != nil {
			n = m[1]
		}
		regions = append(regions, kernelRegion{name: n, body: body})
	}
	return regions
}

// vectorBounds returns the identifiers in body that bound a vectorized loop,
// mapped to whether every declaration of them guards with W::aligned4.
//
// A bound qualifies when some declaration of it divides by four or consults
// aligned4, which is how every vector count in forward.cu is written and how a
// new one would be. It counts as guarded only when all of its declarations
// consult aligned4: forward.cu declares n4 twice in one kernel, and taking the
// guarded one as speaking for both is how an unguarded rewrite of the other
// would slip through.
func vectorBounds(body string) map[string]bool {
	decl := regexp.MustCompile(`int (\w+) = ([^;]+);`)
	inits := make(map[string][]string)
	for _, m := range decl.FindAllStringSubmatch(body, -1) {
		inits[m[1]] = append(inits[m[1]], m[2])
	}
	bounds := make(map[string]bool)
	for name, list := range inits {
		var vector, guarded bool
		guarded = true
		for _, init := range list {
			a := strings.Contains(init, "aligned4")
			if a || strings.Contains(init, "/ 4") {
				vector = true
			}
			guarded = guarded && a
		}
		if vector {
			bounds[name] = guarded
		}
	}
	return bounds
}

// loopBody returns the text of the loop whose header ends at off.
func loopBody(text string, off int) string {
	depth := 0
	for i := off - 1; i < len(text); i++ {
		switch text[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return text[off:i]
			}
		}
	}
	return text[off:]
}

// TestVectorLoadsAreAlignmentGuarded checks that every vectorized weight loop
// is bounded by a count that an alignment test can zero.
//
// A row base w + row*in_dim is aligned to a four-element vector only when
// in_dim is a multiple of four, so a vector load off an arbitrary row faults on
// exactly the dimensions no current model happens to use. That makes the defect
// silent until it is not, which is why it is pinned in the source rather than
// left to a native test that only runs on a machine with a GPU.
//
// The rule: if a loop body performs a W::load4, the identifier bounding that
// loop must have been declared with W::aligned4 in its initializer, in the same
// kernel. The alignment test drives the count to zero, and the scalar remainder
// loop that follows then covers the whole row.
func TestVectorLoadsAreAlignmentGuarded(t *testing.T) {
	src, err := os.ReadFile("forward.cu")
	if err != nil {
		t.Fatal(err)
	}
	var checked int
	for _, r := range kernelRegions(t, string(src)) {
		bounds := vectorBounds(r.body)
		for _, loc := range vectorLoop.FindAllStringSubmatchIndex(r.body, -1) {
			bound := r.body[loc[2]:loc[3]]
			if !strings.Contains(loopBody(r.body, loc[1]), "W::load4(") {
				continue
			}
			checked++
			if !bounds[bound] {
				t.Errorf("%s: vector loop bounded by %q performs W::load4 but %q is not declared with aligned4 in this kernel",
					r.name, bound, bound)
			}
		}
	}
	if checked < 6 {
		t.Fatalf("only %d vectorized weight loops found; expected at least six: EmbeddingLookup, RMSNorm, LinearGEMV, FusedRMSNormQKV (q and kv) and FusedSwiGLUDown", checked)
	}
	t.Logf("%d vectorized weight loops, all alignment-guarded", checked)
}

// TestVectorLoopsHaveScalarTails checks that every vectorized loop is followed
// by a scalar loop covering the elements past the last full vector.
//
// FusedRMSNormQKV and FusedSwiGLUDown dropped their last in_dim%4 elements
// entirely, in float32 as well as half, which is a wrong answer rather than an
// imprecise one.
//
// The check is per bound rather than per kernel. An earlier version searched a
// whole kernel body for one fixed tail spelling, so FusedRMSNormQKVImpl, which
// needs three tails, passed on any one of them: deleting the Q-branch tail left
// it green. It also missed EmbeddingLookupImpl entirely, whose tail is
// grid-strided and does not spell "+ tid". Requiring a tail loop starting at
// bound*4 for each distinct vector bound covers both forms and every kernel,
// including the ones that read no weights.
func TestVectorLoopsHaveScalarTails(t *testing.T) {
	src, err := os.ReadFile("forward.cu")
	if err != nil {
		t.Fatal(err)
	}
	// A tail loop resumes at the first element past the last full vector.
	tail := regexp.MustCompile(`for \(int i = (\w+) \* 4 \+ [^;]+; i < [^;]+;`)

	var checked int
	for _, r := range kernelRegions(t, string(src)) {
		bounds := vectorBounds(r.body)
		if len(bounds) == 0 {
			continue
		}
		tails := make(map[string]int)
		for _, m := range tail.FindAllStringSubmatch(r.body, -1) {
			tails[m[1]]++
		}
		// One vector loop per use of a bound, and one tail for each.
		used := make(map[string]int)
		for _, loc := range vectorLoop.FindAllStringSubmatchIndex(r.body, -1) {
			b := r.body[loc[2]:loc[3]]
			if _, ok := bounds[b]; ok {
				used[b]++
			}
		}
		for b, n := range used {
			checked += n
			if tails[b] < n {
				t.Errorf("%s: %d vectorized loops bounded by %q but %d scalar tails resuming at %s*4; elements past the last full vector are dropped",
					r.name, n, b, tails[b], b)
			}
		}
		for b := range bounds {
			if used[b] == 0 {
				t.Errorf("%s: %q is declared as a vector count but bounds no loop", r.name, b)
			}
		}
	}
	// EmbeddingLookup, RMSNorm (x and weight), LinearGEMV, StoreKVCache,
	// SwiGLU, AddResidual, FusedRMSNormQKV (rms, q, kv) and FusedSwiGLUDown.
	if checked < 11 {
		t.Fatalf("only %d vectorized loops found; expected at least eleven", checked)
	}
	t.Logf("%d vectorized loops, each with a scalar tail", checked)
}
