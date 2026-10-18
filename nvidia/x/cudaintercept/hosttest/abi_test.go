//go:build darwin || linux

package hosttest

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The real-header check reads cuda.h from an operator-supplied absolute path
// and verifies its SHA-256 before compiling. No header is vendored.
const (
	cudaHeaderEnv = "CUDAINTERCEPT_CUDA_H"
	cudaSHA256Env = "CUDAINTERCEPT_CUDA_H_SHA256"
)

// compileABI writes testdata/abi.c, observe (as observe.h) and files into a
// new directory and runs the C compiler there with args before abi.c.
func compileABI(t *testing.T, observe string, files map[string]string, args ...string) (string, error) {
	t.Helper()
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal("ABI check requires a C compiler:", err)
	}
	src, err := os.ReadFile("testdata/abi.c")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	all := map[string]string{"abi.c": string(src), "observe.h": observe}
	for name, data := range files {
		all[name] = data
	}
	for name, data := range all {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	args = append([]string{"-std=c11", "-Wall", "-Wextra", "-Werror", "-I", dir}, args...)
	args = append(args, filepath.Join(dir, "abi.c"))
	t.Logf("compile: %s %v", cc, args)
	out, err := exec.Command(cc, args...).CombinedOutput()
	return string(out), err
}

// edit applies each edit to s. Each old text must occur exactly once.
func edit(t *testing.T, s string, edits ...[2]string) string {
	t.Helper()
	for _, e := range edits {
		if n := strings.Count(s, e[0]); n != 1 {
			t.Fatalf("mutant precondition: %q occurs %d times, want 1", e[0], n)
		}
		s = strings.Replace(s, e[0], e[1], 1)
	}
	return s
}

func readObserveH(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("../observe.h")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestRecordLayout(t *testing.T) {
	observe := readObserveH(t)
	if out, err := compileABI(t, observe, nil, "-fsyntax-only"); err != nil {
		t.Fatalf("record layout: %v\n%s", err, out)
	}
	for _, tt := range []struct {
		name  string
		edits [][2]string
	}{
		{"swap_function_stream", [][2]string{
			{"    uint64_t function;", "    uint64_t stream_;"},
			{"    uint64_t stream;", "    uint64_t function;"},
			{"    uint64_t stream_;", "    uint64_t stream;"},
		}},
		{"widen_reserved", [][2]string{{"    uint32_t reserved;", "    uint64_t reserved;"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mutant := edit(t, observe, tt.edits...)
			out, err := compileABI(t, mutant, nil, "-fsyntax-only")
			if err == nil || !strings.Contains(out, "static assertion failed") {
				t.Fatalf("layout mutant did not fail a static assertion: %v\n%s", err, out)
			}
		})
	}
}

var (
	typedefRE = regexp.MustCompile(`(?s)typedef (\w[\w ]*?) \(\*cuLaunchKernel_t\)\(([^;]*)\);`)
	paramRE   = regexp.MustCompile(`^(.*?)\s*\w+$`)
	failRE    = regexp.MustCompile(`return (\d+); /\* CUDA_ERROR_NOT_INITIALIZED`)
	symbolRE  = regexp.MustCompile(`"cudaintercept-symbol=" "(\w+)"`)
)

// shimABI returns shim-abi.h for the given shim source: the typedef's
// return and parameter types and the fail-closed literal. It reads only the
// typedef; interpose.c asserts that the exported cuLaunchKernel definition
// has the typedef's type, which TestShimDefinition checks.
func shimABI(shim string) (string, error) {
	m := typedefRE.FindStringSubmatch(shim)
	if m == nil {
		return "", fmt.Errorf("no cuLaunchKernel_t typedef")
	}
	var b strings.Builder
	fmt.Fprintf(&b, "#define SHIM_RET %s\n", m[1])
	params := strings.Split(m[2], ",")
	if len(params) != 11 {
		return "", fmt.Errorf("typedef has %d parameters, want 11", len(params))
	}
	for i, p := range params {
		pm := paramRE.FindStringSubmatch(strings.TrimSpace(p))
		if pm == nil || pm[1] == "" {
			return "", fmt.Errorf("cannot parse parameter %q", p)
		}
		fmt.Fprintf(&b, "#define SHIM_P%d %s\n", i, pm[1])
	}
	lits := failRE.FindAllStringSubmatch(shim, -1)
	if len(lits) == 0 {
		return "", fmt.Errorf("no fail-closed literal")
	}
	for _, l := range lits[1:] {
		if l[1] != lits[0][1] {
			return "", fmt.Errorf("fail-closed literals disagree: %s and %s", lits[0][1], l[1])
		}
	}
	fmt.Fprintf(&b, "#define SHIM_NOT_INITIALIZED %s\n", lits[0][1])
	return b.String(), nil
}

func TestShimABI(t *testing.T) {
	const want = `#define SHIM_RET int
#define SHIM_P0 void*
#define SHIM_P1 unsigned int
#define SHIM_P2 unsigned int
#define SHIM_P3 unsigned int
#define SHIM_P4 unsigned int
#define SHIM_P5 unsigned int
#define SHIM_P6 unsigned int
#define SHIM_P7 unsigned int
#define SHIM_P8 void*
#define SHIM_P9 void**
#define SHIM_P10 void**
#define SHIM_NOT_INITIALIZED 3
`
	shim := shimSource(t)
	got, err := shimABI(shim)
	if err != nil || got != want {
		t.Fatalf("shimABI = %q, %v; want %q", got, err, want)
	}
	for _, tt := range []struct {
		name  string
		edits [][2]string
	}{
		{"literals_disagree", [][2]string{{"if (resolving) return 3;", "if (resolving) return 4;"}}},
		{"missing_parameter", [][2]string{{"void* hStream, void** kernelParams, void** extra);", "void* hStream, void** kernelParams);"}}},
		{"no_typedef", [][2]string{{"(*cuLaunchKernel_t)", "(*launch_t)"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := shimABI(edit(t, shim, tt.edits...)); err == nil {
				t.Fatalf("shimABI accepted a broken shim:\n%s", got)
			}
		})
	}
}

// TestShimDefinition checks that interpose.c compiles on its own and that
// its exported cuLaunchKernel definition cannot drift from cuLaunchKernel_t,
// the type the ABI checks read. It needs no cuda.h.
func TestShimDefinition(t *testing.T) {
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal("ABI check requires a C compiler:", err)
	}
	observe := readObserveH(t)
	compile := func(shim string) (string, error) {
		dir := t.TempDir()
		for name, data := range map[string]string{"interpose.c": shim, "observe.h": observe} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
		}
		args := []string{"-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", filepath.Join(dir, "interpose.c")}
		t.Logf("compile: %s %v", cc, args)
		out, err := exec.Command(cc, args...).CombinedOutput()
		return string(out), err
	}
	shim := shimSource(t)
	if out, err := compile(shim); err != nil {
		t.Fatalf("shim: %v\n%s", err, out)
	}
	for _, tt := range []struct {
		name  string
		edits [][2]string
	}{
		{"definition_shmem_unsigned_long", [][2]string{{
			"unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra) {",
			"unsigned long sharedMemBytes, void* hStream, void** kernelParams, void** extra) {",
		}}},
		{"typedef_shmem_unsigned_long", [][2]string{{
			"unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra);",
			"unsigned long sharedMemBytes, void* hStream, void** kernelParams, void** extra);",
		}}},
		{"definition_returns_long", [][2]string{{"\nint cuLaunchKernel(", "\nlong cuLaunchKernel("}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out, err := compile(edit(t, shim, tt.edits...))
			if err == nil || !strings.Contains(out, "cuLaunchKernel definition differs from cuLaunchKernel_t") {
				t.Fatalf("mutant did not fail the definition assertion: %v\n%s", err, out)
			}
		})
	}
}

// TestRealHeaderABI checks the shim typedef against a real cuda.h. Without
// the header it skips with status=unavailable: that outcome proves nothing
// about the ABI and must not be counted as a pass.
func TestRealHeaderABI(t *testing.T) {
	path, want := os.Getenv(cudaHeaderEnv), os.Getenv(cudaSHA256Env)
	if path == "" && want == "" {
		t.Skipf("status=unavailable: %s and %s are unset; the real cuda.h ABI was not checked", cudaHeaderEnv, cudaSHA256Env)
	}
	if path == "" || want == "" || !filepath.IsAbs(path) {
		t.Fatalf("%s must be an absolute path and %s must be set", cudaHeaderEnv, cudaSHA256Env)
	}
	header, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(header)
	if got := hex.EncodeToString(sum[:]); got != strings.ToLower(want) {
		t.Fatalf("%s sha256 = %s, want %s", path, got, want)
	}
	t.Logf("status=checked header=%s sha256=%s", path, want)
	shim := shimSource(t)
	observe := readObserveH(t)
	check := func(shim string, args ...string) (string, error) {
		abi, err := shimABI(shim)
		if err != nil {
			t.Fatal(err)
		}
		files := map[string]string{"cuda.h": string(header), "shim-abi.h": abi}
		return compileABI(t, observe, files, append([]string{"-DCUDAINTERCEPT_HAVE_CUDA_H"}, args...)...)
	}

	t.Run("layout", func(t *testing.T) {
		if out, err := check(shim, "-fsyntax-only"); err != nil {
			t.Fatalf("ABI: %v\n%s", err, out)
		}
	})
	t.Run("symbol", func(t *testing.T) {
		for _, tt := range []struct {
			name string
			args []string
			want string
		}{
			{"default", nil, "cuLaunchKernel"},
			// The shim does not interpose cuLaunchKernel_ptsz.
			{"per_thread_default_stream", []string{"-DCUDA_API_PER_THREAD_DEFAULT_STREAM"}, "cuLaunchKernel_ptsz"},
		} {
			out, err := check(shim, append(tt.args, "-E")...)
			if err != nil {
				t.Fatalf("%s: %v\n%s", tt.name, err, out)
			}
			m := symbolRE.FindStringSubmatch(out)
			if m == nil || m[1] != tt.want {
				t.Fatalf("%s: cuLaunchKernel expands to %v, want %s", tt.name, m, tt.want)
			}
		}
	})
	t.Run("mutants", func(t *testing.T) {
		for _, tt := range []struct {
			name  string
			edits [][2]string
		}{
			{"shmem_unsigned_long", [][2]string{{
				"unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra);",
				"unsigned long sharedMemBytes, void* hStream, void** kernelParams, void** extra);",
			}}},
			{"fail_closed_literal", [][2]string{
				{"if (resolving) return 3;", "if (resolving) return 4;"},
				{"if (!real_cuLaunchKernel) return 3;", "if (!real_cuLaunchKernel) return 4;"},
			}},
		} {
			out, err := check(edit(t, shim, tt.edits...), "-fsyntax-only")
			if err == nil || !strings.Contains(out, "static assertion failed") {
				t.Fatalf("%s: mutant did not fail a static assertion: %v\n%s", tt.name, err, out)
			}
		}
	})
}
