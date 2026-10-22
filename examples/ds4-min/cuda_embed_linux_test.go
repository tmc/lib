//go:build linux

package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCUDAEmbeddedSources(t *testing.T) {
	sources := []struct {
		name   string
		src    string
		expect string
	}{
		{"cudaProbeSource", cudaProbeSource, "ds4_probe"},
		{"cudaExpertSource", cudaExpertSource, "ds4_cuda_ksigns_iq2xs"},
		{"cudaCompressedSource", cudaCompressedSource, "ds4_fill_u32"},
		{"cudaQ80Layout", cudaQ80Layout, "struct ds4_q8_activation"},
		{"cudaQ80QuantBody", cudaQ80QuantBody, "ds4_quant_q8_0"},
		{"cudaQ80MatvecBody", cudaQ80MatvecBody, "ds4_q8_0_matvec_quantized"},
		{"cudaQ81Source", cudaQ81Source, "ds4_quant_q8_1"},
		{"cudaF16DefaultSource", cudaF16DefaultSource, "ds4_f16_matvec_half_input"},
		{"cudaExpertQ81Source", cudaExpertQ81Source, "ds4_iq2_part_q8_1"},
		{"cudaHCSource", cudaHCSource, "ds4_hc_half_input"},
	}
	for _, tc := range sources {
		if len(tc.src) == 0 {
			t.Errorf("%s is empty", tc.name)
		}
		if !strings.Contains(tc.src, tc.expect) {
			t.Errorf("%s does not contain %q", tc.name, tc.expect)
		}
	}
	if !strings.Contains(cudaMathSource, "ds4_probe") || !strings.Contains(cudaMathSource, "ds4_hc_cached_expand") {
		t.Errorf("cudaMathSource does not contain expected kernels")
	}
}

func TestCUDAMathSourceUsesDependencyOrder(t *testing.T) {
	if len(cudaMathFiles) == 0 {
		t.Fatal("cudaMathFiles is empty")
	}
	for _, name := range cudaMathFiles {
		if _, err := cudaSourceFS.ReadFile(name); err != nil {
			t.Errorf("cudaMathFiles entry %q is not embedded: %v", name, err)
		}
		marker := `#line 1 "` + name + `"`
		if !strings.Contains(cudaMathSource, marker) {
			t.Errorf("cudaMathSource missing source marker %q", marker)
		}
	}
	if strings.Contains(cudaMathSource, `#line 1 "cuda_q80_quant.cu"`) {
		t.Error("separately compiled cuda_q80_quant.cu was included in cudaMathSource")
	}
}

func TestCUDAMathCompilationUsesVirtualHeaders(t *testing.T) {
	source := cudaMathCompilationSource()
	if strings.Contains(source, "#line") {
		t.Fatal("NVRTC compilation source should use virtual headers, not synthetic concatenated line offsets")
	}
	for _, name := range cudaMathFiles {
		if !strings.Contains(source, `#include "`+cudaSourceRoot+name+`"`) {
			t.Errorf("compilation source missing virtual header %q", name)
		}
	}
	if got := len(cudaMathHeaderOptions()); got == 0 {
		t.Fatal("virtual header options is empty")
	}
}

// TestCUDAMathReciprocalSqrtIsExplicit requires rsqrtf for every reciprocal
// square root in the math module. Under --use_fast_math, NVRTC lowers
// 1.0f/sqrtf(x) to rsqrt.approx on a fresh compile but to sqrt.approx and
// rcp.approx under -G, and the PTX a process gets can come from an entry in
// the host's CUDA compute cache. The two lowerings differ by an ulp, which
// moved a compressor row by one fp16 step. rsqrtf lowers to rsqrt.approx
// either way.
func TestCUDAMathReciprocalSqrtIsExplicit(t *testing.T) {
	re := regexp.MustCompile(`\b1(\.0*)?f?\s*/\s*sqrtf\s*\(`)
	for _, name := range cudaMathFiles {
		data, err := cudaSourceFS.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(data), "\n") {
			if re.MatchString(line) {
				t.Errorf("%s:%d: use rsqrtf: %s", name, i+1, strings.TrimSpace(line))
			}
		}
	}
}

// TestCUDAMathPTXIndependentOfWorkingDirectory compiles the math module from
// the starting directory and two fresh ones and requires the same PTX, with
// no reciprocal square root lowered as rcp(sqrt). NVRTC keys its compute
// cache on bitcode that records the working directory, so each directory can
// be served a different entry. Run the test binary from several directories
// to widen the check.
func TestCUDAMathPTXIndependentOfWorkingDirectory(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles the math module three times")
	}
	start, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	short := t.TempDir()
	long := filepath.Join(short, "a-longer-working-directory")
	if err := os.Mkdir(long, 0o755); err != nil {
		t.Fatal(err)
	}
	var first, firstDir string
	for _, dir := range []string{start, short, long} {
		t.Chdir(dir)
		cm, err := nvrtcx.Compile(cudaMathCompilationSource(), cudaMathCompileOptions(12, 1)...)
		if err != nil {
			t.Fatalf("compile math module in %s: %v", dir, err)
		}
		ptx := cm.PTX()
		if ptx == "" {
			t.Fatalf("empty PTX in %s", dir)
		}
		if line := rcpOfSqrt(ptx); line != "" {
			t.Errorf("compiled in %s: reciprocal square root lowered as rcp(sqrt); use rsqrtf:\n%s", dir, line)
		}
		if first == "" {
			first, firstDir = ptx, dir
			continue
		}
		if ptx != first {
			al, bl := strings.Split(first, "\n"), strings.Split(ptx, "\n")
			for i := 0; i < len(al) && i < len(bl); i++ {
				if al[i] != bl[i] {
					t.Fatalf("PTX depends on the working directory; first difference at line %d:\n%s: %s\n%s: %s", i+1, firstDir, al[i], dir, bl[i])
				}
			}
			t.Fatalf("PTX depends on the working directory: %d vs %d lines", len(al), len(bl))
		}
	}
}

// rcpOfSqrt returns the first rcp.approx instruction in ptx whose operand is
// the result of a sqrt.approx instruction in the same function, or "" if
// there is none.
func rcpOfSqrt(ptx string) string {
	sqrts := make(map[string]bool)
	for _, line := range strings.Split(ptx, "\n") {
		f := strings.Fields(strings.TrimSpace(line))
		if len(f) > 0 && (f[0] == ".entry" || f[0] == ".func" || f[0] == ".visible" || f[0] == ".weak") {
			clear(sqrts) // registers are scoped to a function
		}
		if len(f) != 3 {
			continue
		}
		dst, src := strings.TrimSuffix(f[1], ","), strings.TrimSuffix(f[2], ";")
		switch {
		case strings.HasPrefix(f[0], "sqrt.approx."):
			sqrts[dst] = true
		case strings.HasPrefix(f[0], "rcp.approx.") && sqrts[src]:
			return strings.TrimSpace(line)
		}
	}
	return ""
}
