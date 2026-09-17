//go:build linux

package main

import (
	"strings"
	"testing"
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
		if !strings.Contains(source, `#include "`+name+`"`) {
			t.Errorf("compilation source missing virtual header %q", name)
		}
	}
	if got := len(cudaMathHeaderOptions()); got != len(cudaMathFiles) {
		t.Fatalf("virtual header options = %d, want %d", got, len(cudaMathFiles))
	}
}
