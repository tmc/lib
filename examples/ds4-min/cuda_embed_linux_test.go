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
