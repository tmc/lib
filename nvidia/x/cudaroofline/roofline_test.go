package cudaroofline_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func TestLookupCeiling(t *testing.T) {
	tests := []struct {
		arch       string
		wantPeak   float64
		wantMem    float64
		shouldFail bool
	}{
		{"sm_90", 67.0, 3350.0, false},
		{"sm_121", 40.0, 273.0, false},
		{"sm_121a", 40.0, 273.0, false},
		{"sm_80", 19.5, 1935.0, false},
		{"sm_999", 0, 0, true},
		{"", 0, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.arch, func(t *testing.T) {
			c, err := cudaroofline.LookupCeiling(tt.arch)
			if tt.shouldFail {
				if err == nil {
					t.Fatalf("expected error for arch %q, got nil", tt.arch)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if c.PeakTFLOPS != tt.wantPeak {
				t.Errorf("PeakTFLOPS = %f, want %f", c.PeakTFLOPS, tt.wantPeak)
			}
			if c.PeakMemoryGBps != tt.wantMem {
				t.Errorf("PeakMemoryGBps = %f, want %f", c.PeakMemoryGBps, tt.wantMem)
			}
		})
	}
}

func TestModelKneeIntensity(t *testing.T) {
	model, err := cudaroofline.NewModel("sm_121")
	if err != nil {
		t.Fatal(err)
	}
	// PeakTFLOPS: 40, PeakMemoryGBps: 273
	// Knee = (40 * 1000) / 273 = 146.52014...
	knee := model.KneeIntensity()
	want := (40.0 * 1000.0) / 273.0
	if diff := knee - want; diff < -0.01 || diff > 0.01 {
		t.Errorf("knee intensity = %f, want %f", knee, want)
	}
}

func TestEvaluateClassifications(t *testing.T) {
	model, err := cudaroofline.NewModel("sm_121")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name       string
		kernel     string
		flops      uint64
		bytes      uint64
		durationNs uint64
		wantClass  string
	}{
		{
			name:       "memory_bound_q8_matvec",
			kernel:     "ds4_q8_0_matvec",
			flops:      1000000,
			bytes:      1000000, // Intensity = 1.0 FLOP/B (well below knee ~146.5)
			durationNs: 4000,    // 1 MB / 4us = 250 GB/s (near 273 GB/s peak, ~91% efficiency)
			wantClass:  cudaroofline.ClassMemoryBound,
		},
		{
			name:       "compute_bound_dense_gemm",
			kernel:     "dense_gemm",
			flops:      1000000000, // 1 GFLOP
			bytes:      1000000,    // 1 MB -> Intensity = 1000 FLOP/B (well above knee)
			durationNs: 28000,      // 1 GFLOP / 28us = 35.7 TFLOPS (near 40 TFLOPS peak, ~89% efficiency)
			wantClass:  cudaroofline.ClassComputeBound,
		},
		{
			name:       "unconstrained_slow_kernel",
			kernel:     "ragged_slow_kernel",
			flops:      1000000,
			bytes:      1000000, // Intensity = 1.0 FLOP/B, ceiling = 0.273 TFLOPS
			durationNs: 100000,  // 1 MB / 100us = 10 GB/s -> attained = 0.01 TFLOPS (< 35% of ceiling)
			wantClass:  cudaroofline.ClassUnconstrained,
		},
		{
			name:       "zero_duration_edge_case",
			kernel:     "noop",
			flops:      0,
			bytes:      0,
			durationNs: 0,
			wantClass:  cudaroofline.ClassUnconstrained,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pt := model.Evaluate(tt.kernel, tt.flops, tt.bytes, tt.durationNs)
			if pt.Classification != tt.wantClass {
				t.Errorf("Classification = %q, want %q (attained=%.3f TFLOPS, intensity=%.2f, eff=%.2f)",
					pt.Classification, tt.wantClass, pt.AttainedTFLOPS, pt.Intensity, pt.Efficiency)
			}
		})
	}
}

func TestRenderSVG(t *testing.T) {
	model, err := cudaroofline.NewModel("sm_90")
	if err != nil {
		t.Fatal(err)
	}

	pt1 := model.Evaluate("matvec", 2000000, 1000000, 500)
	pt2 := model.Evaluate("gemm", 10000000000, 5000000, 160000)

	var buf bytes.Buffer
	if err := model.RenderSVG(&buf, []cudaroofline.KernelPoint{pt1, pt2}); err != nil {
		t.Fatalf("RenderSVG failed: %v", err)
	}

	svg := buf.String()
	for _, expected := range []string{
		"<?xml version=",
		"<svg",
		"Roofline Model (sm_90)",
		"matvec",
		"gemm",
		"Memory Bound",
		"Compute Bound",
		"</svg>",
	} {
		if !strings.Contains(svg, expected) {
			t.Errorf("SVG missing expected substring: %q", expected)
		}
	}
}
