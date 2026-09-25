package advise

import (
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestAdvice_ElementwiseExcludesGEMV(t *testing.T) {
	// Verifies that matrix-vector multiplication (GEMV) kernels like ds4_q8_0_matvec_quantized
	// are excluded from elementwise fusion recommendations, even if they have "quant" in name.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{50000},
					Label: map[string][]string{
						"name":    {"ds4_swiglu_quant_q8_0"},
						"subtype": {"kernel"},
						"stream":  {"1"},
					},
				},
				{
					Value: []int64{30000},
					Label: map[string][]string{
						"name":    {"ds4_q8_0_matvec_quantized"},
						"subtype": {"kernel"},
						"stream":  {"1"},
					},
				},
			},
		},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	for _, item := range report.Items {
		if item.Category == CategoryMemoryCompute && strings.Contains(item.Finding, "ds4_q8_0_matvec_quantized") && strings.Contains(item.Finding, "elementwise") {
			t.Errorf("unexpected elementwise fusion recommendation between SwiGLU and GEMV: %+v", item)
		}
	}
}
