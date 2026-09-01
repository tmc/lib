package metalsample

import (
	"strings"
	"testing"
)

func TestArgmaxSamplingEmbedded(t *testing.T) {
	if !strings.Contains(ArgmaxSamplingMSL, "kernel void ArgmaxSampling") {
		t.Fatal("ArgmaxSamplingMSL does not contain expected kernel signature")
	}
}

func TestSampleArgmaxCPU(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		want   int32
	}{
		{name: "empty", logits: nil, want: 0},
		{name: "single", logits: []float32{4.2}, want: 0},
		{name: "max_at_middle", logits: []float32{1.0, 3.5, 9.2, 2.1, 0.4}, want: 2},
		{name: "negative_logits", logits: []float32{-10.5, -3.2, -1.1, -4.0}, want: 2},
		{name: "ties_favor_lower_index", logits: []float32{5.0, 5.0, 3.0}, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SampleArgmaxCPU(tt.logits)
			if got != tt.want {
				t.Errorf("SampleArgmaxCPU() = %d, want %d", got, tt.want)
			}
		})
	}
}
