package cudasample_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasample"
)

func TestArgmaxTemplate(t *testing.T) {
	tmpl := cudasample.ArgmaxKernelTemplate
	if !strings.Contains(tmpl, "ArgmaxLogits") {
		t.Error("expected ArgmaxLogits in template")
	}
}

func TestSampleArgmaxCPU(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		want   int32
	}{
		{"finite", []float32{-1, .5, 3.2, 1.1}, 2},
		{"leading_nan", []float32{float32(math.NaN()), 4, 3}, 1},
		{"below_old_sentinel", []float32{-2e20, -3e20}, 0},
		{"all_nan", []float32{float32(math.NaN())}, -1},
		{"all_negative_infinity", []float32{float32(math.Inf(-1))}, -1},
		{"empty", nil, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cudasample.SampleArgmaxCPU(tt.logits); got != tt.want {
				t.Fatalf("got %d want %d", got, tt.want)
			}
		})
	}
}
