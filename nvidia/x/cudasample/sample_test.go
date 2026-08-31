package cudasample_test

import (
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
	logits := []float32{-1.0, 0.5, 3.2, 1.1, -0.4}
	idx := cudasample.SampleArgmaxCPU(logits)
	if idx != 2 {
		t.Errorf("expected argmax idx 2, got %d", idx)
	}
}
