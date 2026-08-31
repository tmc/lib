package main

import (
	"strings"
	"testing"
)

const sampleCUDA = `
extern "C" __global__ void VectorAdd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
`

func TestParseCUDA(t *testing.T) {
	kernels, err := parseCUDA(sampleCUDA)
	if err != nil {
		t.Fatalf("parseCUDA failed: %v", err)
	}

	if len(kernels) != 1 {
		t.Fatalf("expected 1 kernel, got %d", len(kernels))
	}

	k := kernels[0]
	if k.Name != "VectorAdd" {
		t.Errorf("expected VectorAdd, got %s", k.Name)
	}

	if len(k.Params) != 4 {
		t.Fatalf("expected 4 params, got %d", len(k.Params))
	}

	code, err := generateGo("mypkg", kernels, sampleCUDA)
	if err != nil {
		t.Fatalf("generateGo failed: %v", err)
	}

	if !strings.Contains(string(code), "func LaunchVectorAdd") {
		t.Errorf("expected func LaunchVectorAdd in generated code:\n%s", string(code))
	}
}
