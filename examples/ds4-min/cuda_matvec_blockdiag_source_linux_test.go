//go:build linux

package main

import (
	"strings"
	"testing"
)

// TestCUDAMatvecQ80BlockDiagIsMatvecQ80WithBasePointers keeps the two CUDA kernels
// in lockstep at the source level, mirroring metal_matvec_blockdiag_source_darwin_test.go.
//
// ds4_q8_0_matvec_quantized_blockdiag's claim is that it is ds4_q8_0_matvec_quantized
// with base pointers moved and no other change, ensuring bit-identical outputs.
// Recover ds4_q8_0_matvec_quantized's body from ds4_q8_0_matvec_quantized_blockdiag's
// by undoing the substitutions and dropping the added line, then require exact equality.
func TestCUDAMatvecQ80BlockDiagIsMatvecQ80WithBasePointers(t *testing.T) {
	plain := cudaKernelBody(t, "ds4_q8_0_matvec_quantized")
	blockDiag := cudaKernelBody(t, "ds4_q8_0_matvec_quantized_blockdiag")

	added := []string{
		"  const ds4_q8_activation *xb = x + (unsigned long long)diag * blocks;",
	}

	rebased := strings.NewReplacer(
		"extern \"C\" __global__ void ds4_q8_0_matvec_quantized_blockdiag(", "extern \"C\" __global__ void ds4_q8_0_matvec_quantized(",
		"const unsigned row = blockIdx.x * 8 + warp_id, diag = blockIdx.y;", "const unsigned row = blockIdx.x * 8 + warp_id;",
		"weight_row = weights + ((unsigned long long)diag * rows + row) * blocks * 34;", "weight_row = weights + (unsigned long long)row * blocks * 34;",
		"xb[b].qs", "x[b].qs",
		"xb[b].d", "x[b].d",
		"out[(unsigned long long)diag * rows + row] = sum;", "out[row] = sum;",
	)

	var recovered []string
	for _, line := range strings.Split(blockDiag, "\n") {
		if cudaSlicesContains(added, line) {
			continue
		}
		recovered = append(recovered, rebased.Replace(line))
	}
	got := strings.Join(recovered, "\n")
	if got != plain {
		t.Errorf("ds4_q8_0_matvec_quantized_blockdiag is no longer ds4_q8_0_matvec_quantized with moved base pointers.\n"+
			"Recovered body:\n%s\n\nds4_q8_0_matvec_quantized body:\n%s\n\n"+
			"If the change was intended, make it in BOTH kernels and update the substitutions above.", got, plain)
	}
}

func cudaSlicesContains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

func cudaKernelBody(t *testing.T, name string) string {
	t.Helper()
	start := strings.Index(cudaQ80MatvecBody, "extern \"C\" __global__ void "+name+"(")
	if start < 0 {
		t.Fatalf("kernel %q not found in cudaQ80MatvecBody", name)
	}
	end := strings.Index(cudaQ80MatvecBody[start:], "\n}\n")
	if end < 0 {
		t.Fatalf("kernel %q has no closing brace", name)
	}
	return cudaQ80MatvecBody[start : start+end+2]
}
