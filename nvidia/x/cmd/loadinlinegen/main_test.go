package main

import (
	"slices"
	"strings"
	"testing"
)

func TestScanTopLevelDecls(t *testing.T) {
	src := `
#include <cuda_runtime.h>
// namespace ignored_comment {
/* void ignored_block_comment() {} */
namespace {
__global__ void anon_helper(float* x) {}
}  // namespace

namespace alpha {
__global__ void inside_alpha(float* x) {}
}  // namespace alpha

struct Config {
	int n;
	void member() {}
};

void forward_only(int n);

__device__ __forceinline__ float top_helper(float x) {
	return x * 2.0f;
}

__global__ void __launch_bounds__(128, 1) top_kernel(
    float* __restrict__ out,
    const float* __restrict__ in) {
	out[0] = top_helper(in[0]);
}
`
	ns, fns := scanTopLevelDecls(src)
	if !slices.Equal(ns, []string{"alpha"}) {
		t.Errorf("namespaces = %v, want [alpha]", ns)
	}
	if !slices.Equal(fns, []string{"top_helper", "top_kernel"}) {
		t.Errorf("funcs = %v, want [top_helper top_kernel]", fns)
	}
}

func TestCheckCollisions(t *testing.T) {
	tests := []struct {
		name  string
		units []sourceUnit
		want  string
	}{
		{
			name: "disjoint namespaces and functions",
			units: []sourceUnit{
				{label: "a.cu", cuda: "namespace ns_a { __global__ void k() {} }\n__device__ float f1(float x) { return x; }"},
				{label: "b.cu", cuda: "namespace ns_b { __global__ void k() {} }\n__device__ float f2(float x) { return x; }"},
			},
			want: "",
		},
		{
			name: "colliding top-level namespace",
			units: []sourceUnit{
				{label: "a.cu", cuda: "namespace shared_ns { void run() {} }"},
				{label: "b.cu", cuda: "namespace shared_ns { void other() {} }"},
			},
			want: "namespace collision: namespace shared_ns defined in both a.cu and b.cu",
		},
		{
			name: "colliding top-level kernel",
			units: []sourceUnit{
				{label: "a.cu", cuda: "__global__ void reduce_kernel(float* x) {}"},
				{label: "b.cu", cuda: "__global__ void __launch_bounds__(128) reduce_kernel(float* x) {}"},
			},
			want: "function collision: reduce_kernel defined in both a.cu and b.cu",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkCollisions(tt.units)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("checkCollisions() = %v, want %q", err, tt.want)
			}
		})
	}
}
