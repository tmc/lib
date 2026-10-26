package egraph

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/tileir"
)

const hostCudaTileHeader = `#ifndef CUDA_TILE_H
#define CUDA_TILE_H

#include <cmath>
#include <cstdint>
#include <vector>

#define __tile_global__
#define __restrict__

namespace cuda {
namespace tiles {

template <int... Dims>
struct shape {};

struct extents {
    int d0;
    int d1;
};

template <int Alignment, typename T>
inline T* assume_aligned(T* ptr) {
    return ptr;
}

template <typename T>
struct tensor_span {
    T* ptr;
    extents ext;
};
template <typename T>
tensor_span(T*, extents) -> tensor_span<T>;

struct irange_iter {
    int val;
    bool operator!=(const irange_iter& other) const { return val != other.val; }
    void operator++() { ++val; }
    int operator*() const { return val; }
};

struct irange {
    int start_val;
    int end_val;
    irange(int s, int e) : start_val(s), end_val(e) {}
    irange_iter begin() const { return {start_val}; }
    irange_iter end() const { return {end_val}; }
};

template <typename T, typename Shape>
struct tile;

template <typename T, int R, int C>
struct tile<T, shape<R, C>> {
    T data[R * C];

    tile() {
        for (int i = 0; i < R * C; ++i) {
            data[i] = static_cast<T>(0);
        }
    }

    template <typename U>
    tile(const tile<U, shape<R, C>>& other) {
        for (int i = 0; i < R * C; ++i) {
            data[i] = static_cast<T>(other.data[i]);
        }
    }

    T& at(int r, int c) { return data[r * C + c]; }
    const T& at(int r, int c) const { return data[r * C + c]; }
};

template <typename TileT>
inline TileT full(float v) {
    TileT out;
    for (auto& elem : out.data) {
        elem = v;
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator+(const tile<T, shape<R, C>>& a, const tile<T, shape<R, C>>& b) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = a.data[i] + b.data[i];
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator+(const tile<T, shape<R, C>>& a, float b) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = a.data[i] + static_cast<T>(b);
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator+(float a, const tile<T, shape<R, C>>& b) {
    return b + a;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator-(const tile<T, shape<R, C>>& a) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = -a.data[i];
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator*(const tile<T, shape<R, C>>& a, const tile<T, shape<R, C>>& b) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = a.data[i] * b.data[i];
    }
    return out;
}

// Broadcast [R, C] * [R, 1] across columns.
template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator*(const tile<T, shape<R, C>>& a, const tile<float, shape<R, 1>>& b) {
    tile<T, shape<R, C>> out;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            out.at(r, c) = static_cast<T>(static_cast<float>(a.at(r, c)) * b.at(r, 0));
        }
    }
    return out;
}

// Broadcast [R, C] * [1, C] across rows.
template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator*(const tile<T, shape<R, C>>& a, const tile<T, shape<1, C>>& b) {
    tile<T, shape<R, C>> out;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            out.at(r, c) = a.at(r, c) * b.at(0, c);
        }
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator*(const tile<T, shape<R, C>>& a, T scalar) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = a.data[i] * scalar;
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator/(const tile<T, shape<R, C>>& a, float b) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = static_cast<T>(static_cast<float>(a.data[i]) / b);
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> operator/(const tile<T, shape<R, C>>& a, const tile<T, shape<R, C>>& b) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = a.data[i] / b.data[i];
    }
    return out;
}

template <typename T, int R, int C>
inline tile<float, shape<R, 1>> row_sum(const tile<T, shape<R, C>>& a) {
    tile<float, shape<R, 1>> out;
    for (int r = 0; r < R; ++r) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
            acc += static_cast<float>(a.at(r, c));
        }
        out.at(r, 0) = acc;
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> rsqrt(const tile<T, shape<R, C>>& a) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = static_cast<T>(1.0f / std::sqrt(static_cast<float>(a.data[i])));
    }
    return out;
}

template <typename T, int R, int C>
inline tile<T, shape<R, C>> exp(const tile<T, shape<R, C>>& a) {
    tile<T, shape<R, C>> out;
    for (int i = 0; i < R * C; ++i) {
        out.data[i] = static_cast<T>(std::exp(static_cast<float>(a.data[i])));
    }
    return out;
}

template <typename TA, typename TB, typename TC, int M, int K, int N>
inline tile<TC, shape<M, N>> mma(const tile<TA, shape<M, K>>& a, const tile<TB, shape<K, N>>& b, const tile<TC, shape<M, N>>& c) {
    tile<TC, shape<M, N>> out = c;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = static_cast<float>(out.at(m, n));
            for (int k = 0; k < K; ++k) {
                sum += static_cast<float>(a.at(m, k)) * static_cast<float>(b.at(k, n));
            }
            out.at(m, n) = static_cast<TC>(sum);
        }
    }
    return out;
}

template <typename SpanT, typename ShapeT>
struct partition_view;

template <typename T, int TR, int TC>
struct partition_view<tensor_span<T>, shape<TR, TC>> {
    tensor_span<T> span;

    partition_view(tensor_span<T> s, shape<TR, TC>) : span(s) {}

    using value_type = typename std::remove_const<T>::type;

    tile<value_type, shape<TR, TC>> load(int tr, int tc) const {
        tile<value_type, shape<TR, TC>> out;
        for (int r = 0; r < TR; ++r) {
            int gr = tr * TR + r;
            for (int c = 0; c < TC; ++c) {
                int gc = tc * TC + c;
                if (gr < span.ext.d0 && gc < span.ext.d1) {
                    out.at(r, c) = span.ptr[gr * span.ext.d1 + gc];
                } else {
                    out.at(r, c) = static_cast<value_type>(0);
                }
            }
        }
        return out;
    }

    void store(const tile<value_type, shape<TR, TC>>& val, int tr, int tc) {
        for (int r = 0; r < TR; ++r) {
            int gr = tr * TR + r;
            for (int c = 0; c < TC; ++c) {
                int gc = tc * TC + c;
                if (gr < span.ext.d0 && gc < span.ext.d1) {
                    span.ptr[gr * span.ext.d1 + gc] = val.at(r, c);
                }
            }
        }
    }
};

template <typename SpanT, int TR, int TC>
partition_view(SpanT, shape<TR, TC>) -> partition_view<SpanT, shape<TR, TC>>;

} // namespace tiles
} // namespace cuda

#endif // CUDA_TILE_H
`

func referenceRMSNormRows(x, w []float32, m, k int, eps float32) []float32 {
	out := make([]float32, m*k)
	for row := 0; row < m; row++ {
		var sq float64
		for col := 0; col < k; col++ {
			v := float64(x[row*k+col])
			sq += v * v
		}
		scale := float32(1.0 / math.Sqrt(sq/float64(k)+float64(eps)))
		for col := 0; col < k; col++ {
			weight := float32(1.0)
			if w != nil {
				weight = w[col]
			}
			out[row*k+col] = x[row*k+col] * scale * weight
		}
	}
	return out
}

func referenceMatMul(a, b []float32, m, k, n int) []float32 {
	out := make([]float32, m*n)
	for r := 0; r < m; r++ {
		for c := 0; c < n; c++ {
			var sum float32
			for i := 0; i < k; i++ {
				sum += a[r*k+i] * b[i*n+c]
			}
			out[r*n+c] = sum
		}
	}
	return out
}

func maxAbsSliceDiff(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	var maxDiff float32
	for i := range a {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

func runDeclarativeKernelHost(t *testing.T, kernelSrc, mainBody string) string {
	t.Helper()
	cxx, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("c++ compiler not available")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "cuda_tile.h"), []byte(hostCudaTileHeader), 0o600); err != nil {
		t.Fatalf("write cuda_tile.h: %v", err)
	}
	fullSrc := kernelSrc + "\n#include <cstdio>\nint main() {\n" + mainBody + "\nreturn 0;\n}\n"
	srcPath := filepath.Join(dir, "runner.cpp")
	binPath := filepath.Join(dir, "runner")
	if err := os.WriteFile(srcPath, []byte(fullSrc), 0o600); err != nil {
		t.Fatalf("write runner.cpp: %v", err)
	}
	cmd := exec.Command(cxx, "-std=c++17", "-O2", "-I", dir, srcPath, "-o", binPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile declarative kernel failed: %v\n%s", err, string(out))
	}
	out, err := exec.Command(binPath).CombinedOutput()
	if err != nil {
		t.Fatalf("run declarative kernel failed: %v\n%s", err, string(out))
	}
	return strings.TrimSpace(string(out))
}

func parseFloatsOutput(t *testing.T, s string) []float32 {
	t.Helper()
	fields := strings.Fields(s)
	out := make([]float32, len(fields))
	for i, f := range fields {
		var v float32
		if _, err := fmt.Sscanf(f, "%f", &v); err != nil {
			t.Fatalf("parse float %q at %d: %v", f, i, err)
		}
		out[i] = v
	}
	return out
}

func formatCPPFloatArray(name string, vals []float32) string {
	parts := make([]string, len(vals))
	for i, v := range vals {
		parts[i] = fmt.Sprintf("%.8ff", v)
	}
	return fmt.Sprintf("alignas(16) const float %s[%d] = {%s};", name, len(vals), strings.Join(parts, ", "))
}

// TestRMSNormWeightAndEpsilonSensitivity verifies that changing RMSNorm weights
// or epsilon changes both generated code and evaluated/executed kernel behavior
// across declarative Tile C++, Tile IR, EmitCUDA, and ReferenceGraph.
func TestRMSNormWeightAndEpsilonSensitivity(t *testing.T) {
	const (
		m = 32 // 2 TM tiles (TM=16)
		n = 32 // 2 TN tiles (TN=16)
		k = 32 // 2 TK tiles (TK=16)
	)

	x := make([]float32, m*k)
	for i := range x {
		x[i] = float32(((i*7)%19)-9) * 0.05
	}
	wNormA := make([]float32, k)
	wNormB := make([]float32, k)
	for i := range wNormA {
		wNormA[i] = 0.4 + 0.1*float32((i*3)%11)
		wNormB[i] = 1.9 - 0.12*float32((i*5)%9)
	}
	wProj := make([]float32, k*n)
	for i := range wProj {
		wProj[i] = float32(((i*11)%23)-11) * 0.04
	}

	buildFusedProjExpr := func(wLeaf, epsLeaf *Expr) *Expr {
		return &Expr{
			Op: "tile_rmsnorm_proj",
			Children: []*Expr{
				{Op: "x"},
				wLeaf,
				epsLeaf,
				{Op: "w_proj"},
			},
		}
	}

	cases := []struct {
		name     string
		exprA    *Expr
		exprB    *Expr
		wSliceA  []float32
		wSliceB  []float32
		epsA     float32
		epsB     float32
		wantCode bool
	}{
		{
			name: "weight_constant_change",
			exprA: buildFusedProjExpr(
				&Expr{Op: "w", Data: float32(0.5)},
				&Expr{Op: "eps", Data: float32(1e-5)},
			),
			exprB: buildFusedProjExpr(
				&Expr{Op: "w", Data: float32(2.25)},
				&Expr{Op: "eps", Data: float32(1e-5)},
			),
			wSliceA:  wNormA,
			wSliceB:  wNormB,
			epsA:     1e-5,
			epsB:     1e-5,
			wantCode: true,
		},
		{
			name: "epsilon_constant_change",
			exprA: buildFusedProjExpr(
				&Expr{Op: "w"},
				&Expr{Op: "eps", Data: float32(1e-6)},
			),
			exprB: buildFusedProjExpr(
				&Expr{Op: "w"},
				&Expr{Op: "eps", Data: float32(0.5)},
			),
			wSliceA:  wNormA,
			wSliceB:  wNormA,
			epsA:     1e-6,
			epsB:     0.5,
			wantCode: true,
		},
		{
			name: "symbolic_weight_and_runtime_buffer_change",
			exprA: buildFusedProjExpr(
				&Expr{Op: "w_alpha"},
				&Expr{Op: "eps_alpha"},
			),
			exprB: buildFusedProjExpr(
				&Expr{Op: "w_beta"},
				&Expr{Op: "eps_beta"},
			),
			wSliceA:  wNormA,
			wSliceB:  wNormB,
			epsA:     1e-5,
			epsB:     0.25,
			wantCode: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			declA, err := GenerateDeclarativeTile(tc.exprA, CodegenOptions{KernelName: "k_test"})
			if err != nil {
				t.Fatalf("GenerateDeclarativeTile(A): %v", err)
			}
			declB, err := GenerateDeclarativeTile(tc.exprB, CodegenOptions{KernelName: "k_test"})
			if err != nil {
				t.Fatalf("GenerateDeclarativeTile(B): %v", err)
			}
			if tc.wantCode && declA == declB {
				t.Errorf("GenerateDeclarativeTile output did not change between A and B")
			}

			irA, err := GenerateTileIR(tc.exprA, CodegenOptions{KernelName: "k_test"})
			if err != nil {
				t.Fatalf("GenerateTileIR(A): %v", err)
			}
			irB, err := GenerateTileIR(tc.exprB, CodegenOptions{KernelName: "k_test"})
			if err != nil {
				t.Fatalf("GenerateTileIR(B): %v", err)
			}
			if tc.wantCode && irA == irB {
				t.Errorf("GenerateTileIR output did not change between A and B")
			}

			cudaA, err := GenerateCUDA(tc.exprA, CodegenOptions{KernelName: "k_test"})
			if err != nil {
				t.Fatalf("GenerateCUDA(A): %v", err)
			}
			cudaB, err := GenerateCUDA(tc.exprB, CodegenOptions{KernelName: "k_test"})
			if err != nil {
				t.Fatalf("GenerateCUDA(B): %v", err)
			}
			if tc.wantCode && cudaA == cudaB {
				t.Errorf("GenerateCUDA output did not change between A and B")
			}

			// Evaluate ReferenceGraph with non-uniform weights and epsilon values.
			refExpr := &Expr{
				Op: "tile_rmsnorm_proj",
				Children: []*Expr{
					{Op: "x"},
					{Op: "w"},
					{Op: "eps"},
					{Op: "w_proj"},
				},
			}
			src, err := EmitCUDA(refExpr, EmitOptions{
				Parameters: []Parameter{
					{Name: "x", Shape: []int{k}},
					{Name: "w", Shape: []int{k}},
					{Name: "eps", Shape: []int{1}},
					{Name: "w_proj", Shape: []int{n, k}},
				},
			})
			if err != nil {
				t.Fatalf("EmitCUDA: %v", err)
			}
			refGraph, err := src.ReferenceGraph()
			if err != nil {
				t.Fatalf("ReferenceGraph: %v", err)
			}
			xRow := x[:k]
			// Note: in EmitCUDA / ReferenceGraph for GEMV, w_proj is [out_dim=n, in_dim=k].
			wProjRowMajor := make([]float32, n*k)
			for r := 0; r < k; r++ {
				for c := 0; c < n; c++ {
					wProjRowMajor[c*k+r] = wProj[r*n+c]
				}
			}
			evalMapA, err := cudaref.Evaluate(refGraph, map[string][]float32{
				"x":      xRow,
				"w":      tc.wSliceA,
				"eps":    {tc.epsA},
				"w_proj": wProjRowMajor,
			})
			if err != nil {
				t.Fatalf("cudaref.Evaluate(A): %v", err)
			}
			evalMapB, err := cudaref.Evaluate(refGraph, map[string][]float32{
				"x":      xRow,
				"w":      tc.wSliceB,
				"eps":    {tc.epsB},
				"w_proj": wProjRowMajor,
			})
			if err != nil {
				t.Fatalf("cudaref.Evaluate(B): %v", err)
			}
			evalA := evalMapA["result"]
			evalB := evalMapB["result"]
			if maxAbsSliceDiff(evalA, evalB) < 1e-4 {
				t.Errorf("ReferenceGraph evaluation did not change between A and B (maxDiff=%g)", maxAbsSliceDiff(evalA, evalB))
			}

			// Execute the declarative Tile C++ kernel on host across M=32, N=32, K=32.
			runtimeDecl, err := GenerateDeclarativeTile(refExpr, CodegenOptions{KernelName: "rms_proj_kernel"})
			if err != nil {
				t.Fatalf("GenerateDeclarativeTile(runtime): %v", err)
			}
			mainBody := fmt.Sprintf(`
%s
%s
%s
%s
alignas(16) float OutA[%d] = {0};
alignas(16) float OutB[%d] = {0};
rms_proj_kernel(X, W_norm_A, W_proj, OutA, %d, %d, %d, %.8ff);
rms_proj_kernel(X, W_norm_B, W_proj, OutB, %d, %d, %d, %.8ff);
for (int i = 0; i < %d; ++i) {
    std::printf("%%.8f ", OutA[i]);
}
for (int i = 0; i < %d; ++i) {
    std::printf("%%.8f ", OutB[i]);
}
`,
				formatCPPFloatArray("X", x),
				formatCPPFloatArray("W_norm_A", tc.wSliceA),
				formatCPPFloatArray("W_norm_B", tc.wSliceB),
				formatCPPFloatArray("W_proj", wProj),
				m*n, m*n,
				m, n, k, tc.epsA,
				m, n, k, tc.epsB,
				m*n, m*n,
			)
			rawOut := runDeclarativeKernelHost(t, runtimeDecl, mainBody)
			vals := parseFloatsOutput(t, rawOut)
			if len(vals) != 2*m*n {
				t.Fatalf("expected %d output floats, got %d", 2*m*n, len(vals))
			}
			gotA := vals[:m*n]
			gotB := vals[m*n:]

			wantA := referenceMatMul(referenceRMSNormRows(x, tc.wSliceA, m, k, tc.epsA), wProj, m, k, n)
			wantB := referenceMatMul(referenceRMSNormRows(x, tc.wSliceB, m, k, tc.epsB), wProj, m, k, n)

			if diff := maxAbsSliceDiff(gotA, wantA); diff > 1e-4 {
				t.Errorf("declarative kernel A mismatch vs reference: maxDiff=%g", diff)
			}
			if diff := maxAbsSliceDiff(gotB, wantB); diff > 1e-4 {
				t.Errorf("declarative kernel B mismatch vs reference: maxDiff=%g", diff)
			}
			if diff := maxAbsSliceDiff(gotA, gotB); diff < 1e-4 {
				t.Errorf("declarative kernel output did not change when RMSNorm weight/eps changed: maxDiff=%g", diff)
			}
			// First row of declarative kernel must also match cudaref.Evaluate on ReferenceGraph.
			if diff := maxAbsSliceDiff(gotA[:n], evalA); diff > 1e-4 {
				t.Errorf("declarative kernel row 0 mismatch vs ReferenceGraph: maxDiff=%g", diff)
			}

			// Negative control: unnormalized X @ W_proj must be rejected.
			unnormalized := referenceMatMul(x, wProj, m, k, n)
			if diff := maxAbsSliceDiff(gotA, unnormalized); diff < 1e-2 {
				t.Errorf("negative control failed: normalized output is indistinguishable from unnormalized matmul (maxDiff=%g)", diff)
			}
		})
	}
}

// TestQKVSplitDistinctOutputsAndReferenceParity verifies that QKV split
// produces distinct Q, K, and V outputs matching the reference graph and
// rejects the buggy Q+K+V sum and unnormalized mutants.
func TestQKVSplitDistinctOutputsAndReferenceParity(t *testing.T) {
	cases := []struct {
		name      string
		m, n, k   int
		eps       float32
		buildExpr func() *Expr
	}{
		{
			name: "unfused_qkv_split_gemv_32x32x32",
			m:    32, n: 32, k: 32,
			eps: 1e-5,
			buildExpr: func() *Expr {
				norm := &Expr{
					Class: 1,
					Op:    "rmsnorm",
					Children: []*Expr{
						{Class: 2, Op: "x"},
						{Class: 3, Op: "w"},
						{Class: 4, Op: "eps"},
					},
				}
				return &Expr{
					Class: 10,
					Op:    "qkv_split",
					Children: []*Expr{
						{Class: 7, Op: "gemv", Children: []*Expr{norm, {Class: 5, Op: "wq"}}},
						{Class: 8, Op: "gemv", Children: []*Expr{norm, {Class: 6, Op: "wk"}}},
						{Class: 9, Op: "gemv", Children: []*Expr{norm, {Class: 11, Op: "wv"}}},
					},
				}
			},
		},
		{
			name: "fused_tile_rmsnorm_qkv_split_16x32x48",
			m:    16, n: 32, k: 48,
			eps: 1e-3,
			buildExpr: func() *Expr {
				return &Expr{
					Op: "tile_rmsnorm_qkv_split",
					Children: []*Expr{
						{Op: "x"},
						{Op: "w"},
						{Op: "eps"},
						{Op: "wq"},
						{Op: "wk"},
						{Op: "wv"},
					},
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := make([]float32, tc.m*tc.k)
			for i := range x {
				x[i] = float32(((i*5)%17)-8) * 0.07
			}
			wNorm := make([]float32, tc.k)
			for i := range wNorm {
				wNorm[i] = 0.5 + 0.15*float32((i*7)%9)
			}
			wq := make([]float32, tc.k*tc.n)
			wk := make([]float32, tc.k*tc.n)
			wv := make([]float32, tc.k*tc.n)
			for i := range wq {
				wq[i] = float32(((i*3)%13)-6) * 0.05
				wk[i] = float32(((i*7+2)%15)-7) * 0.06
				wv[i] = float32(((i*11+5)%19)-9) * 0.04
			}

			expr := tc.buildExpr()
			declSrc, err := GenerateDeclarativeTile(expr, CodegenOptions{KernelName: "qkv_split_kernel"})
			if err != nil {
				t.Fatalf("GenerateDeclarativeTile: %v", err)
			}
			if strings.Contains(declSrc, "sum_q + sum_k + sum_v") {
				t.Fatalf("GenerateDeclarativeTile still contains buggy sum_q + sum_k + sum_v store")
			}
			for _, want := range []string{"out_q.store(sum_q, m, n)", "out_k.store(sum_k, m, n)", "out_v.store(sum_v, m, n)"} {
				if !strings.Contains(declSrc, want) {
					t.Errorf("GenerateDeclarativeTile missing %q", want)
				}
			}

			mainBody := fmt.Sprintf(`
%s
%s
%s
%s
%s
alignas(16) float Out[%d] = {0};
qkv_split_kernel(X, W_norm, W_q, W_k, W_v, Out, %d, %d, %d, %.8ff);
for (int i = 0; i < %d; ++i) {
    std::printf("%%.8f ", Out[i]);
}
`,
				formatCPPFloatArray("X", x),
				formatCPPFloatArray("W_norm", wNorm),
				formatCPPFloatArray("W_q", wq),
				formatCPPFloatArray("W_k", wk),
				formatCPPFloatArray("W_v", wv),
				3*tc.m*tc.n,
				tc.m, tc.n, tc.k, tc.eps,
				3*tc.m*tc.n,
			)
			rawOut := runDeclarativeKernelHost(t, declSrc, mainBody)
			gotAll := parseFloatsOutput(t, rawOut)
			partLen := tc.m * tc.n
			if len(gotAll) != 3*partLen {
				t.Fatalf("expected %d output elements, got %d", 3*partLen, len(gotAll))
			}
			gotQ := gotAll[:partLen]
			gotK := gotAll[partLen : 2*partLen]
			gotV := gotAll[2*partLen : 3*partLen]

			normX := referenceRMSNormRows(x, wNorm, tc.m, tc.k, tc.eps)
			wantQ := referenceMatMul(normX, wq, tc.m, tc.k, tc.n)
			wantK := referenceMatMul(normX, wk, tc.m, tc.k, tc.n)
			wantV := referenceMatMul(normX, wv, tc.m, tc.k, tc.n)

			if diff := maxAbsSliceDiff(gotQ, wantQ); diff > 1e-4 {
				t.Errorf("Q partition mismatch vs reference: maxDiff=%g", diff)
			}
			if diff := maxAbsSliceDiff(gotK, wantK); diff > 1e-4 {
				t.Errorf("K partition mismatch vs reference: maxDiff=%g", diff)
			}
			if diff := maxAbsSliceDiff(gotV, wantV); diff > 1e-4 {
				t.Errorf("V partition mismatch vs reference: maxDiff=%g", diff)
			}

			// Verify Q, K, and V are genuinely distinct.
			if maxAbsSliceDiff(gotQ, gotK) < 1e-2 || maxAbsSliceDiff(gotK, gotV) < 1e-2 || maxAbsSliceDiff(gotQ, gotV) < 1e-2 {
				t.Errorf("expected distinct Q, K, V outputs")
			}

			// Negative control: the old buggy Q + K + V sum must not match Q, K, or V.
			buggySum := make([]float32, partLen)
			for i := range buggySum {
				buggySum[i] = wantQ[i] + wantK[i] + wantV[i]
			}
			if maxAbsSliceDiff(gotQ, buggySum) < 1e-2 {
				t.Errorf("negative control failed: Q partition matches buggy Q+K+V sum")
			}

			// Verify ReferenceGraph parity across all shapes.
			src, err := EmitCUDA(expr, EmitOptions{
				Parameters: []Parameter{
					{Name: "x", Shape: []int{tc.k}},
					{Name: "w", Shape: []int{tc.k}},
					{Name: "eps", Shape: []int{1}},
					{Name: "wq", Shape: []int{tc.n, tc.k}},
					{Name: "wk", Shape: []int{tc.n, tc.k}},
					{Name: "wv", Shape: []int{tc.n, tc.k}},
				},
			})
			if err != nil {
				t.Fatalf("EmitCUDA: %v", err)
			}
			refGraph, err := src.ReferenceGraph()
			if err != nil {
				t.Fatalf("ReferenceGraph: %v", err)
			}
			transpose := func(mat []float32, rows, cols int) []float32 {
				out := make([]float32, rows*cols)
				for r := 0; r < rows; r++ {
					for c := 0; c < cols; c++ {
						out[c*rows+r] = mat[r*cols+c]
					}
				}
				return out
			}
			refMap, err := cudaref.Evaluate(refGraph, map[string][]float32{
				"x":   x[:tc.k],
				"w":   wNorm,
				"eps": {tc.eps},
				"wq":  transpose(wq, tc.k, tc.n),
				"wk":  transpose(wk, tc.k, tc.n),
				"wv":  transpose(wv, tc.k, tc.n),
			})
			if err != nil {
				t.Fatalf("cudaref.Evaluate: %v", err)
			}
			refOut := refMap["result"]
			if len(refOut) != 3*tc.n {
				t.Fatalf("expected ReferenceGraph output len %d, got %d", 3*tc.n, len(refOut))
			}
			if diff := maxAbsSliceDiff(refOut[:tc.n], gotQ[:tc.n]); diff > 1e-4 {
				t.Errorf("ReferenceGraph Q mismatch vs declarative row 0: maxDiff=%g", diff)
			}
			if diff := maxAbsSliceDiff(refOut[tc.n:2*tc.n], gotK[:tc.n]); diff > 1e-4 {
				t.Errorf("ReferenceGraph K mismatch vs declarative row 0: maxDiff=%g", diff)
			}
			if diff := maxAbsSliceDiff(refOut[2*tc.n:3*tc.n], gotV[:tc.n]); diff > 1e-4 {
				t.Errorf("ReferenceGraph V mismatch vs declarative row 0: maxDiff=%g", diff)
			}
		})
	}
}

// TestToTileIRNormalizationLowering verifies that ToTileIR on root and fused
// normalization graphs emits mean division, epsilon addition before rsqrt,
// weight multiplication when present, and final output store(s).
func TestToTileIRNormalizationLowering(t *testing.T) {
	cases := []struct {
		name           string
		expr           *Expr
		wantWeightMult bool
		wantStores     int
		forbidQKVAdd   bool
	}{
		{
			name: "root_rmsnorm_weighted",
			expr: &Expr{
				Op: "rmsnorm",
				Children: []*Expr{
					{Op: "x"},
					{Op: "w"},
					{Op: "eps", Data: float32(1e-5)},
				},
			},
			wantWeightMult: true,
			wantStores:     1,
		},
		{
			name: "root_rmsnorm_unweighted",
			expr: &Expr{
				Op: "rmsnorm",
				Children: []*Expr{
					{Op: "x"},
					{Op: "eps", Data: float32(1e-5)},
				},
			},
			wantWeightMult: false,
			wantStores:     1,
		},
		{
			name: "tile_rmsnorm_proj_weighted",
			expr: &Expr{
				Op: "tile_rmsnorm_proj",
				Children: []*Expr{
					{Op: "x"},
					{Op: "w"},
					{Op: "eps", Data: float32(1e-5)},
					{Op: "w_proj"},
				},
			},
			wantWeightMult: true,
			wantStores:     1,
		},
		{
			name: "tile_rmsnorm_proj_unweighted",
			expr: &Expr{
				Op: "tile_rmsnorm_proj",
				Children: []*Expr{
					{Op: "x"},
					{Op: "eps", Data: float32(1e-5)},
					{Op: "w_proj"},
				},
			},
			wantWeightMult: false,
			wantStores:     1,
		},
		{
			name: "tile_rmsnorm_qkv_weighted",
			expr: &Expr{
				Op: "tile_rmsnorm_qkv",
				Children: []*Expr{
					{Op: "x"},
					{Op: "w"},
					{Op: "eps", Data: float32(1e-5)},
					{Op: "w_qkv"},
				},
			},
			wantWeightMult: true,
			wantStores:     1,
		},
		{
			name: "tile_rmsnorm_qkv_split_weighted",
			expr: &Expr{
				Op: "tile_rmsnorm_qkv_split",
				Children: []*Expr{
					{Op: "x"},
					{Op: "w"},
					{Op: "eps", Data: float32(1e-5)},
					{Op: "wq"},
					{Op: "wk"},
					{Op: "wv"},
				},
			},
			wantWeightMult: true,
			wantStores:     3,
			forbidQKVAdd:   true,
		},
		{
			name: "tile_rmsnorm_gemv_swiglu_weighted",
			expr: &Expr{
				Op: "tile_rmsnorm_gemv_swiglu",
				Children: []*Expr{
					{Op: "x"},
					{Op: "w"},
					{Op: "eps", Data: float32(1e-5)},
					{Op: "w_proj"},
					{Op: "w_gate"},
					{Op: "w_up"},
					{Op: "w_down"},
				},
			},
			wantWeightMult: true,
			wantStores:     1,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mod, err := ToTileIR(tc.expr, CodegenOptions{KernelName: "norm_kernel"})
			if err != nil {
				t.Fatalf("ToTileIR: %v", err)
			}
			irText := mod.String()
			parsed, err := tileir.ParseString(irText)
			if err != nil {
				t.Fatalf("tileir.ParseString failed: %v\nIR:\n%s", err, irText)
			}
			hazards, shapeErrs := parsed.Verify()
			if len(hazards) > 0 {
				t.Errorf("VerifyTokens reported hazards: %v", hazards)
			}
			if len(shapeErrs) > 0 {
				t.Errorf("VerifyShapes reported errors: %v", shapeErrs)
			}
			if len(parsed.Functions) != 1 {
				t.Fatalf("expected 1 function, got %d", len(parsed.Functions))
			}
			ops := parsed.Functions[0].Operations

			findOpIdx := func(opCode, result string) int {
				for i, op := range ops {
					if op.OpCode == opCode {
						if result == "" || (len(op.Results) > 0 && op.Results[0] == result) {
							return i
						}
					}
				}
				return -1
			}

			sqIdx := findOpIdx("cuda_tile.mulf", "%x_sq")
			redIdx := findOpIdx("cuda_tile.reduce_sum", "%x_sq_sum")
			divIdx := findOpIdx("cuda_tile.divf", "%x_mean")
			addEpsIdx := findOpIdx("cuda_tile.addf", "%x_var")
			rsqrtIdx := findOpIdx("cuda_tile.rsqrt", "%r_rms")
			normIdx := findOpIdx("cuda_tile.mulf", "%x_norm")
			scaleIdx := findOpIdx("cuda_tile.mulf", "%x_scaled")
			retIdx := findOpIdx("cuda_tile.return", "")

			if sqIdx < 0 {
				t.Fatalf("missing %%x_sq = cuda_tile.mulf in IR:\n%s", irText)
			}
			if redIdx <= sqIdx {
				t.Fatalf("missing or misordered %%x_sq_sum = cuda_tile.reduce_sum (sq=%d, red=%d):\n%s", sqIdx, redIdx, irText)
			}
			if divIdx <= redIdx {
				t.Fatalf("missing or misordered mean division %%x_mean = cuda_tile.divf (red=%d, div=%d):\n%s", redIdx, divIdx, irText)
			}
			if ops[divIdx].Operands[0] != "%x_sq_sum" || ops[divIdx].Operands[1] != "%cst_dim" {
				t.Errorf("divf operands = %v, want [%%x_sq_sum %%cst_dim]", ops[divIdx].Operands)
			}
			if addEpsIdx <= divIdx {
				t.Fatalf("missing or misordered epsilon addition %%x_var = cuda_tile.addf (div=%d, add=%d):\n%s", divIdx, addEpsIdx, irText)
			}
			if rsqrtIdx <= addEpsIdx {
				t.Fatalf("cuda_tile.rsqrt must occur after epsilon addition (add=%d, rsqrt=%d):\n%s", addEpsIdx, rsqrtIdx, irText)
			}
			if ops[rsqrtIdx].Operands[0] != "%x_var" {
				t.Errorf("rsqrt operand = %v, want [%%x_var] (not raw %%x_sq)", ops[rsqrtIdx].Operands)
			}
			if normIdx <= rsqrtIdx {
				t.Fatalf("missing or misordered %%x_norm = cuda_tile.mulf (rsqrt=%d, norm=%d):\n%s", rsqrtIdx, normIdx, irText)
			}

			if tc.wantWeightMult {
				if scaleIdx <= normIdx {
					t.Errorf("expected weight multiplication %%x_scaled = cuda_tile.mulf after %%x_norm:\n%s", irText)
				}
			} else if scaleIdx >= 0 {
				t.Errorf("unexpected weight multiplication %%x_scaled in unweighted RMSNorm:\n%s", irText)
			}

			storeCount := 0
			for i, op := range ops {
				if op.OpCode == "cuda_tile.store_view_tko" {
					storeCount++
					if retIdx >= 0 && i > retIdx {
						t.Errorf("store_view_tko at %d occurs after return at %d", i, retIdx)
					}
				}
				if tc.forbidQKVAdd && op.OpCode == "cuda_tile.addf" {
					for _, opnd := range op.Operands {
						if opnd == "%q" || opnd == "%k" || opnd == "%v" || opnd == "%kv" {
							t.Errorf("forbidden QKV sum operation in QKV split lowering: %v", op)
						}
					}
				}
			}
			if storeCount != tc.wantStores {
				t.Errorf("store_view_tko count = %d, want %d\nIR:\n%s", storeCount, tc.wantStores, irText)
			}
		})
	}
}
