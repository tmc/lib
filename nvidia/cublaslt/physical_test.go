//go:build cublaslt_physical

// Package cublaslt_test drives cuBLASLt matmul against a float64 CPU
// reference. The inputs are small integers, so the float32 result and the
// float64 reference must agree exactly and any difference is a defect rather
// than rounding. cuBLASLt is worth testing separately from cuBLAS because its
// arguments arrive through opaque descriptors and an attribute setter that
// takes a pointer and a size, which is a shape a generator can get wrong in
// ways that still compile and still return CUBLAS_STATUS_SUCCESS.
//
// Set CUBLASLT_PHYSICAL_PERTURB to run a negative control.
package cublaslt_test

import (
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/driver"
)

func perturb(name string) bool { return os.Getenv("CUBLASLT_PHYSICAL_PERTURB") == name }

// pattern is a deterministic sequence of integers in [-4, 4].
func pattern(n, seed int) []float64 {
	xs := make([]float64, n)
	state := uint32(seed)*2654435761 + 1
	for i := range xs {
		state = state*1664525 + 1013904223
		xs[i] = float64(int32(state>>28)%9 - 4)
	}
	return xs
}

func float32s(xs []float64) []float32 {
	ys := make([]float32, len(xs))
	for i, x := range xs {
		ys[i] = float32(x)
	}
	return ys
}

// refGemm computes alpha*op(A)*op(B) + beta*C in column-major float64.
func refGemm(transa bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64) []float64 {
	at := func(i, j int) float64 {
		if transa {
			return a[j+i*lda]
		}
		return a[i+j*lda]
	}
	out := make([]float64, len(c))
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			var sum float64
			for p := 0; p < k; p++ {
				sum += at(i, p) * b[p+j*ldb]
			}
			out[i+j*m] = alpha*sum + beta*c[i+j*m]
		}
	}
	return out
}

// devSlice uploads xs and returns its device address.
func devSlice[T any](t *testing.T, xs []T) uintptr {
	t.Helper()
	n := uint(len(xs)) * uint(unsafe.Sizeof(xs[0]))
	ptr, err := driver.MemAlloc(n)
	if err != nil {
		t.Fatalf("MemAlloc(%d): %v", n, err)
	}
	t.Cleanup(func() {
		if err := ptr.MemFree(); err != nil {
			t.Errorf("free device buffer: %v", err)
		}
	})
	if err := ptr.MemcpyHtoD(unsafe.Pointer(&xs[0]), n); err != nil {
		t.Fatalf("MemcpyHtoD: %v", err)
	}
	return uintptr(ptr)
}

// TestMatmul runs D = alpha*op(A)*B + beta*C with a separate C and D, which
// exercises all four matrix layouts, and repeats it with A transposed through
// the descriptor attribute setter.
func TestMatmul(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("retain primary context: %v", err)
	}
	defer func() {
		if err := dev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	}()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}

	h, err := cublaslt.Create()
	if err != nil {
		t.Skipf("cuBLASLt unavailable: %v", err)
	}
	defer func() {
		if err := h.Destroy(); err != nil {
			t.Errorf("destroy handle: %v", err)
		}
	}()
	var major int32
	if err := cublaslt.GetProperty(cublaslt.LibraryPropertyType(0), &major); err == nil {
		t.Logf("cuBLASLt major version %d", major)
	}

	const (
		m, n, k = 37, 29, 23
		alpha   = 2.0
		beta    = -1.0
	)
	for _, tc := range []struct {
		name   string
		transa bool
	}{
		{"NN", false},
		{"TN", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			if err := ctx.SetCurrent(); err != nil {
				t.Fatalf("set current context: %v", err)
			}

			// op(A) is m x k, so A is m x k (lda=m) or k x m (lda=k).
			lda, arows, acols := m, m, k
			if tc.transa {
				lda, arows, acols = k, k, m
			}
			a := pattern(arows*acols, 1)
			b := pattern(k*n, 2)
			c := pattern(m*n, 3)
			want := refGemm(tc.transa, m, n, k, alpha, a, lda, b, k, beta, c)

			ad := devSlice(t, float32s(a))
			bd := devSlice(t, float32s(b))
			cd := devSlice(t, float32s(c))
			// D is distinct from C so a binding that confused the two would
			// leave the sentinel behind instead of the result.
			sentinel := make([]float32, m*n)
			for i := range sentinel {
				sentinel[i] = -12345
			}
			dd := devSlice(t, sentinel)

			desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
			if err != nil {
				t.Fatalf("NewMatmulDesc: %v", err)
			}
			defer func() {
				if err := desc.Destroy(); err != nil {
					t.Errorf("destroy desc: %v", err)
				}
			}()
			if tc.transa {
				op := cublaslt.CublasOpT
				// SetAttribute takes a pointer and a byte count; passing the
				// wrong size here is exactly the defect this exercises.
				if err := desc.SetAttribute(cublaslt.MatmulDescTransa,
					uintptr(unsafe.Pointer(&op)), uint(unsafe.Sizeof(op))); err != nil {
					t.Fatalf("SetAttribute(TRANSA): %v", err)
				}
			}

			layout := func(rows, cols, ld int) cublaslt.MatrixLayout {
				l, err := cublaslt.NewMatrixLayout(cublaslt.R32F, uint64(rows), uint64(cols), int64(ld))
				if err != nil {
					t.Fatalf("NewMatrixLayout(%d,%d,%d): %v", rows, cols, ld, err)
				}
				t.Cleanup(func() {
					if err := l.Destroy(); err != nil {
						t.Errorf("destroy layout: %v", err)
					}
				})
				return l
			}
			// The A layout describes the stored matrix, not op(A).
			aDesc := layout(arows, acols, lda)
			bDesc := layout(k, n, k)
			cDesc := layout(m, n, m)
			dDesc := layout(m, n, m)

			al, be := float32(alpha), float32(beta)
			if perturb("alpha") {
				al += 1
			}
			if err := h.Matmul(desc,
				uintptr(unsafe.Pointer(&al)), ad, aDesc, bd, bDesc,
				uintptr(unsafe.Pointer(&be)), cd, cDesc, dd, dDesc,
				nil, 0, 0, 0); err != nil {
				t.Fatalf("Matmul: %v", err)
			}
			if err := driver.CtxSynchronize(); err != nil {
				t.Fatalf("synchronize: %v", err)
			}
			got := make([]float32, m*n)
			if err := driver.DevicePtr(dd).MemcpyDtoH(unsafe.Pointer(&got[0]), uint(len(got))*4); err != nil {
				t.Fatalf("MemcpyDtoH: %v", err)
			}
			for i := range want {
				if float64(got[i]) != want[i] {
					t.Fatalf("D[%d,%d] = %v, want %v (exact)", i%m, i/m, got[i], want[i])
				}
			}
			t.Logf("%s: %dx%dx%d matmul exact against the float64 reference", tc.name, m, n, k)
		})
	}
}
