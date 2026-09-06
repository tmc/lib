//go:build cublas_physical

// Package cublas_test exercises the generated cuBLAS bindings against a real
// GPU. Every numeric assertion is checked against an independent CPU reference
// computed in float64, and the inputs are small integers so the float32 and
// float64 results agree exactly: a wrong leading dimension, a swapped
// parameter, or a truncated integer width shows up as a mismatch rather than
// as accumulated rounding.
//
// Set CUBLAS_PHYSICAL_PERTURB to run the negative control. The value names the
// assertion to sabotage; the corresponding test must then fail.
package cublas_test

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
)

// perturb reports whether the named negative control is active.
func perturb(name string) bool {
	return os.Getenv("CUBLAS_PHYSICAL_PERTURB") == name
}

// current is the primary context retained by setup. Each subtest runs on its
// own goroutine, so bind must make it current again there.
var current driver.Context

// bind pins the calling goroutine to a thread and makes the primary context
// current on it. Without this a subtest allocating device memory fails with
// CUDA_ERROR_INVALID_CONTEXT.
func bind(t *testing.T) {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := current.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}
}

// setup retains the primary context on device 0 and returns a cuBLAS handle.
func setup(t *testing.T) cublas.Handle {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
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
	t.Cleanup(func() {
		if err := dev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	})
	current = ctx
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}
	h, err := cublas.Create()
	if err != nil {
		t.Skipf("cuBLAS unavailable: %v", err)
	}
	t.Cleanup(func() {
		if err := h.Destroy(); err != nil {
			t.Errorf("destroy handle: %v", err)
		}
	})
	version, err := h.GetVersion()
	if err != nil {
		t.Fatalf("get version: %v", err)
	}
	t.Logf("cuBLAS version %d", version)
	return h
}

// devSlice uploads xs and returns the device pointer. The buffer is freed when
// the test ends.
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

// devp reinterprets a device address as the pointer type the generated cuBLAS
// signatures use for buffer arguments. The bindings declare device buffers as
// *float32 and *float64 rather than an opaque address type, so every call site
// must launder a device address through unsafe.Pointer. go vet flags this
// shape; the conversion is confined to this one function and the tests run
// with -vet=off.
func devp[T any](p uintptr) *T {
	return (*T)(unsafe.Pointer(p)) //lint:ignore unsafeptr device address, never a Go pointer
}

// download copies len(xs) elements back from ptr into xs.
func download[T any](t *testing.T, ptr uintptr, xs []T) {
	t.Helper()
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	n := uint(len(xs)) * uint(unsafe.Sizeof(xs[0]))
	if err := driver.DevicePtr(ptr).MemcpyDtoH(unsafe.Pointer(&xs[0]), n); err != nil {
		t.Fatalf("MemcpyDtoH: %v", err)
	}
}

// pattern fills a deterministic sequence of small integers in [-4, 4]. Products
// of two such values summed over the k values used here stay well inside the
// exactly representable range of float32, so the GPU and the float64 reference
// must agree bit for bit.
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

// refGemm computes C = alpha*op(A)*op(B) + beta*C in column-major order using
// float64 arithmetic. It is deliberately a naive triple loop with no shared
// code with the binding under test.
func refGemm(transa, transb bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) []float64 {
	at := func(i, j int) float64 {
		if transa {
			return a[j+i*lda] // op(A)[i,j] = A[j,i], A is k x m
		}
		return a[i+j*lda]
	}
	bt := func(i, j int) float64 {
		if transb {
			return b[j+i*ldb] // op(B)[i,j] = B[j,i], B is n x k
		}
		return b[i+j*ldb]
	}
	out := make([]float64, len(c))
	copy(out, c)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			var sum float64
			for p := 0; p < k; p++ {
				sum += at(i, p) * bt(p, j)
			}
			out[i+j*ldc] = alpha*sum + beta*out[i+j*ldc]
		}
	}
	return out
}

// TestSgemm checks single-precision GEMM in four transpose combinations
// against the float64 reference. The dimensions m, n and k are all distinct so
// that a swapped dimension or leading dimension cannot pass by coincidence.
func TestSgemm(t *testing.T) {
	h := setup(t)
	// PedanticMath keeps cuBLAS on the IEEE fp32 path; the integer inputs are
	// exact under TF32 as well, so this only removes a variable.
	if err := h.SetMathMode(cublas.PedanticMath); err != nil {
		t.Fatalf("set math mode: %v", err)
	}
	const (
		m, n, k = 61, 47, 33
		alpha   = 2.0
		beta    = -3.0
	)
	for _, tc := range []struct {
		name           string
		transa, transb bool
	}{
		{"NN", false, false},
		{"TN", true, false},
		{"NT", false, true},
		{"TT", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bind(t)
			// op(A) is m x k, so A is m x k (lda=m) or k x m (lda=k).
			lda, arows := m, m
			if tc.transa {
				lda, arows = k, k
			}
			ldb, brows := k, k
			if tc.transb {
				ldb, brows = n, n
			}
			acols, bcols := k, n
			if tc.transa {
				acols = m
			}
			if tc.transb {
				bcols = k
			}
			a := pattern(arows*acols, 1)
			b := pattern(brows*bcols, 2)
			c := pattern(m*n, 3)
			want := refGemm(tc.transa, tc.transb, m, n, k, alpha, a, lda, b, ldb, beta, c, m)

			af, bf, cf := float32s(a), float32s(b), float32s(c)
			ad, bd, cd := devSlice(t, af), devSlice(t, bf), devSlice(t, cf)
			opa, opb := cublas.OpN, cublas.OpN
			if tc.transa {
				opa = cublas.OpT
			}
			if tc.transb {
				opb = cublas.OpT
			}
			al, be := float32(alpha), float32(beta)
			if perturb("sgemm-alpha") {
				al += 1e-3
			}
			err := h.Sgemm(opa, opb, m, n, k, &al,
				devp[float32](ad), int32(lda),
				devp[float32](bd), int32(ldb),
				&be, devp[float32](cd), m)
			if err != nil {
				t.Fatalf("Sgemm: %v", err)
			}
			got := make([]float32, m*n)
			download(t, cd, got)
			for i := range want {
				if float64(got[i]) != want[i] {
					t.Fatalf("C[%d,%d] = %v, want %v (exact)", i%m, i/m, got[i], want[i])
				}
			}
			t.Logf("%s: %d x %d x %d exact against float64 reference", tc.name, m, n, k)
		})
	}
}

// TestDgemm repeats the NN case in double precision. Different element width
// means a parameter-width defect that happens to be invisible in the float32
// path shows up here.
func TestDgemm(t *testing.T) {
	h := setup(t)
	const (
		m, n, k = 53, 41, 29
		alpha   = -1.5
		beta    = 0.5
	)
	a := pattern(m*k, 11)
	b := pattern(k*n, 12)
	c := pattern(m*n, 13)
	want := refGemm(false, false, m, n, k, alpha, a, m, b, k, beta, c, m)
	ad, bd, cd := devSlice(t, a), devSlice(t, b), devSlice(t, c)
	al, be := float64(alpha), float64(beta)
	if perturb("dgemm-lda") {
		// A wrong leading dimension is the canonical binding defect.
		al = alpha
		if err := h.Dgemm(cublas.OpN, cublas.OpN, m, n, k, &al,
			devp[float64](ad), m+1,
			devp[float64](bd), k, &be,
			devp[float64](cd), m); err != nil {
			t.Fatalf("Dgemm: %v", err)
		}
	} else if err := h.Dgemm(cublas.OpN, cublas.OpN, m, n, k, &al,
		devp[float64](ad), m,
		devp[float64](bd), k, &be,
		devp[float64](cd), m); err != nil {
		t.Fatalf("Dgemm: %v", err)
	}
	got := make([]float64, m*n)
	download(t, cd, got)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("C[%d,%d] = %v, want %v (exact)", i%m, i/m, got[i], want[i])
		}
	}
	t.Logf("Dgemm %dx%dx%d exact against float64 reference", m, n, k)
}

// TestLevel1 checks the level-1 routines that return a value through an
// out-parameter. Each is compared against a float64 reference, and a
// non-unit increment is used so that a dropped stride argument is visible.
func TestLevel1(t *testing.T) {
	h := setup(t)
	const (
		n    = 97
		incx = 3
		incy = 2
	)
	x := pattern(n*incx, 21)
	y := pattern(n*incy, 22)
	xf, yf := float32s(x), float32s(y)
	xd := devSlice(t, xf)
	yd := devSlice(t, yf)
	px := devp[float32](xd)
	py := devp[float32](yd)

	t.Run("Sdot", func(t *testing.T) {
		var want float64
		for i := 0; i < n; i++ {
			want += x[i*incx] * y[i*incy]
		}
		if perturb("sdot") {
			want++
		}
		got, err := h.Sdot(n, px, incx, py, incy)
		if err != nil {
			t.Fatalf("Sdot: %v", err)
		}
		if float64(got) != want {
			t.Fatalf("Sdot = %v, want %v (exact)", got, want)
		}
		t.Logf("Sdot = %v exact", got)
	})

	t.Run("Sasum", func(t *testing.T) {
		var want float64
		for i := 0; i < n; i++ {
			want += math.Abs(x[i*incx])
		}
		if perturb("sasum") {
			want++
		}
		got, err := h.Sasum(n, px, incx)
		if err != nil {
			t.Fatalf("Sasum: %v", err)
		}
		if float64(got) != want {
			t.Fatalf("Sasum = %v, want %v (exact)", got, want)
		}
		t.Logf("Sasum = %v exact", got)
	})

	t.Run("Snrm2", func(t *testing.T) {
		var sum float64
		for i := 0; i < n; i++ {
			sum += x[i*incx] * x[i*incx]
		}
		want := math.Sqrt(sum)
		if perturb("snrm2") {
			want++
		}
		got, err := h.Snrm2(n, px, incx)
		if err != nil {
			t.Fatalf("Snrm2: %v", err)
		}
		// The square root is the only inexact step; one ulp of float32.
		if math.Abs(float64(got)-want) > 1e-4*want {
			t.Fatalf("Snrm2 = %v, want %v", got, want)
		}
		t.Logf("Snrm2 = %v, reference %v", got, want)
	})

	t.Run("Isamax", func(t *testing.T) {
		// cuBLAS returns a one-based index, and ties resolve to the first.
		best, bestIdx := -1.0, 0
		for i := 0; i < n; i++ {
			if v := math.Abs(x[i*incx]); v > best {
				best, bestIdx = v, i+1
			}
		}
		if perturb("isamax") {
			bestIdx++
		}
		got, err := h.Isamax(n, px, incx)
		if err != nil {
			t.Fatalf("Isamax: %v", err)
		}
		if int(got) != bestIdx {
			t.Fatalf("Isamax = %d, want %d (|x| = %v)", got, bestIdx, best)
		}
		t.Logf("Isamax = %d (one-based), |x| = %v", got, best)
	})

	t.Run("Saxpy", func(t *testing.T) {
		const alpha = 3
		want := make([]float64, len(y))
		copy(want, y)
		for i := 0; i < n; i++ {
			want[i*incy] += alpha * x[i*incx]
		}
		if perturb("saxpy") {
			want[0]++
		}
		// Saxpy writes y, so use a private copy.
		yc := float32s(y)
		ycd := devSlice(t, yc)
		al := float32(alpha)
		if err := h.Saxpy(n, &al, px, incx, devp[float32](ycd), incy); err != nil {
			t.Fatalf("Saxpy: %v", err)
		}
		got := make([]float32, len(yc))
		download(t, ycd, got)
		for i := range want {
			if float64(got[i]) != want[i] {
				t.Fatalf("y[%d] = %v, want %v (exact)", i, got[i], want[i])
			}
		}
		t.Logf("Saxpy over %d elements, incy=%d, exact including untouched gaps", n, incy)
	})
}

// TestPointerModeRoundTrip writes each pointer mode and reads it back. The
// enum crosses the ABI as a 32-bit value in one direction and through an
// out-parameter in the other, so a width mistake in either shows up here.
func TestPointerModeRoundTrip(t *testing.T) {
	h := setup(t)
	for _, want := range []cublas.PointerMode{cublas.PointerModeDevice, cublas.PointerModeHost} {
		if err := h.SetPointerMode(want); err != nil {
			t.Fatalf("SetPointerMode(%v): %v", want, err)
		}
		got, err := h.GetPointerMode()
		if err != nil {
			t.Fatalf("GetPointerMode: %v", err)
		}
		if got != want {
			t.Fatalf("GetPointerMode = %v, want %v", got, want)
		}
	}
	t.Log("pointer mode round-trips through both enum directions")
}

func TestMain(m *testing.M) {
	if v := os.Getenv("CUBLAS_PHYSICAL_PERTURB"); v != "" {
		fmt.Fprintf(os.Stderr, "negative control active: %s\n", v)
	}
	os.Exit(m.Run())
}
