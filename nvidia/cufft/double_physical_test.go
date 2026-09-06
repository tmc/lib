//go:build cufft_physical

package cufft_test

import (
	"math"
	"math/cmplx"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cufft"
	"github.com/tmc/lib/nvidia/driver"
)

// TestDoubleRoundTrip exercises CUDA-owned, 16-byte-aligned Z2Z storage.
// Host DoubleComplex values are only copied as bytes, never passed as native
// complex operand storage. Enable cufft_physical only on a CUDA test host.
func TestDoubleRoundTrip(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	clean := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Errorf("cleanup %s: %v", what, err)
		}
	}
	check("CUDA init", driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check("device", err)
	ctx, err := dev.PrimaryCtxRetain()
	check("retain context", err)
	defer func() { clean("release context", dev.PrimaryCtxRelease()) }()
	check("set context", ctx.SetCurrent())
	version, err := cufft.GetVersion()
	check("version", err)
	const n = 16
	plan, err := cufft.Plan1d(n, cufft.Z2Z, 1)
	check("plan", err)
	defer func() { clean("destroy plan", plan.Destroy()) }()
	input := make([]complex128, n)
	values := make([]cufft.DoubleComplex, n)
	for i := range input {
		input[i] = complex(float64((i*7)%11)-5, float64((i*3)%7)-3)
		values[i] = cufft.DoubleComplex{X: real(input[i]), Y: imag(input[i])}
	}
	bytes := uint(n) * uint(unsafe.Sizeof(values[0]))
	ptr, err := driver.MemAlloc(bytes)
	check("allocate", err)
	defer func() { clean("free buffer", ptr.MemFree()) }()
	if uintptr(ptr)%16 != 0 {
		t.Fatalf("device address %#x is not 16-byte aligned", ptr)
	}
	check("upload", ptr.MemcpyHtoD(unsafe.Pointer(&values[0]), bytes))
	defer func() { clean("synchronize before release", driver.CtxSynchronize()) }()
	check("forward", plan.ExecZ2Z(uintptr(ptr), uintptr(ptr), cufft.Forward))
	check("synchronize forward", driver.CtxSynchronize())
	check("download forward", ptr.MemcpyDtoH(unsafe.Pointer(&values[0]), bytes))
	const tolerance = 1e-10
	for k, v := range values {
		var want complex128
		for j, x := range input {
			angle := -2 * math.Pi * float64(j*k) / n
			want += x * complex(math.Cos(angle), math.Sin(angle))
		}
		got := complex(v.X, v.Y)
		if cmplx.IsNaN(got) || cmplx.IsInf(got) || cmplx.Abs(got-want) > tolerance {
			t.Errorf("forward[%d]=%v, want %v within %g", k, got, want, tolerance)
		}
	}
	check("inverse", plan.ExecZ2Z(uintptr(ptr), uintptr(ptr), cufft.Inverse))
	check("synchronize inverse", driver.CtxSynchronize())
	check("download inverse", ptr.MemcpyDtoH(unsafe.Pointer(&values[0]), bytes))
	for i, v := range values {
		got := complex(v.X, v.Y) / n
		if cmplx.IsNaN(got) || cmplx.IsInf(got) || cmplx.Abs(got-input[i]) > tolerance {
			t.Errorf("normalized inverse[%d]=%v, want %v within %g", i, got, input[i], tolerance)
		}
	}
	t.Logf("cuFFT %d: Z2Z length=%d address=%#x aligned=16 tolerance=%g; forward CPU DFT and normalized inverse checked", version, n, ptr, tolerance)
}
