//go:build cufft_physical

package cufft_test

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cufft"
	"github.com/tmc/lib/nvidia/driver"
)

// TestImpulseRoundTrip requires CUDA when the cufft_physical tag is enabled.
// cuFFT's inverse is unnormalized, so the recovered impulse has amplitude n.
func TestImpulseRoundTrip(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	check(driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check(err)
	ctx, err := dev.PrimaryCtxRetain()
	check(err)
	defer func() {
		if err := dev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	}()
	check(ctx.SetCurrent())
	version, err := cufft.GetVersion()
	check(err)
	t.Logf("cuFFT version %d", version)

	const n = 64
	plan, err := cufft.Plan1d(n, cufft.C2C, 1)
	check(err)
	defer func() {
		if err := plan.Destroy(); err != nil {
			t.Errorf("destroy plan: %v", err)
		}
	}()
	values := make([]cufft.Complex, n)
	values[0].X = 1
	bytes := uint(n) * uint(unsafe.Sizeof(values[0]))
	ptr, err := driver.MemAlloc(bytes)
	check(err)
	defer func() {
		if err := ptr.MemFree(); err != nil {
			t.Errorf("free device buffer: %v", err)
		}
	}()
	check(ptr.MemcpyHtoD(unsafe.Pointer(&values[0]), bytes))
	check(plan.ExecC2C(uintptr(ptr), uintptr(ptr), cufft.Forward))
	check(driver.CtxSynchronize())
	check(ptr.MemcpyDtoH(unsafe.Pointer(&values[0]), bytes))
	for i, v := range values {
		if math.IsNaN(float64(v.X)) || math.IsNaN(float64(v.Y)) || math.Abs(float64(v.X)-1) > 1e-5 || math.Abs(float64(v.Y)) > 1e-5 {
			t.Fatalf("forward[%d] = %v, want (1, 0)", i, v)
		}
	}
	check(plan.ExecC2C(uintptr(ptr), uintptr(ptr), cufft.Inverse))
	check(driver.CtxSynchronize())
	check(ptr.MemcpyDtoH(unsafe.Pointer(&values[0]), bytes))
	for i, v := range values {
		want := float64(0)
		if i == 0 {
			want = n
		}
		if math.IsNaN(float64(v.X)) || math.IsNaN(float64(v.Y)) || math.Abs(float64(v.X)-want) > 1e-4 || math.Abs(float64(v.Y)) > 1e-4 {
			t.Fatalf("inverse[%d] = %v, want (%g, 0)", i, v, want)
		}
	}
}
