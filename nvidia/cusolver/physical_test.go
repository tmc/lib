package cusolver_test

import (
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/driver"
)

// TestCholeskySolve checks factorization, solve, and a non-positive-definite
// input. Once enabled, unavailable hardware or libraries fail the test.
func TestCholeskySolve(t *testing.T) {
	if os.Getenv("CUSOLVER_PHYSICAL_TEST") != "1" {
		t.Skip("set CUSOLVER_PHYSICAL_TEST=1 for CUDA numerical validation")
	}
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
	ctx, err := driver.NewContext(nil, 0, dev)
	check(err)
	defer func() { check(ctx.Destroy()) }()
	h, err := cusolver.Create()
	check(err)
	defer func() { check(h.Destroy()) }()
	// Device addresses are opaque to Go. The generated API uses typed pointers;
	// reinterpret the address bits without dereferencing device memory on host.
	floatPtr := func(p driver.DevicePtr) *float32 { return *(**float32)(unsafe.Pointer(&p)) }
	intPtr := func(p driver.DevicePtr) *int32 { return *(**int32)(unsafe.Pointer(&p)) }
	// Cleanup allocations before destroying the context.
	func() {
		var allocations []driver.DevicePtr
		defer func() {
			for i := len(allocations) - 1; i >= 0; i-- {
				check(allocations[i].MemFree())
			}
		}()
		check := func(err error) {
			t.Helper()
			if err != nil {
				t.Fatal(err)
			}
		}
		alloc := func(n uint) driver.DevicePtr {
			p, err := driver.MemAlloc(n)
			check(err)
			allocations = append(allocations, p)
			return p
		}
		// A=[4 1; 1 3], x=[1 2], b=[6 7], column-major.
		a := []float32{4, 1, 1, 3}
		b := []float32{6, 7}
		da, db, di := alloc(16), alloc(8), alloc(4)
		check(da.MemcpyHtoD(unsafe.Pointer(&a[0]), 16))
		check(db.MemcpyHtoD(unsafe.Pointer(&b[0]), 8))
		lower := cusolver.FillMode(cublas.FillModeLower)
		n, err := h.SpotrfBufferSize(lower, 2, floatPtr(da), 2)
		check(err)
		if n <= 0 || n > 1<<26 {
			t.Fatalf("invalid workspace size %d", n)
		}
		dw := alloc(uint(n) * 4)
		info := func() int32 {
			var v int32
			check(driver.CtxSynchronize())
			check(di.MemcpyDtoH(unsafe.Pointer(&v), 4))
			return v
		}
		check(h.Spotrf(lower, 2, floatPtr(da), 2, floatPtr(dw), n, intPtr(di)))
		if v := info(); v != 0 {
			t.Fatalf("potrf info=%d", v)
		}
		check(h.Spotrs(lower, 2, 1, floatPtr(da), 2, floatPtr(db), 2, intPtr(di)))
		if v := info(); v != 0 {
			t.Fatalf("potrs info=%d", v)
		}
		check(db.MemcpyDtoH(unsafe.Pointer(&b[0]), 8))
		for i, want := range []float32{1, 2} {
			if math.IsNaN(float64(b[i])) || math.Abs(float64(b[i]-want)) > 1e-5 {
				t.Fatalf("x[%d]=%g want %g", i, b[i], want)
			}
		}
		t.Logf("solution=%v", b)
		a = []float32{1, 2, 2, 1}
		check(da.MemcpyHtoD(unsafe.Pointer(&a[0]), 16))
		check(h.Spotrf(lower, 2, floatPtr(da), 2, floatPtr(dw), n, intPtr(di)))
		if v := info(); v != 2 {
			t.Fatalf("indefinite matrix info=%d want 2", v)
		}
	}()
}
