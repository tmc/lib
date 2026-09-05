package cusparse_test

import (
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cusparse"
	"github.com/tmc/lib/nvidia/driver"
)

func TestSpMVNumerical(t *testing.T) {
	if os.Getenv("CUSPARSE_PHYSICAL_TEST") != "1" {
		t.Skip("set CUSPARSE_PHYSICAL_TEST=1 for CUDA numerical validation")
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
	h, err := cusparse.Create()
	check(err)
	defer func() { check(h.Destroy()) }()
	var allocations []driver.DevicePtr
	defer func() {
		for i := len(allocations) - 1; i >= 0; i-- {
			check(allocations[i].MemFree())
		}
	}()
	alloc := func(n uint, src unsafe.Pointer) uintptr {
		if n == 0 {
			return 0
		}
		p, err := driver.MemAlloc(n)
		check(err)
		allocations = append(allocations, p)
		if src != nil {
			check(p.MemcpyHtoD(src, n))
		}
		return uintptr(p)
	}
	// A = [2 0 -1; 0 3 0; 4 0 5]. Check y = 2*A*x - y.
	rows := []int32{0, 2, 3, 5}
	cols := []int32{0, 2, 1, 0, 2}
	values := []float32{2, -1, 3, 4, 5}
	x := []float32{1, 2, 3}
	y := []float32{7, 8, 9}
	dr := alloc(16, unsafe.Pointer(&rows[0]))
	dc := alloc(20, unsafe.Pointer(&cols[0]))
	dv := alloc(20, unsafe.Pointer(&values[0]))
	dx := alloc(12, unsafe.Pointer(&x[0]))
	dy := alloc(12, unsafe.Pointer(&y[0]))
	a, err := cusparse.CreateCsr(3, 3, 5, dr, dc, dv, cusparse.Index32I, cusparse.Index32I, cusparse.IndexBaseZero, cusparse.R32F)
	check(err)
	defer func() { check(cusparse.ConstSpMatDescr(a).DestroySpMat()) }()
	vx, err := cusparse.CreateDnVec(3, dx, cusparse.R32F)
	check(err)
	defer func() { check(cusparse.ConstDnVecDescr(vx).DestroyDnVec()) }()
	vy, err := cusparse.CreateDnVec(3, dy, cusparse.R32F)
	check(err)
	defer func() { check(cusparse.ConstDnVecDescr(vy).DestroyDnVec()) }()
	alpha, beta := float32(2), float32(-1)
	// Pin host scalars because the generated API represents their addresses
	// as uintptr, which does not retain the underlying Go objects.
	var pin runtime.Pinner
	pin.Pin(&alpha)
	pin.Pin(&beta)
	defer pin.Unpin()
	ap, bp := uintptr(unsafe.Pointer(&alpha)), uintptr(unsafe.Pointer(&beta))
	size, err := h.SpMVBufferSize(cusparse.OperationNonTranspose, ap, cusparse.ConstSpMatDescr(a), cusparse.ConstDnVecDescr(vx), bp, vy, cusparse.R32F, cusparse.SpmvAlgDefault)
	check(err)
	if size > 1<<28 {
		t.Fatalf("invalid workspace size %d", size)
	}
	work := alloc(size, nil)
	check(h.SpMV(cusparse.OperationNonTranspose, ap, cusparse.ConstSpMatDescr(a), cusparse.ConstDnVecDescr(vx), bp, vy, cusparse.R32F, cusparse.SpmvAlgDefault, work))
	check(driver.CtxSynchronize())
	check(driver.DevicePtr(dy).MemcpyDtoH(unsafe.Pointer(&y[0]), 12))
	for i, want := range []float32{-9, 4, 29} {
		if math.IsNaN(float64(y[i])) || math.Abs(float64(y[i]-want)) > 1e-5 {
			t.Fatalf("y[%d]=%g want %g", i, y[i], want)
		}
	}
	t.Logf("scaled CSR product=%v", y)
}
