//go:build cutensor_physical

package cutensor_test

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cutensor"
	"github.com/tmc/lib/nvidia/driver"
)

func TestPhysicalContraction(t *testing.T) {
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	cleanup := func(f func() error) {
		t.Cleanup(func() {
			if err := f(); err != nil {
				t.Error(err)
			}
		})
	}
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
	cleanup(ctx.Destroy)
	h, err := cutensor.Create()
	if err != nil {
		t.Fatal(err)
	}
	cleanup(h.Destroy)

	compute32f, err := cutensor.ComputeDesc32F()
	if err != nil {
		t.Fatal(err)
	}

	const m, n, k = 3, 4, 5
	newDesc := func(extent ...int64) cutensor.TensorDescriptor {
		d, err := h.CreateTensorDescriptor(uint32(len(extent)), &extent[0],
			nil, cutensor.R32F, 128)
		if err != nil {
			t.Fatal(err)
		}
		cleanup(d.Destroy)
		return d
	}
	descA := newDesc(m, k)
	descB := newDesc(k, n)
	descC := newDesc(m, n)

	modeA := []int32{'m', 'k'}
	modeB := []int32{'k', 'n'}
	modeC := []int32{'m', 'n'}
	op, err := h.CreateContraction(
		descA, &modeA[0], cutensor.OpIdentity,
		descB, &modeB[0], cutensor.OpIdentity,
		descC, &modeC[0], cutensor.OpIdentity,
		descC, &modeC[0], compute32f)
	if err != nil {
		t.Fatal(err)
	}
	cleanup(op.Destroy)

	pref, err := h.CreatePlanPreference(cutensor.AlgoDefault, cutensor.JitModeNone)
	if err != nil {
		t.Fatal(err)
	}
	cleanup(pref.Destroy)

	wsSize, err := h.EstimateWorkspaceSize(op, pref, cutensor.WorkspaceDefault)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := h.CreatePlan(op, pref, wsSize)
	if err != nil {
		t.Fatal(err)
	}
	cleanup(plan.Destroy)

	a, b, c := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
	for i := range a {
		a[i] = float32(i%7 - 3)
	}
	for i := range b {
		b[i] = float32(i%5 - 2)
	}
	for i := range c {
		c[i] = float32(i + 1)
	}
	want := make([]float32, len(c))
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			v := float32(0)
			for l := 0; l < k; l++ {
				v += a[i+l*m] * b[l+j*k]
			}
			want[i+j*m] = 2*v - c[i+j*m]
		}
	}
	alloc := func(v []float32) driver.DevicePtr {
		p, e := driver.MemAlloc(uint(len(v) * 4))
		check(e)
		cleanup(p.MemFree)
		check(p.MemcpyHtoD(unsafe.Pointer(&v[0]), uint(len(v)*4)))
		return p
	}
	pa, pb, pc := alloc(a), alloc(b), alloc(c)
	var workspace driver.DevicePtr
	if wsSize > 0 {
		workspace, err = driver.MemAlloc(uint(wsSize))
		check(err)
		cleanup(workspace.MemFree)
	}
	alpha, beta := float32(2), float32(-1)
	check(h.Contract(plan, unsafe.Pointer(&alpha), uintptr(pa), uintptr(pb), unsafe.Pointer(&beta), uintptr(pc), uintptr(pc), uintptr(workspace), wsSize, 0))
	check(driver.CtxSynchronize())
	check(pc.MemcpyDtoH(unsafe.Pointer(&c[0]), uint(len(c)*4)))
	for i, v := range c {
		if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > 1e-5 {
			t.Errorf("output %d = %v, want %v", i, v, want[i])
		}
	}
	t.Logf("2*A*B-C matched %d CPU results: %v", len(c), c)
}
