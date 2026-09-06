//go:build linux

package cudss_test

import (
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cudss"
	"github.com/tmc/lib/nvidia/driver"
)

// TestPhysicalMultiGPU exercises cuDSS 0.8 single-process MG setup and solve.
// It shares the hand-computed CSR oracle from solve_physical_test.go.
// Run with CUDSS_MULTIGPU_PHYSICAL_TEST=1 and an external timeout.
func TestPhysicalMultiGPU(t *testing.T) {
	if os.Getenv("CUDSS_MULTIGPU_PHYSICAL_TEST") != "1" {
		t.Skip("set CUDSS_MULTIGPU_PHYSICAL_TEST=1")
	}
	check := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	cleanup := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Errorf("cleanup %s: %v", what, err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check("initialize CUDA", driver.Init(0))
	devices := [2]int32{0, 1}
	var contexts [2]driver.Context
	var streams [2]cudss.Stream
	var uuids [2]driver.Uuid
	var pin runtime.Pinner
	pin.Pin(&devices)
	pin.Pin(&streams)
	defer pin.Unpin()
	defer func() {
		for i := range contexts {
			if contexts[i] == 0 {
				continue
			}
			cleanup("set stream context", contexts[i].SetCurrent())
			if streams[i] != 0 {
				cleanup("destroy stream", driver.Stream(streams[i]).Destroy())
			}
			cleanup("release primary context", driver.Device(i).PrimaryCtxRelease())
		}
		if !t.Failed() {
			t.Log("checked cleanup complete")
		}
	}()
	for i := range contexts {
		dev := driver.Device(i)
		check("UUID", dev.GetUuid(&uuids[i]))
		var err error
		contexts[i], err = dev.PrimaryCtxRetain()
		check("retain context", err)
		check("select device", contexts[i].SetCurrent())
		stream, err := driver.NewStream(0)
		check("create stream", err)
		streams[i] = cudss.Stream(stream)
		t.Logf("device=%d uuid=%x stream=%x", i, uuids[i], streams[i])
	}
	if uuids[0] == uuids[1] {
		t.Fatal("device UUIDs must differ")
	}
	check("select first MG device", contexts[0].SetCurrent())
	handle, err := cudss.CreateMg(2, &devices[0])
	check("create two-device cuDSS handle", err)
	defer func() { cleanup("destroy handle", handle.Destroy()) }()
	check("set per-device streams", handle.SetMgStreams(&streams[0], 2))
	t.Log("CreateMg and SetMgStreams succeeded for devices [0 1]")

	// Everything past this point is the package under test doing real work.
	// A failure here is a failure, not a missing prerequisite.
	var frees []driver.DevicePtr
	defer func() {
		cleanup("restore allocation context", contexts[0].SetCurrent())
		for _, d := range frees {
			if err := d.MemFree(); err != nil {
				t.Errorf("free device memory: %v", err)
			}
		}
	}()
	upload := func(name string, p unsafe.Pointer, n uint) driver.DevicePtr {
		t.Helper()
		d, err := driver.MemAlloc(n)
		if err != nil {
			t.Fatalf("allocate %d bytes for %s: %v", n, name, err)
		}
		frees = append(frees, d)
		if err := d.MemcpyHtoD(p, n); err != nil {
			t.Fatalf("upload %s: %v", name, err)
		}
		return d
	}
	x := make([]float64, system.n)
	for i := range x {
		x[i] = math.NaN()
	}
	dRow := upload("row offsets", unsafe.Pointer(&system.rowOffsets[0]), uint(len(system.rowOffsets))*4)
	dCol := upload("column indices", unsafe.Pointer(&system.colIndices[0]), uint(len(system.colIndices))*4)
	dVal := upload("values", unsafe.Pointer(&system.values[0]), uint(len(system.values))*8)
	dB := upload("right-hand side", unsafe.Pointer(&system.b[0]), uint(len(system.b))*8)
	dX := upload("solution", unsafe.Pointer(&x[0]), uint(len(x))*8)

	config, err := cudss.NewConfig()
	if err != nil {
		t.Fatalf("create config: %v", err)
	}
	defer func() { cleanup("destroy config", config.Destroy()) }()
	check("set device count", config.SetDeviceCount(2))
	check("set device indices", config.SetDeviceIndices(devices[:]))
	count, err := config.DeviceCount()
	check("get device count", err)
	var indices [2]int32
	n, err := config.DeviceIndices(indices[:])
	check("get device indices", err)
	if count != 2 || n != 2 || indices != devices {
		t.Fatalf("MG config count=%d indices=%v n=%d", count, indices, n)
	}
	// Use default reordering: cuDSS MG does not support COLAMD.
	data, err := cudss.NewData(handle)
	if err != nil {
		t.Fatalf("create data: %v", err)
	}
	defer func() { cleanup("destroy data", data.Destroy(handle)) }()

	a, err := cudss.NewMatrixCsr(system.n, system.n, int64(len(system.values)),
		uintptr(dRow), 0, uintptr(dCol), uintptr(dVal),
		cudss.R32I, cudss.R32I, cudss.R64F,
		cudss.MtypeSpd, cudss.MviewFull, cudss.BaseZero)
	if err != nil {
		t.Fatalf("create CSR matrix: %v", err)
	}
	defer func() { cleanup("destroy a", a.Destroy()) }()
	rhs, err := cudss.NewMatrixDn(system.n, 1, system.n, uintptr(dB), cudss.R64F, cudss.LayoutColMajor)
	if err != nil {
		t.Fatalf("create right-hand side: %v", err)
	}
	defer func() { cleanup("destroy rhs", rhs.Destroy()) }()
	sol, err := cudss.NewMatrixDn(system.n, 1, system.n, uintptr(dX), cudss.R64F, cudss.LayoutColMajor)
	if err != nil {
		t.Fatalf("create solution: %v", err)
	}
	defer func() { cleanup("destroy sol", sol.Destroy()) }()

	for _, phase := range []cudss.Phase{cudss.PhaseAnalysis, cudss.PhaseFactorization, cudss.PhaseSolve} {
		if err := handle.Execute(phase, config, data, a, sol, rhs); err != nil {
			t.Fatalf("%v: %v", phase, err)
		}
		// MG phases are synchronous. Also check both user streams before
		// reading the phase status.
		for i := range contexts {
			check("select phase context", contexts[i].SetCurrent())
			check("synchronize phase stream", driver.Stream(streams[i]).Synchronize())
		}
		check("restore first device", contexts[0].SetCurrent())
		info, err := data.Info(handle)
		if err != nil {
			t.Fatalf("read CUDSS_DATA_INFO after %v: %v", phase, err)
		}
		if info != 0 {
			t.Fatalf("%v reported device-side status %d, want 0", phase, info)
		}
	}

	if err := dX.MemcpyDtoH(unsafe.Pointer(&x[0]), uint(len(x))*8); err != nil {
		t.Fatalf("download solution: %v", err)
	}

	// Solution error against the exact answer, and residual against the CPU
	// reference. Both are checked: a solver can return a small residual for a
	// badly scaled system while being far from the true solution, and this
	// system is well conditioned enough that neither excuse applies.
	const tol = 1e-12
	for i, got := range x {
		if math.IsNaN(got) || math.Abs(got-system.want[i]) > tol {
			t.Errorf("x[%d] = %.17g, want %.17g", i, got, system.want[i])
		}
	}
	if r := residual(x); r > tol {
		t.Errorf("residual |b - A*x| = %.17g, want <= %g", r, tol)
	}

	t.Logf("solution=%v residual=%g", x, residual(x))

}
