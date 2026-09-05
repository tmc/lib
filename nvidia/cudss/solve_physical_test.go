//go:build linux
// +build linux

package cudss_test

import (
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cudss"
	"github.com/tmc/lib/nvidia/driver"
)

// This file is the physical consumer test. It has never been run: the machine
// it was written on has no NVIDIA GPU, no CUDA driver and no libcudss, so
// every path below stops at the first missing prerequisite.
//
// Running it needs all of:
//
//   - an NVIDIA GPU with compute capability 7.0 or newer, and the CUDA driver
//     (libcuda.so.1) that exposes it;
//   - libcudss.so.0 from the cuDSS 0.8 runtime on the loader path, or named by
//     CUDSS_LIB_PATH.
//
// By default a missing prerequisite skips with that prerequisite named. That
// skip is not evidence of anything, and a skipped run must never be reported
// as a passing one. Set CUDSS_REQUIRE_GPU=1 to turn every skip below into a
// failure, which is how this should be run anywhere the hardware is supposed
// to be present.

// requirement stops the test for a missing prerequisite: a failure when
// CUDSS_REQUIRE_GPU is set, a skip naming the prerequisite otherwise.
func requirement(t *testing.T, what string, err error) {
	t.Helper()
	if err == nil {
		return
	}
	if os.Getenv("CUDSS_REQUIRE_GPU") != "" {
		t.Fatalf("CUDSS_REQUIRE_GPU is set but %s is unavailable: %v", what, err)
	}
	t.Skipf("physical prerequisite unavailable: %s: %v", what, err)
}

// system is the test problem: the 4x4 matrix with 4 on the diagonal and 1 on
// the first off-diagonals, which is symmetric, positive definite and strictly
// diagonally dominant, so it is nonsingular and needs no pivoting. It is held
// as full CSR with 0-based indices.
//
// b was computed by hand as A*want, so want is the exact solution and the
// reference below is not a second implementation of the same mistake.
var system = struct {
	n          int64
	rowOffsets []int32
	colIndices []int32
	values     []float64
	b          []float64
	want       []float64
}{
	n:          4,
	rowOffsets: []int32{0, 2, 5, 8, 10},
	colIndices: []int32{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
	values:     []float64{4, 1, 1, 4, 1, 1, 4, 1, 1, 4},
	b:          []float64{6, 12, 18, 19},
	want:       []float64{1, 2, 3, 4},
}

// residual returns the infinity norm of b - A*x, computed on the host from the
// same CSR arrays that were handed to the solver.
func residual(x []float64) float64 {
	var worst float64
	for i := range system.b {
		sum := 0.0
		for k := system.rowOffsets[i]; k < system.rowOffsets[i+1]; k++ {
			sum += system.values[k] * x[system.colIndices[k]]
		}
		if d := math.Abs(system.b[i] - sum); d > worst {
			worst = d
		}
	}
	return worst
}

func TestSolveCSR(t *testing.T) {
	requirement(t, "CUDA driver (libcuda.so.1)", driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	requirement(t, "an NVIDIA GPU at device ordinal 0", err)
	ctx, err := dev.PrimaryCtxRetain()
	requirement(t, "a CUDA primary context on device 0", err)
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}

	handle, err := cudss.NewHandle()
	requirement(t, "libcudss.so.0 (cuDSS 0.8 runtime)", err)
	defer handle.Destroy()

	// Everything past this point is the package under test doing real work.
	// A failure here is a failure, not a missing prerequisite.
	var frees []driver.DevicePtr
	defer func() {
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
	dRow := upload("row offsets", unsafe.Pointer(&system.rowOffsets[0]), uint(len(system.rowOffsets))*4)
	dCol := upload("column indices", unsafe.Pointer(&system.colIndices[0]), uint(len(system.colIndices))*4)
	dVal := upload("values", unsafe.Pointer(&system.values[0]), uint(len(system.values))*8)
	dB := upload("right-hand side", unsafe.Pointer(&system.b[0]), uint(len(system.b))*8)
	dX := upload("solution", unsafe.Pointer(&x[0]), uint(len(x))*8)

	config, err := cudss.NewConfig()
	if err != nil {
		t.Fatalf("create config: %v", err)
	}
	defer config.Destroy()
	data, err := cudss.NewData(handle)
	if err != nil {
		t.Fatalf("create data: %v", err)
	}
	defer data.Destroy(handle)

	a, err := cudss.NewMatrixCsr(system.n, system.n, int64(len(system.values)),
		uintptr(dRow), 0, uintptr(dCol), uintptr(dVal),
		cudss.R32I, cudss.R32I, cudss.R64F,
		cudss.MtypeSpd, cudss.MviewFull, cudss.BaseZero)
	if err != nil {
		t.Fatalf("create CSR matrix: %v", err)
	}
	defer a.Destroy()
	rhs, err := cudss.NewMatrixDn(system.n, 1, system.n, uintptr(dB), cudss.R64F, cudss.LayoutColMajor)
	if err != nil {
		t.Fatalf("create right-hand side: %v", err)
	}
	defer rhs.Destroy()
	sol, err := cudss.NewMatrixDn(system.n, 1, system.n, uintptr(dX), cudss.R64F, cudss.LayoutColMajor)
	if err != nil {
		t.Fatalf("create solution: %v", err)
	}
	defer sol.Destroy()

	for _, phase := range []cudss.Phase{cudss.PhaseAnalysis, cudss.PhaseFactorization, cudss.PhaseSolve} {
		if err := handle.Execute(phase, config, data, a, sol, rhs); err != nil {
			t.Fatalf("%v: %v", phase, err)
		}
		// Execute only enqueues; the device-side status of a phase is not
		// meaningful until the stream has drained.
		if err := driver.CtxSynchronize(); err != nil {
			t.Fatalf("synchronize after %v: %v", phase, err)
		}
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
		if math.Abs(got-system.want[i]) > tol {
			t.Errorf("x[%d] = %.17g, want %.17g", i, got, system.want[i])
		}
	}
	if r := residual(x); r > tol {
		t.Errorf("residual |b - A*x| = %.17g, want <= %g", r, tol)
	}

	// The typed parameter accessors are the part of this package that a wrong
	// size would corrupt silently, and only a real libcudss can show that the
	// sizes are right. Reading them here is the check.
	nnz, err := data.LUNonzeros(handle)
	if err != nil {
		t.Fatalf("read CUDSS_DATA_LU_NNZ: %v", err)
	}
	if nnz < int64(len(system.values)) {
		t.Errorf("factor holds %d nonzeros, fewer than the %d in A", nnz, len(system.values))
	}
	estimates, err := data.MemoryEstimates(handle)
	if err != nil {
		t.Fatalf("read CUDSS_DATA_MEMORY_ESTIMATES: %v", err)
	}
	if estimates[0] <= 0 {
		t.Errorf("memory estimates = %v, want a positive first entry", estimates)
	}
	perm := make([]int32, system.n)
	n, err := cudss.RowPermutation(handle, data, perm)
	if err != nil {
		t.Fatalf("read CUDSS_DATA_PERM_ROW: %v", err)
	}
	if n != int(system.n) {
		t.Errorf("row permutation has %d entries, want %d", n, system.n)
	}
	seen := make(map[int32]bool, n)
	for _, p := range perm[:n] {
		if p < 0 || p >= int32(system.n) || seen[p] {
			t.Fatalf("row permutation %v is not a permutation of 0..%d", perm[:n], system.n-1)
		}
		seen[p] = true
	}
}
