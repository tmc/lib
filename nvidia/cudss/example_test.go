//go:build linux
// +build linux

package cudss_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cudss"
	"github.com/tmc/lib/nvidia/driver"
)

// Example solves a 4x4 symmetric positive definite system in CSR form on one
// GPU, running the three phases in order: analysis, factorization, solve.
//
// The device allocations are made with the driver package. cuDSS records the
// addresses it is given and does not copy them, so every allocation below has
// to stay alive and unmodified until the last Execute has completed on the
// stream -- which is why the frees are deferred to the end of the function and
// not to the end of each phase.
//
// It prints an error and returns when libcudss or a CUDA device is not
// available, so it is useful as a compile-time check on a machine with no GPU.
func Example() {
	if err := driver.Init(0); err != nil {
		fmt.Println("CUDA unavailable:", err)
		return
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		fmt.Println("no CUDA device:", err)
		return
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		fmt.Println("context:", err)
		return
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		fmt.Println("context:", err)
		return
	}

	// A is the 4x4 matrix with 4 on the diagonal and 1 on the first
	// off-diagonals, stored as full (not triangular) CSR with 0-based indices.
	rowOffsets := []int32{0, 2, 5, 8, 10}
	colIndices := []int32{0, 1, 0, 1, 2, 1, 2, 3, 2, 3}
	values := []float64{4, 1, 1, 4, 1, 1, 4, 1, 1, 4}
	// b = A * [1 2 3 4].
	b := []float64{6, 12, 18, 19}
	x := make([]float64, 4)

	upload := func(p unsafe.Pointer, n uint) (driver.DevicePtr, error) {
		d, err := driver.MemAlloc(n)
		if err != nil {
			return 0, err
		}
		if err := d.MemcpyHtoD(p, n); err != nil {
			d.MemFree()
			return 0, err
		}
		return d, nil
	}

	var frees []driver.DevicePtr
	defer func() {
		for _, d := range frees {
			d.MemFree()
		}
	}()
	alloc := func(p unsafe.Pointer, n uint) driver.DevicePtr {
		d, err := upload(p, n)
		if err != nil {
			panic(err)
		}
		frees = append(frees, d)
		return d
	}
	dRow := alloc(unsafe.Pointer(&rowOffsets[0]), uint(len(rowOffsets))*4)
	dCol := alloc(unsafe.Pointer(&colIndices[0]), uint(len(colIndices))*4)
	dVal := alloc(unsafe.Pointer(&values[0]), uint(len(values))*8)
	dB := alloc(unsafe.Pointer(&b[0]), uint(len(b))*8)
	dX := alloc(unsafe.Pointer(&x[0]), uint(len(x))*8)

	handle, err := cudss.NewHandle()
	if err != nil {
		fmt.Println("cuDSS unavailable:", err)
		return
	}
	defer handle.Destroy()

	config, err := cudss.NewConfig()
	if err != nil {
		fmt.Println("config:", err)
		return
	}
	defer config.Destroy()
	// Every configuration parameter has a documented C type. The accessor
	// takes the matching Go type and sizes the call from it, so the byte count
	// the library is given always describes the storage it is given.
	if err := config.SetPivotType(cudss.PivotNone); err != nil {
		fmt.Println("pivot type:", err)
		return
	}

	data, err := cudss.NewData(handle)
	if err != nil {
		fmt.Println("data:", err)
		return
	}
	defer data.Destroy(handle)

	a, err := cudss.NewMatrixCsr(4, 4, 10,
		uintptr(dRow), 0, uintptr(dCol), uintptr(dVal),
		cudss.R32I, cudss.R32I, cudss.R64F,
		cudss.MtypeSpd, cudss.MviewFull, cudss.BaseZero)
	if err != nil {
		fmt.Println("matrix:", err)
		return
	}
	defer a.Destroy()
	rhs, err := cudss.NewMatrixDn(4, 1, 4, uintptr(dB), cudss.R64F, cudss.LayoutColMajor)
	if err != nil {
		fmt.Println("rhs:", err)
		return
	}
	defer rhs.Destroy()
	sol, err := cudss.NewMatrixDn(4, 1, 4, uintptr(dX), cudss.R64F, cudss.LayoutColMajor)
	if err != nil {
		fmt.Println("solution:", err)
		return
	}
	defer sol.Destroy()

	for _, phase := range []cudss.Phase{
		cudss.PhaseAnalysis,
		cudss.PhaseFactorization,
		cudss.PhaseSolve,
	} {
		if err := handle.Execute(phase, config, data, a, sol, rhs); err != nil {
			fmt.Printf("%v: %v\n", phase, err)
			return
		}
	}
	// Execute enqueues work; nothing above guarantees it has run.
	if err := driver.CtxSynchronize(); err != nil {
		fmt.Println("synchronize:", err)
		return
	}
	if err := dX.MemcpyDtoH(unsafe.Pointer(&x[0]), uint(len(x))*8); err != nil {
		fmt.Println("download:", err)
		return
	}

	nnz, err := data.LUNonzeros(handle)
	if err != nil {
		fmt.Println("lu nnz:", err)
		return
	}
	fmt.Printf("x = %.3f, factor nnz = %d\n", x, nnz)
}
