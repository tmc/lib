package cutensornet_test

import (
	"errors"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cutensornet"
	"github.com/tmc/lib/nvidia/driver"
)

// This file holds the physical consumer test: a small deterministic tensor
// network contracted on a GPU and compared against a result computed
// independently in Go.
//
// IT HAS NEVER BEEN RUN. The machine this package was written on has no NVIDIA
// GPU, no CUDA driver and no cuQuantum installation, so nothing below is
// evidence that the bindings work. It is written so that someone with the
// hardware can run it and find out.
//
// It is opt-in: without CUTENSORNET_PHYSICAL_TEST=1 it skips, naming every
// prerequisite. With the variable set it fails loudly at the first missing
// piece rather than skipping, so a machine that was supposed to have the
// hardware cannot report a pass by quietly doing nothing.

// physicalPrereqs lists what the physical test needs. It is printed by the skip
// so that a skipped run says exactly what is missing.
const physicalPrereqs = `the physical contraction test needs, on the machine running it:
  - an NVIDIA GPU of compute capability 7.0 (Volta) or later
  - a CUDA driver, providing libcuda.so.1
  - libcutensornet.so.2 from cuQuantum (cuTensorNet 2.x), on the loader path
    or named by CUTENSORNET_LIB_PATH
  - libcutensor.so.2, which libcutensornet.so.2 links against
  - enough free device memory for the contraction workspace, a few MiB
set CUTENSORNET_PHYSICAL_TEST=1 to run it; it then fails rather than skips`

// TestContractionMatchesIndependentResult contracts C[m,n] = A[m,k] B[k,n] in
// double precision on the device and compares the result against the same
// product computed in Go.
//
// The operands hold small integers, so every intermediate is exact in float64
// and the comparison is for equality, not for a tolerance. A tolerance here
// would let a wrong contraction order or a misread stride pass.
func TestContractionMatchesIndependentResult(t *testing.T) {
	if os.Getenv("CUTENSORNET_PHYSICAL_TEST") != "1" {
		t.Skip(physicalPrereqs)
	}
	// The CUDA driver API binds a context to the calling OS thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := driver.Init(0); err != nil {
		t.Fatalf("cuInit: %v", err)
	}
	n, err := driver.DeviceGetCount()
	if err != nil {
		t.Fatalf("cuDeviceGetCount: %v", err)
	}
	if n == 0 {
		t.Fatal("no CUDA device present; the physical test cannot run here")
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Fatalf("cuDeviceGet(0): %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("cuDevicePrimaryCtxRetain: %v", err)
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("cuCtxSetCurrent: %v", err)
	}

	// The network: C[m,n] = sum_k A[m,k] B[k,n], with M=2, K=3, N=2.
	const (
		M, K, N = 2, 3, 2
		modeM   = 'm'
		modeK   = 'k'
		modeN   = 'n'
	)
	// Column-major (Fortran) storage, stated explicitly rather than relying on
	// the library's default, so the host layout below is unambiguous.
	a := cutensornet.Tensor{
		Modes:   []int32{modeM, modeK},
		Extents: []int64{M, K},
		Strides: []int64{1, M},
	}
	b := cutensornet.Tensor{
		Modes:   []int32{modeK, modeN},
		Extents: []int64{K, N},
		Strides: []int64{1, K},
	}
	c := cutensornet.Tensor{
		Modes:   []int32{modeM, modeN},
		Extents: []int64{M, N},
		Strides: []int64{1, M},
	}

	// Column-major host data. hostA[m + k*M], hostB[k + n*K].
	hostA := []float64{
		1, 4, // column k=0: A[0,0]=1 A[1,0]=4
		2, 5, // column k=1
		3, 6, // column k=2
	}
	hostB := []float64{
		7, 9, 11, // column n=0
		8, 10, 12, // column n=1
	}
	want := make([]float64, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float64
			for k := 0; k < K; k++ {
				sum += hostA[i+k*M] * hostB[k+j*K]
			}
			want[i+j*M] = sum
		}
	}

	h, err := cutensornet.Create()
	if err != nil {
		t.Fatalf("cutensornetCreate: %v", err)
	}
	defer h.Destroy()
	t.Logf("cuTensorNet version %d, built against CUDA runtime %d, loaded from %s",
		cutensornet.GetVersion(), cutensornet.GetCudartVersion(), cutensornet.Path())

	net, err := h.NewNetwork([]cutensornet.Tensor{a, b}, &c, cutensornet.R64F, cutensornet.Compute64F)
	if err != nil {
		t.Fatalf("NewNetwork: %v", err)
	}
	defer func() {
		if err := net.Close(); err != nil {
			t.Errorf("Network.Close: %v", err)
		}
	}()

	optCfg, err := h.CreateContractionOptimizerConfig()
	if err != nil {
		t.Fatalf("CreateContractionOptimizerConfig: %v", err)
	}
	defer optCfg.Destroy()
	info, err := h.CreateContractionOptimizerInfo(net.Descriptor())
	if err != nil {
		t.Fatalf("CreateContractionOptimizerInfo: %v", err)
	}
	defer info.Destroy()

	const workspaceLimit = 64 << 20
	if err := h.ContractionOptimize(net.Descriptor(), optCfg, workspaceLimit, info); err != nil {
		t.Fatalf("ContractionOptimize: %v", err)
	}
	slices, err := info.Int64(h, cutensornet.ContractionOptimizerInfoNumSlices)
	if err != nil {
		t.Fatalf("read CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES: %v", err)
	}
	if slices < 1 {
		t.Fatalf("optimizer reported %d slices, want at least 1", slices)
	}

	ws, err := h.NewWorkspace()
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	if err := ws.ComputeContractionSizes(net, info); err != nil {
		t.Fatalf("WorkspaceComputeContractionSizes: %v", err)
	}
	scratch, err := ws.Size(cutensornet.WorksizePrefMin, cutensornet.MemspaceDevice, cutensornet.WorkspaceScratch)
	if err != nil {
		t.Fatalf("WorkspaceGetMemorySize: %v", err)
	}
	if scratch < 0 {
		t.Fatalf("workspace size = %d, want >= 0", scratch)
	}
	if scratch > 0 {
		dW, err := driver.MemAlloc(uint(scratch))
		if err != nil {
			t.Fatalf("cuMemAlloc(%d) for the workspace: %v", scratch, err)
		}
		if err := ws.Provide(cutensornet.MemspaceDevice, cutensornet.WorkspaceScratch,
			uintptr(dW), scratch, func() {
				if err := dW.MemFree(); err != nil {
					t.Errorf("cuMemFree(workspace): %v", err)
				}
			}); err != nil {
			dW.MemFree()
			t.Fatalf("WorkspaceSetMemory: %v", err)
		}
	}

	dA, err := driver.MemAlloc(uint(len(hostA)) * 8)
	if err != nil {
		t.Fatalf("cuMemAlloc(A): %v", err)
	}
	defer dA.MemFree()
	dB, err := driver.MemAlloc(uint(len(hostB)) * 8)
	if err != nil {
		t.Fatalf("cuMemAlloc(B): %v", err)
	}
	defer dB.MemFree()
	dC, err := driver.MemAlloc(uint(M*N) * 8)
	if err != nil {
		t.Fatalf("cuMemAlloc(C): %v", err)
	}
	defer dC.MemFree()
	if err := dA.MemcpyHtoD(unsafe.Pointer(&hostA[0]), uint(len(hostA))*8); err != nil {
		t.Fatalf("cuMemcpyHtoD(A): %v", err)
	}
	if err := dB.MemcpyHtoD(unsafe.Pointer(&hostB[0]), uint(len(hostB))*8); err != nil {
		t.Fatalf("cuMemcpyHtoD(B): %v", err)
	}

	plan, err := ws.NewPlan(net, info)
	if err != nil {
		t.Fatalf("CreateContractionPlan: %v", err)
	}

	// The ownership contract, checked against the real library: while the plan
	// is live neither the workspace nor the network may be released.
	if err := ws.Close(); !errors.Is(err, cutensornet.ErrWorkspaceInUse) {
		t.Errorf("Workspace.Close with a live plan = %v, want ErrWorkspaceInUse", err)
	}
	if err := net.Close(); !errors.Is(err, cutensornet.ErrNetworkInUse) {
		t.Errorf("Network.Close with a live plan = %v, want ErrNetworkInUse", err)
	}

	stream, err := driver.NewStream(0)
	if err != nil {
		t.Fatalf("cuStreamCreate: %v", err)
	}
	defer stream.Destroy()

	in := []uintptr{uintptr(dA), uintptr(dB)}
	if err := plan.Contract(in, uintptr(dC), false, 0, cutensornet.Stream(stream)); err != nil {
		t.Fatalf("ContractSlices: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("cuStreamSynchronize: %v", err)
	}

	got := make([]float64, M*N)
	if err := dC.MemcpyDtoH(unsafe.Pointer(&got[0]), uint(M*N)*8); err != nil {
		t.Fatalf("cuMemcpyDtoH(C): %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("C[%d] = %v, want %v (whole result %v, want %v)", i, got[i], want[i], got, want)
		}
	}

	// Release in the order the contract requires: plan, then workspace, then
	// network. The stream is already synchronized, so the workspace buffer is
	// no longer in use by the device.
	if err := plan.Close(); err != nil {
		t.Errorf("Plan.Close: %v", err)
	}
	if err := ws.Close(); err != nil {
		t.Errorf("Workspace.Close: %v", err)
	}
}
