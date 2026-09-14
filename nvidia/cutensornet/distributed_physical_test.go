//go:build linux && cgo && nvidiampi

package cutensornet_test

import (
	"errors"
	"github.com/tmc/lib/nvidia/cutensornet"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/mpitest"
	"os"
	"runtime"
	"testing"
	"unsafe"
)

// TestDistributedContraction requires a CUDA-aware MPI and the matching provider.
// Run with mpiexec -n 2 and CUTENSORNET_DISTRIBUTED_TEST=1.
func TestDistributedContraction(t *testing.T) {
	if os.Getenv("CUTENSORNET_DISTRIBUTED_TEST") != "1" {
		t.Skip(physicalPrereqs)
	}
	// The CUDA driver API binds a context to the calling OS thread.
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
	mpi, err := mpitest.Init()
	check("MPI init", err)
	defer func() { clean("MPI finalize", mpi.Close()) }()
	if mpi.Size != 2 {
		t.Fatalf("need 2 ranks, got %d", mpi.Size)
	}

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
	dev, err := driver.NewDeviceGet(mpi.Rank)
	if err != nil {
		t.Fatalf("cuDeviceGet(0): %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("cuDevicePrimaryCtxRetain: %v", err)
	}
	defer func() { clean("dev.PrimaryCtxRelease()", dev.PrimaryCtxRelease()) }()
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
	defer func() { clean("handle destroy", h.Destroy()) }()
	check("distributed reset with duplicated MPI communicator", h.DistributedResetConfiguration(mpi.Pointer(), mpi.Bytes()))
	size, err := h.DistributedGetNumRanks()
	check("rank count", err)
	rank, err := h.DistributedGetProcRank()
	check("rank", err)
	if size != 2 || int(rank) != mpi.Rank {
		t.Fatalf("rank=%d size=%d", rank, size)
	}
	uuid, err := dev.GetUuid()
	check("UUID", err)
	t.Logf("rank=%d device=%d uuid=%x", rank, dev, uuid)
	ids := make([]byte, 32)
	check("gather UUID", mpi.AllGather(unsafe.Slice((*byte)(unsafe.Pointer(&uuid)), 16), ids))
	if string(ids[:16]) == string(ids[16:]) {
		t.Fatal("GPU UUIDs must differ")
	}
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
	defer func() { clean("optimizer config", optCfg.Destroy()) }()
	check("minimum slices", optCfg.SetInt32(h, cutensornet.ContractionOptimizerConfigSlicerMinSlices, 2))
	info, err := h.CreateContractionOptimizerInfo(net.Descriptor())
	if err != nil {
		t.Fatalf("CreateContractionOptimizerInfo: %v", err)
	}
	defer func() { clean("info.Destroy()", info.Destroy()) }()

	const workspaceLimit = 64 << 20
	if err := h.ContractionOptimize(net.Descriptor(), optCfg, workspaceLimit, info); err != nil {
		t.Fatalf("ContractionOptimize: %v", err)
	}
	slices, err := info.Int64(h, cutensornet.ContractionOptimizerInfoNumSlices)
	if err != nil {
		t.Fatalf("read CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES: %v", err)
	}
	if slices < 2 {
		t.Fatalf("optimizer reported %d slices, want at least 2", slices)
	}

	ws, err := h.NewWorkspace()
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	defer func() { clean("workspace close", ws.Close()) }()
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
			clean("free rejected workspace", dW.MemFree())
			t.Fatalf("WorkspaceSetMemory: %v", err)
		}
	}

	dA, err := driver.MemAlloc(uint(len(hostA)) * 8)
	if err != nil {
		t.Fatalf("cuMemAlloc(A): %v", err)
	}
	defer func() { clean("dA.MemFree()", dA.MemFree()) }()
	dB, err := driver.MemAlloc(uint(len(hostB)) * 8)
	if err != nil {
		t.Fatalf("cuMemAlloc(B): %v", err)
	}
	defer func() { clean("dB.MemFree()", dB.MemFree()) }()
	dC, err := driver.MemAlloc(uint(M*N) * 8)
	if err != nil {
		t.Fatalf("cuMemAlloc(C): %v", err)
	}
	defer func() { clean("dC.MemFree()", dC.MemFree()) }()
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

	defer func() { clean("plan close", plan.Close()) }()

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
	defer func() {
		clean("stream cleanup synchronize", stream.Synchronize())
		clean("stream.Destroy()", stream.Destroy())
	}()

	in := []uintptr{uintptr(dA), uintptr(dB)}
	if err := plan.Contract(in, uintptr(dC), false, 0, cutensornet.Stream(stream)); err != nil {
		t.Fatalf("ContractSlices: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("cuStreamSynchronize: %v", err)
	}

	check("distributed synchronize", h.DistributedSynchronize())
	got := make([]float64, M*N)
	if err := dC.MemcpyDtoH(unsafe.Pointer(&got[0]), uint(M*N)*8); err != nil {
		t.Fatalf("cuMemcpyDtoH(C): %v", err)
	}
	t.Logf("rank=%d slices=%d result=%v", mpi.Rank, slices, got)
	for i := range want {
		if os.Getenv("CUTENSORNET_BAD_ORACLE") == "1" {
			want[i]++
		}
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
