//go:build linux && cgo && nvidiampi

package cudensitymat_test

import (
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cudensitymat"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/mpitest"
)

// TestDistributedNorm checks a partitioned eight-amplitude state's global norm.
// Run under mpiexec -n 2 with CUDENSITYMAT_DISTRIBUTED_TEST=1 and a matching
// CUDA-aware MPI provider.
func TestDistributedNorm(t *testing.T) {
	if os.Getenv("CUDENSITYMAT_DISTRIBUTED_TEST") != "1" {
		t.Skip("set CUDENSITYMAT_DISTRIBUTED_TEST=1")
	}
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
		t.Fatalf("need two ranks, got %d", mpi.Size)
	}
	check("CUDA init", driver.Init(0))
	dev := driver.Device(mpi.Rank)
	uuid, err := dev.GetUuid()
	check("UUID", err)
	t.Logf("rank=%d device=%d uuid=%x", mpi.Rank, dev, uuid)
	ids := make([]byte, 32)
	check("gather UUID", mpi.AllGather(unsafe.Slice((*byte)(unsafe.Pointer(&uuid)), 16), ids))
	if string(ids[:16]) == string(ids[16:]) {
		t.Fatal("GPU UUIDs must differ")
	}
	ctx, err := dev.PrimaryCtxRetain()
	check("context", err)
	defer func() { clean("context release", dev.PrimaryCtxRelease()) }()
	check("select context", ctx.SetCurrent())
	stream, err := driver.NewStream(0)
	check("stream", err)
	defer func() { clean("stream destroy", stream.Destroy()) }()
	h, err := cudensitymat.Create()
	check("handle", err)
	defer func() { clean("handle destroy", h.Destroy()) }()
	check("import duplicated native MPI communicator", h.ResetDistributedConfiguration(cudensitymat.DistributedProviderMpi, mpi.Pointer(), mpi.Bytes()))
	extents := [3]int64{2, 2, 2}
	state, err := h.CreateState(cudensitymat.StatePurityPure, 3, &extents[0], 1, cudensitymat.C64F)
	check("state", err)
	var allocations []driver.DevicePtr
	defer func() {
		for _, p := range allocations {
			clean("free device buffer", p.MemFree())
		}
	}()
	defer func() { clean("state destroy", state.Destroy()) }()
	alloc := func(n uint) driver.DevicePtr {
		t.Helper()
		p, e := driver.MemAlloc(n)
		check("allocate", e)
		allocations = append(allocations, p)
		return p
	}
	count, err := state.GetNumComponents(h)
	check("component count", err)
	if count != 1 {
		t.Fatalf("expected one local component, got %d", count)
	}
	var bytes uint
	check("storage size", state.GetComponentStorageSize(h, 1, &bytes))
	if bytes == 0 || bytes > 4096 {
		t.Fatalf("storage bound %d", bytes)
	}
	_, nmodes, batchMode, err := state.GetComponentNumModes(h, 0)
	check("mode count", err)
	if nmodes < 3 || nmodes > 4 {
		t.Fatalf("mode count %d", nmodes)
	}
	if !((nmodes == 3 && batchMode == -1) || (nmodes == 4 && batchMode == 3)) {
		t.Fatalf("unsupported batch mode %d for %d modes", batchMode, nmodes)
	}
	shape := make([]int64, nmodes)
	offset := make([]int64, nmodes)
	var global, actualModes int32
	check("component info", state.GetComponentInfo(h, 0, &global, &actualModes, &shape[0], &offset[0]))
	if actualModes != nmodes {
		t.Fatal("inconsistent mode count")
	}
	n := int64(1)
	for _, e := range shape {
		n *= e
	}
	if n <= 0 || n >= 8 || uint(n*16) > bytes {
		t.Fatalf("expected distributed nonempty slice: shape=%v bytes=%d", shape, bytes)
	}
	if batchMode >= 0 && (shape[batchMode] != 1 || offset[batchMode] != 0) {
		t.Fatal("expected singleton batch mode")
	}
	coverage := make([]byte, 8)
	host := make([]complex128, bytes/16)
	for i := int64(0); i < n; i++ {
		q := i
		g := int64(0)
		stride := int64(1)
		for m, e := range shape {
			coord := q%e + offset[m]
			q /= e
			if m < 3 {
				if coord < 0 || coord >= 2 {
					t.Fatal("bad component offset")
				}
				g += coord * stride
				stride *= 2
			}
		}
		coverage[g]++
		host[i] = complex(float64(g+1), 0)
	}
	allCoverage := make([]byte, 16)
	check("gather index coverage", mpi.AllGather(coverage, allCoverage))
	for i := 0; i < 8; i++ {
		if int(allCoverage[i])+int(allCoverage[8+i]) != 1 {
			t.Fatalf("global index %d coverage %d + %d, want 1", i, allCoverage[i], allCoverage[8+i])
		}
	}
	ptr := alloc(bytes)
	check("upload slice", ptr.MemcpyHtoD(unsafe.Pointer(&host[0]), bytes))
	address := uintptr(ptr)
	check("attach", state.AttachComponentStorage(h, 1, &address, &bytes))
	out := alloc(8)
	sentinel := math.NaN()
	check("norm sentinel", out.MemcpyHtoD(unsafe.Pointer(&sentinel), 8))
	check("global norm", state.ComputeNorm(h, uintptr(out), cudensitymat.Stream(stream)))
	check("synchronize", stream.Synchronize())
	var got float64
	check("download norm", out.MemcpyDtoH(unsafe.Pointer(&got), 8))
	want := float64(204)
	if os.Getenv("CUDENSITYMAT_BAD_ORACLE") == "1" {
		want++
	}
	if math.IsNaN(got) || math.Abs(got-want) > 1e-10 {
		t.Fatalf("global norm %g want %g", got, want)
	}
	t.Logf("rank=%d shape=%v offsets=%v local=%d global norm=%g", mpi.Rank, shape, offset, n, got)
}
