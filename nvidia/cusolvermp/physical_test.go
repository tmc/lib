//go:build linux

package cusolvermp_test

import (
	"encoding/hex"
	"os"
	"runtime"
	"strconv"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/cusolvermp"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// This file holds the physical consumer test: the one that runs a real
// distributed factorization and solve on real GPUs and checks the answer.
//
// It has never been executed. The author had no GPU, no multi-rank launcher
// and no cuSOLVERMp runtime, so nothing below has been observed to work. Treat
// a first run as debugging, not as validation.
//
// Running it requires all of the following, and the test names each one it
// cannot find rather than skipping quietly:
//
//   - libcusolverMp.so.0 from the cuSOLVERMp 0.9.1 redistributable, plus the
//     libnccl.so.2 and libcuda.so.1 it links against;
//   - at least two NVIDIA GPUs with a driver new enough for the CUDA 12
//     build, reachable from this process;
//   - a launcher that starts one process per GPU and gives every process the
//     same NCCL unique ID, its own rank, and the grid shape, through the
//     environment variables below.
//
// A single-rank run is accepted and exercises the same code path on a 1x1
// grid, but it does not test anything distributed. A run with fewer ranks than
// CUSOLVERMP_GRID_ROWS*CUSOLVERMP_GRID_COLS fails.
//
// Environment:
//
//	CUSOLVERMP_PHYSICAL_TEST   set to any non-empty value to opt in
//	CUSOLVERMP_RANK            this process's rank, 0-based
//	CUSOLVERMP_WORLD_SIZE      total rank count
//	CUSOLVERMP_DEVICE          CUDA device ordinal this rank owns
//	CUSOLVERMP_GRID_ROWS       process grid rows
//	CUSOLVERMP_GRID_COLS       process grid columns
//	CUSOLVERMP_NCCL_UNIQUE_ID  hex of the 128-byte ncclUniqueId from rank 0
const physicalEnv = "CUSOLVERMP_PHYSICAL_TEST"

// TestPhysicalGetrfSolve factorizes a diagonally scaled system whose solution
// is known exactly and checks the answer that comes back.
//
// A = 2*I of order n and B = 2, so X must be all ones. That is a weak numerical
// test on purpose: the point is to exercise the distributed path end to end,
// not the accuracy of the factorization.
func TestPhysicalGetrfSolve(t *testing.T) {
	if os.Getenv(physicalEnv) == "" {
		t.Skipf("physical test not requested: set %s=1 and run under a launcher that provides %s, %s, %s, %s, %s and %s, on a host with libcusolverMp.so.0 and one NVIDIA GPU per rank",
			physicalEnv, "CUSOLVERMP_RANK", "CUSOLVERMP_WORLD_SIZE", "CUSOLVERMP_DEVICE",
			"CUSOLVERMP_GRID_ROWS", "CUSOLVERMP_GRID_COLS", "CUSOLVERMP_NCCL_UNIQUE_ID")
	}
	// Past this point a missing prerequisite is a failure. The caller asked
	// for a physical run, so reporting success without one would be a lie.

	// cuSOLVERMp binds the calling thread's device and context.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	rank := mustEnvInt32(t, "CUSOLVERMP_RANK")
	world := mustEnvInt32(t, "CUSOLVERMP_WORLD_SIZE")
	device := mustEnvInt32(t, "CUSOLVERMP_DEVICE")
	nprow := mustEnvInt32(t, "CUSOLVERMP_GRID_ROWS")
	npcol := mustEnvInt32(t, "CUSOLVERMP_GRID_COLS")
	if nprow*npcol != world {
		t.Fatalf("grid %dx%d does not cover %d ranks", nprow, npcol, world)
	}
	if rank < 0 || rank >= world {
		t.Fatalf("rank %d is outside a world of %d", rank, world)
	}
	uniqueID := mustEnvUniqueID(t, "CUSOLVERMP_NCCL_UNIQUE_ID")

	if err := cusolvermp.Load(""); err != nil {
		t.Fatalf("libcusolverMp is required for a physical run: %v", err)
	}
	for _, symbol := range []string{
		"cusolverMpCreate", "cusolverMpCreateDeviceGrid", "cusolverMpCreateMatrixDesc",
		"cusolverMpMalloc", "cusolverMpNUMROC", "cusolverMpMatrixScatterH2D",
		"cusolverMpMatrixGatherD2H", "cusolverMpGetrf", "cusolverMpGetrs",
	} {
		if !cusolvermp.Has(symbol) {
			t.Fatalf("%s is missing from %s; this build cannot run the LU path", symbol, cusolvermp.Path())
		}
	}

	if err := driver.Init(0); err != nil {
		t.Fatalf("CUDA driver is required for a physical run: %v", err)
	}
	count, err := driver.DeviceGetCount()
	if err != nil {
		t.Fatalf("device count: %v", err)
	}
	if int(device) >= count {
		t.Fatalf("this rank was assigned device %d but only %d are visible", device, count)
	}
	dev, err := driver.NewDeviceGet(int(device))
	if err != nil {
		t.Fatalf("device %d: %v", device, err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("retain context on device %d: %v", device, err)
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set context: %v", err)
	}
	stream, err := driver.NewStream(0)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Destroy()

	// Every rank joins the same communicator. This blocks until all of them
	// do, which is why a launcher that starts too few ranks hangs rather than
	// failing: the launcher, not this test, has to bound that.
	comm, err := nccl.NewCommInitRank(world, uniqueID, rank)
	if err != nil {
		t.Fatalf("nccl communicator: %v", err)
	}
	defer comm.Destroy()

	handle, err := cusolvermp.NewHandle(device, cusolver.Stream(stream))
	if err != nil {
		t.Fatalf("handle: %v", err)
	}
	defer handle.Destroy()

	grid, err := handle.CreateDeviceGrid(comm, nprow, npcol, cusolvermp.GridMappingRowMajor)
	if err != nil {
		t.Fatalf("grid: %v", err)
	}
	defer grid.Destroy()

	const (
		n      int64 = 512
		nrhs   int64 = 1
		mb, nb int64 = 64, 64
	)
	// Row-major mapping, so consecutive ranks advance along a grid row.
	myrow, mycol := uint32(rank/npcol), uint32(rank%npcol)

	localRowsA := numrocOrFail(t, n, mb, myrow, uint32(nprow))
	localColsA := numrocOrFail(t, n, nb, mycol, uint32(npcol))
	localRowsB := localRowsA
	localColsB := numrocOrFail(t, nrhs, nb, mycol, uint32(npcol))

	descA, err := grid.CreateMatrixDesc(cusolvermp.R64F, n, n, mb, nb, 0, 0, max64(localRowsA, 1))
	if err != nil {
		t.Fatalf("descriptor A: %v", err)
	}
	defer descA.Destroy()
	descB, err := grid.CreateMatrixDesc(cusolvermp.R64F, n, nrhs, mb, nb, 0, 0, max64(localRowsB, 1))
	if err != nil {
		t.Fatalf("descriptor B: %v", err)
	}
	defer descB.Destroy()

	// Device blocks. The grid allocator registers them for collective
	// transfer as it allocates; each one is freed explicitly below. A rank
	// that owns no elements still allocates a byte, because a zero-sized
	// allocation has no address to hand over.
	dA := mallocOrFail(t, grid, uint(max64(localRowsA*localColsA, 1))*8)
	defer grid.Free(dA)
	dB := mallocOrFail(t, grid, uint(max64(localRowsB*localColsB, 1))*8)
	defer grid.Free(dB)
	dIpiv := mallocOrFail(t, grid, uint(max64(localRowsA, 1))*8)
	defer grid.Free(dIpiv)
	dInfo := mallocOrFail(t, grid, 4)
	defer grid.Free(dInfo)

	// Host buffers for the scatter and the gather. They are page-locked CUDA
	// allocations, not Go memory: the library reads and writes them after the
	// enqueuing call has returned.
	hA := hostAllocOrFail(t, uint(n*n)*8)
	defer driver.MemFreeHost(uintptr(hA))
	hB := hostAllocOrFail(t, uint(n*nrhs)*8)
	defer driver.MemFreeHost(uintptr(hB))

	if rank == 0 {
		a := hostFloats(hA, n*n)
		for i := range a {
			a[i] = 0
		}
		for i := int64(0); i < n; i++ {
			a[i*n+i] = 2 // column-major, so the diagonal is at i*n+i either way
		}
		b := hostFloats(hB, n*nrhs)
		for i := range b {
			b[i] = 2
		}
	}

	// Scatter is collective: every rank calls it, the host buffer is read on
	// rank 0 only.
	if err := cusolvermp.MatrixScatterH2D(handle, n, n, dA, 1, 1, descA, 0, hA, n); err != nil {
		t.Fatalf("scatter A: %v", err)
	}
	if err := cusolvermp.MatrixScatterH2D(handle, n, nrhs, dB, 1, 1, descB, 0, hB, n); err != nil {
		t.Fatalf("scatter B: %v", err)
	}

	deviceBytes, hostBytes, err := cusolvermp.GetrfBufferSize(handle, n, n, dA, 1, 1, descA, dIpiv, cusolvermp.R64F)
	if err != nil {
		t.Fatalf("getrf workspace: %v", err)
	}
	solveDeviceBytes, solveHostBytes, err := cusolvermp.GetrsBufferSize(handle, cusolvermp.OpN, n, nrhs, dA, 1, 1, descA, dIpiv, dB, 1, 1, descB, cusolvermp.R64F)
	if err != nil {
		t.Fatalf("getrs workspace: %v", err)
	}
	// One workspace serves both calls because they do not overlap in time.
	// Two operations in flight would need two.
	if solveDeviceBytes > deviceBytes {
		deviceBytes = solveDeviceBytes
	}
	if solveHostBytes > hostBytes {
		hostBytes = solveHostBytes
	}
	dWork := mallocOrFail(t, grid, max(deviceBytes, 1))
	defer grid.Free(dWork)
	hWork := hostAllocOrFail(t, max(hostBytes, 1))
	defer driver.MemFreeHost(uintptr(hWork))

	if err := cusolvermp.Getrf(handle, n, n, dA, 1, 1, descA, dIpiv, cusolvermp.R64F, dWork, deviceBytes, hWork, hostBytes, dInfo); err != nil {
		t.Fatalf("getrf: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("synchronize after getrf: %v", err)
	}
	if info := readInfo(t, dInfo); info != 0 {
		t.Fatalf("getrf reported info = %d; a positive value is the one-based index of a zero pivot", info)
	}

	if err := cusolvermp.Getrs(handle, cusolvermp.OpN, n, nrhs, dA, 1, 1, descA, dIpiv, dB, 1, 1, descB, cusolvermp.R64F, dWork, deviceBytes, hWork, hostBytes, dInfo); err != nil {
		t.Fatalf("getrs: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("synchronize after getrs: %v", err)
	}
	if info := readInfo(t, dInfo); info != 0 {
		t.Fatalf("getrs reported info = %d", info)
	}

	if err := cusolvermp.MatrixGatherD2H(handle, n, nrhs, dB, 1, 1, descB, 0, hB, n); err != nil {
		t.Fatalf("gather X: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("synchronize after gather: %v", err)
	}
	if rank == 0 {
		x := hostFloats(hB, n*nrhs)
		for i, got := range x {
			if got < 1-1e-9 || got > 1+1e-9 {
				t.Fatalf("x[%d] = %v, want 1", i, got)
			}
		}
		t.Logf("solved a %dx%d system on a %dx%d grid over %d ranks", n, n, nprow, npcol, world)
	}
}

func mustEnvInt32(t *testing.T, name string) int32 {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		t.Fatalf("%s is set but %s is not; a physical run needs the launcher to supply it", physicalEnv, name)
	}
	n, err := strconv.ParseInt(value, 10, 32)
	if err != nil {
		t.Fatalf("%s = %q: %v", name, value, err)
	}
	return int32(n)
}

func mustEnvUniqueID(t *testing.T, name string) nccl.UniqueId {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		t.Fatalf("%s is set but %s is not; rank 0 must publish nccl.GetUniqueId as hex and the launcher must give it to every rank", physicalEnv, name)
	}
	raw, err := hex.DecodeString(value)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	var id nccl.UniqueId
	if len(raw) != len(id.Internal) {
		t.Fatalf("%s decodes to %d bytes, want %d", name, len(raw), len(id.Internal))
	}
	copy(id.Internal[:], raw)
	return id
}

func numrocOrFail(t *testing.T, n, nb int64, iproc, nprocs uint32) int64 {
	t.Helper()
	got, err := cusolvermp.NUMROC(n, nb, iproc, 0, nprocs)
	if err != nil {
		t.Fatalf("NUMROC: %v", err)
	}
	return got
}

func mallocOrFail(t *testing.T, grid cusolvermp.Grid, size uint) cusolvermp.DevicePtr {
	t.Helper()
	ptr, err := grid.Malloc(size)
	if err != nil {
		t.Fatalf("allocate %d device bytes: %v", size, err)
	}
	return ptr
}

func hostAllocOrFail(t *testing.T, size uint) cusolvermp.HostPtr {
	t.Helper()
	var p uintptr
	if err := driver.MemAllocHost(&p, size); err != nil {
		t.Fatalf("allocate %d host bytes: %v", size, err)
	}
	return cusolvermp.HostPtr(p)
}

// readInfo copies the device int the factorization and the solve report their
// status through. A nil error from Getrf means the work was enqueued; this is
// what says whether it succeeded.
func readInfo(t *testing.T, dInfo cusolvermp.DevicePtr) int32 {
	t.Helper()
	var info int32
	if err := driver.DevicePtr(dInfo).MemcpyDtoH(unsafe.Pointer(&info), 4); err != nil {
		t.Fatalf("read info: %v", err)
	}
	return info
}

// hostFloats views a page-locked host allocation as a float64 slice. The
// conversion from an address is confined to this one place: the memory comes
// from driver.MemAllocHost, so it is not Go memory, the collector never moves
// it, and driver.MemFreeHost is what ends its lifetime. go vet's unsafeptr
// check cannot see that and reports the conversion.
func hostFloats(p cusolvermp.HostPtr, n int64) []float64 {
	return unsafe.Slice((*float64)(unsafe.Pointer(uintptr(p))), n)
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
