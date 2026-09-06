package cublasmp_test

import (
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cublasmp"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// job describes the rank layout the launcher put in the environment.
type job struct {
	launcher   string
	rank, size int
	local      int
}

// launchers lists the environment variables each supported launcher sets, in
// the order they are checked: rank, size, local rank.
var launchers = []struct {
	name              string
	rank, size, local string
}{
	{"Open MPI", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK"},
	{"Slurm", "SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"},
	{"MPICH", "PMI_RANK", "PMI_SIZE", "MPI_LOCALRANKID"},
	{"torchrun", "RANK", "WORLD_SIZE", "LOCAL_RANK"},
}

// findJob reports the rank layout of the surrounding multi-rank job, or false
// when this process was not started by a launcher.
func findJob() (job, bool) {
	for _, l := range launchers {
		rank, errRank := strconv.Atoi(os.Getenv(l.rank))
		size, errSize := strconv.Atoi(os.Getenv(l.size))
		if errRank != nil || errSize != nil {
			continue
		}
		local, err := strconv.Atoi(os.Getenv(l.local))
		if err != nil {
			local = rank
		}
		return job{launcher: l.name, rank: rank, size: size, local: local}, true
	}
	return job{}, false
}

// uniqueIDPath is the file the ranks exchange the NCCL bootstrap identifier
// through. It must name a path visible to every rank of the job.
const uniqueIDPath = "CUBLASMP_TEST_UNIQUE_ID"

// exchangeUniqueID gets the NCCL bootstrap identifier to every rank. Rank 0
// creates it and writes it to a file on shared storage; the others wait for
// that file. NCCL needs an out-of-band channel for this and cuBLASMp does not
// provide one, so the test asks the operator for a path rather than inventing
// a transport.
func exchangeUniqueID(t *testing.T, j job) nccl.UniqueId {
	t.Helper()
	path := os.Getenv(uniqueIDPath)
	if path == "" {
		t.Fatalf("running under %s but %s is unset: point it at a path on storage every rank can see", j.launcher, uniqueIDPath)
	}
	if j.rank == 0 {
		id, err := nccl.GetUniqueId()
		if err != nil {
			t.Fatalf("nccl.GetUniqueId: %v", err)
		}
		if err := os.WriteFile(path, id.Internal[:], 0600); err != nil {
			t.Fatalf("publish unique id: %v", err)
		}
		return id
	}
	deadline := time.Now().Add(2 * time.Minute)
	for {
		data, err := os.ReadFile(path)
		if err == nil && len(data) == len(nccl.UniqueId{}.Internal) {
			var id nccl.UniqueId
			copy(id.Internal[:], data)
			return id
		}
		if time.Now().After(deadline) {
			t.Fatalf("rank %d never saw a unique id at %s: %v", j.rank, path, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// numroc returns the number of rows or columns of a block-cyclic dimension of
// length n, block size nb, held by process iproc of nprocs, with the first
// block on process isrcproc. It is the ScaLAPACK routine of the same name.
func numroc(n, nb, iproc, isrcproc, nprocs int64) int64 {
	dist := (nprocs + iproc - isrcproc) % nprocs
	blocks := n / nb
	local := (blocks / nprocs) * nb
	switch extra := blocks % nprocs; {
	case dist < extra:
		local += nb
	case dist == extra:
		local += n % nb
	}
	return local
}

// globalIndex maps a local index to its global index for process iproc, with
// the first block on process 0.
func globalIndex(local, nb, iproc, nprocs int64) int64 {
	return (local/nb)*nb*nprocs + iproc*nb + local%nb
}

// element is the value at global position (row, col) of the test matrix B. It
// varies in both indices so that a transposed or misdistributed tile does not
// compare equal by accident.
func element(row, col int64) float32 {
	return float32((row*31+col*17)%101) + 1
}

// TestPhysicalDistributedGemm multiplies a distributed identity by a
// distributed matrix across every rank of a live job and checks the result
// element by element on each rank. It is the only test in this package that
// touches a GPU, and it never passes without having done so: on a machine with
// no launcher it skips with the prerequisites named, and once a launcher is
// present every missing piece is fatal.
//
// A rank that fails leaves its peers waiting in a collective, so a failure
// shows up as a hung job until the launcher's timeout fires. That is the
// intended behavior: a distributed test that let one rank exit quietly would
// report success for a job that never ran.
func TestPhysicalDistributedGemm(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	j, ok := findJob()
	if !ok {
		t.Skipf("no multi-rank launcher in the environment: run under mpirun, srun, or torchrun on a host with at least two NVIDIA GPUs, a CUDA 13 driver, libcublasmp.so.0, libnccl.so.2, and %s pointing at shared storage", uniqueIDPath)
	}
	if j.size < 2 {
		t.Fatalf("%s reports a job of %d rank(s): a distributed GEMM needs at least 2", j.launcher, j.size)
	}
	t.Logf("%s: rank %d of %d, local rank %d", j.launcher, j.rank, j.size, j.local)

	// One CUDA context per rank, on the device this rank owns.
	if err := driver.Init(0); err != nil {
		t.Fatalf("driver.Init: %v", err)
	}
	count, err := driver.DeviceGetCount()
	if err != nil {
		t.Fatalf("driver.DeviceGetCount: %v", err)
	}
	if j.local >= count {
		t.Fatalf("local rank %d has no device: this host exposes %d", j.local, count)
	}
	device, err := driver.NewDeviceGet(j.local)
	if err != nil {
		t.Fatalf("driver.NewDeviceGet(%d): %v", j.local, err)
	}
	var uuid driver.Uuid
	if err := device.GetUuid(&uuid); err != nil {
		t.Fatal(err)
	}
	t.Logf("rank %d device %d UUID %x", j.rank, device, uuid)
	ctx, err := device.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("PrimaryCtxRetain: %v", err)
	}
	defer device.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("SetCurrent: %v", err)
	}

	comm, err := nccl.NewCommInitRank(int32(j.size), exchangeUniqueID(t, j), int32(j.rank))
	if err != nil {
		t.Fatalf("nccl.NewCommInitRank: %v", err)
	}
	defer comm.Destroy()

	// A 1 x size grid puts rank r in process column r under either layout, so
	// the local tile this test builds does not depend on the layout
	// convention.
	const (
		m  int64 = 512
		n  int64 = 384
		mb int64 = 64
		nb int64 = 64
	)
	nprow, npcol := int64(1), int64(j.size)
	grid, err := cublasmp.NewGrid(nprow, npcol, cublasmp.ColMajor, comm)
	if err != nil {
		t.Fatalf("NewGrid: %v", err)
	}
	defer grid.Destroy()

	handle, err := cublasmp.NewHandle(cublas.Stream(0))
	if err != nil {
		t.Fatalf("NewHandle: %v", err)
	}
	defer handle.Destroy()

	rank := int64(j.rank)
	colsA := numroc(m, nb, rank, 0, npcol)
	colsB := numroc(n, nb, rank, 0, npcol)

	// A is the global identity, so C = A*B is B and every rank can check its
	// own tile without a gather.
	hostA := make([]float32, m*colsA)
	for c := int64(0); c < colsA; c++ {
		if row := globalIndex(c, nb, rank, npcol); row < m {
			hostA[row+c*m] = 1
		}
	}
	hostB := make([]float32, m*colsB)
	for c := int64(0); c < colsB; c++ {
		col := globalIndex(c, nb, rank, npcol)
		for r := int64(0); r < m; r++ {
			hostB[r+c*m] = element(r, col)
		}
	}
	// C starts as a value the GEMM must overwrite, since beta is zero.
	hostC := make([]float32, m*colsB)
	for i := range hostC {
		hostC[i] = -12345
	}

	upload := func(name string, values []float32) driver.DevicePtr {
		t.Helper()
		bytes := uint(len(values)) * uint(unsafe.Sizeof(float32(0)))
		ptr, err := driver.MemAlloc(bytes)
		if err != nil {
			t.Fatalf("allocate %s: %v", name, err)
		}
		if err := ptr.MemcpyHtoD(unsafe.Pointer(&values[0]), bytes); err != nil {
			t.Fatalf("upload %s: %v", name, err)
		}
		return ptr
	}
	deviceA := upload("A", hostA)
	defer deviceA.MemFree()
	deviceB := upload("B", hostB)
	defer deviceB.MemFree()
	deviceC := upload("C", hostC)
	defer deviceC.MemFree()

	descA, err := cublasmp.NewMatrixDescriptor(m, m, mb, nb, 0, 0, m, cublas.R32F, grid)
	if err != nil {
		t.Fatalf("descriptor A: %v", err)
	}
	defer descA.Destroy()
	descB, err := cublasmp.NewMatrixDescriptor(m, n, mb, nb, 0, 0, m, cublas.R32F, grid)
	if err != nil {
		t.Fatalf("descriptor B: %v", err)
	}
	defer descB.Destroy()
	descC, err := cublasmp.NewMatrixDescriptor(m, n, mb, nb, 0, 0, m, cublas.R32F, grid)
	if err != nil {
		t.Fatalf("descriptor C: %v", err)
	}
	defer descC.Destroy()

	alpha, beta := float32(1), float32(0)
	var pins runtime.Pinner
	pins.Pin(&alpha)
	pins.Pin(&beta)
	defer pins.Unpin()

	const ia, ja, ib, jb, ic, jc = 1, 1, 1, 1, 1, 1
	deviceBytes, hostBytes, err := handle.GemmBufferSize(
		cublas.OpN, cublas.OpN, m, n, m,
		unsafe.Pointer(&alpha), uintptr(deviceA), ia, ja, descA,
		uintptr(deviceB), ib, jb, descB,
		unsafe.Pointer(&beta), uintptr(deviceC), ic, jc, descC,
		cublas.Compute32F)
	if err != nil {
		t.Fatalf("GemmBufferSize: %v", err)
	}
	t.Logf("workspace: %d bytes on device, %d on host", deviceBytes, hostBytes)

	var deviceWork uintptr
	if deviceBytes > 0 {
		var ptr driver.DevicePtr
		ptr, err = driver.MemAlloc(deviceBytes)
		deviceWork = uintptr(ptr)
		if err != nil {
			t.Fatalf("allocate device workspace: %v", err)
		}
		defer driver.DevicePtr(deviceWork).MemFree()
	}
	hostWork := make([]byte, hostBytes+1)
	pins.Pin(&hostWork[0])

	if err := handle.Gemm(
		cublas.OpN, cublas.OpN, m, n, m,
		unsafe.Pointer(&alpha), uintptr(deviceA), ia, ja, descA,
		uintptr(deviceB), ib, jb, descB,
		unsafe.Pointer(&beta), uintptr(deviceC), ic, jc, descC,
		cublas.Compute32F,
		deviceWork, deviceBytes, unsafe.Pointer(&hostWork[0]), hostBytes); err != nil {
		t.Fatalf("Gemm: %v", err)
	}
	// Gemm is asynchronous on the handle's stream. The workspaces and alpha
	// and beta must stay live until it has completed, which is what this
	// synchronize establishes.
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("CtxSynchronize: %v", err)
	}

	got := make([]float32, len(hostC))
	if err := deviceC.MemcpyDtoH(unsafe.Pointer(&got[0]), uint(len(got))*uint(unsafe.Sizeof(float32(0)))); err != nil {
		t.Fatalf("download C: %v", err)
	}
	for c := int64(0); c < colsB; c++ {
		col := globalIndex(c, nb, rank, npcol)
		for r := int64(0); r < m; r++ {
			if want, have := element(r, col), got[r+c*m]; want != have {
				t.Fatalf("rank %d: C[%d,%d] = %v, want %v", j.rank, r, col, have, want)
			}
		}
	}
	t.Logf("rank %d verified %d elements", j.rank, len(got))
}
