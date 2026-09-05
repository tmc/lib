package cusolvermp_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cusolvermp"
)

// Example shows the shape of one rank's work in a distributed LU solve: build
// the handle for the device this rank owns, wrap the already-initialized NCCL
// communicator in a process grid, describe the local block of each distributed
// matrix, size the workspaces, factorize, and solve.
//
// Every call below is collective over the grid. All ranks must reach every one
// of them, in this order, with matching global arguments; a rank that returns
// early leaves the others waiting. The addresses are device addresses supplied
// by the caller, which this package will not allocate, free or trace.
//
// This example is not executed. It needs libcusolverMp, an NVIDIA GPU per rank,
// and a launcher that starts one rank per GPU with a shared NCCL communicator.
func Example() {
	// comm is an initialized nccl.Comm with nprow*npcol ranks; stream is the
	// CUDA stream this rank drives. Both are caller-owned and must outlive
	// everything created below.
	var (
		comm         = exampleComm()
		stream       = exampleStream()
		device int32 = 0 // the ordinal this rank owns
		nprow  int32 = 2
		npcol  int32 = 2
	)

	handle, err := cusolvermp.NewHandle(device, stream)
	if err != nil {
		fmt.Println("cuSOLVERMp unavailable:", err)
		return
	}
	defer handle.Destroy()

	grid, err := handle.CreateDeviceGrid(comm, nprow, npcol, cusolvermp.GridMappingRowMajor)
	if err != nil {
		fmt.Println("grid:", err)
		return
	}
	defer grid.Destroy()

	// The global system is n-by-n with nrhs right-hand sides, blocked mb-by-nb
	// and distributed over the grid. Each rank holds only its own block.
	const (
		n, nrhs int64 = 4096, 1
		mb, nb  int64 = 256, 256
	)
	myrow, mycol := exampleCoords(nprow, npcol)

	// NUMROC gives this rank's local extents, which is what the leading
	// dimension has to accommodate. A smaller lld is a silent memory error.
	localRowsA, err := cusolvermp.NUMROC(n, mb, myrow, 0, uint32(nprow))
	if err != nil {
		fmt.Println("numroc:", err)
		return
	}
	localColsA, err := cusolvermp.NUMROC(n, nb, mycol, 0, uint32(npcol))
	if err != nil {
		fmt.Println("numroc:", err)
		return
	}

	descA, err := grid.CreateMatrixDesc(cusolvermp.R64F, n, n, mb, nb, 0, 0, localRowsA)
	if err != nil {
		fmt.Println("descriptor:", err)
		return
	}
	defer descA.Destroy()
	descB, err := grid.CreateMatrixDesc(cusolvermp.R64F, n, nrhs, mb, nb, 0, 0, localRowsA)
	if err != nil {
		fmt.Println("descriptor:", err)
		return
	}
	defer descB.Destroy()

	// The grid's allocator hands back device addresses that are already
	// registered for collective transfer. They are freed explicitly: nothing
	// here is traced by the garbage collector.
	dA, err := grid.Malloc(uint(localRowsA * localColsA * 8))
	if err != nil {
		fmt.Println("allocate:", err)
		return
	}
	defer grid.Free(dA)
	dB, err := grid.Malloc(uint(localRowsA * 8))
	if err != nil {
		fmt.Println("allocate:", err)
		return
	}
	defer grid.Free(dB)
	dIpiv, err := grid.Malloc(uint(localRowsA * 8))
	if err != nil {
		fmt.Println("allocate:", err)
		return
	}
	defer grid.Free(dIpiv)
	dInfo, err := grid.Malloc(4)
	if err != nil {
		fmt.Println("allocate:", err)
		return
	}
	defer grid.Free(dInfo)

	// Workspace sizes are specific to these arguments and must be re-queried
	// whenever any of them change.
	deviceBytes, hostBytes, err := cusolvermp.GetrfBufferSize(handle, n, n, dA, 1, 1, descA, dIpiv, cusolvermp.R64F)
	if err != nil {
		fmt.Println("workspace:", err)
		return
	}
	dWork, err := grid.Malloc(deviceBytes)
	if err != nil {
		fmt.Println("allocate:", err)
		return
	}
	defer grid.Free(dWork)
	hWork := exampleHostBuffer(hostBytes)

	// Matrix indices are one-based, following ScaLAPACK.
	if err := cusolvermp.Getrf(handle, n, n, dA, 1, 1, descA, dIpiv, cusolvermp.R64F, dWork, deviceBytes, hWork, hostBytes, dInfo); err != nil {
		fmt.Println("getrf:", err)
		return
	}
	if err := cusolvermp.Getrs(handle, cusolvermp.OpN, n, nrhs, dA, 1, 1, descA, dIpiv, dB, 1, 1, descB, cusolvermp.R64F, dWork, deviceBytes, hWork, hostBytes, dInfo); err != nil {
		fmt.Println("getrs:", err)
		return
	}

	// Both calls only enqueued work. Synchronize the stream, then read dInfo
	// back from the device: a nil error above does not mean the factorization
	// succeeded. The solution overwrites dB in place.
	exampleSynchronize(stream)
	if info := exampleReadInfo(dInfo); info != 0 {
		fmt.Println("factorization reported", info)
		return
	}
	fmt.Println("solved")
}

// ExampleNUMROC shows how a rank's grid coordinates and its local block extents
// follow from the grid shape. The arithmetic is ScaLAPACK's, and it is what the
// leading dimension passed to CreateMatrixDesc has to accommodate.
//
// This part needs no library: it is the same formula NUMROC evaluates.
func ExampleNUMROC() {
	const (
		n, nb        int64  = 1000, 64
		nprow, npcol uint32 = 2, 3
	)
	// A rank's coordinates in a row-major grid.
	for rank := uint32(0); rank < nprow*npcol; rank++ {
		myrow, mycol := rank/npcol, rank%npcol
		rows := numroc(n, nb, myrow, 0, nprow)
		cols := numroc(n, nb, mycol, 0, npcol)
		fmt.Printf("rank %d at (%d,%d) holds %dx%d\n", rank, myrow, mycol, rows, cols)
	}
	// Output:
	// rank 0 at (0,0) holds 512x360
	// rank 1 at (0,1) holds 512x320
	// rank 2 at (0,2) holds 512x320
	// rank 3 at (1,0) holds 488x360
	// rank 4 at (1,1) holds 488x320
	// rank 5 at (1,2) holds 488x320
}

// numroc is the ScaLAPACK formula cusolverMpNUMROC evaluates, repeated here so
// the example runs without the library.
func numroc(n, nb int64, iproc, isrcproc, nprocs uint32) int64 {
	mydist := int64((nprocs + iproc - isrcproc) % nprocs)
	nblocks := n / nb
	nroc := (nblocks / int64(nprocs)) * nb
	extra := nblocks % int64(nprocs)
	switch {
	case mydist < extra:
		nroc += nb
	case mydist == extra:
		nroc += n % nb
	}
	return nroc
}
