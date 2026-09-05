package cusolvermp_test

import (
	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/cusolvermp"
	"github.com/tmc/lib/nvidia/nccl"
)

// These stand in for the parts of a distributed program that are outside this
// package: the communicator, the stream, this rank's coordinates, host
// workspace, and reading a device int back. Example above shows the cuSOLVERMp
// calls; it is not executed, and these return zero values.

func exampleComm() nccl.Comm { return 0 }

func exampleStream() cusolver.Stream { return 0 }

func exampleCoords(nprow, npcol int32) (myrow, mycol uint32) { return 0, 0 }

func exampleHostBuffer(size uint) cusolvermp.HostPtr { return 0 }

func exampleSynchronize(stream cusolver.Stream) {}

func exampleReadInfo(dInfo cusolvermp.DevicePtr) int32 { return 0 }
