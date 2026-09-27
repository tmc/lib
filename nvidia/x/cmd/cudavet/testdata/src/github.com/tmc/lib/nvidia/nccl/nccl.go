// Package nccl is a declaration-only stand-in for the real nccl package.
package nccl

import "github.com/tmc/lib/nvidia/driver"

type DataType int32

const Float32 DataType = 7

type RedOp int32

const Sum RedOp = 0

type UniqueId struct{ Internal [128]byte }

type Comm uintptr

func CommInitRank(nranks int32, commId UniqueId, rank int32) (Comm, error) { return 0, nil }
func (c Comm) Destroy() error                                              { return nil }
func (c Comm) UserRank() (int32, error)                                    { return 0, nil }

func GroupStart() error { return nil }
func GroupEnd() error   { return nil }

func (c Comm) AllReduce(sendbuff driver.DevicePtr, recvbuff driver.DevicePtr, count uint, datatype DataType, op RedOp, stream driver.Stream) error {
	return nil
}

func (c Comm) Send(sendbuff driver.DevicePtr, count uint, datatype DataType, peer int32, stream driver.Stream) error {
	return nil
}

func (c Comm) Recv(recvbuff driver.DevicePtr, count uint, datatype DataType, peer int32, stream driver.Stream) error {
	return nil
}
