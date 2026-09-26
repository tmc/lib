// Package nccl is a declaration-only stand-in for the real nccl package.
package nccl

import "github.com/tmc/lib/nvidia/driver"

type DataType int32

const Float32 DataType = 7

type RedOp int32

const Sum RedOp = 0

type UniqueID [128]byte

type Comm uintptr

func CommInitRank(nranks int32, id UniqueID, rank int32) (Comm, error) { return 0, nil }
func (c Comm) Destroy() error                                          { return nil }
func (c Comm) UserRank() (int32, error)                                { return 0, nil }

func GroupStart() error { return nil }
func GroupEnd() error   { return nil }

func (c Comm) AllReduce(send, recv driver.DevicePtr, count uint, dt DataType, op RedOp, s driver.Stream) error {
	return nil
}

func (c Comm) Send(send driver.DevicePtr, count uint, dt DataType, peer int32, s driver.Stream) error {
	return nil
}

func (c Comm) Recv(recv driver.DevicePtr, count uint, dt DataType, peer int32, s driver.Stream) error {
	return nil
}
