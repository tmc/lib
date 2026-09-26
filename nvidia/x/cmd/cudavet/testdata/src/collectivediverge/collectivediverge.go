package collectivediverge

import (
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

func rooted(c nccl.Comm, rank int32, buf driver.DevicePtr, s driver.Stream) error {
	if rank == 0 {
		return c.AllReduce(buf, buf, 4, nccl.Float32, nccl.Sum, s) // want `AllReduce is called only when rank == 0; the ranks that skip it leave the others waiting`
	}
	return nil
}

func others(c nccl.Comm, rank int32, buf driver.DevicePtr, s driver.Stream) error {
	if rank != 0 {
		return nil
	} else {
		return c.AllReduce(buf, buf, 4, nccl.Float32, nccl.Sum, s) // want `AllReduce is called only when !\(rank != 0\); the ranks that skip it leave the others waiting`
	}
}

func both(c nccl.Comm, rank int32, in, out driver.DevicePtr, s driver.Stream) error {
	if rank == 0 {
		return c.AllReduce(in, in, 4, nccl.Float32, nccl.Sum, s)
	} else {
		return c.AllReduce(in, out, 4, nccl.Float32, nccl.Sum, s)
	}
}

func pointToPoint(c nccl.Comm, rank int32, buf driver.DevicePtr, s driver.Stream) error {
	if rank == 0 {
		return c.Send(buf, 4, nccl.Float32, 1, s)
	}
	return c.Recv(buf, 4, nccl.Float32, 0, s)
}
