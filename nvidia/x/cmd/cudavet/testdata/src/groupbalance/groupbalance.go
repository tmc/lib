package groupbalance

import (
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

func leaky(c nccl.Comm, buf driver.DevicePtr, s driver.Stream) error {
	if err := nccl.GroupStart(); err != nil { // want `GroupStart is not matched by GroupEnd on every path out of the function; an open group hangs the next collective`
		return err
	}
	if err := c.AllReduce(buf, buf, 4, nccl.Float32, nccl.Sum, s); err != nil {
		return err
	}
	return nccl.GroupEnd()
}

func balanced(c nccl.Comm, buf driver.DevicePtr, s driver.Stream) error {
	if err := nccl.GroupStart(); err != nil {
		return err
	}
	err := c.AllReduce(buf, buf, 4, nccl.Float32, nccl.Sum, s)
	if endErr := nccl.GroupEnd(); err == nil {
		err = endErr
	}
	return err
}

func deferred(c nccl.Comm, buf driver.DevicePtr, s driver.Stream) error {
	if err := nccl.GroupStart(); err != nil {
		return err
	}
	defer nccl.GroupEnd()
	return c.AllReduce(buf, buf, 4, nccl.Float32, nccl.Sum, s)
}

func panics(c nccl.Comm, buf driver.DevicePtr, s driver.Stream) {
	nccl.GroupStart()
	if err := c.AllReduce(buf, buf, 4, nccl.Float32, nccl.Sum, s); err != nil {
		panic(err)
	}
	nccl.GroupEnd()
}

// optional opens and closes the group under the same condition.
func optional(grouped bool, c nccl.Comm, buf driver.DevicePtr, s driver.Stream) error {
	if grouped {
		if err := nccl.GroupStart(); err != nil {
			return err
		}
	}
	err := c.AllReduce(buf, buf, 4, nccl.Float32, nccl.Sum, s)
	if grouped {
		return nccl.GroupEnd()
	}
	return err
}
