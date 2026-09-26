package streammismatch

import (
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
)

func mismatch(h cublas.Handle, s1, s2 driver.Stream, x, y driver.DevicePtr, host []float32) error {
	if err := h.SetStream(s1); err != nil {
		return err
	}
	if err := x.CopyFromHostAsync(host, s2); err != nil {
		return err
	}
	a := float32(1)
	return h.Saxpy(4, &a, x, 1, y, 1) // want `x was last written on stream s2, but h runs on stream s1; record an event on s2 and have s1 wait for it`
}

func waited(h cublas.Handle, s1, s2 driver.Stream, e driver.Event, x, y driver.DevicePtr, host []float32) error {
	if err := h.SetStream(s1); err != nil {
		return err
	}
	if err := x.CopyFromHostAsync(host, s2); err != nil {
		return err
	}
	if err := e.Record(s2); err != nil {
		return err
	}
	if err := s1.WaitEvent(e, 0); err != nil {
		return err
	}
	a := float32(1)
	return h.Saxpy(4, &a, x, 1, y, 1)
}

func same(h cublas.Handle, s driver.Stream, x, y driver.DevicePtr, host []float32) error {
	if err := h.SetStream(s); err != nil {
		return err
	}
	if err := x.CopyFromHostAsync(host, s); err != nil {
		return err
	}
	a := float32(1)
	return h.Saxpy(4, &a, x, 1, y, 1)
}
