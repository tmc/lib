package readbeforesync

import "github.com/tmc/lib/nvidia/driver"

func early(d driver.DevicePtr, s driver.Stream, buf []float32) (float32, error) {
	if err := d.CopyToHostAsync(buf, s); err != nil {
		return 0, err
	}
	x := buf[0] // want `buf is read before the stream is synchronized; CopyToHostAsync may not have finished`
	return x, s.Synchronize()
}

func late(d driver.DevicePtr, s driver.Stream, buf []float32) (float32, error) {
	if err := d.CopyToHostAsync(buf, s); err != nil {
		return 0, err
	}
	if err := s.Synchronize(); err != nil {
		return 0, err
	}
	return buf[0], nil
}

func defaultStream(d driver.DevicePtr, buf []float32) (float32, error) {
	if err := d.CopyToHostAsync(buf, 0); err != nil {
		return 0, err
	}
	return buf[0], nil
}

func sizeOnly(d driver.DevicePtr, s driver.Stream, buf []float32) (int, error) {
	if err := d.CopyToHostAsync(buf, s); err != nil {
		return 0, err
	}
	n := len(buf)
	return n, s.Synchronize()
}
