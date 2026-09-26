package asynchost

import "github.com/tmc/lib/nvidia/driver"

func overwrite(d driver.DevicePtr, s driver.Stream, buf []float32) error {
	if err := d.CopyFromHostAsync(buf, s); err != nil {
		return err
	}
	buf[0] = 1 // want `buf is written while CopyFromHostAsync may still be using it; synchronize the stream first`
	return s.Synchronize()
}

func dies(d driver.DevicePtr, s driver.Stream) error {
	buf := make([]float32, 4)
	return d.CopyToHostAsync(buf, s) // want `buf dies when the function returns but CopyToHostAsync may still be using it; synchronize the stream before returning`
}

func synchronized(d driver.DevicePtr, s driver.Stream, buf []float32) error {
	if err := d.CopyFromHostAsync(buf, s); err != nil {
		return err
	}
	if err := s.Synchronize(); err != nil {
		return err
	}
	buf[0] = 1
	return nil
}

func wait(s driver.Stream) error { return s.Synchronize() }

func helper(d driver.DevicePtr, s driver.Stream) ([]float32, error) {
	buf := make([]float32, 4)
	if err := d.CopyToHostAsync(buf, s); err != nil {
		return nil, err
	}
	if err := wait(s); err != nil {
		return nil, err
	}
	return buf, nil
}
