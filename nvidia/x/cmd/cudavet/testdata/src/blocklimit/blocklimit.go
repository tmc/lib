package blocklimit

import "github.com/tmc/lib/nvidia/driver"

func big(f driver.Function, s driver.Stream) error {
	return f.LaunchKernel(driver.Dims(4, 1, 1), driver.Dims(2048, 1, 1), 0, s, nil) // want `block of 2048 threads exceeds the limit of 1024`
}

func shared(f driver.Function, s driver.Stream) error {
	return f.LaunchKernel(driver.Dims(4, 1, 1), driver.Dims(256, 1, 1), 64<<10, s, nil) // want `65536 bytes of dynamic shared memory exceeds the default 48 KiB; raise FuncAttributeMaxDynamicSharedSizeBytes first`
}

func config(f driver.Function, s driver.Stream) error {
	cfg := driver.LaunchConfig{
		GridDim:  driver.Dims(4, 1, 1),
		BlockDim: driver.Dim3{X: 32, Y: 32, Z: 2}, // want `block of 2048 threads exceeds the limit of 1024`
		HStream:  s,
	}
	return f.LaunchKernelEx(&cfg, nil)
}

func raised(f driver.Function, s driver.Stream) error {
	if err := f.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, 64<<10); err != nil {
		return err
	}
	return f.LaunchKernel(driver.Dims(4, 1, 1), driver.Dims(1024, 1, 1), 64<<10, s, nil)
}
