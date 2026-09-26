package gridtrunc

import "github.com/tmc/lib/nvidia/driver"

func direct(f driver.Function, s driver.Stream, n uint32) error {
	return f.LaunchKernel(driver.Dims(n/256, 1, 1), driver.Dims(256, 1, 1), 0, s, nil) // want `grid dimension n / 256 truncates, skipping the last partial block; use driver.GridFor or round up`
}

func viaVar(f driver.Function, s driver.Stream, n uint32) error {
	blocks := n / 128
	cfg := driver.LaunchConfig{
		GridDim:  driver.Dim3{X: blocks, Y: 1, Z: 1}, // want `grid dimension n / 128 truncates, skipping the last partial block; use driver.GridFor or round up`
		BlockDim: driver.Dim3{X: 128, Y: 1, Z: 1},
		HStream:  s,
	}
	return f.LaunchKernelEx(&cfg, nil, nil)
}

func rounded(f driver.Function, s driver.Stream, n uint32) error {
	if err := f.LaunchKernel(driver.GridFor(n, 256), driver.Dims(256, 1, 1), 0, s, nil); err != nil {
		return err
	}
	return f.LaunchKernel(driver.Dims((n+255)/256, 1, 1), driver.Dims(256, 1, 1), 0, s, nil)
}

func block(f driver.Function, s driver.Stream, n uint32) error {
	threads := driver.Dims(n/4, 1, 1)
	return f.LaunchKernel(driver.Dims(4, 1, 1), threads, 0, s, nil)
}
