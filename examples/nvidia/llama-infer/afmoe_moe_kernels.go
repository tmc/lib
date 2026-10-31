package main

import (
	_ "embed"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed afmoe_moe.cu
var afmoeMoECUDA string

// afmoeMoEKernels owns an independent scalar-token module. Activation buffers
// use float32 storage for BF16 values. Expert tables hold128 uint64 addresses
// to original BF16 gate/up/down matrices; they do not require a packed model.
// The host must validate dimensions and allocate all outputs before launch.
type afmoeMoEKernels struct {
	mod            driver.Module
	ptx            string
	routerLinearFn driver.Function
	routeFn        driver.Function
	gateUpFn       driver.Function
	downFn         driver.Function
	selectedGateFn driver.Function
	selectedDownFn driver.Function
	combineFn      driver.Function
	clearFaultFn   driver.Function
	dev            *cuda.Device
}

func compileAFMoEMoEKernels(dev *cuda.Device) (*afmoeMoEKernels, error) {
	compiled, err := nvrtcx.Compile(afmoeMoECUDA,
		nvrtcx.WithName("afmoe_moe.cu"), nvrtcx.WithDeviceArch(dev))
	if err != nil {
		return nil, fmt.Errorf("compile afmoe moe: %w", releaseErrorOwner(err))
	}
	loaded, err := compiled.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load afmoe moe: %w", releaseErrorOwner(err))
	}
	// Loading restores the caller's context. Function lookup and any cleanup
	// still need this module's device context, including on another OS thread.
	m := &afmoeMoEKernels{mod: loaded, dev: dev}
	err = dev.WithContext(func() error {
		var lookupErr error
		m, lookupErr = lookupAFMoEMoEKernels(loaded, loaded.Function, (*driver.Module).Close)
		return lookupErr
	})
	if m != nil {
		m.ptx = compiled.PTX()
		m.dev = dev
	}
	if err != nil {
		return m, fmt.Errorf("initialize afmoe moe functions: %w", err)
	}
	return m, nil
}

func lookupAFMoEMoEKernels(mod driver.Module, lookup func(string) (driver.Function, error), closeModule func(*driver.Module) error) (*afmoeMoEKernels, error) {
	m := &afmoeMoEKernels{mod: mod}
	for _, entry := range []struct {
		name string
		dst  *driver.Function
	}{
		{"AFMoERouterLinearBF16", &m.routerLinearFn},
		{"AFMoERoute", &m.routeFn},
		{"AFMoEMLPGateUpBF16", &m.gateUpFn},
		{"AFMoEMLPDownBF16", &m.downFn},
		{"AFMoESelectedGateUpBF16", &m.selectedGateFn},
		{"AFMoESelectedDownBF16", &m.selectedDownFn},
		{"AFMoECombine", &m.combineFn},
		{"AFMoEClearFault", &m.clearFaultFn},
	} {
		fn, err := lookup(entry.name)
		if err == nil && fn == 0 {
			err = fmt.Errorf("zero function")
		}
		if err != nil {
			err = fmt.Errorf("lookup afmoe moe kernel %s: %w", entry.name, err)
			if cerr := m.close(closeModule); cerr != nil {
				return m, errors.Join(err, cerr)
			}
			return nil, err
		}
		*entry.dst = fn
	}
	return m, nil
}

// Close releases the module. A failed close retains its ownership so the
// caller can retry; successful and zero closes are idempotent.
func (m *afmoeMoEKernels) Close() error {
	if m != nil && m.mod != 0 && m.dev != nil {
		return m.dev.WithContext(func() error { return m.close((*driver.Module).Close) })
	}
	return m.close((*driver.Module).Close)
}

func (m *afmoeMoEKernels) close(closeModule func(*driver.Module) error) error {
	if m == nil || m.mod == 0 {
		return nil
	}
	if err := closeModule(&m.mod); err != nil {
		return fmt.Errorf("close afmoe moe module: %w", err)
	}
	*m = afmoeMoEKernels{}
	return nil
}
