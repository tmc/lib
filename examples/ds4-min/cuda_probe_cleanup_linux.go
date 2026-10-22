//go:build linux

package main

import "github.com/tmc/lib/nvidia/driver"

type cudaProbeResources struct {
	module      driver.Module
	buffer      func() error
	run         func(func() error) error
	synchronize func() error
	unload      func(driver.Module) error
	drained     bool
}

func (r *cudaFoundation) probeResources() *cudaProbeResources {
	return &cudaProbeResources{
		run:         r.device.WithContext,
		synchronize: driver.CtxSynchronize,
		unload:      func(module driver.Module) error { return module.Close() },
	}
}

func (p *cudaProbeResources) Close() error {
	if p.module == 0 && p.buffer == nil {
		return nil
	}
	return p.run(func() error {
		if !p.drained {
			if err := p.synchronize(); err != nil {
				return err
			}
			p.drained = true
		}
		if p.buffer != nil {
			if err := p.buffer(); err != nil {
				return err
			}
			p.buffer = nil
		}
		if p.module != 0 {
			if err := p.unload(p.module); err != nil {
				return err
			}
			p.module = 0
		}
		return nil
	})
}
