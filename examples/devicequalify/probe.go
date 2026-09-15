package main

import "github.com/tmc/lib/nvidia/x/cuda"

// probeCUDA asks the driver itself whether a usable device exists. This is
// the gate: nvidia-smi listing a GPU is a claim about enumeration, while
// this is the same call the measurement is about to make.
func probeCUDA() cudaProbe {
	var p cudaProbe
	if err := cuda.Init(); err != nil {
		p.InitError = err.Error()
		return p
	}
	p.InitOK = true
	n, err := cuda.DeviceCount()
	if err != nil {
		p.ProbeError = err.Error()
		return p
	}
	p.DeviceCount = n
	if n < 1 {
		return p
	}
	devices, err := cuda.Devices()
	if err != nil || len(devices) == 0 {
		if err != nil {
			p.ProbeError = err.Error()
		} else {
			p.ProbeError = "device count is positive but no device was returned"
		}
		return p
	}
	d := devices[0]
	if p.Name, err = d.Name(); err != nil {
		p.ProbeError = "name: " + err.Error()
		return p
	}
	if p.Major, p.Minor, err = d.ComputeCapability(); err != nil {
		p.ProbeError = "compute capability: " + err.Error()
		return p
	}
	if p.TotalMem, err = d.TotalMem(); err != nil {
		p.ProbeError = "total memory: " + err.Error()
		return p
	}
	return p
}
