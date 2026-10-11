package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// probeCUDA asks the driver itself whether a usable device exists. This is
// the gate: nvidia-smi listing a GPU is a claim about enumeration, while
// this is the same call the measurement is about to make.
func probeCUDA(uuid string) cudaProbe {
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
	var d driver.Device
	found := false
	for i := 0; i < n; i++ {
		candidate, e := driver.DeviceGet(i)
		if e != nil {
			p.ProbeError = e.Error()
			return p
		}
		u, e := candidate.Uuid()
		if e != nil {
			p.ProbeError = "UUID: " + e.Error()
			return p
		}
		identity := fmt.Sprintf("GPU-%x-%x-%x-%x-%x", u.Bytes[:4], u.Bytes[4:6], u.Bytes[6:8], u.Bytes[8:10], u.Bytes[10:])
		if identity == uuid {
			if found {
				p.ProbeError = "duplicate CUDA device UUID"
				return p
			}
			d, p.UUID, found = candidate, identity, true
		}
	}
	if !found {
		p.ProbeError = "requested UUID is not a visible CUDA device"
		return p
	}
	if p.Name, err = d.Name(); err != nil {
		p.ProbeError = "name: " + err.Error()
		return p
	}
	if p.Major, p.Minor, err = d.ComputeCapability(); err != nil {
		p.ProbeError = "compute capability: " + err.Error()
		return p
	}
	mem, err := d.TotalMem()
	if err != nil {
		p.ProbeError = "total memory: " + err.Error()
		return p
	}
	p.TotalMem = uint64(mem)
	return p
}
