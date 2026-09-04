package cudamlx

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// Device holds one CUDA device, its cached attributes, and its retained
// primary context. Port of cu::Device in device.cpp.
type Device struct {
	ordinal                 int
	raw                     driver.Device
	ctx                     driver.Context
	name                    string
	computeCapabilityMajor  int
	computeCapabilityMinor  int
	concurrentManagedAccess bool
	hostNativeAtomic        bool
	managedMemory           bool
	memoryPools             bool
}

var (
	devicesOnce sync.Once
	devices     []*Device
	devicesErr  error
)

// Devices initializes the driver and returns all CUDA devices. The
// devices and their primary contexts live for the process, mirroring the
// intentionally leaked device list in device.cpp.
func Devices() ([]*Device, error) {
	devicesOnce.Do(func() {
		if err := driver.Init(0); err != nil {
			devicesErr = fmt.Errorf("cuda init: %w", err)
			return
		}
		n, err := driver.DeviceGetCount()
		if err != nil {
			devicesErr = fmt.Errorf("device count: %w", err)
			return
		}
		for i := 0; i < n; i++ {
			d, err := newDevice(i)
			if err != nil {
				devicesErr = err
				return
			}
			devices = append(devices, d)
		}
	})
	return devices, devicesErr
}

// GetDevice returns the device with the given CUDA ordinal.
func GetDevice(ordinal int) (*Device, error) {
	ds, err := Devices()
	if err != nil {
		return nil, err
	}
	if ordinal < 0 || ordinal >= len(ds) {
		return nil, fmt.Errorf("no cuda device %d", ordinal)
	}
	return ds[ordinal], nil
}

func newDevice(ordinal int) (*Device, error) {
	raw, err := driver.NewDeviceGet(ordinal)
	if err != nil {
		return nil, fmt.Errorf("device %d: %w", ordinal, err)
	}
	d := &Device{ordinal: ordinal, raw: raw}
	attrs := []struct {
		attr driver.DeviceAttribute
		dst  *int
	}{
		{driver.DevAttrComputeCapabilityMajor, &d.computeCapabilityMajor},
		{driver.DevAttrComputeCapabilityMinor, &d.computeCapabilityMinor},
	}
	for _, a := range attrs {
		v, err := raw.GetAttribute(a.attr)
		if err != nil {
			return nil, fmt.Errorf("device %d attribute %v: %w", ordinal, a.attr, err)
		}
		*a.dst = v
	}
	bools := []struct {
		attr driver.DeviceAttribute
		dst  *bool
	}{
		{driver.DevAttrConcurrentManagedAccess, &d.concurrentManagedAccess},
		{driver.DevAttrHostNativeAtomicSupported, &d.hostNativeAtomic},
		{driver.DevAttrManagedMemory, &d.managedMemory},
		{driver.DevAttrMemoryPoolsSupported, &d.memoryPools},
	}
	for _, a := range bools {
		v, err := raw.GetAttribute(a.attr)
		if err != nil {
			return nil, fmt.Errorf("device %d attribute %v: %w", ordinal, a.attr, err)
		}
		*a.dst = v == 1
	}
	d.ctx, err = raw.PrimaryCtxRetain()
	if err != nil {
		return nil, fmt.Errorf("device %d primary context: %w", ordinal, err)
	}
	return d, nil
}

func (d *Device) Ordinal() int                  { return d.ordinal }
func (d *Device) Raw() driver.Device            { return d.raw }
func (d *Device) Context() driver.Context       { return d.ctx }
func (d *Device) ComputeCapabilityMajor() int   { return d.computeCapabilityMajor }
func (d *Device) ComputeCapabilityMinor() int   { return d.computeCapabilityMinor }
func (d *Device) ConcurrentManagedAccess() bool { return d.concurrentManagedAccess }
func (d *Device) HostNativeAtomic() bool        { return d.hostNativeAtomic }
func (d *Device) ManagedMemory() bool           { return d.managedMemory }
func (d *Device) MemoryPools() bool             { return d.memoryPools }

// MakeCurrent binds the device's primary context to the calling thread.
// The C++ caches the current device in a thread_local; goroutines migrate
// between threads, so the Go version always sets it. Callers doing a
// sequence of context-dependent driver calls should hold the OS thread —
// see WithCurrent.
func (d *Device) MakeCurrent() error {
	return d.ctx.SetCurrent()
}

// WithCurrent locks the goroutine to its OS thread, makes the device's
// context current, and runs fn.
func (d *Device) WithCurrent(fn func() error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := d.ctx.SetCurrent(); err != nil {
		return err
	}
	return fn()
}
