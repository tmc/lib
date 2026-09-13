package cuda

import (
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// Device represents a CUDA-capable GPU device.
type Device struct {
	ordinal int
	raw     driver.Device
	mu      sync.RWMutex
	ctx     *Context

	defaultOnce   sync.Once
	defaultStream *Stream
}

// NewDevice returns a handle to the CUDA device with the specified ordinal.
func NewDevice(ordinal int) (*Device, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	dev, err := driver.NewDeviceGet(ordinal)
	if err != nil {
		return nil, fmt.Errorf("cuda: device %d: %w", ordinal, err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		return nil, fmt.Errorf("cuda: device %d retain primary context: %w", ordinal, err)
	}
	d := &Device{
		ordinal: ordinal,
		raw:     dev,
	}
	d.ctx = &Context{raw: ctx, device: d, isPrimary: true}
	return d, nil
}

// Ordinal returns the device ordinal index.
func (d *Device) Ordinal() int {
	return d.ordinal
}

// Raw returns the underlying driver.Device handle.
func (d *Device) Raw() driver.Device {
	return d.raw
}

// Name returns the human-readable identifier of the device.
func (d *Device) Name() (string, error) {
	name, err := d.raw.GetName()
	if err != nil {
		return "", fmt.Errorf("cuda: device %d get name: %w", d.ordinal, err)
	}
	return name, nil
}

// TotalMem returns the total available device global memory in bytes.
func (d *Device) TotalMem() (uint64, error) {
	bytes, err := d.raw.TotalMem()
	if err != nil {
		return 0, fmt.Errorf("cuda: device %d total mem: %w", d.ordinal, err)
	}
	return uint64(bytes), nil
}

// ComputeCapability returns the major and minor compute capability revision numbers.
func (d *Device) ComputeCapability() (major, minor int, err error) {
	maj, err := d.raw.GetAttribute(driver.DevAttrComputeCapabilityMajor)
	if err != nil {
		return 0, 0, fmt.Errorf("cuda: device %d compute capability major: %w", d.ordinal, err)
	}
	min, err := d.raw.GetAttribute(driver.DevAttrComputeCapabilityMinor)
	if err != nil {
		return 0, 0, fmt.Errorf("cuda: device %d compute capability minor: %w", d.ordinal, err)
	}
	return int(maj), int(min), nil
}

// Attribute queries an integer hardware or driver attribute on the device.
func (d *Device) Attribute(attrib driver.DeviceAttribute) (int, error) {
	val, err := d.raw.GetAttribute(attrib)
	if err != nil {
		return 0, fmt.Errorf("cuda: device %d get attribute %v: %w", d.ordinal, attrib, err)
	}
	return int(val), nil
}

// MultiprocessorCount returns the number of physical Streaming Multiprocessors (SMs) on the device.
func (d *Device) MultiprocessorCount() (int, error) {
	return d.Attribute(driver.DevAttrMultiprocessorCount)
}

// PrimaryContext returns and retains the primary context associated with this device.
func (d *Device) PrimaryContext() (*Context, error) {
	ctx, err := d.raw.PrimaryCtxRetain()
	if err != nil {
		return nil, fmt.Errorf("cuda: device %d retain primary context: %w", d.ordinal, err)
	}
	return &Context{
		raw:       ctx,
		device:    d,
		isPrimary: true,
	}, nil
}

// WithContext executes the given closure with the device's primary context bound
// to the current OS thread, guaranteeing OS thread affinity for the duration of the call.
func (d *Device) WithContext(fn func() error) error {
	if d == nil {
		return fmt.Errorf("cuda: nil or closed device")
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.ctx == nil {
		return fmt.Errorf("cuda: nil or closed device")
	}
	return d.ctx.Run(fn)
}

// Close releases the primary-context reference retained by NewDevice.
// A Device must not be used after Close returns.
func (d *Device) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.ctx == nil {
		return nil
	}
	err := d.ctx.Release()
	if err == nil {
		d.ctx = nil
	}
	return err
}

// DefaultStream returns the device's legacy default stream, the stream that carries
// work submitted without a stream of its own. The returned Stream is never nil and is
// shared by every caller, so code can always name a stream instead of branching on nil.
// [Stream.Destroy] on it drains registered cleanups without destroying the handle.
func (d *Device) DefaultStream() *Stream {
	d.defaultOnce.Do(func() {
		d.defaultStream = &Stream{device: d, isDefault: true}
	})
	return d.defaultStream
}

// NewStream creates a new asynchronous stream on the device with default flags.
func (d *Device) NewStream() (*Stream, error) {
	return d.NewStreamWithFlags(0)
}

// NewStreamWithFlags creates a new asynchronous stream on the device with specified flags (e.g. driver.StreamNonBlocking).
func (d *Device) NewStreamWithFlags(flags uint32) (*Stream, error) {
	var s Stream
	err := d.WithContext(func() error {
		raw, err := driver.NewStream(flags)
		if err != nil {
			return err
		}
		s = Stream{
			raw:    raw,
			device: d,
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: create stream: %w", err)
	}
	return &s, nil
}

// NewEvent creates a new CUDA event on the device with default flags.
func (d *Device) NewEvent() (*Event, error) {
	return d.NewEventWithFlags(0)
}

// NewEventWithFlags creates a new CUDA event on the device with specified flags (e.g. driver.EventDisableTiming, driver.EventBlockingSync).
func (d *Device) NewEventWithFlags(flags uint32) (*Event, error) {
	var e Event
	err := d.WithContext(func() error {
		raw, err := driver.NewEvent(flags)
		if err != nil {
			return err
		}
		e = Event{
			raw:    raw,
			device: d,
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: create event: %w", err)
	}
	return &e, nil
}
