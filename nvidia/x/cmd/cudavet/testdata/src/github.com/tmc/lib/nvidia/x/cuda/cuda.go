// Package cuda is a declaration-only stand-in for the real x/cuda package,
// whose Stream and Event wrap a driver handle and release it with Close.
package cuda

import (
	"context"

	"github.com/tmc/lib/nvidia/driver"
)

type Device struct{}

func NewDevice(ordinal int) (*Device, error)        { return nil, nil }
func (d *Device) Raw() driver.Device                { return 0 }
func (d *Device) DefaultStream() *Stream            { return nil }
func (d *Device) NewStream() (*Stream, error)       { return nil, nil }
func (d *Device) NewEvent() (*Event, error)         { return nil, nil }
func (d *Device) WithContext(fn func() error) error { return nil }
func (d *Device) PrimaryContext() (*Context, error) { return nil, nil }
func (d *Device) Close() error                      { return nil }

type Stream struct{}

func (s *Stream) Raw() driver.Stream                    { return 0 }
func (s *Stream) Query() (bool, error)                  { return false, nil }
func (s *Stream) Synchronize(ctx context.Context) error { return nil }
func (s *Stream) WaitEvent(e *Event) error              { return nil }
func (s *Stream) Close() error                          { return nil }

type Event struct{}

func (e *Event) Raw() driver.Event                     { return 0 }
func (e *Event) Record(s *Stream) error                { return nil }
func (e *Event) Query() (bool, error)                  { return false, nil }
func (e *Event) Synchronize(ctx context.Context) error { return nil }
func (e *Event) Close() error                          { return nil }

type Context struct{}

func (c *Context) Synchronize() error              { return nil }
func (c *Context) Run(fn func() error) (err error) { return nil }
