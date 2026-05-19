//go:build !linux

package fuse

type Device struct{}

func OpenDevice(path string) (*Device, error) { return nil, ErrUnsupported }
func (d *Device) FD() int                     { return -1 }
func (d *Device) Read(p []byte) (int, error)  { return 0, ErrUnsupported }
func (d *Device) Write(p []byte) (int, error) { return 0, ErrUnsupported }
func (d *Device) Close() error                { return ErrUnsupported }
