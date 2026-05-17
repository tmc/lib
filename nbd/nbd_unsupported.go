//go:build !linux

package nbd

// KernelDevice is unavailable on non-Linux platforms.
type KernelDevice struct{}

func OpenKernelDevice(path string) (*KernelDevice, error) { return nil, ErrUnsupported }
func (d *KernelDevice) FD() int                           { return -1 }
func (d *KernelDevice) Close() error                      { return ErrUnsupported }
func (d *KernelDevice) SetSocket(fd int) error            { return ErrUnsupported }
func (d *KernelDevice) SetBlockSize(size uint64) error    { return ErrUnsupported }
func (d *KernelDevice) SetSize(size uint64) error         { return ErrUnsupported }
func (d *KernelDevice) SetSizeBlocks(blocks uint64) error { return ErrUnsupported }
func (d *KernelDevice) SetFlags(flags uint64) error       { return ErrUnsupported }
func (d *KernelDevice) SetTimeout(seconds uint64) error   { return ErrUnsupported }
func (d *KernelDevice) Disconnect() error                 { return ErrUnsupported }
func (d *KernelDevice) ClearSocket() error                { return ErrUnsupported }
func (d *KernelDevice) ClearQueue() error                 { return ErrUnsupported }
func (d *KernelDevice) DoIt() error                       { return ErrUnsupported }
