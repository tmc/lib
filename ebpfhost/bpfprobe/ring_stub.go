//go:build !linux

package bpfprobe

import (
	"context"
	"errors"
)

// Ring reads records from a Linux BPF ring buffer. The zero value is not usable.
type Ring struct{}

// OpenRing returns an error on platforms other than Linux.
func OpenRing(fd int, capacity uint32) (*Ring, error) {
	return nil, errors.New("bpf ring: unsupported platform")
}

// Read returns an error on platforms other than Linux.
func (*Ring) Read(context.Context) ([]byte, error) {
	return nil, errors.New("bpf ring: unsupported platform")
}

// Close releases a Ring.
func (*Ring) Close() error { return nil }
