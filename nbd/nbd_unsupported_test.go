//go:build !linux

package nbd

import (
	"errors"
	"testing"
)

func TestUnsupportedKernelDevice(t *testing.T) {
	if _, err := OpenKernelDevice("/dev/nbd0"); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("OpenKernelDevice error = %v, want ErrUnsupported", err)
	}
}
