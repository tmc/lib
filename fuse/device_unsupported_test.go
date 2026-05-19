//go:build !linux

package fuse

import (
	"errors"
	"testing"
)

func TestUnsupportedOpenDevice(t *testing.T) {
	if _, err := OpenDevice("/dev/fuse"); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("OpenDevice error = %v, want ErrUnsupported", err)
	}
}
