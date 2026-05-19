//go:build linux

package fuse

import "testing"

func TestOpenDeviceEmptyPathDoesNotPanic(t *testing.T) {
	dev, err := OpenDevice("/definitely/not/fuse")
	if err == nil {
		_ = dev.Close()
		t.Fatal("OpenDevice unexpectedly succeeded")
	}
}
