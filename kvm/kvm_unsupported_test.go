//go:build !linux

package kvm

import (
	"errors"
	"testing"
)

func TestUnsupportedPlatform(t *testing.T) {
	if _, err := Open(); err == nil {
		t.Fatal("Open succeeded without an explicit device")
	} else if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("Open error = %v, want ErrUnsupported", err)
	}
}
