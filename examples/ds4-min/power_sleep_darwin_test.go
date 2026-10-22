//go:build darwin

package main

import (
	"testing"
	"unsafe"
)

func TestPowerDarwinTimespecLayout(t *testing.T) {
	var x powerTimespec
	if unsafe.Sizeof(x) != 16 || unsafe.Offsetof(x.sec) != 0 || unsafe.Offsetof(x.nsec) != 8 {
		t.Fatal("Darwin nanosleep timespec layout")
	}
	if err := powerSleepSupported(); err != nil {
		t.Fatal(err)
	}
	// Resolving the borrowed symbol is not a real-sleep/EINTR receipt.
	if darwinPowerSleep.call == nil {
		t.Fatal("missing typed nanosleep binding")
	}
}
