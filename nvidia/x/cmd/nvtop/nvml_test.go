package main

import (
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestArchName(t *testing.T) {
	tests := []struct {
		arch nvml.DeviceArchitecture
		want opt[string]
	}{
		{2, some("Kepler")},
		{10, some("Blackwell")},
		{13, some("Rubin")},
		{15, some("NPU3")},
		{14, opt[string]{}},
		{0xffffffff, opt[string]{}}, // NVML_DEVICE_ARCH_UNKNOWN
	}
	for _, tt := range tests {
		if got := archName(tt.arch); got != tt.want {
			t.Errorf("archName(%d) = %+v, want %+v", tt.arch, got, tt.want)
		}
	}
}

func TestNVLinkVersion(t *testing.T) {
	// NVML's enumeration: 1.0, 2.0, 2.2, 3.0, 3.1, 4.0, 5.0, 6.0.
	// Unknown versions pass through.
	want := []uint32{0, 1, 2, 2, 3, 3, 4, 5, 6, 9}
	for raw, w := range want {
		if got := nvlinkVersion(uint32(raw)); got != w {
			t.Errorf("nvlinkVersion(%d) = %d, want %d", raw, got, w)
		}
	}
}

func TestNVLinkRate(t *testing.T) {
	t0 := time.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name string
		a, b nvlinkSample
		want opt[uint64]
	}{
		{"first", nvlinkSample{}, nvlinkSample{tx: 100, at: t0}, opt[uint64]{}},
		{"rate", nvlinkSample{tx: 100, rx: 50, at: t0}, nvlinkSample{tx: 300, rx: 250, at: t0.Add(2 * time.Second)}, some[uint64](200)},
		{"reset", nvlinkSample{tx: 300, rx: 50, at: t0}, nvlinkSample{tx: 10, rx: 150, at: t0.Add(time.Second)}, some[uint64](100)},
		{"no time", nvlinkSample{tx: 1, at: t0}, nvlinkSample{tx: 2, at: t0}, opt[uint64]{}},
	}
	for _, tt := range tests {
		if got := nvlinkRate(tt.a, tt.b); got != tt.want {
			t.Errorf("%s: nvlinkRate = %+v, want %+v", tt.name, got, tt.want)
		}
	}
}
