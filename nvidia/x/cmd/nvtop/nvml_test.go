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

// fields returns the values nvlinkCounters.read would read for two
// links: the data counters, each link's flit and data CRC errors, and
// the ECC errors. A negative value is one NVML did not report.
func fields(v ...int64) []nvml.FieldValue {
	f := make([]nvml.FieldValue, len(v))
	for i, x := range v {
		if x < 0 {
			f[i].NvmlReturn = nvml.ErrNotSupported
			continue
		}
		f[i].Value.Data[0] = uint64(x)
	}
	return f
}

func TestNVLinkCountersUpdate(t *testing.T) {
	t0 := time.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC)
	none := opt[nvlinkErrors]{}
	steps := []struct {
		v    []nvml.FieldValue
		rate opt[uint64]
		errs opt[nvlinkErrors]
	}{
		// The first read sets the error baselines.
		{fields(100, 100, 5, 1, 5, 1, 50), opt[uint64]{}, some(nvlinkErrors{})},
		{fields(200, 300, 6, 1, 7, 2, 49), some[uint64](300), some(nvlinkErrors{flit: 3, crc: 1})},
		// A link that does not answer leaves its sum unchanged.
		{fields(300, 400, -1, 1, 9, 9, 52), some[uint64](200), some(nvlinkErrors{flit: 3, crc: 8, ecc: 2})},
		// A failed read keeps the counts but restarts the rate.
		{nil, opt[uint64]{}, some(nvlinkErrors{flit: 3, crc: 8, ecc: 2})},
		{fields(500, 400, 6, 1, 7, 2, 50), some[uint64](200), some(nvlinkErrors{flit: 3, crc: 1})},
	}
	var c nvlinkCounters
	for i, st := range steps {
		rate, errs := c.update(st.v, 2, t0.Add(time.Duration(i)*time.Second))
		if rate != st.rate || errs != st.errs {
			t.Errorf("step %d: update = %+v, %+v; want %+v, %+v", i, rate, errs, st.rate, st.errs)
		}
	}

	c = nvlinkCounters{}
	if _, errs := c.update(fields(1, 1, -1, -1, -1, -1, -1), 2, t0); errs != none {
		t.Errorf("no error counters: errs = %+v, want none", errs)
	}
}
