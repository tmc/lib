package cudaroofline

import (
	"strings"
	"testing"
)

func TestLookupDeviceCeiling(t *testing.T) {
	tests := []struct {
		device    string
		residency Residency
		arch      string
		gbps      float64
		wantErr   string
	}{
		{"NVIDIA H200", DeviceResident, "sm_90", 2740.48, ""},
		{"  nvidia   H200 ", DeviceResident, "sm_90", 2740.48, ""},
		{"NVIDIA GB10", HostPageTableResident, "sm_121", 162.4, ""},
		{"NVIDIA H200", HostPageTableResident, "", 0, `no ceiling for "NVIDIA H200" at "host_page_table" residency (have: device)`},
		{"NVIDIA H200 NVL", DeviceResident, "", 0, `no measured ceiling for device "NVIDIA H200 NVL" (have: NVIDIA GB10, NVIDIA H200)`},
		{"NVIDIA H100 80GB HBM3", DeviceResident, "", 0, "no measured ceiling"},
	}
	for _, tt := range tests {
		c, err := LookupDeviceCeiling(tt.device, tt.residency)
		if tt.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("LookupDeviceCeiling(%q, %q) error = %v, want %q", tt.device, tt.residency, err, tt.wantErr)
			}
			continue
		}
		if err != nil || c.Architecture != tt.arch || c.PeakMemoryGBps != tt.gbps || c.Device == "" {
			t.Errorf("LookupDeviceCeiling(%q, %q) = %+v, %v", tt.device, tt.residency, c, err)
		}
	}
}

// TestDeviceCeilingsAreMeasured holds the device table to its reason for
// existing: every memory figure in it was measured on the named part, and
// every entry is a valid ceiling for its architecture.
func TestDeviceCeilingsAreMeasured(t *testing.T) {
	for k, c := range knownDeviceCeilings {
		if c.MemoryProvenance != Measured {
			t.Errorf("%s: memory provenance %q, want measured", c.Device, c.MemoryProvenance)
		}
		if normalizeDevice(c.Device) != k.name || c.Residency != k.residency {
			t.Errorf("%s: key %+v does not match entry", c.Device, k)
		}
		if err := ValidateCeiling(c); err != nil {
			t.Errorf("%s: %v", c.Device, err)
		}
	}
	// The GB10 device rows must agree with the sm_121 rows they repeat.
	for _, r := range []Residency{DeviceResident, HostPageTableResident} {
		d, _ := LookupDeviceCeiling("NVIDIA GB10", r)
		a, _ := LookupCeiling("sm_121", r)
		if d.PeakMemoryGBps != a.PeakMemoryGBps || d.PeakTFLOPS != a.PeakTFLOPS {
			t.Errorf("GB10 %s: device %v GB/s, arch %v GB/s", r, d.PeakMemoryGBps, a.PeakMemoryGBps)
		}
	}
}
