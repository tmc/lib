package cudamem_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestLocationInfoValidation(t *testing.T) {
	tests := []struct {
		name         string
		ptr          uintptr
		size         uint64
		summaryGran  uint64
		samplingGran uint64
		wantErr      bool
	}{
		{name: "nil_ptr", ptr: 0, size: 1024, summaryGran: 256, samplingGran: 64, wantErr: true},
		{name: "zero_size", ptr: 0x1000, size: 0, summaryGran: 256, samplingGran: 64, wantErr: false},
		{name: "zero_summary_gran", ptr: 0x1000, size: 1024, summaryGran: 0, samplingGran: 64, wantErr: true},
		{name: "summary_exceeds_size", ptr: 0x1000, size: 1024, summaryGran: 2048, samplingGran: 64, wantErr: true},
		{name: "sampling_exceeds_summary", ptr: 0x1000, size: 1024, summaryGran: 256, samplingGran: 512, wantErr: true},
		{name: "valid_partition", ptr: 0x1000, size: 1024, summaryGran: 256, samplingGran: 64, wantErr: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			locs, err := cudamem.LocationInfo(tt.ptr, tt.size, tt.summaryGran, tt.samplingGran)
			if (err != nil) != tt.wantErr {
				t.Fatalf("LocationInfo() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && tt.size > 0 {
				wantRegions := int(tt.size / tt.summaryGran)
				if len(locs) != wantRegions {
					t.Errorf("len(locs) = %d, want %d", len(locs), wantRegions)
				}
			}
		})
	}
}

func TestFormatLocationSummary(t *testing.T) {
	locs := []cudamem.MemLocation{
		{Type: cudamem.LocationTypeDevice, ID: 0},
		{Type: cudamem.LocationTypeDevice, ID: 0},
		{Type: cudamem.LocationTypeHost, ID: 0},
	}
	summary := cudamem.FormatLocationSummary(locs)
	if summary == "" || summary == "empty" {
		t.Errorf("FormatLocationSummary = %q, want populated string", summary)
	}
}

func TestMemLocationTypeString(t *testing.T) {
	tests := []struct {
		locType cudamem.MemLocationType
		want    string
	}{
		{cudamem.LocationTypeDevice, "device"},
		{cudamem.LocationTypeHost, "host"},
		{cudamem.LocationTypeHostNUMA, "host_numa"},
		{cudamem.LocationTypeHostNUMACurrent, "host_numa_current"},
		{cudamem.LocationTypeInvalid, "none"},
	}

	for _, tt := range tests {
		if got := tt.locType.String(); got != tt.want {
			t.Errorf("%v.String() = %q, want %q", tt.locType, got, tt.want)
		}
	}
}
