package cudamem_test

import (
	"errors"
	"strings"
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
		wantErr      string // substring of the expected error, "" for no error
	}{
		{name: "nil_ptr", ptr: 0, size: 1024, summaryGran: 256, samplingGran: 64, wantErr: "nil pointer"},
		{name: "zero_size", ptr: 0x1000, size: 0, summaryGran: 256, samplingGran: 64},
		{name: "zero_summary_gran", ptr: 0x1000, size: 1024, summaryGran: 0, samplingGran: 64, wantErr: "invalid summary granularity"},
		{name: "summary_exceeds_size", ptr: 0x1000, size: 1024, summaryGran: 2048, samplingGran: 64, wantErr: "invalid summary granularity"},
		{name: "sampling_exceeds_summary", ptr: 0x1000, size: 1024, summaryGran: 256, samplingGran: 512, wantErr: "sampling granularity exceeds"},
		{name: "valid_partition", ptr: 0x1000, size: 1024, summaryGran: 256, samplingGran: 64, wantErr: "cuMemGetLocationInfo"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			locs, err := cudamem.LocationInfo(tt.ptr, tt.size, tt.summaryGran, tt.samplingGran)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("LocationInfo() error = %v, want nil", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("LocationInfo() error = %v, want error containing %q", err, tt.wantErr)
			}
			// No argument produces a location: the query is unimplemented, so a
			// non-empty result would be fabricated.
			if len(locs) != 0 {
				t.Errorf("LocationInfo() = %v, want no locations", locs)
			}
		})
	}
}

// TestLocationInfoUnsupported checks that a well-formed query reports the
// driver entry point as unavailable rather than inventing a residency.
func TestLocationInfoUnsupported(t *testing.T) {
	locs, err := cudamem.LocationInfo(0x1000, 4096, 1024, 256)
	if !errors.Is(err, errors.ErrUnsupported) {
		t.Fatalf("LocationInfo() error = %v, want one matching errors.ErrUnsupported", err)
	}
	if locs != nil {
		t.Errorf("LocationInfo() = %v, want nil", locs)
	}
}

// TestSliceLocationInfoUnsupported checks the same for the slice method, which
// is the exported surface most likely to be cited as evidence about residency.
func TestSliceLocationInfoUnsupported(t *testing.T) {
	var s cudamem.Slice[byte] // zero slice: empty range, no device needed
	locs, err := s.LocationInfo(1024, 256)
	if err != nil || locs != nil {
		t.Fatalf("empty Slice.LocationInfo() = %v, %v, want nil, nil", locs, err)
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
