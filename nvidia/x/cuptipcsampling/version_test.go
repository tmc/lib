package cuptipcsampling

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestQualifiedVersions(t *testing.T) {
	for _, tt := range []struct {
		version         uint32
		sampling, query bool
	}{
		{27, false, true},
		{130001, true, true},
		{130101, true, true},
		{130202, true, true},
		{130301, true, true},
		{130200, true, true},
		{130400, true, true},
		{0, false, false},
		{28, false, false},
		{120901, false, false},
		// Inside CUDA 13 but unmeasured: a range gate would admit these.
		{130002, false, false},
		{130100, false, false},
		{130201, false, false},
		{130300, false, false},
		{130500, false, false},
		{139999, false, false},
		{140000, false, false},
	} {
		if got := qualifiedSamplingVersion(tt.version); got != tt.sampling {
			t.Errorf("qualifiedSamplingVersion(%d) = %v, want %v", tt.version, got, tt.sampling)
		}
		if got := qualifiedQueryVersion(tt.version); got != tt.query {
			t.Errorf("qualifiedQueryVersion(%d) = %v, want %v", tt.version, got, tt.query)
		}
	}
}

// An unwitnessed version must be refused before any native configuration call.
func TestPublicStartRejectsUnwitnessedVersions(t *testing.T) {
	for _, version := range []uint32{0, 27, 28, 120901, 130300, 130500, 140000} {
		t.Run(fmt.Sprint(version), func(t *testing.T) {
			api := samplingControlAPI{
				version: func() (uint32, error) { return version, nil },
				set: func(*cupti.PCSamplingConfigurationInfoParams) error {
					t.Fatal("configured an unqualified version")
					return nil
				},
				get:     func(*cupti.PCSamplingConfigurationInfoParams) error { return nil },
				enable:  func(*cupti.PCSamplingEnableParams) error { return nil },
				start:   func(*cupti.PCSamplingStartParams) error { return nil },
				stop:    func(*cupti.PCSamplingStopParams) error { return nil },
				disable: func(*cupti.PCSamplingDisableParams) error { return nil },
			}
			inventory := ReasonInventory{CUPTIVersion: version, Reasons: []StallReason{{Index: 7, Name: "reason"}}}
			s, err := startSession(5, testPublicConfig(), api,
				func(cupti.Context, int) (ReasonInventory, error) { return inventory, nil },
				func(a, b, c int) (*samplingNames, error) {
					return newSamplingNames(a, b, c, func(uintptr, int) (string, error) { return "name", nil }, func(uintptr) {})
				},
				func(*cupti.PCSamplingGetDataParams) error { return nil })
			if err == nil {
				t.Fatalf("session=%v accepted CUPTI %d", s, version)
			}
			if s != nil {
				t.Cleanup(func() { _ = s.Close() })
			}
		})
	}
}
