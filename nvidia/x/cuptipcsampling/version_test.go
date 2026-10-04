package cuptipcsampling

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
)

func TestQualifiedVersions(t *testing.T) {
	for _, tt := range []struct {
		version         uint32
		sampling, query bool
	}{
		{24, false, true},
		{26, false, true},
		{28, false, true},
		{27, false, true},
		{130001, true, true},
		{130101, true, true},
		{130202, true, true},
		{130301, true, true},
		{130200, true, true},
		{130400, true, true},
		{0, false, false},
		{20, false, false},
		{120901, false, false},
		{130000, false, false},
		{140000, false, false},
		// Unmeasured releases use the newest measured layout below them.
		// Sampling has no measured CUDA 12 layout.
		{25, false, true},
		{29, false, true},
		{130002, true, true},
		{130100, true, true},
		{130201, true, true},
		{130300, true, true},
		{130401, true, true},
		{130500, true, true},
		{139999, true, true},
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
	for _, version := range []uint32{0, 27, 28, 120901, 130000, 140000} {
		t.Run(fmt.Sprint(version), func(t *testing.T) {
			api := samplingControlAPI{
				device:  func(driver.Context) (uint32, error) { return 0, nil },
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
				func(driver.Context, int) (ReasonInventory, error) { return inventory, nil },
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

// TestGeneratedVersionIsQualified pins the qualification table to the version
// the Go types were generated from. Both gates are asserted: the query gate
// admits a superset, so checking only it would pass even if the sampling gate
// had dropped the generated version.
//
// The table holds version literals, so nothing otherwise connects it to
// [cupti.ApiVersion] -- a regeneration moves the constant and leaves every
// literal, and neither the compiler nor any other test notices. Whatever else
// the table admits, it must admit the version whose struct shapes these types
// actually are.
func TestGeneratedVersionIsQualified(t *testing.T) {
	if !qualifiedSamplingVersion(cupti.ApiVersion) {
		t.Errorf("qualifiedSamplingVersion(cupti.ApiVersion = %d) = false, want true", cupti.ApiVersion)
	}
	if !qualifiedQueryVersion(cupti.ApiVersion) {
		t.Errorf("qualifiedQueryVersion(cupti.ApiVersion = %d) = false, want true", cupti.ApiVersion)
	}
}
