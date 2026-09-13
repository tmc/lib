package cuptipcsampling

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestQualifiedVersions(t *testing.T) {
	for _, tt := range []struct {
		version         uint32
		sampling, query bool
	}{
		{27, false, true},
		{130001, true, true},
		{130200, false, true},
		{130400, true, true},
		{0, false, false},
		{28, false, false},
		{120901, false, false},
		{130002, false, false},
		{130300, false, false},
		{130500, false, false},
	} {
		if got := qualifiedSamplingVersion(tt.version); got != tt.sampling {
			t.Errorf("qualifiedSamplingVersion(%d) = %v, want %v", tt.version, got, tt.sampling)
		}
		if got := qualifiedQueryVersion(tt.version); got != tt.query {
			t.Errorf("qualifiedQueryVersion(%d) = %v, want %v", tt.version, got, tt.query)
		}
	}
}

// fakeSamplingAPI answers every control call successfully and reports version.
func fakeSamplingAPI(version uint32) samplingControlAPI {
	var saved []cupti.PCSamplingConfigurationInfo
	return samplingControlAPI{
		version: func() (uint32, error) { return version, nil },
		set: func(p *cupti.PCSamplingConfigurationInfoParams) error {
			a := unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes))
			for i := range a {
				a[i].AttributeStatus = 0
			}
			saved = append(saved, a...)
			return nil
		},
		get: func(p *cupti.PCSamplingConfigurationInfoParams) error {
			a := unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes))
			for i := range a {
				for _, v := range saved {
					if a[i].AttributeType == v.AttributeType {
						a[i] = v
					}
				}
			}
			return nil
		},
		enable:  func(*cupti.PCSamplingEnableParams) error { return nil },
		start:   func(*cupti.PCSamplingStartParams) error { return nil },
		stop:    func(*cupti.PCSamplingStopParams) error { return nil },
		disable: func(*cupti.PCSamplingDisableParams) error { return nil },
	}
}

// A session under CUDA 13.4's CUPTI must configure, drain and close exactly as
// one under 13.0: cupti_pcsampling.h is byte-identical between the two, so the
// output layout the drain reads is the same storage. The drained version is the
// one the control observed, not a constant, which is what keeps Final's
// CUPTIVersion honest on a host whose toolkit moved.
func TestPublicStartAdmitsQualifiedVersions(t *testing.T) {
	for _, version := range []uint32{130001, 130400} {
		t.Run(fmtVersion(version), func(t *testing.T) {
			inventory := ReasonInventory{CUPTIVersion: version, Reasons: []StallReason{{Index: 7, Name: "reason"}}}
			s, err := startSession(5, testPublicConfig(), fakeSamplingAPI(version),
				func(cupti.Context, int) (ReasonInventory, error) { return inventory, nil },
				func(a, b, c int) (*samplingNames, error) {
					return newSamplingNames(a, b, c, func(uintptr, int) (string, error) { return "name", nil }, func(uintptr) {})
				},
				func(*cupti.PCSamplingGetDataParams) error { return nil })
			if s == nil || err != nil {
				t.Fatalf("session=%v err=%v", s, err)
			}
			t.Cleanup(func() { _ = s.Close() })
			if b, err := s.Read(); err != nil || !b.Valid {
				t.Fatalf("batch=%+v err=%v", b, err)
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if b, ok := s.Final(); !ok || !b.Valid || b.CUPTIVersion != version {
				t.Fatalf("final=%+v available=%v", b, ok)
			}
		})
	}
}

// An unwitnessed version must be refused before any native call, including
// 130200, which has a recorded reason layout but no sampling output witness.
func TestPublicStartRejectsUnwitnessedVersions(t *testing.T) {
	for _, version := range []uint32{27, 130200, 130300, 130500} {
		t.Run(fmtVersion(version), func(t *testing.T) {
			api := fakeSamplingAPI(version)
			api.set = func(*cupti.PCSamplingConfigurationInfoParams) error {
				t.Fatal("configured an unqualified version")
				return nil
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
				if err := s.Close(); err != nil && !errors.Is(err, errors.ErrUnsupported) {
					t.Logf("cleanup: %v", err)
				}
			}
		})
	}
}

func fmtVersion(v uint32) string {
	switch v {
	case 130001:
		return "CUDA 13.0"
	case 130400:
		return "CUDA 13.4"
	}
	return "CUPTI " + itoa(v)
}

func itoa(v uint32) string {
	if v == 0 {
		return "0"
	}
	var b [10]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}
