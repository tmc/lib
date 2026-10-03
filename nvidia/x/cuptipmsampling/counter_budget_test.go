package cuptipmsampling

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/internal/cuptiowner"
)

func TestSessionCounterImageBudgetBeforeAllocation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		size  uint
		limit uint
		want  bool
	}{
		{"at limit", 16, 16, true},
		{"over limit", 17, 16, false},
		{"native size overflow", ^uint(0), 0, false},
		{"legacy unspecified limit", 16, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := mockSessionConfig()
			cfg.MaxCounterDataBytes = tc.limit
			api := mockSessionAPI()
			initialized := false
			api.size = func(p *cupti.PmSamplingGetCounterDataSizeParams) error { p.CounterDataSize = tc.size; return nil }
			api.image = func(p *cupti.PmSamplingCounterDataImageInitializeParams) error {
				initialized = true
				if !tc.want {
					t.Fatal("oversized image reached native initialization")
				}
				if p.CounterDataSize != tc.size {
					t.Fatal("native size changed")
				}
				return nil
			}
			s, err := newSession(cfg, api)
			if (err == nil) != tc.want || initialized != tc.want {
				t.Fatalf("initialized=%v session=%v err=%v", initialized, s != nil, err)
			}
			if s != nil {
				if err := s.Close(); err != nil {
					t.Fatal(err)
				}
			}
			lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
			if err != nil {
				t.Fatalf("allocation refusal retained successful-cleanup lease: %v", err)
			}
			lease.Release()
		})
	}
}

func TestCounterImageRefusalRetainsFailedCleanupOwner(t *testing.T) {
	cfg := mockSessionConfig()
	cfg.MaxCounterDataBytes = 16
	api := mockSessionAPI()
	api.size = func(p *cupti.PmSamplingGetCounterDataSizeParams) error { p.CounterDataSize = 17; return nil }
	api.image = func(*cupti.PmSamplingCounterDataImageInitializeParams) error {
		t.Fatal("oversized counter image allocated/initialized")
		return nil
	}
	disables := 0
	api.disable = func(*cupti.PmSamplingDisableParams) error {
		disables++
		if disables == 1 {
			return errors.New("injected disable failure")
		}
		return nil
	}
	s, err := newSession(cfg, api)
	if s == nil {
		t.Fatalf("failed cleanup discarded owner: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	if err == nil || !strings.Contains(err.Error(), "allocation limit") || !strings.Contains(err.Error(), "injected disable failure") {
		t.Fatalf("original refusal and cleanup error not retained: %v", err)
	}
	if disables != 1 || s.closed || !s.closing || s.lease == nil || s.handle == 0 || len(s.counterData) != 0 {
		t.Fatalf("cleanup state lost: disables=%d closed=%v closing=%v lease=%v handle=%d bytes=%d", disables, s.closed, s.closing, s.lease != nil, s.handle, len(s.counterData))
	}
	if lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex); err == nil {
		lease.Release()
		t.Fatal("failed disable released device lease")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if disables != 2 || !s.closed || s.lease != nil || s.handle != 0 {
		t.Fatal("retry did not release owner")
	}
	lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
	if err != nil {
		t.Fatalf("successful retry retained lease: %v", err)
	}
	lease.Release()
}
