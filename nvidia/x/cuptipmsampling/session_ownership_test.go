package cuptipmsampling

import (
	"errors"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cuptiowner"
	"testing"
)

func mockSessionAPI() sessionAPI {
	return sessionAPI{
		version: func() (uint32, error) { return 130200, nil }, initialize: func() error { return nil },
		enable: func(p *cupti.PmSamplingEnableParams) error { p.PPmSamplingObject = 17; return nil },
		config: func(*cupti.PmSamplingSetConfigParams) error { return nil },
		size:   func(p *cupti.PmSamplingGetCounterDataSizeParams) error { p.CounterDataSize = 16; return nil },
		image:  func(*cupti.PmSamplingCounterDataImageInitializeParams) error { return nil },
		start:  func(*cupti.PmSamplingStartParams) error { return nil }, stop: func(*cupti.PmSamplingStopParams) error { return nil }, disable: func(*cupti.PmSamplingDisableParams) error { return nil },
	}
}
func mockSessionConfig() Config {
	return Config{DeviceIndex: 73001, SamplingInterval: 1000, MaxSamples: 4, ConfigImage: []byte{1, 2}, MetricNames: []string{"counter"}}
}
func TestSessionDeviceConflict(t *testing.T) {
	cfg := mockSessionConfig()
	lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	api := mockSessionAPI()
	api.initialize = func() error { t.Fatal("conflict initialized native profiler"); return nil }
	api.enable = func(*cupti.PmSamplingEnableParams) error { t.Fatal("conflict enabled native sampler"); return nil }
	if s, err := newSession(cfg, api); err == nil || s != nil {
		t.Fatalf("s=%v err=%v", s, err)
	}
}
func TestSessionCleanupRetry(t *testing.T) {
	fail := errors.New("native cleanup failed")
	cfg := mockSessionConfig()
	api := mockSessionAPI()
	stopFails, disableFails := true, true
	stops, disables := 0, 0
	api.stop = func(*cupti.PmSamplingStopParams) error {
		stops++
		if stopFails {
			return fail
		}
		return nil
	}
	api.disable = func(*cupti.PmSamplingDisableParams) error {
		disables++
		if disableFails {
			return fail
		}
		return nil
	}
	s, err := newSession(cfg, api)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		stopFails = false
		disableFails = false
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	}()
	if err = s.Start(); err != nil {
		t.Fatal(err)
	}
	for _, stage := range []string{"stop", "disable"} {
		if !errors.Is(s.Close(), fail) {
			t.Fatalf("%s cleanup succeeded", stage)
		}
		if s.closed || s.handle != 17 || s.lease == nil || len(s.counterData) != 16 {
			t.Fatalf("lost cleanup state %+v", s)
		}
		if lease, e := cuptiowner.AcquireDevice(cfg.DeviceIndex); e == nil {
			lease.Release()
			t.Fatal("failed cleanup released device")
		}
		if s.Start() == nil {
			t.Fatal("reuse after failed close admitted")
		}
		stopFails = false
	}
	disableFails = false
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	if stops != 2 || disables != 2 {
		t.Fatalf("stops=%d disables=%d", stops, disables)
	}
	lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
	if err != nil {
		t.Fatal(err)
	}
	lease.Release()
}
func TestSessionConstructorCleanupRetry(t *testing.T) {
	for _, stage := range []string{"enable-handle", "config", "size", "image"} {
		t.Run(stage, func(t *testing.T) {
			cfg := mockSessionConfig()
			api := mockSessionAPI()
			fail := errors.New("construction failed")
			cleanup := errors.New("disable failed")
			disableFails := true
			switch stage {
			case "enable-handle":
				api.enable = func(p *cupti.PmSamplingEnableParams) error { p.PPmSamplingObject = 17; return fail }
			case "config":
				api.config = func(*cupti.PmSamplingSetConfigParams) error { return fail }
			case "size":
				api.size = func(*cupti.PmSamplingGetCounterDataSizeParams) error { return fail }
			case "image":
				api.image = func(*cupti.PmSamplingCounterDataImageInitializeParams) error { return fail }
			}
			api.disable = func(*cupti.PmSamplingDisableParams) error {
				if disableFails {
					return cleanup
				}
				return nil
			}
			s, err := newSession(cfg, api)
			if s == nil || !errors.Is(err, fail) || !errors.Is(err, cleanup) {
				t.Fatalf("s=%v err=%v", s, err)
			}
			defer func() {
				disableFails = false
				if err := s.Close(); err != nil {
					t.Error(err)
				}
			}()
			if lease, e := cuptiowner.AcquireDevice(cfg.DeviceIndex); e == nil {
				lease.Release()
				t.Fatal("constructor released pending device")
			}
			if s.Start() == nil {
				t.Fatal("partial constructor reused")
			}
			disableFails = false
			if err = s.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
func TestSessionConstructorCleanupSuccess(t *testing.T) {
	cfg := mockSessionConfig()
	api := mockSessionAPI()
	api.config = func(*cupti.PmSamplingSetConfigParams) error { return errors.New("config") }
	s, err := newSession(cfg, api)
	if s != nil || err == nil {
		t.Fatalf("s=%v err=%v", s, err)
	}
	lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
	if err != nil {
		t.Fatal(err)
	}
	lease.Release()
}

func TestSessionEnableWithoutHandleRetainsClaim(t *testing.T) {
	cfg := mockSessionConfig()
	api := mockSessionAPI()
	api.enable = func(*cupti.PmSamplingEnableParams) error { return nil }
	s, err := newSession(cfg, api)
	if s == nil || err == nil {
		t.Fatalf("s=%v err=%v", s, err)
	}
	defer s.lease.Release()
	if s.Close() == nil {
		t.Fatal("unknown native cleanup reported success")
	}
	if lease, e := cuptiowner.AcquireDevice(cfg.DeviceIndex); e == nil {
		lease.Release()
		t.Fatal("unknown native owner released")
	}
}
