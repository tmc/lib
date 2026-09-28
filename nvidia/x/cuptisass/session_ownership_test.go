package cuptisass

import (
	"errors"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/internal/cuptiowner"
	"strings"
	"testing"
)

func mockSessionAPI() sessionAPI {
	return sessionAPI{context: func(_ uintptr, fn func() error) error { return fn() }, version: func() error { return nil }, initialize: func() error { return nil }, device: func(uintptr) (uint32, error) { return 74001, nil }, config: func(*cupti.SassMetricsSetConfigParams) error { return nil }, unset: func(*cupti.SassMetricsUnsetConfigParams) error { return nil }, enable: func(*cupti.SassMetricsEnableParams) error { return nil }, disable: func(*cupti.SassMetricsDisableParams) error { return nil }}
}
func mockSessionConfig() Config {
	return Config{DeviceIndex: 74001, Metrics: []MetricConfig{{MetricID: 1, Granularity: GranularityGPU}}}
}
func TestSessionDeviceConflict(t *testing.T) {
	cfg := mockSessionConfig()
	lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	api := mockSessionAPI()
	api.initialize = func() error { t.Fatal("conflict initialized profiler"); return nil }
	api.config = func(*cupti.SassMetricsSetConfigParams) error { t.Fatal("conflict changed config"); return nil }
	if s, err := newSession(1, cfg, api); s != nil || err == nil {
		t.Fatalf("s=%v err=%v", s, err)
	}
}
func TestSessionContextDeviceMismatch(t *testing.T) {
	cfg := mockSessionConfig()
	api := mockSessionAPI()
	api.device = func(uintptr) (uint32, error) { return cfg.DeviceIndex + 1, nil }
	api.config = func(*cupti.SassMetricsSetConfigParams) error { t.Fatal("mismatch changed config"); return nil }
	if s, err := newSession(1, cfg, api); s != nil || err == nil {
		t.Fatalf("s=%v err=%v", s, err)
	}
}
func TestSessionCleanupRetry(t *testing.T) {
	cfg := mockSessionConfig()
	api := mockSessionAPI()
	fail := errors.New("cleanup")
	disableFails, unsetFails := true, true
	disables, unsets := 0, 0
	api.disable = func(*cupti.SassMetricsDisableParams) error {
		disables++
		if disableFails {
			return fail
		}
		return nil
	}
	api.unset = func(*cupti.SassMetricsUnsetConfigParams) error {
		unsets++
		if unsetFails {
			return fail
		}
		return nil
	}
	s, err := newSession(1, cfg, api)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		disableFails = false
		unsetFails = false
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	}()
	if err = s.Enable(); err != nil {
		t.Fatal(err)
	}
	for _, stage := range []string{"disable", "unset"} {
		if !errors.Is(s.Close(), fail) {
			t.Fatalf("%s cleanup succeeded", stage)
		}
		if s.closed || !s.configured || s.lease == nil {
			t.Fatal("cleanup ownership lost")
		}
		if lease, e := cuptiowner.AcquireDevice(cfg.DeviceIndex); e == nil {
			lease.Release()
			t.Fatal("failed cleanup released device")
		}
		if s.Enable() == nil {
			t.Fatal("reuse after close accepted")
		}
		disableFails = false
	}
	unsetFails = false
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	if disables != 2 || unsets != 2 {
		t.Fatalf("disables=%d unsets=%d", disables, unsets)
	}
	lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
	if err != nil {
		t.Fatal(err)
	}
	lease.Release()
}
func TestSessionFailedConfigDoesNotUnsetForeignState(t *testing.T) {
	cfg := mockSessionConfig()
	api := mockSessionAPI()
	api.config = func(*cupti.SassMetricsSetConfigParams) error { return errors.New("native config conflict") }
	api.unset = func(*cupti.SassMetricsUnsetConfigParams) error {
		t.Fatal("unsuccessful config unset foreign state")
		return nil
	}
	s, err := newSession(1, cfg, api)
	if s != nil || err == nil {
		t.Fatalf("s=%v err=%v", s, err)
	}
	lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
	if err != nil {
		t.Fatal(err)
	}
	lease.Release()
}

func TestSessionEnableErrorRetainsOnlyOwnedConfiguration(t *testing.T) {
	cfg := mockSessionConfig()
	api := mockSessionAPI()
	fail := errors.New("native conflict")
	unsetFails := true
	api.enable = func(*cupti.SassMetricsEnableParams) error { return fail }
	api.disable = func(*cupti.SassMetricsDisableParams) error {
		t.Fatal("disabled instrumentation not successfully enabled by this session")
		return nil
	}
	api.unset = func(*cupti.SassMetricsUnsetConfigParams) error {
		if unsetFails {
			return fail
		}
		return nil
	}
	s, err := newSession(1, cfg, api)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		unsetFails = false
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	}()
	if err = s.Enable(); !errors.Is(err, fail) {
		t.Fatal(err)
	}
	if err = s.Close(); !errors.Is(err, fail) {
		t.Fatal(err)
	}
	if !s.configured || s.lease == nil {
		t.Fatal("owned configuration was lost")
	}
	if lease, e := cuptiowner.AcquireDevice(cfg.DeviceIndex); e == nil {
		lease.Release()
		t.Fatal("failed unset released claim")
	}
	unsetFails = false
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionReconfigure(t *testing.T) {
	cfg := mockSessionConfig()
	cfg.DeviceIndex = 74002
	api := mockSessionAPI()
	api.device = func(uintptr) (uint32, error) { return cfg.DeviceIndex, nil }
	configs := 0
	api.config = func(*cupti.SassMetricsSetConfigParams) error {
		configs++
		if configs > 1 {
			return cupti.ErrOutOfMemory
		}
		return nil
	}
	s, err := newSession(1, cfg, api)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	s, err = newSession(1, cfg, api)
	if s != nil || err == nil {
		t.Fatalf("second session: s=%v err=%v", s, err)
	}
	if !errors.Is(err, cupti.ErrOutOfMemory) || !strings.Contains(err.Error(), "already configured") {
		t.Errorf("second session error = %v, want the one-configuration explanation wrapping ErrOutOfMemory", err)
	}
}

func TestSessionConfigurationContext(t *testing.T) {
	for _, failAt := range []string{"", "restore config", "restore unset"} {
		t.Run(failAt, func(t *testing.T) {
			api := mockSessionAPI()
			active := false
			restores := 0
			configures, unsets := 0, 0
			failure := errors.New("restore context")
			api.context = func(ctx uintptr, fn func() error) error {
				if ctx != 17 || active {
					t.Fatalf("invalid context entry %d active=%v", ctx, active)
				}
				active = true
				err := fn()
				active = false
				restores++
				if failAt == "restore config" && restores == 1 || failAt == "restore unset" && restores == 2 {
					return errors.Join(err, failure)
				}
				return err
			}
			api.config = func(*cupti.SassMetricsSetConfigParams) error {
				if !active {
					t.Fatal("config outside context")
				}
				configures++
				return nil
			}
			api.unset = func(*cupti.SassMetricsUnsetConfigParams) error {
				if !active {
					t.Fatal("unset outside context")
				}
				unsets++
				return nil
			}
			s, err := newSession(17, mockSessionConfig(), api)
			if failAt == "restore config" {
				if !errors.Is(err, failure) || s != nil || configures != 1 || unsets != 1 {
					t.Fatalf("session=%v error=%v config=%d unset=%d", s, err, configures, unsets)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := s.Close(); err != nil {
					t.Error(err)
				}
			})
			err = s.Close()
			if failAt == "restore unset" {
				if !errors.Is(err, failure) || s.lease == nil || s.configured {
					t.Fatalf("lost cleanup state: error=%v configured=%v", err, s.configured)
				}
				if err := s.Close(); err != nil {
					t.Fatal(err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			if configures != 1 || unsets != 1 {
				t.Fatalf("config=%d unset=%d", configures, unsets)
			}
		})
	}
}
