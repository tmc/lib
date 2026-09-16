package cuptipcsampling

import (
	"errors"
	"reflect"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func testSamplingControl(t *testing.T) (*samplingControl, *[]string) {
	t.Helper()
	o, inventory := testSamplingOptions()
	c, err := newSamplingConfig(o, inventory, testSamplingBuffer(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.release)
	var calls []string
	a := samplingControlAPI{
		device:  func(cupti.Context) (uint32, error) { return 0, nil },
		version: func() (uint32, error) { calls = append(calls, "version"); return 130001, nil },
		set: func(*cupti.PCSamplingConfigurationInfoParams) error {
			calls = append(calls, "set")
			for i := range c.attrs {
				c.attrs[i].AttributeStatus = 0
			}
			return nil
		},
		get: func(p *cupti.PCSamplingConfigurationInfoParams) error {
			calls = append(calls, "get")
			attrs := unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes))
			for i := range attrs {
				for _, a := range c.attrs {
					if attrs[i].AttributeType == a.AttributeType {
						attrs[i] = a
					}
				}
			}
			return nil
		},
		enable: func(p *cupti.PCSamplingEnableParams) error {
			calls = append(calls, "enable")
			if p.Size != 24 || p.Ctx != 5 {
				t.Fatal("enable params")
			}
			return nil
		},
		start: func(p *cupti.PCSamplingStartParams) error {
			calls = append(calls, "start")
			if p.Size != 24 || p.Ctx != 5 {
				t.Fatal("start params")
			}
			return nil
		},
		stop: func(p *cupti.PCSamplingStopParams) error {
			calls = append(calls, "stop")
			if p.Size != 24 || p.Ctx != 5 {
				t.Fatal("stop params")
			}
			return nil
		},
		disable: func(p *cupti.PCSamplingDisableParams) error {
			calls = append(calls, "disable")
			if p.Size != 24 || p.Ctx != 5 {
				t.Fatal("disable params")
			}
			return nil
		},
	}
	return &samplingControl{ctx: 5, config: c, api: a}, &calls
}

func TestSamplingControlLifecycle(t *testing.T) {
	s, calls := testSamplingControl(t)
	if err := s.prepare(); err != nil {
		t.Fatal(err)
	}
	if s.canRelease() {
		t.Fatal("enabled storage releasable")
	}
	if err := s.start(); err != nil {
		t.Fatal(err)
	}
	if err := s.start(); err == nil {
		t.Fatal("double start")
	}
	if err := s.shutdown(); err != nil {
		t.Fatal(err)
	}
	if err := s.shutdown(); err != nil {
		t.Fatal(err)
	}
	if !s.canRelease() || s.starts != 1 || s.stops != 1 {
		t.Fatalf("state=%+v", s)
	}
	if !reflect.DeepEqual(*calls, []string{"version", "set", "get", "enable", "start", "stop", "disable"}) {
		t.Fatalf("calls=%v", *calls)
	}
	if err := s.start(); err == nil {
		t.Fatal("start after shutdown")
	}
}

func TestSamplingControlPartialEnable(t *testing.T) {
	s, calls := testSamplingControl(t)
	failure := errors.New("enable failed")
	s.api.enable = func(*cupti.PCSamplingEnableParams) error { *calls = append(*calls, "enable-failed"); return failure }
	if err := s.prepare(); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if s.canRelease() {
		t.Fatal("released uncertain native ownership")
	}
	if err := s.shutdown(); err != nil {
		t.Fatal(err)
	}
	if !s.canRelease() {
		t.Fatal("disable did not end ownership")
	}
}

func TestSamplingControlDisableRetry(t *testing.T) {
	s, _ := testSamplingControl(t)
	if err := s.prepare(); err != nil {
		t.Fatal(err)
	}
	if err := s.start(); err != nil {
		t.Fatal(err)
	}
	stopFailure, disableFailure := errors.New("stop failure"), errors.New("disable failure")
	s.api.stop = func(*cupti.PCSamplingStopParams) error { return stopFailure }
	s.api.disable = func(*cupti.PCSamplingDisableParams) error { return disableFailure }
	err := s.shutdown()
	if !errors.Is(err, stopFailure) || !errors.Is(err, disableFailure) || s.canRelease() {
		t.Fatalf("release=%v err=%v", s.canRelease(), err)
	}
	s.api.stop = func(*cupti.PCSamplingStopParams) error { return nil }
	s.api.disable = func(*cupti.PCSamplingDisableParams) error { return nil }
	if err := s.shutdown(); err != nil || !s.canRelease() {
		t.Fatalf("release=%v err=%v", s.canRelease(), err)
	}
	if !s.failed {
		t.Fatal("retry erased collection failure")
	}
}

func TestSamplingControlVersionGate(t *testing.T) {
	s, calls := testSamplingControl(t)
	s.api.version = func() (uint32, error) { return 27, nil }
	if err := s.prepare(); err == nil || !s.canRelease() || len(*calls) != 0 {
		t.Fatalf("calls=%v release=%v err=%v", *calls, s.canRelease(), err)
	}
}
