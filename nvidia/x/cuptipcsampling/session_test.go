package cuptipcsampling

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func testSamplingSession(t *testing.T) *samplingSession {
	t.Helper()
	control, _ := testSamplingControl(t)
	control.config.output.reset()
	drain := testSamplingBuffer(t)
	names, err := newSamplingNames(8, 20, 160, func(uintptr, int) (string, error) { return "kernel", nil })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(names.release)
	s := &samplingSession{control: control, drain: drain, names: names, maxReads: 2}
	s.query = func(*cupti.PCSamplingGetDataParams) error {
		drain.data.TotalNumPcs = 1
		drain.pcs[0].FunctionName = 11
		return nil
	}
	t.Cleanup(func() {
		if !s.closed {
			_, _ = s.finish(true)
		}
	})
	return s
}

func TestSamplingSessionFinalBuffer(t *testing.T) {
	s := testSamplingSession(t)
	if err := s.begin(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.read(); err != nil {
		t.Fatal(err)
	}
	s.control.api.disable = func(*cupti.PCSamplingDisableParams) error {
		out := s.control.config.output
		out.data.TotalNumPcs = 1
		out.pcs[0].FunctionName = 22
		out.pcs[0].CubinCrc = 99
		return nil
	}
	got, err := s.finish(true)
	if err != nil || len(got.pcs) != 1 || got.pcs[0].crc != 99 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	if !s.names.released || !s.drain.released || !s.control.config.output.released {
		t.Fatal("finish retained storage")
	}
	if _, err := s.finish(true); err != nil {
		t.Fatal("non-idempotent finish")
	}
}

func TestSamplingSessionDisableRetryRetainsStorage(t *testing.T) {
	s := testSamplingSession(t)
	if err := s.begin(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.read(); err != nil {
		t.Fatal(err)
	}
	failure := errors.New("disable failed")
	s.control.api.disable = func(*cupti.PCSamplingDisableParams) error { return failure }
	if _, err := s.finish(true); !errors.Is(err, failure) || s.closed || s.drain.released || s.names.released {
		t.Fatalf("closed=%v names released=%v err=%v", s.closed, s.names.released, err)
	}
	s.control.api.disable = func(*cupti.PCSamplingDisableParams) error { return nil }
	if _, err := s.finish(true); err == nil || !s.closed || !s.control.failed {
		t.Fatal("retry lost previous failure or ownership")
	}
}

func TestSamplingSessionBudgetAndLoss(t *testing.T) {
	s := testSamplingSession(t)
	s.maxReads = 1
	if err := s.begin(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.read(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.read(); err == nil {
		t.Fatal("exceeded read budget")
	}
	s.control.api.disable = func(*cupti.PCSamplingDisableParams) error {
		s.control.config.output.data.DroppedSamples = 1<<63 + 1
		return nil
	}
	got, err := s.finish(true)
	if err == nil || got.dropped != 1<<63+1 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestSamplingSessionSharingGate(t *testing.T) {
	s := testSamplingSession(t)
	if err := s.begin(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.finish(false); err == nil || s.control.disabled || s.names.released {
		t.Fatal("tore down shared data early")
	}
	if _, err := s.finish(true); err != nil {
		t.Fatal(err)
	}
}

func TestSamplingSessionEarlierLoss(t *testing.T) {
	s := testSamplingSession(t)
	if err := s.begin(); err != nil {
		t.Fatal(err)
	}
	s.query = func(*cupti.PCSamplingGetDataParams) error { s.drain.data.DroppedSamples = 3; return nil }
	got, err := s.read()
	if err != nil || got.dropped != 3 {
		t.Fatalf("batch=%+v err=%v", got, err)
	}
	if _, err := s.finish(true); err == nil || !s.lossObserved {
		t.Fatal("clean final buffer erased prior loss")
	}
}
