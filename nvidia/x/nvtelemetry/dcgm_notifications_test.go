package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/dcgm"
	"io"
	"testing"
)

type fakeNotifications struct {
	queue  chan dcgm.PolicyCallbackResponse
	fail   error
	closed int
}

type changingPolicyClock struct{ *FakeClock }

func (c changingPolicyClock) NowMonotonic() int64 {
	n := c.FakeClock.NowMonotonic()
	c.SetEpochID("changed")
	return n
}

func TestPolicyClockChangesDuringCapture(t *testing.T) {
	f := &fakeNotifications{queue: make(chan dcgm.PolicyCallbackResponse, 1)}
	f.queue <- dcgm.PolicyCallbackResponse{Version: 0x2000020, GpuId: 7}
	s, err := NewDCGMPolicySource(f, changingPolicyClock{policyClock()}, policyConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := s.Next(t.Context()); err == nil {
		t.Fatal("clock epoch change admitted")
	}
}

func (f *fakeNotifications) Notifications() <-chan dcgm.PolicyCallbackResponse { return f.queue }
func (f *fakeNotifications) Dropped() uint64                                   { return 3 }
func (f *fakeNotifications) Close() error {
	if f.fail != nil {
		return f.fail
	}
	f.closed++
	close(f.queue)
	return nil
}
func policyClock() *FakeClock { c := NewFakeClock(10, 100); c.SetEpochID("instance"); return c }
func policyConfig() DCGMPolicyConfig {
	return DCGMPolicyConfig{SourceID: "owned-source", ProducerVersion: "fixture-version", NativeOrigin: "engine", ClockDomain: "synthetic_fake_clock", ClockEpoch: "instance", DeviceIDs: map[uint32]string{7: "GPU-a"}}
}
func ExampleNewDCGMPolicySource() {
	f := &fakeNotifications{queue: make(chan dcgm.PolicyCallbackResponse, 1)}
	f.queue <- dcgm.PolicyCallbackResponse{Version: 0x2000020, GpuId: 7, Val: [2]uint64{17, 23}}
	source, _ := NewDCGMPolicySource(f, policyClock(), policyConfig())
	event, _ := source.Next(context.Background())
	fmt.Println(event.DeviceUUID, event.RawPayload, event.LayoutKnown, event.Dropped)
	source.Close()
	// Output: GPU-a [17 23] true 3
}
func TestPolicySourcePublicDiagnostic(t *testing.T) {
	f := &fakeNotifications{queue: make(chan dcgm.PolicyCallbackResponse, 2)}
	c := policyConfig()
	s, err := NewDCGMPolicySource(f, policyClock(), c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Next(nil); err == nil {
		t.Fatal("nil context admitted")
	}
	if _, err := NewDCGMPolicySource(f, RealClock{}, c); err == nil {
		t.Fatal("clock mismatch admitted")
	}
	c.DeviceIDs[7] = "mutated"
	f.queue <- dcgm.PolicyCallbackResponse{Version: 0x2000020, GpuId: 7, Condition: dcgm.PolicyCondXid, Val: [2]uint64{9007199254740993, 23}}
	event, err := s.Next(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if !event.LossUnknown {
		t.Fatal("upstream loss claimed known")
	}
	for _, mutate := range []func(*DCGMPolicyEvent){
		func(e *DCGMPolicyEvent) { e.LossUnknown = false },
		func(e *DCGMPolicyEvent) { e.SourceID = string([]byte{0xff}) },
		func(e *DCGMPolicyEvent) { e.DeviceUUID = string([]byte{0xff}) },
	} {
		bad := event
		mutate(&bad)
		if WriteDCGMPolicyEvent(io.Discard, bad) == nil {
			t.Fatal("invalid contract admitted")
		}
	}
	badConfig := policyConfig()
	badConfig.SourceID = string([]byte{0xff})
	if _, err := NewDCGMPolicySource(f, policyClock(), badConfig); err == nil {
		t.Fatal("invalid UTF-8 identity admitted")
	}
	if WriteDCGMPolicyEvent(nil, event) == nil {
		t.Fatal("nil writer admitted")
	}
	var buf bytes.Buffer
	if err := WriteDCGMPolicyEvent(&buf, event); err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		DeviceUUID string    `json:"device_uuid"`
		RawPayload [2]uint64 `json:"raw_payload"`
	}
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.DeviceUUID != "GPU-a" || decoded.RawPayload[0] != 9007199254740993 {
		t.Fatal(decoded)
	}
	f.queue <- dcgm.PolicyCallbackResponse{Version: 1, GpuId: 99}
	unknown, err := s.Next(t.Context())
	if err != nil || unknown.LayoutKnown || unknown.DeviceKnown {
		t.Fatal(unknown, err)
	}
	if err := WriteDCGMPolicyEvent(io.Discard, unknown); err != nil {
		t.Fatal(err)
	}
	f.fail = errors.New("unregister failed")
	if s.Close() == nil {
		t.Fatal("failure lost")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := s.Next(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	f.fail = nil
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil || f.closed != 1 {
		t.Fatal(err, f.closed)
	}
	if _, err := s.Next(t.Context()); err != io.EOF {
		t.Fatal(err)
	}
	event.LayoutKnown = false
	if WriteDCGMPolicyEvent(io.Discard, event) == nil {
		t.Fatal("contradiction accepted")
	}
	var nilReader *fakeNotifications
	if _, err := NewDCGMPolicySource(nilReader, policyClock(), policyConfig()); err == nil {
		t.Fatal("typed nil accepted")
	}
}
func ExampleWriteDCGMPolicyEvent() {
	err := WriteDCGMPolicyEvent(io.Discard, DCGMPolicyEvent{})
	fmt.Println(err != nil)
	// Output: true
}
func ExampleDCGMPolicySource_Close() {
	var s DCGMPolicySource
	fmt.Println(errors.Is(s.Close(), ErrDCGMPolicySourceClosed))
	// Output: true
}
