package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/dcgm"
)

func copyFixed256(s string) [256]byte {
	var out [256]byte
	copy(out[:255], s)
	return out
}

func fieldPolicyV3Config() DCGMFieldPolicyV3Config {
	return DCGMFieldPolicyV3Config{
		SourceID:        "owned-v3-source",
		ProducerVersion: "dcgm-4.7.0-2",
		NativeOrigin:    "hostengine",
		ClockDomain:     "synthetic_fake_clock",
		ClockEpoch:      "instance",
		FieldID:         uint16(dcgm.FieldDevGpuTempCelsius),
		PolicyID:        42,
		DeviceIDs:       map[uint32]string{7: "GPU-a"},
	}
}

func sampleViolation() dcgm.FieldPolicyViolation {
	return dcgm.FieldPolicyViolation{
		PolicyId:       42,
		PolicyName:     copyFixed256("thermal_ceiling"),
		FieldName:      copyFixed256("DCGM_FI_DEV_GPU_TEMP"),
		FieldId:        uint16(dcgm.FieldDevGpuTempCelsius),
		Entity:         dcgm.GroupEntityPair{EntityGroupId: dcgm.FeGpu, EntityId: 7},
		CurrentValue:   88.5,
		ThresholdValue: 85.0,
		PolicyOperator: dcgm.PolicyOpGt,
		ChannelMask:    0x3,
		Timestamp:      1700000000123456,
	}
}

func ExampleNewDCGMFieldPolicyV3Source() {
	var token uint64
	sub, _ := NewDCGMFieldPolicyV3NotificationsWithBinder(1, uint16(dcgm.FieldDevGpuTempCelsius), 42, 2, DCGMFieldPolicyV3Binder{
		Register: func(_ dcgm.Handle, _ uint16, _ uint64, _ dcgm.FpRecvPolicyViolation, u uint64) error {
			token = u
			return nil
		},
		Unregister: func(dcgm.Handle, uint16, uint64) error { return nil },
	})
	v := sampleViolation()
	DispatchDCGMFieldPolicyV3Callback(&v, token)
	source, _ := NewDCGMFieldPolicyV3Source(sub, policyClock(), fieldPolicyV3Config())
	event, _ := source.Next(context.Background())
	fmt.Println(event.Schema, event.DeviceUUID, event.PolicyName, event.CurrentValueFloat64(), event.LayoutKnown, event.LossUnknown)
	_ = source.Close()
	// Output: nvtelemetry.dcgm-field-policy.v3 GPU-a thermal_ceiling 88.5 true true
}

func ExampleWriteDCGMFieldPolicyV3Event() {
	err := WriteDCGMFieldPolicyV3Event(io.Discard, DCGMFieldPolicyV3Event{})
	fmt.Println(err != nil)
	// Output: true
}

func TestDCGMFieldPolicyV3NotificationsLifecycleAndOwnership(t *testing.T) {
	// Missing symbols prove registration was refused before the native call.
	symBinder := DCGMFieldPolicyV3Binder{
		Register: func(dcgm.Handle, uint16, uint64, dcgm.FpRecvPolicyViolation, uint64) error {
			return &dcgm.SymbolError{Symbol: "dcgmPolicyRegister_v3", Err: errors.New("missing")}
		},
		Unregister: func(dcgm.Handle, uint16, uint64) error { return nil },
	}
	if sub, err := NewDCGMFieldPolicyV3NotificationsWithBinder(1, 203, 42, 4, symBinder); err == nil || sub != nil {
		t.Fatalf("expected missing symbol refusal with nil sub, got sub=%v err=%v", sub, err)
	}

	// 2. Invalid configuration refusals.
	validBinder := DCGMFieldPolicyV3Binder{
		Register:   func(dcgm.Handle, uint16, uint64, dcgm.FpRecvPolicyViolation, uint64) error { return nil },
		Unregister: func(dcgm.Handle, uint16, uint64) error { return nil },
	}
	for _, tc := range []struct {
		h   dcgm.Handle
		fid uint16
		pid uint64
		cap int
		b   DCGMFieldPolicyV3Binder
	}{
		{0, 203, 42, 4, validBinder},
		{1, 0, 42, 4, validBinder},
		{1, 203, 0, 4, validBinder},
		{1, 203, 42, 0, validBinder},
		{1, 203, 42, 4097, validBinder},
		{1, 203, 42, 4, DCGMFieldPolicyV3Binder{Unregister: validBinder.Unregister}},
		{1, 203, 42, 4, DCGMFieldPolicyV3Binder{Register: validBinder.Register}},
	} {
		if _, err := NewDCGMFieldPolicyV3NotificationsWithBinder(tc.h, tc.fid, tc.pid, tc.cap, tc.b); err == nil {
			t.Fatalf("accepted invalid v3 notification config: %+v", tc)
		}
	}

	// 3. Native registration error preserves non-nil subscription so caller can Close it.
	var regToken uint64
	nativeFailBinder := DCGMFieldPolicyV3Binder{
		Register: func(_ dcgm.Handle, _ uint16, _ uint64, _ dcgm.FpRecvPolicyViolation, u uint64) error {
			regToken = u
			return dcgm.ErrGenericError
		},
		Unregister: func(dcgm.Handle, uint16, uint64) error { return nil },
	}
	failedSub, err := NewDCGMFieldPolicyV3NotificationsWithBinder(99, 203, 42, 1, nativeFailBinder)
	if err == nil || failedSub == nil {
		t.Fatalf("expected non-nil subscription on native registration error, got sub=%v err=%v", failedSub, err)
	}
	v := sampleViolation()
	DispatchDCGMFieldPolicyV3Callback(&v, regToken)
	got := <-failedSub.Violations()
	if got.PolicyId != 42 {
		t.Fatalf("expected side-effect callback before Close, got %+v", got)
	}
	if err := failedSub.Close(); err != nil {
		t.Fatalf("close failed registration subscription: %v", err)
	}

	// 4. Overlap refusal, queue overflow drops, nil callback drops, failed unregister retry, and late callback retirement.
	var activeToken uint64
	var unregisterErr error
	var unregisterCalls int
	binder := DCGMFieldPolicyV3Binder{
		Register: func(_ dcgm.Handle, _ uint16, _ uint64, cb dcgm.FpRecvPolicyViolation, u uint64) error {
			if cb == 0 {
				t.Fatal("zero callback trampoline pointer")
			}
			activeToken = u
			return nil
		},
		Unregister: func(dcgm.Handle, uint16, uint64) error {
			unregisterCalls++
			return unregisterErr
		},
	}
	sub, err := NewDCGMFieldPolicyV3NotificationsWithBinder(2, 203, 42, 1, binder)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewDCGMFieldPolicyV3NotificationsWithBinder(2, 203, 42, 1, binder); err == nil {
		t.Fatal("duplicate (handle, fieldID, policyID) registration admitted")
	}

	DispatchDCGMFieldPolicyV3Callback(nil, activeToken)
	if sub.Dropped() != 1 {
		t.Fatalf("expected 1 drop for nil violation pointer, got %d", sub.Dropped())
	}
	DispatchDCGMFieldPolicyV3Callback(&v, activeToken)
	DispatchDCGMFieldPolicyV3Callback(&v, activeToken)
	if sub.Dropped() != 2 {
		t.Fatalf("expected 2 drops after queue overflow, got %d", sub.Dropped())
	}

	// Failed unregister must preserve ownership and keep token active.
	unregisterErr = dcgm.ErrInUse
	if err := sub.Close(); !errors.Is(err, dcgm.ErrInUse) {
		t.Fatalf("expected ErrInUse from Close, got %v", err)
	}
	<-sub.Violations()
	DispatchDCGMFieldPolicyV3Callback(&v, activeToken)
	if got := <-sub.Violations(); got.PolicyId != 42 {
		t.Fatalf("subscription retired despite failed unregister: %+v", got)
	}

	// Successful retry retires token and closes channel; late callbacks are ignored.
	unregisterErr = nil
	if err := sub.Close(); err != nil {
		t.Fatalf("retry Close failed: %v", err)
	}
	if err := sub.Close(); err != nil || unregisterCalls != 2 {
		t.Fatalf("idempotent Close failed: err=%v calls=%d", err, unregisterCalls)
	}
	DispatchDCGMFieldPolicyV3Callback(&v, activeToken)
	if _, ok := <-sub.Violations(); ok {
		t.Fatal("late callback delivered after Close")
	}
}

func TestDCGMFieldPolicyV3SourceEndToEndAndRefusals(t *testing.T) {
	var token uint64
	var unregErr error
	binder := DCGMFieldPolicyV3Binder{
		Register: func(_ dcgm.Handle, _ uint16, _ uint64, _ dcgm.FpRecvPolicyViolation, u uint64) error {
			token = u
			return nil
		},
		Unregister: func(dcgm.Handle, uint16, uint64) error {
			return unregErr
		},
	}
	sub, err := NewDCGMFieldPolicyV3NotificationsWithBinder(1, uint16(dcgm.FieldDevGpuTempCelsius), 42, 4, binder)
	if err != nil {
		t.Fatal(err)
	}

	cfg := fieldPolicyV3Config()
	if _, err := NewDCGMFieldPolicyV3Source(sub, RealClock{}, cfg); err == nil {
		t.Fatal("clock provenance mismatch admitted")
	}
	badID := cfg
	badID.SourceID = string([]byte{0xff})
	if _, err := NewDCGMFieldPolicyV3Source(sub, policyClock(), badID); err == nil {
		t.Fatal("invalid UTF-8 source identity admitted")
	}
	var nilReader *DCGMFieldPolicyV3Notifications
	if _, err := NewDCGMFieldPolicyV3Source(nilReader, policyClock(), cfg); err == nil {
		t.Fatal("typed nil reader admitted")
	}

	source, err := NewDCGMFieldPolicyV3Source(sub, policyClock(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Mutating caller map after construction must not affect source inventory.
	cfg.DeviceIDs[7] = "GPU-mutated"

	if _, err := source.Next(nil); err == nil {
		t.Fatal("nil context admitted")
	}

	// Verify v2 and v3 coexist without interference.
	v2Fake := &fakeNotifications{queue: make(chan dcgm.PolicyCallbackResponse, 1)}
	v2Fake.queue <- dcgm.PolicyCallbackResponse{Version: 0x02000020, GpuId: 7, Val: [2]uint64{11, 22}}
	v2Source, err := NewDCGMPolicySource(v2Fake, policyClock(), policyConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer v2Source.Close()
	v2Event, err := v2Source.Next(t.Context())
	if err != nil || v2Event.Schema != "nvtelemetry.dcgm-policy.v1" {
		t.Fatalf("v2 coexistence failed: %+v %v", v2Event, err)
	}

	// Known v3 event round-trip.
	v := sampleViolation()
	DispatchDCGMFieldPolicyV3Callback(&v, token)
	event, err := source.Next(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if event.Schema != DCGMFieldPolicyV3Schema ||
		event.DeviceUUID != "GPU-a" ||
		!event.LayoutKnown ||
		!event.DeviceKnown ||
		!event.LossUnknown ||
		event.CurrentValueBits != math.Float64bits(88.5) ||
		event.ThresholdValueBits != math.Float64bits(85.0) ||
		event.PolicyOperator != dcgm.PolicyOpGt ||
		event.ViolationTimestamp != 1700000000123456 {
		t.Fatalf("unexpected v3 event: %+v", event)
	}

	var buf bytes.Buffer
	if err := WriteDCGMFieldPolicyV3Event(&buf, event); err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadDCGMFieldPolicyV3Event(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != event {
		t.Fatalf("round-trip mismatch:\ngot:  %+v\nwant: %+v", decoded, event)
	}

	// Unknown layout (mismatched PolicyID / invalid operator / unterminated string) degrades cleanly to LayoutKnown=false, DeviceKnown=false.
	badLayout := sampleViolation()
	badLayout.PolicyId = 999
	for i := range badLayout.PolicyName {
		badLayout.PolicyName[i] = 'x' // no NUL terminator
	}
	DispatchDCGMFieldPolicyV3Callback(&badLayout, token)
	unknownEvt, err := source.Next(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if unknownEvt.LayoutKnown || unknownEvt.DeviceKnown || unknownEvt.DeviceUUID != "" || unknownEvt.PolicyName != "" || unknownEvt.FieldName != "" {
		t.Fatalf("expected unknown layout to scrub unverified strings and device, got %+v", unknownEvt)
	}
	var unknownBuf bytes.Buffer
	if err := WriteDCGMFieldPolicyV3Event(&unknownBuf, unknownEvt); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadDCGMFieldPolicyV3Event(&unknownBuf); err != nil {
		t.Fatal(err)
	}

	// Non-GPU entity (FeSwitch) with valid layout sets LayoutKnown=true, DeviceKnown=false.
	switchViolation := sampleViolation()
	switchViolation.Entity = dcgm.GroupEntityPair{EntityGroupId: dcgm.FeSwitch, EntityId: 7}
	DispatchDCGMFieldPolicyV3Callback(&switchViolation, token)
	switchEvt, err := source.Next(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if !switchEvt.LayoutKnown || switchEvt.DeviceKnown || switchEvt.DeviceUUID != "" {
		t.Fatalf("expected FeSwitch event to have LayoutKnown=true and DeviceKnown=false, got %+v", switchEvt)
	}

	// Writer and reader contract refusal controls.
	for name, mutate := range map[string]func(*DCGMFieldPolicyV3Event){
		"wrong_schema":       func(e *DCGMFieldPolicyV3Event) { e.Schema = "nvtelemetry.dcgm-policy.v1" },
		"negative_received":  func(e *DCGMFieldPolicyV3Event) { e.ReceivedNanos = -1 },
		"loss_claimed_known": func(e *DCGMFieldPolicyV3Event) { e.LossUnknown = false },
		"device_contradict":  func(e *DCGMFieldPolicyV3Event) { e.DeviceKnown = false },
		"unknown_layout_dev": func(e *DCGMFieldPolicyV3Event) { e.LayoutKnown = false },
		"non_gpu_device":     func(e *DCGMFieldPolicyV3Event) { e.EntityGroup = dcgm.FeSwitch },
		"invalid_operator":   func(e *DCGMFieldPolicyV3Event) { e.PolicyOperator = dcgm.PolicyOpCount },
		"negative_violation": func(e *DCGMFieldPolicyV3Event) { e.ViolationTimestamp = -1 },
		"invalid_utf8_src":   func(e *DCGMFieldPolicyV3Event) { e.SourceID = string([]byte{0xff}) },
		"invalid_utf8_pol":   func(e *DCGMFieldPolicyV3Event) { e.PolicyName = string([]byte{0xff}) },
	} {
		bad := event
		mutate(&bad)
		if err := WriteDCGMFieldPolicyV3Event(io.Discard, bad); err == nil {
			t.Fatalf("writer accepted invalid event (%s): %+v", name, bad)
		}
	}
	if _, err := ReadDCGMFieldPolicyV3Event(strings.NewReader(`{"schema":"nvtelemetry.dcgm-field-policy.v3","unknown_field":1}`)); err == nil {
		t.Fatal("reader accepted unknown JSON field")
	}

	// Close failure preserves source for retry.
	unregErr = errors.New("busy")
	if err := source.Close(); err == nil {
		t.Fatal("expected Close error when reader unregister fails")
	}
	unregErr = nil
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := source.Next(t.Context()); err != io.EOF {
		t.Fatalf("expected EOF after Close, got %v", err)
	}
}

func TestDCGMFieldPolicyV3ClockEpochChange(t *testing.T) {
	var token uint64
	sub, err := NewDCGMFieldPolicyV3NotificationsWithBinder(1, uint16(dcgm.FieldDevGpuTempCelsius), 42, 1, DCGMFieldPolicyV3Binder{
		Register: func(_ dcgm.Handle, _ uint16, _ uint64, _ dcgm.FpRecvPolicyViolation, u uint64) error {
			token = u
			return nil
		},
		Unregister: func(dcgm.Handle, uint16, uint64) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()
	v := sampleViolation()
	DispatchDCGMFieldPolicyV3Callback(&v, token)
	source, err := NewDCGMFieldPolicyV3Source(sub, changingPolicyClock{policyClock()}, fieldPolicyV3Config())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := source.Next(t.Context()); err == nil {
		t.Fatal("clock epoch change during capture admitted")
	}
}

func TestDCGMFieldPolicyV3EncodedBounds(t *testing.T) {
	s := strings.Repeat("\x01", 256)
	e := DCGMFieldPolicyV3Event{
		Schema: DCGMFieldPolicyV3Schema, SourceID: s, ProducerVersion: s,
		NativeOrigin: s, ClockDomain: s, ClockEpoch: s, DeviceUUID: s,
		PolicyID: 42, FieldID: 203, EntityGroup: dcgm.FeGpu, EntityID: 7,
		PolicyName: s[:255], FieldName: s[:255], PolicyOperator: dcgm.PolicyOpGt,
		LayoutKnown: true, DeviceKnown: true, LossUnknown: true,
	}
	var b bytes.Buffer
	if err := WriteDCGMFieldPolicyV3Event(&b, e); err != nil {
		t.Fatal(err)
	}
	if b.Len() <= 8192 || b.Len() > dcgmFieldPolicyV3MaxBytes {
		t.Fatalf("encoded size = %d", b.Len())
	}
	data := append([]byte(nil), b.Bytes()...)
	for _, size := range []int{len(data), dcgmFieldPolicyV3MaxBytes, dcgmFieldPolicyV3MaxBytes + 1} {
		padded := append(append([]byte(nil), data...), bytes.Repeat([]byte(" "), size-len(data))...)
		got, err := ReadDCGMFieldPolicyV3Event(bytes.NewReader(padded))
		if size > dcgmFieldPolicyV3MaxBytes {
			if err == nil {
				t.Fatal("reader accepted oversized line")
			}
		} else if err != nil || got != e {
			t.Fatalf("roundtrip at %d bytes = %+v, %v", size, got, err)
		}
	}
	b.Reset()
	e.SourceID += "x"
	if err := WriteDCGMFieldPolicyV3Event(&b, e); err == nil || b.Len() != 0 {
		t.Fatalf("invalid event write = %d bytes, %v", b.Len(), err)
	}
}

func TestDCGMFieldPolicyV3UnknownRegistrationError(t *testing.T) {
	postCall := errors.New("post-call bookkeeping")
	unregisterFailure := errors.New("unregister failed")
	active := false
	var token uint64
	binder := DCGMFieldPolicyV3Binder{
		Register: func(_ dcgm.Handle, _ uint16, _ uint64, _ dcgm.FpRecvPolicyViolation, u uint64) error {
			active, token = true, u
			return fmt.Errorf("register bookkeeping: %w", postCall)
		},
		Unregister: func(dcgm.Handle, uint16, uint64) error {
			if unregisterFailure != nil {
				return unregisterFailure
			}
			active = false
			return nil
		},
	}
	sub, err := NewDCGMFieldPolicyV3NotificationsWithBinder(41, 203, 42, 1, binder)
	if sub == nil || !errors.Is(err, postCall) {
		t.Fatalf("owner = %v, error = %v", sub, err)
	}
	defer func() { unregisterFailure = nil; _ = sub.Close() }()
	if err := sub.Close(); !errors.Is(err, unregisterFailure) || !active {
		t.Fatalf("failed close lost owner: %v", err)
	}
	if duplicate, err := NewDCGMFieldPolicyV3NotificationsWithBinder(41, 203, 42, 1, binder); duplicate != nil || err == nil {
		t.Fatal("duplicate key admitted")
	}
	v := sampleViolation()
	DispatchDCGMFieldPolicyV3Callback(&v, token)
	select {
	case got := <-sub.Violations():
		if got != v {
			t.Fatal("callback changed")
		}
	default:
		t.Fatal("unknown registration error retired callback")
	}
	unregisterFailure = nil
	if err := sub.Close(); err != nil || active {
		t.Fatalf("close = %v, active = %v", err, active)
	}
}
