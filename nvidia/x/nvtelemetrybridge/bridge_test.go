package nvtelemetrybridge

import (
	"encoding/binary"
	"errors"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type memoryMap struct {
	mu       sync.Mutex
	data     []byte
	closes   int
	fail     error
	closeErr error
}

func (m *memoryMap) Lookup(key, value []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.data) == 0 {
		return errors.New("no entry")
	}
	copy(value, m.data)
	return nil
}
func (m *memoryMap) Update(key, value []byte, flags uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.fail != nil {
		return m.fail
	}
	m.data = append([]byte(nil), value...)
	return nil
}
func (m *memoryMap) Close() error { m.mu.Lock(); defer m.mu.Unlock(); m.closes++; return m.closeErr }
func validInfo() Info {
	return Info{ID: 17, Type: 1, KeySize: KeySize, ValueSize: ValueSize, MaxEntries: 1}
}
func fixture() Snapshot {
	return Snapshot{Owner: [16]byte{1}, Generation: 1, PublishedUnix: 1_000_000_000, EpochID: "fixture-epoch", ClockDomain: "synthetic_fixture", SchemaVersion: 2, Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-test", MIGScope: nvtelemetry.MIGScopeMIGInstance, MIGParentUUID: "GPU-parent", HasMIGInstanceID: true, MIGInstanceID: 0, HasMIGComputeInstanceID: true, MIGComputeInstanceID: 3}}, Sample: nvtelemetry.Sample{Seq: 1, QueryStartNanos: 0, QueryEndNanos: 20, Readings: []nvtelemetry.Reading{{Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules, DeviceUUID: "GPU-test", Evidence: nvtelemetry.EvidenceMeasured, IsInt: true, IntValue: 1<<54 + 1, Value: float64(1<<54 + 1), HasQueryBounds: true, QueryStartNanos: 0, QueryEndNanos: 10}}}}
}
func TestLayoutRoundTrip(t *testing.T) {
	s := fixture()
	data, err := Encode(s)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != ValueSize || got.Sample.Readings[0].IntValue != 1<<54+1 || !got.Devices[0].HasMIGInstanceID || got.Devices[0].MIGInstanceID != 0 || got.EpochID != s.EpochID {
		t.Fatalf("lost identity/integer evidence: %+v", got)
	}
}
func TestRejectLayoutContradictions(t *testing.T) {
	data, err := Encode(fixture())
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		mutate func([]byte) []byte
	}{
		{"short", func(b []byte) []byte { return b[:len(b)-1] }},
		{"version", func(b []byte) []byte { b[8] = 2; return b }},
		{"device count", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[192:], math.MaxUint32); return b }},
		{"reading count", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[12:], math.MaxUint32); return b }},
		{"generation", func(b []byte) []byte { binary.LittleEndian.PutUint64(b[16:], 0); return b }},
		{"owner", func(b []byte) []byte { clear(b[48:64]); return b }},
		{"partial write", func(b []byte) []byte { clear(b[64:128]); return b }},
		{"reserved", func(b []byte) []byte { b[HeaderSize+208] = 1; return b }},
		{"unused slot", func(b []byte) []byte { b[ValueSize-1] = 1; return b }},
		{"unknown device", func(b []byte) []byte {
			binary.LittleEndian.PutUint32(b[HeaderSize+MaxDevices*DeviceSize:], 8)
			return b
		}},
		{"unknown flags", func(b []byte) []byte { b[HeaderSize+MaxDevices*DeviceSize+7] = 128; return b }},
		{"nan", func(b []byte) []byte {
			binary.LittleEndian.PutUint64(b[HeaderSize+MaxDevices*DeviceSize+128:], math.Float64bits(math.NaN()))
			return b
		}},
		{"out of pass", func(b []byte) []byte { put64(b[HeaderSize+MaxDevices*DeviceSize+168:], 21); return b }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bad := tt.mutate(append([]byte(nil), data...))
			if _, err := Decode(bad); err == nil {
				t.Fatal("accepted contradiction")
			}
		})
	}
}
func TestPublishRefusalsPreserveEntry(t *testing.T) {
	m := new(memoryMap)
	b, err := New(m, validInfo(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	s := fixture()
	manifest := &nvtelemetry.Manifest{SchemaVersion: 2, ClockMonotonicEpochID: s.EpochID, ClockMonotonicDomain: s.ClockDomain, Devices: s.Devices}
	if err := b.Publish(manifest, s.Sample, time.Unix(1, 0)); err != nil {
		t.Fatal(err)
	}
	s.Sample.Readings[0].QueryEndNanos = 21
	if err := b.Publish(manifest, s.Sample, time.Unix(2, 0)); err == nil {
		t.Fatal("accepted bounds")
	}
	got, err := readAt(b, time.Unix(1, 0), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if got.Generation != 1 {
		t.Fatal("refusal replaced publication")
	}
	m.fail = errors.New("permission denied")
	s.Sample.Readings[0].QueryEndNanos = 10
	if err := b.Publish(manifest, s.Sample, time.Unix(1, 1)); err == nil {
		t.Fatal("ignored update error")
	}
	m.fail = nil
	if err := b.Publish(manifest, s.Sample, time.Unix(1, 2)); err != nil {
		t.Fatal(err)
	}
	got, err = readAt(b, time.Unix(1, 2), time.Second)
	if err != nil || got.Generation != 2 {
		t.Fatalf("failed write advanced generation: %v %+v", err, got)
	}
}
func TestStaleRestartAndOwnership(t *testing.T) {
	m := new(memoryMap)
	s := fixture()
	manifest := &nvtelemetry.Manifest{SchemaVersion: 2, ClockMonotonicEpochID: s.EpochID, ClockMonotonicDomain: s.ClockDomain, Devices: s.Devices}
	removed := 0
	b, err := New(m, validInfo(), Options{OwnPin: true, PinPath: "private/pin", RemovePin: func(path string, id uint32) error {
		if path != "private/pin" || id != 17 {
			t.Fatal("wrong removal identity")
		}
		removed++
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Publish(manifest, s.Sample, time.Unix(1, 0)); err != nil {
		t.Fatal(err)
	}
	first, err := readAt(b, time.Unix(1, 0), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	for _, now := range []time.Time{time.Unix(0, 0), time.Unix(3, 0)} {
		if _, err := readAt(b, now, time.Second); !errors.Is(err, ErrStale) {
			t.Fatalf("stale/future accepted: %v", err)
		}
	}
	next, err := New(m, validInfo(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := next.Publish(manifest, s.Sample, time.Unix(2, 0)); err != nil {
		t.Fatal(err)
	}
	second, err := readAt(next, time.Unix(2, 0), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if first.Owner == second.Owner || second.Generation != 1 {
		t.Fatal("restart identity not distinct")
	}
	if err := next.Close(); err != nil {
		t.Fatal(err)
	}
	if removed != 0 {
		t.Fatal("reused pin removed")
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if removed != 1 || m.closes != 2 {
		t.Fatalf("ownership cleanup removed=%d closes=%d", removed, m.closes)
	}
	if _, err := readAt(b, time.Unix(2, 0), time.Second); !errors.Is(err, ErrClosed) {
		t.Fatal("read after close")
	}
}
func TestMapAdmission(t *testing.T) {
	for _, kind := range []uint32{2, 5, 9} {
		info := validInfo()
		info.Type = kind
		if _, err := New(new(memoryMap), info, Options{}); !errors.Is(err, ErrLayout) {
			t.Fatal("accepted non-hash map")
		}
	}
	info := validInfo()
	info.BTFValueTypeID = 1
	if _, err := New(new(memoryMap), info, Options{}); !errors.Is(err, ErrLayout) {
		t.Fatal("accepted BTF-backed map")
	}
	info = validInfo()
	info.ValueSize--
	if _, err := New(new(memoryMap), info, Options{}); !errors.Is(err, ErrLayout) {
		t.Fatal("accepted old size")
	}
	if _, err := New(new(memoryMap), validInfo(), Options{OwnPin: true}); err == nil {
		t.Fatal("adopted cleanup without identity protocol")
	}
	b, err := New(new(memoryMap), validInfo(), Options{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	s := fixture()
	if err := b.Publish(&nvtelemetry.Manifest{SchemaVersion: 2, Devices: s.Devices, ClockMonotonicEpochID: s.EpochID, ClockMonotonicDomain: s.ClockDomain}, s.Sample, time.Now()); err == nil {
		t.Fatal("read-only publish")
	}
}

func readAt(b *Bridge, now time.Time, maxAge time.Duration) (Snapshot, error) {
	b.now = func() time.Time { return now }
	return b.Read(maxAge)
}

type advancingMap struct {
	memoryMap
	afterLookup func()
}

func (m *advancingMap) Lookup(key, value []byte) error {
	err := m.memoryMap.Lookup(key, value)
	if m.afterLookup != nil {
		m.afterLookup()
	}
	return err
}
func TestFreshnessClockAfterLookup(t *testing.T) {
	s := fixture()
	s.PublishedUnix = time.Unix(2, 0).UnixNano()
	data, err := Encode(s)
	if err != nil {
		t.Fatal(err)
	}
	clock := time.Unix(1, 0)
	m := &advancingMap{memoryMap: memoryMap{data: data}, afterLookup: func() { clock = time.Unix(2, 0) }}
	b, err := New(m, validInfo(), Options{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	b.now = func() time.Time { return clock }
	if _, err := b.Read(time.Second); err != nil {
		t.Fatalf("concurrent publication was mistaken for future data: %v", err)
	}
}

func TestNilAndZeroBridge(t *testing.T) {
	var store *memoryMap
	if _, err := New(store, validInfo(), Options{}); !errors.Is(err, ErrLayout) {
		t.Fatalf("typed nil store: %v", err)
	}
	var zero Bridge
	for _, b := range []*Bridge{nil, &zero} {
		if err := b.Publish(nil, nvtelemetry.Sample{}, time.Now()); !errors.Is(err, ErrClosed) {
			t.Fatalf("uninitialized publish: %v", err)
		}
		if _, err := b.Read(time.Second); !errors.Is(err, ErrClosed) {
			t.Fatalf("uninitialized read: %v", err)
		}
		if err := b.Close(); !errors.Is(err, ErrClosed) {
			t.Fatalf("uninitialized close: %v", err)
		}
	}
}

func TestCloseFailureAndExplicitPinRetry(t *testing.T) {
	pinErr := errors.New("pin permission denied")
	closeErr := errors.New("ambiguous descriptor close")
	m := &memoryMap{closeErr: closeErr}
	attempts := 0
	b, err := New(m, validInfo(), Options{OwnPin: true, PinPath: "owned/snapshot", RemovePin: func(path string, id uint32) error {
		attempts++
		if attempts == 1 {
			return pinErr
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); !errors.Is(err, pinErr) || !errors.Is(err, closeErr) {
		t.Fatalf("lost first failures: %v", err)
	}
	if _, err := b.Read(time.Second); !errors.Is(err, ErrClosed) {
		t.Fatal("access after failed close")
	}
	if err := b.Close(); errors.Is(err, pinErr) || !errors.Is(err, closeErr) {
		t.Fatalf("retry cleared descriptor error or retained cleared pin error: %v", err)
	}
	if err := b.Close(); !errors.Is(err, closeErr) {
		t.Fatalf("hidden descriptor failure: %v", err)
	}
	if attempts != 2 || m.closes != 1 {
		t.Fatalf("pin attempts=%d descriptor attempts=%d", attempts, m.closes)
	}
}
