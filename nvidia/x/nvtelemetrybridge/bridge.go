package nvtelemetrybridge

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// Layout defines a single-key, non-per-CPU BPF_MAP_TYPE_HASH snapshot.
const (
	Version     = 1
	KeySize     = 4
	MaxDevices  = 8
	MaxReadings = 128
	HeaderSize  = 208
	DeviceSize  = 256
	ReadingSize = 256
	ValueSize   = HeaderSize + MaxDevices*DeviceSize + MaxReadings*ReadingSize
)

var (
	ErrLayout   = errors.New("incompatible telemetry map layout")
	ErrSnapshot = errors.New("invalid telemetry map snapshot")
	ErrStale    = errors.New("stale telemetry map snapshot")
	ErrClosed   = errors.New("telemetry bridge closed")
)

// Store is a map descriptor. Lookup copies a whole value; Update must replace
// the hash entry, never mutate its existing value in place.
type Store interface {
	Lookup(key, value []byte) error
	Update(key, value []byte, flags uint64) error
	Close() error
}

// Info is the inspected kernel map identity and layout. BTF-backed layouts are
// refused: this wire schema is independent of BTF and does not permit relocation.
type Info struct {
	ID, Type, KeySize, ValueSize, MaxEntries                   uint32
	BTFID, BTFKeyTypeID, BTFValueTypeID, BTFVmlinuxValueTypeID uint32
}

// Options selects descriptor access and pin retention. RemovePin must verify
// the expected kernel map ID under the caller's namespace ownership protocol.
// A nil callback is permitted only when the pin is retained or not owned.
type Options struct {
	ReadOnly  bool
	OwnPin    bool
	RetainPin bool
	PinPath   string
	RemovePin func(path string, expectedID uint32) error
}

// Bridge owns its supplied descriptor. Its zero value is not usable.
type Bridge struct {
	mu                       sync.Mutex
	store                    Store
	info                     Info
	opts                     Options
	owner                    [16]byte
	generation               uint64
	closed                   bool
	descriptorCloseAttempted bool
	descriptorCloseErr       error
	pinRemovalPending        bool
	now                      func() time.Time
}

// New checks the map layout and adopts store on success. The caller closes
// store if New fails. A newly created hash map has no snapshot until Publish.
func New(store Store, info Info, opts Options) (*Bridge, error) {
	if nilStore(store) || info.Type != 1 || info.KeySize != KeySize || info.ValueSize != ValueSize || info.MaxEntries != 1 || info.BTFID != 0 || info.BTFKeyTypeID != 0 || info.BTFValueTypeID != 0 || info.BTFVmlinuxValueTypeID != 0 {
		return nil, ErrLayout
	}
	if opts.OwnPin && !opts.RetainPin && (opts.RemovePin == nil || opts.PinPath == "" || info.ID == 0) {
		return nil, errors.New("owned pin removal requires path, map identity and callback")
	}
	b := &Bridge{store: store, info: info, opts: opts, now: time.Now, pinRemovalPending: opts.OwnPin && !opts.RetainPin}
	if !opts.ReadOnly {
		if _, err := rand.Read(b.owner[:]); err != nil {
			return nil, fmt.Errorf("create publisher identity: %w", err)
		}
	}
	return b, nil
}

// Snapshot is one publication. Owner and Generation identify a publisher
// incarnation and publication; a changed Owner denotes a restart or takeover.
type Snapshot struct {
	Owner         [16]byte
	Generation    uint64
	PublishedUnix int64
	EpochID       string
	ClockDomain   string
	SchemaVersion int
	Devices       []nvtelemetry.DeviceMetadata
	Sample        nvtelemetry.Sample
}

// Publish replaces the whole map value. Oversized or contradictory snapshots
// are refused before the map is updated. now is the publication wall time.
func (b *Bridge) Publish(manifest *nvtelemetry.Manifest, sample nvtelemetry.Sample, now time.Time) error {
	if b == nil {
		return ErrClosed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed || b.store == nil {
		return ErrClosed
	}
	if b.opts.ReadOnly {
		return errors.New("read-only telemetry bridge")
	}
	if manifest == nil || manifest.ClockMonotonicEpochID == "" || manifest.ClockMonotonicDomain == "" {
		return fmt.Errorf("%w: missing monotonic epoch identity", ErrSnapshot)
	}
	if b.generation == math.MaxUint64 {
		return errors.New("publication generation overflow")
	}
	s := Snapshot{Owner: b.owner, Generation: b.generation + 1, PublishedUnix: now.UnixNano(), EpochID: manifest.ClockMonotonicEpochID, ClockDomain: manifest.ClockMonotonicDomain, SchemaVersion: manifest.SchemaVersion, Devices: manifest.Devices, Sample: sample}
	value, err := Encode(s)
	if err != nil {
		return err
	}
	if err := b.store.Update(make([]byte, KeySize), value, 0); err != nil {
		return fmt.Errorf("publish telemetry snapshot: %w", err)
	}
	b.generation++
	return nil
}

// Read copies and validates a single hash entry. A positive maxAge is required;
// publications in the future are refused rather than assumed synchronized.
// Wall time is sampled after Lookup so concurrent publication cannot create a
// false future timestamp from a caller clock sampled before the lookup.
func (b *Bridge) Read(maxAge time.Duration) (Snapshot, error) {
	if b == nil {
		return Snapshot{}, ErrClosed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed || b.store == nil {
		return Snapshot{}, ErrClosed
	}
	if maxAge <= 0 {
		return Snapshot{}, errors.New("positive snapshot age limit required")
	}
	value := make([]byte, ValueSize)
	if err := b.store.Lookup(make([]byte, KeySize), value); err != nil {
		return Snapshot{}, fmt.Errorf("read telemetry snapshot: %w", err)
	}
	s, err := Decode(value)
	if err != nil {
		return Snapshot{}, err
	}
	age := b.now().Sub(time.Unix(0, s.PublishedUnix))
	if age < 0 || age > maxAge {
		return Snapshot{}, ErrStale
	}
	return s, nil
}

// Close releases the descriptor once. Reused and retained pins survive. An
// owned removal failure is returned alongside a descriptor-close failure. A
// later explicit Close retries a failed pin removal only; descriptor close is
// never retried, and its first error remains visible on subsequent calls.
func (b *Bridge) Close() error {
	if b == nil {
		return ErrClosed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.store == nil {
		return ErrClosed
	}
	b.closed = true
	var pinErr error
	if b.pinRemovalPending {
		pinErr = b.opts.RemovePin(b.opts.PinPath, b.info.ID)
		if pinErr == nil {
			b.pinRemovalPending = false
		}
	}
	if !b.descriptorCloseAttempted {
		b.descriptorCloseAttempted = true
		b.descriptorCloseErr = b.store.Close()
	}
	return errors.Join(pinErr, b.descriptorCloseErr)
}

func putString(dst []byte, s string) error {
	if len(s) >= len(dst) || strings.IndexByte(s, 0) >= 0 {
		return fmt.Errorf("%w: string exceeds layout or contains NUL", ErrSnapshot)
	}
	copy(dst, s)
	return nil
}
func getString(src []byte) (string, error) {
	n := 0
	for n < len(src) && src[n] != 0 {
		n++
	}
	if n == len(src) {
		return "", ErrSnapshot
	}
	for _, v := range src[n:] {
		if v != 0 {
			return "", ErrSnapshot
		}
	}
	return string(src[:n]), nil
}
func put64(dst []byte, n int64) { binary.LittleEndian.PutUint64(dst, uint64(n)) }
func get64(src []byte) int64    { return int64(binary.LittleEndian.Uint64(src)) }

func nilStore(store Store) bool {
	if store == nil {
		return true
	}
	v := reflect.ValueOf(store)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}
