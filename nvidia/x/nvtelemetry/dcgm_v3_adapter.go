package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"
	"unicode/utf8"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/dcgm"
)

// DCGMFieldPolicyV3Schema identifies the separately versioned DCGM 4.7+ field-policy v3 diagnostic event record.
const DCGMFieldPolicyV3Schema = "nvtelemetry.dcgm-field-policy.v3"

// ErrDCGMFieldPolicyV3SourceClosed reports a closed or uninitialized v3 field-policy source.
var ErrDCGMFieldPolicyV3SourceClosed = errors.New("dcgm field policy v3 source: closed or uninitialized")

// DCGMFieldPolicyV3Reader supplies copied native v3 field-policy violation responses.
// Implementations must keep their connection and registration live until Close succeeds.
type DCGMFieldPolicyV3Reader interface {
	Violations() <-chan dcgm.FieldPolicyViolation
	Dropped() uint64
	Close() error
}

// DCGMFieldPolicyV3Binder binds the read-only v3 field-policy registration pair.
// Policy creation, mutation, triggering, and remediation are excluded.
type DCGMFieldPolicyV3Binder struct {
	Register   func(handle dcgm.Handle, fieldID uint16, policyID uint64, callback dcgm.FpRecvPolicyViolation, userData uint64) error
	Unregister func(handle dcgm.Handle, fieldID uint16, policyID uint64) error
}

// DCGMFieldPolicyV3Notifications owns one v3 field-policy registration (handle, fieldID, policyID).
// Close must succeed before its connection, host engine, or library is closed.
// The zero value is not usable.
type DCGMFieldPolicyV3Notifications struct {
	closeMu    sync.Mutex
	mu         sync.Mutex
	handle     dcgm.Handle
	fieldID    uint16
	policyID   uint64
	token      uint64
	unregister func(dcgm.Handle, uint16, uint64) error
	closed     bool
	queue      chan dcgm.FieldPolicyViolation
	dropped    atomic.Uint64
}

var fieldPolicyV3Callbacks = struct {
	sync.Mutex
	next   uint64
	active map[uint64]*DCGMFieldPolicyV3Notifications
}{active: make(map[uint64]*DCGMFieldPolicyV3Notifications)}

var fieldPolicyV3CallbackLookup sync.Map // uint64 -> *DCGMFieldPolicyV3Notifications

var fieldPolicyV3Trampoline struct {
	once sync.Once
	ptr  dcgm.FpRecvPolicyViolation
	err  error
}

// NewDCGMFieldPolicyV3Notifications registers a v3 field-policy violation callback
// using the generated dcgm.PolicyRegisterV3 and dcgm.PolicyUnregisterV3 bindings.
// It never creates, sets, or triggers policies.
func NewDCGMFieldPolicyV3Notifications(handle dcgm.Handle, fieldID uint16, policyID uint64, capacity int) (*DCGMFieldPolicyV3Notifications, error) {
	return NewDCGMFieldPolicyV3NotificationsWithBinder(handle, fieldID, policyID, capacity, DCGMFieldPolicyV3Binder{
		Register:   dcgm.PolicyRegisterV3,
		Unregister: dcgm.PolicyUnregisterV3,
	})
}

// NewDCGMFieldPolicyV3NotificationsWithBinder registers a v3 field-policy callback
// using an explicit register/unregister pair. Capacity must be between 1 and 4096.
// At most 64 v3 subscriptions can be active in a process. Duplicate (handle, fieldID, policyID)
// registrations are refused because dcgmPolicyUnregister_v3 unregisters by key rather than token.
// On a registration error, a non-nil subscription is returned alongside the error
// unless a dcgm.SymbolError proves registration was never called. The caller must
// Close a returned subscription before releasing the connection.
func NewDCGMFieldPolicyV3NotificationsWithBinder(handle dcgm.Handle, fieldID uint16, policyID uint64, capacity int, binder DCGMFieldPolicyV3Binder) (*DCGMFieldPolicyV3Notifications, error) {
	if handle == 0 || fieldID == 0 || policyID == 0 || capacity < 1 || capacity > 4096 || binder.Register == nil || binder.Unregister == nil {
		return nil, fmt.Errorf("dcgm field policy v3: invalid notification configuration")
	}
	fieldPolicyV3Trampoline.once.Do(func() {
		defer func() {
			if v := recover(); v != nil {
				fieldPolicyV3Trampoline.err = fmt.Errorf("dcgm field policy v3: create callback trampoline: %v", v)
			}
		}()
		fieldPolicyV3Trampoline.ptr = dcgm.FpRecvPolicyViolation(purego.NewCallback(fieldPolicyV3Callback))
	})
	if fieldPolicyV3Trampoline.err != nil {
		return nil, fieldPolicyV3Trampoline.err
	}

	s := &DCGMFieldPolicyV3Notifications{
		handle:     handle,
		fieldID:    fieldID,
		policyID:   policyID,
		unregister: binder.Unregister,
		queue:      make(chan dcgm.FieldPolicyViolation, capacity),
	}

	fieldPolicyV3Callbacks.Lock()
	if len(fieldPolicyV3Callbacks.active) >= 64 || fieldPolicyV3Callbacks.next == math.MaxUint64 {
		fieldPolicyV3Callbacks.Unlock()
		return nil, fmt.Errorf("dcgm field policy v3: notification capacity exhausted")
	}
	for _, prior := range fieldPolicyV3Callbacks.active {
		if prior.handle == handle && prior.fieldID == fieldID && prior.policyID == policyID {
			fieldPolicyV3Callbacks.Unlock()
			return nil, fmt.Errorf("dcgm field policy v3: registration key already active")
		}
	}
	fieldPolicyV3Callbacks.next++
	s.token = fieldPolicyV3Callbacks.next
	fieldPolicyV3Callbacks.active[s.token] = s
	fieldPolicyV3CallbackLookup.Store(s.token, s)
	fieldPolicyV3Callbacks.Unlock()

	if err := binder.Register(handle, fieldID, policyID, fieldPolicyV3Trampoline.ptr, s.token); err != nil {
		var symErr *dcgm.SymbolError
		if errors.As(err, &symErr) {
			s.retire()
			return nil, fmt.Errorf("dcgm field policy v3: register: %w", err)
		}
		return s, fmt.Errorf("dcgm field policy v3: register: %w", err)
	}
	return s, nil
}

// DispatchDCGMFieldPolicyV3Callback routes a native or fixture v3 violation callback
// by its numeric registration token, copying the value before enqueueing.
func DispatchDCGMFieldPolicyV3Callback(violation *dcgm.FieldPolicyViolation, token uint64) int32 {
	return fieldPolicyV3Callback(violation, token)
}

func fieldPolicyV3Callback(violation *dcgm.FieldPolicyViolation, token uint64) int32 {
	v, ok := fieldPolicyV3CallbackLookup.Load(token)
	if !ok {
		return 0
	}
	s := v.(*DCGMFieldPolicyV3Notifications)
	if !s.mu.TryLock() {
		s.dropped.Add(1)
		return 0
	}
	defer s.mu.Unlock()
	if s.closed {
		return 0
	}
	if violation == nil {
		s.dropped.Add(1)
		return 0
	}
	copied := *violation
	select {
	case s.queue <- copied:
	default:
		s.dropped.Add(1)
	}
	return 0
}

// Violations returns the channel of copied v3 violation structs. Close closes the
// channel after unregistration succeeds; queued violations remain readable.
func (s *DCGMFieldPolicyV3Notifications) Violations() <-chan dcgm.FieldPolicyViolation {
	if s == nil {
		return nil
	}
	return s.queue
}

// Dropped reports callbacks discarded because the queue was full, a concurrent
// callback or Close held the lock, or a nil violation pointer was delivered.
func (s *DCGMFieldPolicyV3Notifications) Dropped() uint64 {
	if s == nil {
		return 0
	}
	return s.dropped.Load()
}

// Close unregisters the (handle, fieldID, policyID) subscription and retires its token.
// A failed unregister preserves ownership and keeps the subscription active for retry.
// Successful unregister does not prove vendor worker quiescence; late callbacks for
// retired tokens are ignored by the permanent trampoline.
func (s *DCGMFieldPolicyV3Notifications) Close() error {
	if s == nil || s.queue == nil || s.unregister == nil {
		return fmt.Errorf("dcgm field policy v3: invalid notifications subscription")
	}
	s.closeMu.Lock()
	defer s.closeMu.Unlock()
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		return nil
	}
	if err := s.unregister(s.handle, s.fieldID, s.policyID); err != nil {
		return fmt.Errorf("dcgm field policy v3: unregister: %w", err)
	}
	s.retire()
	return nil
}

func (s *DCGMFieldPolicyV3Notifications) retire() {
	fieldPolicyV3Callbacks.Lock()
	delete(fieldPolicyV3Callbacks.active, s.token)
	fieldPolicyV3CallbackLookup.Delete(s.token)
	fieldPolicyV3Callbacks.Unlock()
	s.mu.Lock()
	if !s.closed {
		s.closed = true
		close(s.queue)
	}
	s.mu.Unlock()
}

// DCGMFieldPolicyV3Config identifies one read-only v3 field-policy callback source.
// DeviceIDs must be a caller-verified inventory of physical GPU entity IDs (1..64 GPUs).
// Optional non-zero FieldID and PolicyID pin the expected registration key; callbacks
// outside that key or with malformed strings/enums are marked LayoutKnown=false.
type DCGMFieldPolicyV3Config struct {
	SourceID        string
	ProducerVersion string
	NativeOrigin    string
	ClockDomain     string
	ClockEpoch      string
	FieldID         uint16
	PolicyID        uint64
	DeviceIDs       map[uint32]string
}

// DCGMFieldPolicyV3Event preserves a v3 field-policy violation payload with raw
// 64-bit IEEE-754 bit patterns and provenance. ReceivedNanos is consumer arrival
// time, never DCGM or sensor acquisition time. ViolationTimestamp is the raw
// callback field without independent sensor-clock verification. LossUnknown is
// always true because zero helper drops do not prove no upstream DCGM loss.
type DCGMFieldPolicyV3Event struct {
	Schema             string                `json:"schema"`
	SourceID           string                `json:"source_id"`
	ProducerVersion    string                `json:"producer_version"`
	NativeOrigin       string                `json:"native_origin"`
	ClockDomain        string                `json:"clock_domain"`
	ClockEpoch         string                `json:"clock_epoch"`
	ReceivedNanos      int64                 `json:"received_nanos"`
	DeviceUUID         string                `json:"device_uuid,omitempty"`
	PolicyID           uint64                `json:"policy_id"`
	PolicyName         string                `json:"policy_name,omitempty"`
	FieldID            uint16                `json:"field_id"`
	FieldName          string                `json:"field_name,omitempty"`
	EntityGroup        dcgm.FieldEntityGroup `json:"entity_group"`
	EntityID           uint32                `json:"entity_id"`
	CurrentValueBits   uint64                `json:"current_value_bits"`
	ThresholdValueBits uint64                `json:"threshold_value_bits"`
	PolicyOperator     dcgm.PolicyOperator   `json:"policy_operator"`
	ChannelMask        uint32                `json:"channel_mask"`
	ViolationTimestamp int64                 `json:"violation_timestamp"`
	LayoutKnown        bool                  `json:"layout_known"`
	DeviceKnown        bool                  `json:"device_known"`
	Dropped            uint64                `json:"dropped"`
	LossUnknown        bool                  `json:"loss_unknown"`
}

// CurrentValueFloat64 returns the IEEE-754 float64 value of CurrentValueBits.
func (e DCGMFieldPolicyV3Event) CurrentValueFloat64() float64 {
	return math.Float64frombits(e.CurrentValueBits)
}

// ThresholdValueFloat64 returns the IEEE-754 float64 value of ThresholdValueBits.
func (e DCGMFieldPolicyV3Event) ThresholdValueFloat64() float64 {
	return math.Float64frombits(e.ThresholdValueBits)
}

// DCGMFieldPolicyV3Source owns a DCGMFieldPolicyV3Reader. The zero value is not usable.
// Close failure preserves the live source for an explicit caller retry.
type DCGMFieldPolicyV3Source struct {
	mu     sync.Mutex
	reader DCGMFieldPolicyV3Reader
	clock  Clock
	config DCGMFieldPolicyV3Config
	closed bool
}

// NewDCGMFieldPolicyV3Source takes ownership of a v3 reader after validating
// clock provenance and device inventory.
func NewDCGMFieldPolicyV3Source(reader DCGMFieldPolicyV3Reader, clock Clock, c DCGMFieldPolicyV3Config) (*DCGMFieldPolicyV3Source, error) {
	if dcgmNil(reader) || dcgmNil(clock) || reader.Violations() == nil || len(c.DeviceIDs) == 0 || len(c.DeviceIDs) > 64 {
		return nil, fmt.Errorf("dcgm field policy v3 source: invalid reader or inventory")
	}
	meta, ok := clock.(ClockMetadata)
	if !ok || meta.MonotonicDomain() != c.ClockDomain || meta.MonotonicEpochID() != c.ClockEpoch {
		return nil, fmt.Errorf("dcgm field policy v3 source: clock provenance mismatch")
	}
	for _, s := range []string{c.SourceID, c.ProducerVersion, c.NativeOrigin, c.ClockDomain, c.ClockEpoch} {
		if s == "" || len(s) > 256 || !utf8.ValidString(s) {
			return nil, fmt.Errorf("dcgm field policy v3 source: invalid identity")
		}
	}
	ids := make(map[uint32]string, len(c.DeviceIDs))
	seen := make(map[string]bool, len(c.DeviceIDs))
	for id, uuid := range c.DeviceIDs {
		if uuid == "" || len(uuid) > 256 || !utf8.ValidString(uuid) || seen[uuid] {
			return nil, fmt.Errorf("dcgm field policy v3 source: invalid device identity")
		}
		ids[id] = uuid
		seen[uuid] = true
	}
	c.DeviceIDs = ids
	return &DCGMFieldPolicyV3Source{reader: reader, clock: clock, config: c}, nil
}

func extractNullTerminatedUTF8(buf [256]byte) (string, bool) {
	idx := bytes.IndexByte(buf[:], 0)
	if idx < 0 {
		return "", false
	}
	s := string(buf[:idx])
	if !utf8.ValidString(s) {
		return "", false
	}
	return s, true
}

func validFieldPolicyV3Operator(op dcgm.PolicyOperator) bool {
	return op >= dcgm.PolicyOpGt && op < dcgm.PolicyOpCount
}

func validFieldPolicyV3EntityGroup(g dcgm.FieldEntityGroup) bool {
	return g >= dcgm.FeNone && g < dcgm.FeCount
}

// Next waits for one copied v3 violation event or context cancellation.
func (s *DCGMFieldPolicyV3Source) Next(ctx context.Context) (DCGMFieldPolicyV3Event, error) {
	if ctx == nil {
		return DCGMFieldPolicyV3Event{}, fmt.Errorf("dcgm field policy v3 source: nil context")
	}
	if s == nil {
		return DCGMFieldPolicyV3Event{}, ErrDCGMFieldPolicyV3SourceClosed
	}
	s.mu.Lock()
	reader, clock, c := s.reader, s.clock, s.config
	s.mu.Unlock()
	if reader == nil {
		return DCGMFieldPolicyV3Event{}, ErrDCGMFieldPolicyV3SourceClosed
	}
	select {
	case <-ctx.Done():
		return DCGMFieldPolicyV3Event{}, ctx.Err()
	case v, ok := <-reader.Violations():
		if !ok {
			return DCGMFieldPolicyV3Event{}, io.EOF
		}
		meta := clock.(ClockMetadata)
		if meta.MonotonicDomain() != c.ClockDomain || meta.MonotonicEpochID() != c.ClockEpoch {
			return DCGMFieldPolicyV3Event{}, fmt.Errorf("dcgm field policy v3 source: clock provenance changed")
		}
		now := clock.NowMonotonic()
		if meta.MonotonicDomain() != c.ClockDomain || meta.MonotonicEpochID() != c.ClockEpoch {
			return DCGMFieldPolicyV3Event{}, fmt.Errorf("dcgm field policy v3 source: clock provenance changed during capture")
		}
		if now < 0 {
			return DCGMFieldPolicyV3Event{}, fmt.Errorf("dcgm field policy v3 source: negative arrival time")
		}

		policyName, policyNameOK := extractNullTerminatedUTF8(v.PolicyName)
		fieldName, fieldNameOK := extractNullTerminatedUTF8(v.FieldName)
		layoutKnown := v.PolicyId != 0 &&
			v.FieldId != 0 &&
			policyNameOK &&
			fieldNameOK &&
			validFieldPolicyV3Operator(v.PolicyOperator) &&
			validFieldPolicyV3EntityGroup(v.Entity.EntityGroupId) &&
			v.Timestamp >= 0 &&
			(c.FieldID == 0 || v.FieldId == c.FieldID) &&
			(c.PolicyID == 0 || v.PolicyId == c.PolicyID)

		var uuid string
		var deviceKnown bool
		if layoutKnown && v.Entity.EntityGroupId == dcgm.FeGpu {
			uuid, deviceKnown = c.DeviceIDs[uint32(v.Entity.EntityId)]
		}
		if !layoutKnown {
			policyName = ""
			fieldName = ""
		}

		return DCGMFieldPolicyV3Event{
			Schema:             DCGMFieldPolicyV3Schema,
			SourceID:           c.SourceID,
			ProducerVersion:    c.ProducerVersion,
			NativeOrigin:       c.NativeOrigin,
			ClockDomain:        c.ClockDomain,
			ClockEpoch:         c.ClockEpoch,
			ReceivedNanos:      now,
			DeviceUUID:         uuid,
			PolicyID:           v.PolicyId,
			PolicyName:         policyName,
			FieldID:            v.FieldId,
			FieldName:          fieldName,
			EntityGroup:        v.Entity.EntityGroupId,
			EntityID:           uint32(v.Entity.EntityId),
			CurrentValueBits:   math.Float64bits(v.CurrentValue),
			ThresholdValueBits: math.Float64bits(v.ThresholdValue),
			PolicyOperator:     v.PolicyOperator,
			ChannelMask:        v.ChannelMask,
			ViolationTimestamp: v.Timestamp,
			LayoutKnown:        layoutKnown,
			DeviceKnown:        deviceKnown,
			Dropped:            reader.Dropped(),
			LossUnknown:        true,
		}, nil
	}
}

// Close unregisters only the owned v3 reader; it never closes the caller's
// connection or shuts down the DCGM engine or library.
func (s *DCGMFieldPolicyV3Source) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if s.reader == nil {
		return ErrDCGMFieldPolicyV3SourceClosed
	}
	if err := s.reader.Close(); err != nil {
		return fmt.Errorf("dcgm field policy v3 source: close: %w", err)
	}
	s.closed = true
	return nil
}

func validateDCGMFieldPolicyV3Event(e DCGMFieldPolicyV3Event) error {
	if e.Schema != DCGMFieldPolicyV3Schema || e.ReceivedNanos < 0 || !e.LossUnknown || e.DeviceKnown != (e.DeviceUUID != "") || (!e.LayoutKnown && e.DeviceKnown) {
		return fmt.Errorf("dcgm field policy v3 event: invalid metadata")
	}
	if e.DeviceKnown && e.EntityGroup != dcgm.FeGpu {
		return fmt.Errorf("dcgm field policy v3 event: non-gpu entity cannot claim device uuid")
	}
	for _, s := range []string{e.SourceID, e.ProducerVersion, e.NativeOrigin, e.ClockDomain, e.ClockEpoch} {
		if s == "" || len(s) > 256 || !utf8.ValidString(s) {
			return fmt.Errorf("dcgm field policy v3 event: invalid identity")
		}
	}
	if len(e.DeviceUUID) > 256 || !utf8.ValidString(e.DeviceUUID) {
		return fmt.Errorf("dcgm field policy v3 event: invalid device")
	}
	if len(e.PolicyName) > 255 || !utf8.ValidString(e.PolicyName) || len(e.FieldName) > 255 || !utf8.ValidString(e.FieldName) {
		return fmt.Errorf("dcgm field policy v3 event: invalid policy or field name")
	}
	if e.LayoutKnown {
		if e.PolicyID == 0 || e.FieldID == 0 || !validFieldPolicyV3Operator(e.PolicyOperator) || !validFieldPolicyV3EntityGroup(e.EntityGroup) || e.ViolationTimestamp < 0 {
			return fmt.Errorf("dcgm field policy v3 event: invalid known-layout fields")
		}
	} else if e.PolicyName != "" || e.FieldName != "" {
		return fmt.Errorf("dcgm field policy v3 event: unknown layout retains unverified names")
	}
	return nil
}

// dcgmFieldPolicyV3MaxBytes includes the newline and bounds escaped JSON, not field bytes.
const dcgmFieldPolicyV3MaxBytes = 16384

// WriteDCGMFieldPolicyV3Event validates and writes one JSON diagnostic line.
// Lines exceeding 16384 encoded bytes are refused before writing.
func WriteDCGMFieldPolicyV3Event(w io.Writer, e DCGMFieldPolicyV3Event) error {
	if dcgmNil(w) {
		return fmt.Errorf("dcgm field policy v3 event: nil writer")
	}
	if err := validateDCGMFieldPolicyV3Event(e); err != nil {
		return err
	}
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if len(b) > dcgmFieldPolicyV3MaxBytes {
		return fmt.Errorf("dcgm field policy v3 event: invalid byte length")
	}
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}

// ReadDCGMFieldPolicyV3Event strictly decodes and validates one JSON diagnostic event.
func ReadDCGMFieldPolicyV3Event(r io.Reader) (DCGMFieldPolicyV3Event, error) {
	if dcgmNil(r) {
		return DCGMFieldPolicyV3Event{}, fmt.Errorf("dcgm field policy v3 event: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, dcgmFieldPolicyV3MaxBytes+1))
	if err != nil {
		return DCGMFieldPolicyV3Event{}, err
	}
	if len(b) == 0 || len(b) > dcgmFieldPolicyV3MaxBytes {
		return DCGMFieldPolicyV3Event{}, fmt.Errorf("dcgm field policy v3 event: invalid byte length")
	}
	var e DCGMFieldPolicyV3Event
	if err := decodeOperationsJSON(b, &e); err != nil {
		return DCGMFieldPolicyV3Event{}, fmt.Errorf("dcgm field policy v3 event: decode: %w", err)
	}
	if err := validateDCGMFieldPolicyV3Event(e); err != nil {
		return DCGMFieldPolicyV3Event{}, err
	}
	return e, nil
}
