package nvtelemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/dcgm"
)

// ErrDCGMPolicySourceClosed reports an uninitialized source.
var ErrDCGMPolicySourceClosed = errors.New("dcgm policy source: closed or uninitialized")

// DCGMNotificationReader supplies copied native responses. Implementations must
// keep their connection and group live until Close succeeds.
type DCGMNotificationReader interface {
	Notifications() <-chan dcgm.PolicyCallbackResponse
	Dropped() uint64
	Close() error
}

// DCGMPolicyConfig identifies one read-only native callback source. DeviceIDs
// must be a caller-verified inventory of the registered group, at most 64 GPUs.
type DCGMPolicyConfig struct {
	SourceID        string
	ProducerVersion string
	NativeOrigin    string
	ClockDomain     string
	ClockEpoch      string
	DeviceIDs       map[uint32]string
}

// DCGMPolicyEvent preserves the v2 callback payload without interpreting the
// condition-dependent union as a reading or sensor timestamp. ReceivedNanos is
// consumer arrival time, not DCGM acquisition time. Loss is cumulative.
type DCGMPolicyEvent struct {
	Schema          string               `json:"schema"`
	SourceID        string               `json:"source_id"`
	ProducerVersion string               `json:"producer_version"`
	NativeOrigin    string               `json:"native_origin"`
	ClockDomain     string               `json:"clock_domain"`
	ClockEpoch      string               `json:"clock_epoch"`
	ReceivedNanos   int64                `json:"received_nanos"`
	DeviceUUID      string               `json:"device_uuid,omitempty"`
	Version         uint32               `json:"version"`
	Condition       dcgm.PolicyCondition `json:"condition"`
	RawPayload      [2]uint64            `json:"raw_payload"`
	GPUIndex        uint32               `json:"gpu_index"`
	LayoutKnown     bool                 `json:"layout_known"`
	DeviceKnown     bool                 `json:"device_known"`
	Dropped         uint64               `json:"dropped"`
	LossUnknown     bool                 `json:"loss_unknown"`
}

// DCGMPolicySource owns a notification reader. The zero value is not usable.
// Close failure preserves the live source for an explicit caller retry.
type DCGMPolicySource struct {
	mu     sync.Mutex
	reader DCGMNotificationReader
	clock  Clock
	config DCGMPolicyConfig
	closed bool
}

// NewDCGMPolicySource takes ownership after validation, without registering,
// setting or triggering native policy itself. Unknown layouts/devices remain
// raw diagnostic events, never admitted numeric or health measurements.
func NewDCGMPolicySource(reader DCGMNotificationReader, clock Clock, c DCGMPolicyConfig) (*DCGMPolicySource, error) {
	if dcgmNil(reader) || dcgmNil(clock) || reader.Notifications() == nil || len(c.DeviceIDs) == 0 || len(c.DeviceIDs) > 64 {
		return nil, fmt.Errorf("dcgm policy source: invalid reader or inventory")
	}
	meta, ok := clock.(ClockMetadata)
	if !ok || meta.MonotonicDomain() != c.ClockDomain || meta.MonotonicEpochID() != c.ClockEpoch {
		return nil, fmt.Errorf("dcgm policy source: clock provenance mismatch")
	}
	for _, s := range []string{c.SourceID, c.ProducerVersion, c.NativeOrigin, c.ClockDomain, c.ClockEpoch} {
		if s == "" || len(s) > 256 || !utf8.ValidString(s) {
			return nil, fmt.Errorf("dcgm policy source: invalid identity")
		}
	}
	ids := make(map[uint32]string, len(c.DeviceIDs))
	seen := make(map[string]bool, len(c.DeviceIDs))
	for id, uuid := range c.DeviceIDs {
		if uuid == "" || len(uuid) > 256 || !utf8.ValidString(uuid) || seen[uuid] {
			return nil, fmt.Errorf("dcgm policy source: invalid device identity")
		}
		ids[id] = uuid
		seen[uuid] = true
	}
	c.DeviceIDs = ids
	return &DCGMPolicySource{reader: reader, clock: clock, config: c}, nil
}

// Next waits for one copied event or cancellation. Successful Close ends the
// reader after buffered callbacks drain. Concurrent Next calls divide events;
// they do not each receive a broadcast.
func (s *DCGMPolicySource) Next(ctx context.Context) (DCGMPolicyEvent, error) {
	if ctx == nil {
		return DCGMPolicyEvent{}, fmt.Errorf("dcgm policy source: nil context")
	}
	if s == nil {
		return DCGMPolicyEvent{}, ErrDCGMPolicySourceClosed
	}
	s.mu.Lock()
	reader, clock, c := s.reader, s.clock, s.config
	s.mu.Unlock()
	if reader == nil {
		return DCGMPolicyEvent{}, ErrDCGMPolicySourceClosed
	}
	select {
	case <-ctx.Done():
		return DCGMPolicyEvent{}, ctx.Err()
	case r, ok := <-reader.Notifications():
		if !ok {
			return DCGMPolicyEvent{}, io.EOF
		}
		meta := clock.(ClockMetadata)
		if meta.MonotonicDomain() != c.ClockDomain || meta.MonotonicEpochID() != c.ClockEpoch {
			return DCGMPolicyEvent{}, fmt.Errorf("dcgm policy source: clock provenance changed")
		}
		now := clock.NowMonotonic()
		if meta.MonotonicDomain() != c.ClockDomain || meta.MonotonicEpochID() != c.ClockEpoch {
			return DCGMPolicyEvent{}, fmt.Errorf("dcgm policy source: clock provenance changed during capture")
		}
		if now < 0 {
			return DCGMPolicyEvent{}, fmt.Errorf("dcgm policy source: negative arrival time")
		}
		uuid, known := c.DeviceIDs[r.GpuId]
		if r.Version != 0x02000020 {
			uuid, known = "", false
		}
		return DCGMPolicyEvent{Schema: "nvtelemetry.dcgm-policy.v1", SourceID: c.SourceID, ProducerVersion: c.ProducerVersion, NativeOrigin: c.NativeOrigin, ClockDomain: c.ClockDomain, ClockEpoch: c.ClockEpoch, ReceivedNanos: now, DeviceUUID: uuid, Version: r.Version, Condition: r.Condition, RawPayload: r.Val, GPUIndex: r.GpuId, LayoutKnown: r.Version == 0x02000020, DeviceKnown: known, Dropped: reader.Dropped(), LossUnknown: true}, nil
	}
}

// Close unregisters only the owned source; it never closes its connection,
// destroys its group or shuts down the DCGM engine or library.
func (s *DCGMPolicySource) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if s.reader == nil {
		return ErrDCGMPolicySourceClosed
	}
	if err := s.reader.Close(); err != nil {
		return fmt.Errorf("dcgm policy source: close: %w", err)
	}
	s.closed = true
	return nil
}

// WriteDCGMPolicyEvent writes one bounded diagnostic JSON line. It does not
// translate opaque union payloads into normalized health, logs or metrics.
func WriteDCGMPolicyEvent(w io.Writer, e DCGMPolicyEvent) error {
	if dcgmNil(w) {
		return fmt.Errorf("dcgm policy event: nil writer")
	}
	if e.Schema != "nvtelemetry.dcgm-policy.v1" || e.ReceivedNanos < 0 || !e.LossUnknown || e.LayoutKnown != (e.Version == 0x02000020) || e.DeviceKnown != (e.DeviceUUID != "") || (!e.LayoutKnown && e.DeviceKnown) {
		return fmt.Errorf("dcgm policy event: invalid metadata")
	}
	for _, s := range []string{e.SourceID, e.ProducerVersion, e.NativeOrigin, e.ClockDomain, e.ClockEpoch} {
		if s == "" || len(s) > 256 || !utf8.ValidString(s) {
			return fmt.Errorf("dcgm policy event: invalid identity")
		}
	}
	if len(e.DeviceUUID) > 256 || !utf8.ValidString(e.DeviceUUID) {
		return fmt.Errorf("dcgm policy event: invalid device")
	}
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}
