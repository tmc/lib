package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/nvml"
)

// NVLinkDevice is the selected read-only NVML surface. nvml.Device implements
// it. The caller owns NVML initialization and the device handle.
type NVLinkDevice interface {
	UUID() (string, error)
	NvLinkState(uint32) (nvml.EnableState, error)
	NvLinkRemotePciInfo(uint32) (nvml.PciInfo, error)
	NvLinkRemoteDeviceType(uint32, *nvml.IntNvLinkDeviceType) error
	NvLinkVersion(uint32) (uint32, error)
}

// NVLinkPeer is a caller-established physical GPU PCI-to-UUID assignment.
// BusID must be the canonical eight-hex-digit domain form. Several assignments
// for one PCI address are permitted, but ambiguous addresses never yield UUIDs.
type NVLinkPeer struct {
	BusID string `json:"bus_id"`
	UUID  string `json:"uuid"`
}

// NVLinkConfig explicitly selects links and supplies source and topology
// provenance. ValidUntilNanos is exclusive; the reader cannot extend validity.
// NVMLVersion must be supplied from the caller's initialized NVML library.
type NVLinkConfig struct {
	NodeID, SourceID, NVMLVersion, TopologyEpoch string
	Links                                        []uint32
	Peers                                        []NVLinkPeer
	ValidFromNanos, ValidUntilNanos              int64
}

// NVLinkQuery identifies availability and the query interval of one native call.
// State is available, unsupported, denied, or failed. It is not link activity.
// NativeCode is a bounded canonical decimal NVML failure result, never error
// text or success code zero. Unsupported (3/13) and denied (4) classifications
// must match; other known or future failure codes have state failed. Device loss
// refuses a trusted snapshot, matching the live reader.
type NVLinkQuery struct {
	State      string `json:"state"`
	NativeCode string `json:"native_code,omitempty"`
	StartNanos int64  `json:"start_nanos"`
	EndNanos   int64  `json:"end_nanos"`
}

// NVLinkObservation retains raw per-link facts. RemoteTypeKnown distinguishes
// a recognized enum from future unknown numeric values. PeerState is mapped,
// unknown, ambiguous, or non_gpu. Version is nominal, never measured traffic.
// PciSubSystemId is deliberately omitted: NVML says it is indeterminate here.
type NVLinkObservation struct {
	Link            uint32      `json:"link"`
	StateQuery      NVLinkQuery `json:"state_query"`
	RawState        uint32      `json:"raw_state"`
	PCIQuery        NVLinkQuery `json:"pci_query"`
	RemotePCI       string      `json:"remote_pci,omitempty"`
	RawRemotePCI    string      `json:"raw_remote_pci,omitempty"`
	TypeQuery       NVLinkQuery `json:"type_query"`
	RawRemoteType   int32       `json:"raw_remote_type"`
	RemoteTypeKnown bool        `json:"remote_type_known"`
	VersionQuery    NVLinkQuery `json:"version_query"`
	LinkVersion     uint32      `json:"link_version"`
	PeerState       string      `json:"peer_state"`
	PeerUUID        string      `json:"peer_uuid,omitempty"`
}

// NVLinkSnapshot is version 1 bounded topology evidence. PeerAssignments are
// caller provenance, not discovery. Raw unmatched links remain in this report;
// a health graph cannot represent unknown endpoints as fabricated GPU UUIDs.
type NVLinkSnapshot struct {
	Version         int                 `json:"version"`
	NodeID          string              `json:"node_id"`
	SourceID        string              `json:"source_id"`
	NVMLVersion     string              `json:"nvml_version"`
	DeviceUUID      string              `json:"device_uuid"`
	TopologyEpoch   string              `json:"topology_epoch"`
	ClockDomain     string              `json:"clock_domain"`
	ClockEpoch      string              `json:"clock_epoch"`
	AtNanos         int64               `json:"at_nanos"`
	ValidFromNanos  int64               `json:"valid_from_nanos"`
	ValidUntilNanos int64               `json:"valid_until_nanos"`
	PeerAssignments []NVLinkPeer        `json:"peer_assignments"`
	Links           []NVLinkObservation `json:"links"`
}

// NVLinkReader reads only explicitly selected links. It allocates no NVML
// resources and needs no Close. Calls cannot be interrupted inside NVML;
// cancellation is checked between calls. It is not safe for concurrent use.
// Its zero value is not usable.
type NVLinkReader struct {
	device NVLinkDevice
	clock  Clock
	config NVLinkConfig
	uuid   string
}

func nvlinkNil(v any) bool {
	if v == nil {
		return true
	}
	r := reflect.ValueOf(v)
	switch r.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan:
		return r.IsNil()
	}
	return false
}
func nvlinkID(s string) bool {
	if s == "" || len(s) > 256 || !utf8.ValidString(s) {
		return false
	}
	for _, r := range s {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			return false
		}
	}
	return true
}
func pciID(s string) bool {
	if len(s) != 16 || s[8] != ':' || s[11] != ':' || s[14] != '.' {
		return false
	}
	for i, b := range []byte(s) {
		if i == 8 || i == 11 || i == 14 {
			continue
		}
		if !(b >= '0' && b <= '9' || b >= 'a' && b <= 'f') {
			return false
		}
	}
	dev, e := strconv.ParseUint(s[12:14], 16, 8)
	return e == nil && dev <= 31 && s[15] <= '7'
}
func peerUUID(pci string, typ int32, typeAvailable bool, peers []NVLinkPeer) (string, string) {
	if typeAvailable && (typ == int32(nvml.NvlinkDeviceTypeSwitch) || typ == int32(nvml.NvlinkDeviceTypeIbmnpu)) {
		return "non_gpu", ""
	}
	if typeAvailable && typ != int32(nvml.NvlinkDeviceTypeGpu) {
		return "unknown", ""
	}
	if !typeAvailable || pci == "" {
		return "unknown", ""
	}
	ids := map[string]bool{}
	for _, p := range peers {
		if p.BusID == pci {
			ids[p.UUID] = true
		}
	}
	if len(ids) > 1 {
		return "ambiguous", ""
	}
	for id := range ids {
		return "mapped", id
	}
	return "unknown", ""
}
func checkPeers(peers []NVLinkPeer) error {
	if len(peers) > 64 {
		return fmt.Errorf("NVLink peer assignments exceed limit")
	}
	seen := map[string]bool{}
	byUUID := map[string]string{}
	for _, p := range peers {
		if !pciID(p.BusID) || !nvlinkID(p.UUID) || strings.HasPrefix(p.UUID, "MIG-") {
			return fmt.Errorf("invalid NVLink peer assignment")
		}
		if previous, ok := byUUID[p.UUID]; ok && previous != p.BusID {
			return fmt.Errorf("NVLink UUID has contradictory PCI assignments")
		}
		byUUID[p.UUID] = p.BusID
		key := p.BusID + "/" + p.UUID
		if seen[key] {
			return fmt.Errorf("duplicate NVLink peer assignment")
		}
		seen[key] = true
	}
	return nil
}

// NewNVLinkReader binds caller-selected link indices to a physical device UUID.
// Known MIG UUIDs are refused: physical PCI identity cannot identify a MIG instance.
// It copies configuration and refuses invalid or duplicate indices before calls.
func NewNVLinkReader(d NVLinkDevice, c Clock, config NVLinkConfig) (*NVLinkReader, error) {
	if nvlinkNil(d) || nvlinkNil(c) || len(config.Links) == 0 || len(config.Links) > 32 || !nvlinkID(config.NodeID) || !nvlinkID(config.SourceID) || !nvlinkID(config.NVMLVersion) || !nvlinkID(config.TopologyEpoch) || config.ValidFromNanos < 0 || config.ValidUntilNanos <= config.ValidFromNanos {
		return nil, fmt.Errorf("invalid NVLink reader configuration")
	}
	if native, ok := d.(nvml.Device); ok && native == 0 {
		return nil, fmt.Errorf("NVLink device handle is zero")
	}
	if e := checkPeers(config.Peers); e != nil {
		return nil, e
	}
	seen := map[uint32]bool{}
	for _, l := range config.Links {
		if l >= 32 || seen[l] {
			return nil, fmt.Errorf("invalid or duplicate NVLink index")
		}
		seen[l] = true
	}
	domain, epoch := clockMetadata(c)
	if !nvlinkID(domain) || !nvlinkID(epoch) {
		return nil, fmt.Errorf("NVLink clock identity is unavailable")
	}
	uuid, e := d.UUID()
	if e != nil {
		return nil, fmt.Errorf("query NVLink device identity: %w", e)
	}
	if !nvlinkID(uuid) || strings.HasPrefix(uuid, "MIG-") {
		return nil, fmt.Errorf("invalid NVLink device UUID")
	}
	config.Links = append([]uint32(nil), config.Links...)
	config.Peers = append([]NVLinkPeer(nil), config.Peers...)
	return &NVLinkReader{d, c, config, uuid}, nil
}

func nvlinkQuery(clock Clock, fn func() error) NVLinkQuery {
	q := NVLinkQuery{State: "available", StartNanos: clock.NowMonotonic()}
	e := fn()
	q.EndNanos = clock.NowMonotonic()
	if e != nil {
		q.State = "failed"
		switch {
		case errors.Is(e, nvml.ErrNotSupported), errors.Is(e, nvml.ErrFunctionNotFound):
			q.State = "unsupported"
		case errors.Is(e, nvml.ErrNoPermission):
			q.State = "denied"
		}
		var code nvml.Result
		if errors.As(e, &code) {
			q.NativeCode = strconv.FormatInt(int64(code), 10)
		}
	}
	return q
}

// Read queries state, remote PCI, device type and nominal version for each
// selected link. Unsupported fields remain unavailable without fabricated zero
// facts. UUID is checked before and after the batch; identity loss returns no
// trusted report. Query intervals are host bounds, not GPU sensor timestamps.
func (r *NVLinkReader) Read(ctx context.Context) (*NVLinkSnapshot, error) {
	if r == nil || nvlinkNil(r.device) || nvlinkNil(ctx) {
		return nil, fmt.Errorf("NVLink reader or context is nil")
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	if now := r.clock.NowMonotonic(); now < r.config.ValidFromNanos || now >= r.config.ValidUntilNanos {
		return nil, fmt.Errorf("NVLink topology validity expired or not started")
	}
	uuid, e := r.device.UUID()
	if e != nil {
		return nil, e
	}
	if uuid != r.uuid {
		return nil, fmt.Errorf("NVLink device identity changed")
	}
	domain, epoch := clockMetadata(r.clock)
	s := &NVLinkSnapshot{Version: 1, NodeID: r.config.NodeID, SourceID: r.config.SourceID, NVMLVersion: r.config.NVMLVersion, DeviceUUID: r.uuid, TopologyEpoch: r.config.TopologyEpoch, ClockDomain: domain, ClockEpoch: epoch, ValidFromNanos: r.config.ValidFromNanos, ValidUntilNanos: r.config.ValidUntilNanos, PeerAssignments: append([]NVLinkPeer(nil), r.config.Peers...)}
	for _, l := range r.config.Links {
		o := NVLinkObservation{Link: l}
		var state nvml.EnableState
		var pci nvml.PciInfo
		var typ nvml.IntNvLinkDeviceType
		var version uint32
		for _, q := range []struct {
			dst *NVLinkQuery
			fn  func() error
		}{{&o.StateQuery, func() error { var e error; state, e = r.device.NvLinkState(l); return e }}, {&o.PCIQuery, func() error { var e error; pci, e = r.device.NvLinkRemotePciInfo(l); return e }}, {&o.TypeQuery, func() error { return r.device.NvLinkRemoteDeviceType(l, &typ) }}, {&o.VersionQuery, func() error { var e error; version, e = r.device.NvLinkVersion(l); return e }}} {
			if e := ctx.Err(); e != nil {
				return nil, e
			}
			*q.dst = nvlinkQuery(r.clock, q.fn)
			if q.dst.NativeCode == strconv.FormatInt(int64(nvml.ErrGpuIsLost), 10) {
				return nil, fmt.Errorf("query NVLink device: %w", nvml.ErrGpuIsLost)
			}
		}
		if o.StateQuery.State == "available" {
			o.RawState = uint32(state)
		}
		if o.PCIQuery.State == "available" {
			n := bytes.IndexByte(pci.BusId[:], 0)
			if n < 0 {
				return nil, fmt.Errorf("unterminated NVLink remote PCI identity")
			}
			o.RawRemotePCI = string(pci.BusId[:n])
			o.RemotePCI = strings.ToLower(o.RawRemotePCI)
			if !pciID(o.RemotePCI) {
				return nil, fmt.Errorf("invalid NVLink remote PCI identity")
			}
			domain, e1 := strconv.ParseUint(o.RemotePCI[:8], 16, 32)
			bus, e2 := strconv.ParseUint(o.RemotePCI[9:11], 16, 32)
			dev, e3 := strconv.ParseUint(o.RemotePCI[12:14], 16, 32)
			if e1 != nil || e2 != nil || e3 != nil || uint32(domain) != pci.Domain || uint32(bus) != pci.Bus || uint32(dev) != pci.Device {
				return nil, fmt.Errorf("NVLink PCI numeric identity contradicts bus ID")
			}
		}
		if o.TypeQuery.State == "available" {
			o.RawRemoteType = int32(typ)
			o.RemoteTypeKnown = typ == nvml.NvlinkDeviceTypeGpu || typ == nvml.NvlinkDeviceTypeIbmnpu || typ == nvml.NvlinkDeviceTypeSwitch || typ == nvml.NvlinkDeviceTypeUnknown
		}
		if o.VersionQuery.State == "available" {
			o.LinkVersion = version
		}
		o.PeerState, o.PeerUUID = peerUUID(o.RemotePCI, o.RawRemoteType, o.TypeQuery.State == "available", s.PeerAssignments)
		s.Links = append(s.Links, o)
	}
	if e = ctx.Err(); e != nil {
		return nil, e
	}
	uuid, e = r.device.UUID()
	if e != nil {
		return nil, e
	}
	if uuid != r.uuid {
		return nil, fmt.Errorf("NVLink device identity changed")
	}
	if newDomain, newEpoch := clockMetadata(r.clock); newDomain != domain || newEpoch != epoch {
		return nil, fmt.Errorf("NVLink clock identity changed")
	}
	s.AtNanos = r.clock.NowMonotonic()
	if e = s.Validate(); e != nil {
		return nil, e
	}
	return s, nil
}

// Validate checks bounded source, peer assignments, query ordering and presence
// flags. An active link with unresolved peer identity never becomes a UUID edge.
func (s NVLinkSnapshot) Validate() error {
	if s.Version != 1 || !nvlinkID(s.NodeID) || !nvlinkID(s.SourceID) || !nvlinkID(s.NVMLVersion) || !nvlinkID(s.DeviceUUID) || strings.HasPrefix(s.DeviceUUID, "MIG-") || !nvlinkID(s.TopologyEpoch) || !nvlinkID(s.ClockDomain) || !nvlinkID(s.ClockEpoch) || s.ValidFromNanos < 0 || s.ValidUntilNanos <= s.ValidFromNanos || s.AtNanos < s.ValidFromNanos || s.AtNanos >= s.ValidUntilNanos || len(s.Links) == 0 || len(s.Links) > 32 {
		return fmt.Errorf("invalid NVLink snapshot identity, bounds or validity")
	}
	if e := checkPeers(s.PeerAssignments); e != nil {
		return e
	}
	seen := map[uint32]bool{}
	last := s.ValidFromNanos
	for _, o := range s.Links {
		if o.Link >= 32 || seen[o.Link] {
			return fmt.Errorf("invalid or duplicate NVLink observation")
		}
		seen[o.Link] = true
		for _, q := range []NVLinkQuery{o.StateQuery, o.PCIQuery, o.TypeQuery, o.VersionQuery} {
			if !healthSourceState(q.State) || q.StartNanos < last || q.EndNanos < q.StartNanos || q.EndNanos > s.AtNanos {
				return fmt.Errorf("invalid NVLink query state or ordering")
			}
			last = q.EndNanos
			if q.NativeCode != "" {
				if q.State == "available" {
					return fmt.Errorf("available NVLink query reports native error")
				}
				code, e := strconv.ParseInt(q.NativeCode, 10, 32)
				if e != nil || q.NativeCode != strconv.FormatInt(code, 10) || code == 0 {
					return fmt.Errorf("invalid NVLink native error code")
				}
				expected := "failed"
				switch nvml.Result(code) {
				case nvml.ErrNotSupported, nvml.ErrFunctionNotFound:
					expected = "unsupported"
				case nvml.ErrNoPermission:
					expected = "denied"
				case nvml.ErrGpuIsLost:
					return fmt.Errorf("NVLink report contains device loss")
				}
				if q.State != expected {
					return fmt.Errorf("NVLink native error contradicts query state")
				}
			}
		}
		if (o.StateQuery.State != "available" && o.RawState != 0) || (o.PCIQuery.State != "available" && (o.RemotePCI != "" || o.RawRemotePCI != "")) || (o.TypeQuery.State != "available" && (o.RawRemoteType != 0 || o.RemoteTypeKnown)) || (o.VersionQuery.State != "available" && o.LinkVersion != 0) {
			return fmt.Errorf("unavailable NVLink field contains value")
		}
		if o.PCIQuery.State == "available" && (!pciID(o.RemotePCI) || strings.ToLower(o.RawRemotePCI) != o.RemotePCI) {
			return fmt.Errorf("invalid NVLink PCI identity")
		}
		known := o.RawRemoteType == int32(nvml.NvlinkDeviceTypeGpu) || o.RawRemoteType == int32(nvml.NvlinkDeviceTypeIbmnpu) || o.RawRemoteType == int32(nvml.NvlinkDeviceTypeSwitch) || o.RawRemoteType == int32(nvml.NvlinkDeviceTypeUnknown)
		if o.TypeQuery.State == "available" && o.RemoteTypeKnown != known {
			return fmt.Errorf("NVLink remote type presence mismatch")
		}
		state, id := peerUUID(o.RemotePCI, o.RawRemoteType, o.TypeQuery.State == "available", s.PeerAssignments)
		if o.PeerState != state || o.PeerUUID != id || id == s.DeviceUUID {
			return fmt.Errorf("NVLink peer assignment contradicts evidence")
		}
	}
	return nil
}

// ReadNVLinkSnapshot reads one strict bounded version 1 report. maxBytes must
// be positive and at most 64 KiB. Unknown fields and duplicate names are refused.
func ReadNVLinkSnapshot(r io.Reader, maxBytes int64) (*NVLinkSnapshot, error) {
	if nvlinkNil(r) || maxBytes <= 0 || maxBytes > 64<<10 {
		return nil, fmt.Errorf("invalid NVLink reader or byte limit")
	}
	b, e := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if e != nil {
		return nil, e
	}
	if int64(len(b)) > maxBytes {
		return nil, fmt.Errorf("NVLink report exceeds byte limit")
	}
	var s NVLinkSnapshot
	if e = decodeOperationsJSON(b, &s); e != nil {
		return nil, e
	}
	if e = s.Validate(); e != nil {
		return nil, e
	}
	return &s, nil
}

// JoinNVLinkHealth adds mapped active GPU edges after explicit node, clock and
// topology joins. Raw unresolved/non-GPU/inactive observations stay in report;
// callers must retain it alongside the health graph. No throughput is derived.
func JoinNVLinkHealth(s HealthSnapshot, report NVLinkSnapshot, l HealthLimits) (*HealthSnapshot, error) {
	if e := s.Validate(l); e != nil {
		return nil, e
	}
	if e := report.Validate(); e != nil {
		return nil, e
	}
	if s.NodeID != report.NodeID || s.TopologyEpoch != report.TopologyEpoch || s.ClockDomain != report.ClockDomain || s.ClockEpoch != report.ClockEpoch || s.AtNanos < report.AtNanos || s.AtNanos >= report.ValidUntilNanos || s.DeviceStates[report.DeviceUUID] != "present" {
		return nil, fmt.Errorf("NVLink health identity, clock or topology mismatch")
	}
	for _, source := range s.Sources {
		if source.ID == report.SourceID {
			return nil, fmt.Errorf("duplicate NVLink health source")
		}
	}
	out := s
	out.Sources = append(append([]HealthSource(nil), s.Sources...), HealthSource{ID: report.SourceID, Kind: "nvml_nvlink", Version: report.NVMLVersion, State: "available", HasLastSuccess: true, LastSuccessNanos: report.AtNanos})
	out.Edges = append([]TopologyEdge(nil), s.Edges...)
	out.Observations = append([]HealthObservation(nil), s.Observations...)
	for i := range out.Observations {
		out.Observations[i].Reading = cloneReadings([]Reading{out.Observations[i].Reading})[0]
	}
	out.DeviceStates = map[string]string{}
	for id, state := range s.DeviceStates {
		out.DeviceStates[id] = state
	}
	for _, o := range report.Links {
		if o.PeerState != "mapped" || o.StateQuery.State != "available" || o.RawState != uint32(nvml.FeatureEnabled) {
			continue
		}
		if s.DeviceStates[o.PeerUUID] != "present" {
			return nil, fmt.Errorf("NVLink mapped peer is absent from health device scope")
		}
		capability := "nominal_version_unknown"
		if o.VersionQuery.State == "available" {
			capability = "nominal_version=" + strconv.FormatUint(uint64(o.LinkVersion), 10)
		}
		out.Edges = append(out.Edges, TopologyEdge{SourceID: report.SourceID, From: report.DeviceUUID, To: o.PeerUUID, Kind: "nvlink_remote_pci", Scope: "link/" + strconv.FormatUint(uint64(o.Link), 10) + "/" + o.RemotePCI, State: "available", Capability: capability, ValidFromNanos: report.ValidFromNanos, ValidUntilNanos: report.ValidUntilNanos})
	}
	if e := out.Validate(l); e != nil {
		return nil, e
	}
	return &out, nil
}
