package nvtelemetrybridge

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

var magic = []byte("NVMLBPF1")
var scopes = []string{"", nvtelemetry.MIGScopeUnknown, nvtelemetry.MIGScopeDisabled, nvtelemetry.MIGScopePhysicalParent, nvtelemetry.MIGScopeMIGInstance, nvtelemetry.MIGScopeUnsupported, nvtelemetry.MIGScopePermissionDenied}
var evidence = []string{"", nvtelemetry.EvidenceMeasured, nvtelemetry.EvidenceDerived, nvtelemetry.EvidenceUnavailable}

// Encode returns the fixed little-endian layout. Unused slots and reserved
// bytes are zero. IDs, metric names, units and domains are NUL-terminated UTF-8.
func Encode(s Snapshot) ([]byte, error) {
	if s.Generation == 0 || s.Owner == ([16]byte{}) || s.PublishedUnix <= 0 || s.EpochID == "" || s.ClockDomain == "" || s.SchemaVersion < 1 || s.SchemaVersion > nvtelemetry.SchemaVersion || len(s.Devices) == 0 || len(s.Devices) > MaxDevices || len(s.Sample.Readings) > MaxReadings || s.Sample.QueryEndNanos < s.Sample.QueryStartNanos || s.Sample.ClockDiscontinuity {
		return nil, ErrSnapshot
	}
	data := make([]byte, ValueSize)
	copy(data, magic)
	binary.LittleEndian.PutUint32(data[8:], Version)
	binary.LittleEndian.PutUint32(data[12:], uint32(len(s.Sample.Readings)))
	binary.LittleEndian.PutUint64(data[16:], s.Generation)
	put64(data[24:], s.PublishedUnix)
	put64(data[32:], s.Sample.QueryStartNanos)
	put64(data[40:], s.Sample.QueryEndNanos)
	copy(data[48:64], s.Owner[:])
	if err := putString(data[64:128], s.EpochID); err != nil {
		return nil, err
	}
	binary.LittleEndian.PutUint32(data[192:], uint32(len(s.Devices)))
	binary.LittleEndian.PutUint32(data[196:], uint32(s.SchemaVersion))
	binary.LittleEndian.PutUint64(data[200:], s.Sample.Seq)
	if err := putString(data[128:192], s.ClockDomain); err != nil {
		return nil, err
	}
	deviceIndex := make(map[string]uint32)
	for i, d := range s.Devices {
		if d.UUID == "" || d.Status == nvtelemetry.DeviceStatusUnidentifiable {
			return nil, ErrSnapshot
		}
		if _, ok := deviceIndex[d.UUID]; ok {
			return nil, ErrSnapshot
		}
		deviceIndex[d.UUID] = uint32(i)
		p := data[HeaderSize+i*DeviceSize : HeaderSize+(i+1)*DeviceSize]
		if err := putString(p[:96], d.UUID); err != nil {
			return nil, err
		}
		if err := putString(p[96:192], d.MIGParentUUID); err != nil {
			return nil, err
		}
		if d.MIGInstanceID < 0 || d.MIGComputeInstanceID < 0 || uint64(d.MIGInstanceID) > math.MaxUint32 || uint64(d.MIGComputeInstanceID) > math.MaxUint32 {
			return nil, ErrSnapshot
		}
		binary.LittleEndian.PutUint32(p[192:], uint32(d.MIGInstanceID))
		binary.LittleEndian.PutUint32(p[196:], uint32(d.MIGComputeInstanceID))
		var flags uint32
		if d.HasMIGInstanceID {
			flags |= 1
		}
		if d.HasMIGComputeInstanceID {
			flags |= 2
		}
		binary.LittleEndian.PutUint32(p[200:], flags)
		scope := -1
		for j, v := range scopes {
			if v == d.MIGScope {
				scope = j
				break
			}
		}
		if scope < 0 {
			return nil, ErrSnapshot
		}
		binary.LittleEndian.PutUint32(p[204:], uint32(scope))
	}
	seen := make(map[string]bool)
	for i, r := range s.Sample.Readings {
		idx, ok := deviceIndex[r.DeviceUUID]
		if !ok || r.Name == "" || r.Unit == "" || math.IsNaN(r.Value) || math.IsInf(r.Value, 0) {
			return nil, ErrSnapshot
		}
		key := r.DeviceUUID + "\x00" + r.Name
		if seen[key] {
			return nil, ErrSnapshot
		}
		seen[key] = true
		if r.HasQueryBounds && (r.QueryEndNanos < r.QueryStartNanos || r.QueryStartNanos < s.Sample.QueryStartNanos || r.QueryEndNanos > s.Sample.QueryEndNanos) {
			return nil, ErrSnapshot
		}
		if r.HasWallClockUncertainty && r.WallClockUncertaintyNanos < 0 {
			return nil, ErrSnapshot
		}
		if r.HasSourceTimestamp && r.SourceTimestampDomain == "" {
			return nil, ErrSnapshot
		}
		var flags uint32
		if r.IsInt {
			flags |= 1
		}
		if r.HasSourceTimestamp {
			flags |= 2
		}
		if r.HasWallClockUncertainty {
			flags |= 4
		}
		if r.HasQueryBounds {
			flags |= 8
		}
		kind := -1
		for j, v := range evidence {
			if j > 0 && r.Evidence == v {
				kind = j
				break
			}
		}
		if kind < 0 || (kind == 3) != (r.UnavailableReason != "") {
			return nil, ErrSnapshot
		}
		flags |= uint32(kind) << 4
		base := HeaderSize + MaxDevices*DeviceSize + i*ReadingSize
		p := data[base : base+ReadingSize]
		binary.LittleEndian.PutUint32(p, idx)
		binary.LittleEndian.PutUint32(p[4:], flags)
		if err := putString(p[8:72], r.Name); err != nil {
			return nil, err
		}
		if err := putString(p[72:96], r.Unit); err != nil {
			return nil, err
		}
		if err := putString(p[96:128], r.UnavailableReason); err != nil {
			return nil, err
		}
		binary.LittleEndian.PutUint64(p[128:], math.Float64bits(r.Value))
		binary.LittleEndian.PutUint64(p[136:], r.IntValue)
		put64(p[144:], r.TimestampMonotonic)
		put64(p[152:], r.TimestampUnix)
		put64(p[160:], r.QueryStartNanos)
		put64(p[168:], r.QueryEndNanos)
		put64(p[176:], r.SourceTimestamp)
		put64(p[184:], r.WallClockUncertaintyNanos)
		if err := putString(p[192:256], r.SourceTimestampDomain); err != nil {
			return nil, err
		}
	}
	return data, nil
}

// Decode rejects incompatible versions, bounds, identity, nonfinite values,
// unknown flag bits, and nonzero reserved or unused bytes.
func Decode(data []byte) (Snapshot, error) {
	var s Snapshot
	if len(data) != ValueSize || !bytes.Equal(data[:8], magic) || binary.LittleEndian.Uint32(data[8:]) != Version {
		return s, ErrLayout
	}
	nr32 := binary.LittleEndian.Uint32(data[12:])
	nd32 := binary.LittleEndian.Uint32(data[192:])
	if nr32 > MaxReadings || nd32 == 0 || nd32 > MaxDevices {
		return s, ErrSnapshot
	}
	nr, nd := int(nr32), int(nd32)
	s.Generation = binary.LittleEndian.Uint64(data[16:])
	s.PublishedUnix = get64(data[24:])
	s.Sample.QueryStartNanos = get64(data[32:])
	s.Sample.QueryEndNanos = get64(data[40:])
	copy(s.Owner[:], data[48:64])
	var err error
	s.EpochID, err = getString(data[64:128])
	if err != nil {
		return s, err
	}
	s.ClockDomain, err = getString(data[128:192])
	if err != nil {
		return s, err
	}
	s.SchemaVersion = int(binary.LittleEndian.Uint32(data[196:]))
	s.Sample.Seq = binary.LittleEndian.Uint64(data[200:])
	for i := 0; i < nd; i++ {
		p := data[HeaderSize+i*DeviceSize : HeaderSize+(i+1)*DeviceSize]
		var d nvtelemetry.DeviceMetadata
		d.UUID, err = getString(p[:96])
		if err != nil {
			return s, err
		}
		d.MIGParentUUID, err = getString(p[96:192])
		if err != nil {
			return s, err
		}
		d.MIGInstanceID = int(binary.LittleEndian.Uint32(p[192:]))
		d.MIGComputeInstanceID = int(binary.LittleEndian.Uint32(p[196:]))
		flags := binary.LittleEndian.Uint32(p[200:])
		if flags&^uint32(3) != 0 {
			return s, ErrSnapshot
		}
		d.HasMIGInstanceID = flags&1 != 0
		d.HasMIGComputeInstanceID = flags&2 != 0
		scope := int(binary.LittleEndian.Uint32(p[204:]))
		if scope >= len(scopes) {
			return s, ErrSnapshot
		}
		d.MIGScope = scopes[scope]
		s.Devices = append(s.Devices, d)
	}
	for i := 0; i < nr; i++ {
		base := HeaderSize + MaxDevices*DeviceSize + i*ReadingSize
		p := data[base : base+ReadingSize]
		idx := int(binary.LittleEndian.Uint32(p))
		if idx >= nd {
			return s, ErrSnapshot
		}
		flags := binary.LittleEndian.Uint32(p[4:])
		kind := int((flags >> 4) & 3)
		if flags&^uint32(63) != 0 || kind == 0 {
			return s, ErrSnapshot
		}
		r := nvtelemetry.Reading{DeviceUUID: s.Devices[idx].UUID, Evidence: evidence[kind], Backend: "nvml_bridge", IsInt: flags&1 != 0, HasSourceTimestamp: flags&2 != 0, HasWallClockUncertainty: flags&4 != 0, HasQueryBounds: flags&8 != 0}
		r.Name, err = getString(p[8:72])
		if err != nil {
			return s, err
		}
		r.Unit, err = getString(p[72:96])
		if err != nil {
			return s, err
		}
		r.UnavailableReason, err = getString(p[96:128])
		if err != nil {
			return s, err
		}
		r.Value = math.Float64frombits(binary.LittleEndian.Uint64(p[128:]))
		r.IntValue = binary.LittleEndian.Uint64(p[136:])
		r.TimestampMonotonic = get64(p[144:])
		r.TimestampUnix = get64(p[152:])
		r.QueryStartNanos = get64(p[160:])
		r.QueryEndNanos = get64(p[168:])
		r.SourceTimestamp = get64(p[176:])
		r.WallClockUncertaintyNanos = get64(p[184:])
		r.SourceTimestampDomain, err = getString(p[192:256])
		if err != nil {
			return s, err
		}
		s.Sample.Readings = append(s.Sample.Readings, r)
	}
	canonical, err := Encode(s)
	if err != nil {
		return s, err
	}
	if !bytes.Equal(canonical, data) {
		return s, fmt.Errorf("%w: noncanonical reserved data", ErrSnapshot)
	}
	return s, nil
}
