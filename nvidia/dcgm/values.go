package dcgm

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"time"
	"unsafe"
)

// This file is hand-written. It adds the two things the generated wrappers
// cannot: buffers sized from the header's own bounds, and readings that keep
// the per-field status and timestamp attached to the value.
//
// The generated wrappers take the pointers the C API declares. That is right
// for a binding -- the generator has no way to know how many elements DCGM
// will write through a given pointer -- but it makes every call site
// responsible for a bound it has to look up in the header. The wrappers here
// allocate from the generated limits instead, so the bound and the buffer
// cannot disagree.

// A FieldValue is one reading. It is the interface the two generated value
// structures have in common, so a caller can handle
// [Handle.LatestValues] and [Handle.EntityLatestValues] results the same way.
type FieldValue interface {
	// ID is the DCGM field this reading is for.
	ID() FieldID
	// Type is the DCGM_FT_* discriminator selecting the payload arm.
	Type() uint16
	// Err reports the per-field status. It is nil when DCGM returned the
	// field successfully, which does not mean the field carries data: check
	// Blank as well.
	Err() error
	// Time is when the sample was taken, or the zero Time when DCGM
	// reported no timestamp.
	Time() time.Time
	// Blank reports that the field carries no data even though Err is nil.
	Blank() bool
}

var (
	_ FieldValue = (*FieldValueV1)(nil)
	_ FieldValue = (*FieldValueV2)(nil)
)

// ID is the DCGM field this reading is for.
func (v *FieldValueV1) ID() FieldID { return FieldID(v.FieldId) }

// Type is the DCGM_FT_* discriminator selecting which payload accessor is
// meaningful.
func (v *FieldValueV1) Type() uint16 { return v.FieldType }

// Err reports the per-field status DCGM returned for this reading.
//
// It is separate from the error the call itself returned: a call can succeed
// and still report a per-field failure here, and a field whose Err is nil can
// still be [FieldValueV1.Blank].
func (v *FieldValueV1) Err() error { return toError(Result(v.Status)) }

// Time is when the sample was taken. DCGM reports microseconds since the Unix
// epoch, and reports zero for a field it never sampled; Time returns the zero
// [time.Time] for that so a caller cannot mistake it for 1970.
func (v *FieldValueV1) Time() time.Time { return fieldTime(v.Ts) }

// Blank reports that DCGM returned this field successfully but with no data.
//
// This is the case a caller most often gets wrong. DCGM does not report "the
// GPU does not support this field" through the status; it returns success and
// writes a sentinel into the value. Reading the value without this check turns
// "no data" into an enormous reading.
func (v *FieldValueV1) Blank() bool { return blank(v.FieldType, v.payload()) }

// Int64 returns the reading of an integer or timestamp field.
//
// ok is false when the field is not integer-typed or carries no data, so a
// blank cannot be mistaken for a value.
func (v *FieldValueV1) Int64() (n int64, ok bool) { return fieldInt64(v.FieldType, v.payload()) }

// Float64 returns the reading of a double-typed field. ok is false when the
// field is not double-typed or carries no data.
func (v *FieldValueV1) Float64() (f float64, ok bool) { return fieldFloat64(v.FieldType, v.payload()) }

// Text returns the reading of a string-typed field. ok is false when the field
// is not string-typed or carries no data.
func (v *FieldValueV1) Text() (s string, ok bool) { return fieldText(v.FieldType, v.payload()) }

// Blob returns a copy of the raw payload bytes of a binary-typed field. It
// returns nil for any other field type.
//
// The copy is deliberate: the payload lives in an array inside v, and handing
// out a slice of it would alias a structure the caller is likely to reuse for
// the next sample.
func (v *FieldValueV1) Blob() []byte { return fieldBlob(v.FieldType, v.payload()) }

// payload views the value union as bytes.
func (v *FieldValueV1) payload() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&v.Value[0])), MaxBlobLength)
}

// ID is the DCGM field this reading is for.
func (v *FieldValueV2) ID() FieldID { return FieldID(v.FieldId) }

// Type is the DCGM_FT_* discriminator selecting which payload accessor is
// meaningful.
func (v *FieldValueV2) Type() uint16 { return v.FieldType }

// Err reports the per-field status DCGM returned for this reading. See
// [FieldValueV1.Err].
func (v *FieldValueV2) Err() error { return toError(Result(v.Status)) }

// Time is when the sample was taken, or the zero [time.Time] when DCGM
// reported no timestamp.
func (v *FieldValueV2) Time() time.Time { return fieldTime(v.Ts) }

// Blank reports that DCGM returned this field successfully but with no data.
// See [FieldValueV1.Blank].
func (v *FieldValueV2) Blank() bool { return blank(v.FieldType, v.payload()) }

// Entity is the entity this reading belongs to.
func (v *FieldValueV2) Entity() GroupEntityPair {
	return GroupEntityPair{EntityGroupId: v.EntityGroupId, EntityId: v.EntityId}
}

// Int64 returns the reading of an integer or timestamp field. ok is false when
// the field is not integer-typed or carries no data.
func (v *FieldValueV2) Int64() (n int64, ok bool) { return fieldInt64(v.FieldType, v.payload()) }

// Float64 returns the reading of a double-typed field. ok is false when the
// field is not double-typed or carries no data.
func (v *FieldValueV2) Float64() (f float64, ok bool) { return fieldFloat64(v.FieldType, v.payload()) }

// Text returns the reading of a string-typed field. ok is false when the field
// is not string-typed or carries no data.
func (v *FieldValueV2) Text() (s string, ok bool) { return fieldText(v.FieldType, v.payload()) }

// Blob returns a copy of the raw payload bytes of a binary-typed field, or nil
// for any other field type.
func (v *FieldValueV2) Blob() []byte { return fieldBlob(v.FieldType, v.payload()) }

// payload views the value union as bytes.
func (v *FieldValueV2) payload() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&v.Value[0])), MaxBlobLength)
}

// fieldTime converts DCGM's microseconds-since-epoch to a time.Time, mapping
// the "never sampled" zero to the zero Time.
func fieldTime(usec int64) time.Time {
	if usec == 0 {
		return time.Time{}
	}
	return time.UnixMicro(usec)
}

// payloadInt64 reads the i64 arm of the value union.
func payloadInt64(p []byte) int64 {
	return int64(*(*uint64)(unsafe.Pointer(&p[0])))
}

// payloadFloat64 reads the dbl arm of the value union.
func payloadFloat64(p []byte) float64 {
	return math.Float64frombits(*(*uint64)(unsafe.Pointer(&p[0])))
}

// payloadText reads the str arm of the value union, which is NUL-terminated
// within DCGM_MAX_STR_LENGTH bytes.
func payloadText(p []byte) string {
	s := p[:MaxStrLength]
	if i := bytes.IndexByte(s, 0); i >= 0 {
		s = s[:i]
	}
	return string(s)
}

// blank applies DCGM's own blank test for the field's type.
//
// The test is an inequality rather than an equality because the header encodes
// the specific reason -- not found, not supported, not permissioned -- as the
// blank value plus a small offset, and a caller that only compares against the
// blank value itself misses all three.
func blank(fieldType uint16, p []byte) bool {
	switch fieldType {
	case FieldTypeInt64, FieldTypeTimestamp:
		return payloadInt64(p) >= Int64Blank
	case FieldTypeDouble:
		return payloadFloat64(p) >= Fp64Blank
	case FieldTypeString:
		// The header's test accepts any value bracketed by "<<<" and
		// ">>>", not only the exact StrBlank spelling.
		s := payloadText(p)
		return strings.HasPrefix(s, "<<<") && strings.Contains(s, ">>>")
	case FieldTypeBinary:
		// A binary field has no sentinel; an empty payload is the only
		// thing that can be said about it, and DCGM does not promise
		// even that. Report not-blank and let the caller decide.
		return false
	}
	return false
}

func fieldInt64(fieldType uint16, p []byte) (int64, bool) {
	switch fieldType {
	case FieldTypeInt64, FieldTypeTimestamp:
	default:
		return 0, false
	}
	n := payloadInt64(p)
	if n >= Int64Blank {
		return 0, false
	}
	return n, true
}

func fieldFloat64(fieldType uint16, p []byte) (float64, bool) {
	if fieldType != FieldTypeDouble {
		return 0, false
	}
	f := payloadFloat64(p)
	if f >= Fp64Blank {
		return 0, false
	}
	return f, true
}

func fieldText(fieldType uint16, p []byte) (string, bool) {
	if fieldType != FieldTypeString {
		return "", false
	}
	s := payloadText(p)
	if strings.HasPrefix(s, "<<<") && strings.Contains(s, ">>>") {
		return "", false
	}
	return s, true
}

func fieldBlob(fieldType uint16, p []byte) []byte {
	if fieldType != FieldTypeBinary {
		return nil
	}
	out := make([]byte, MaxBlobLength)
	copy(out, p)
	return out
}

// fieldIDsToUint16 copies field ids into the plain uint16 slice the generated
// entry points take. FieldID has uint16 as its underlying type, but []FieldID
// is not []uint16, and reinterpreting one as the other would trade a copy of
// at most DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP entries for an unsafe assumption
// about slice representation.
func fieldIDsToUint16(fields []FieldID) []uint16 {
	out := make([]uint16, len(fields))
	for i, f := range fields {
		out[i] = uint16(f)
	}
	return out
}

// ErrTooManyFields reports a field list longer than one field group can hold.
var ErrTooManyFields = fmt.Errorf("dcgm: more than %d fields, which is DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP", MaxFieldIDsPerFieldGroup)

// Devices returns the GPU ids the host engine knows about, whether or not DCGM
// supports them.
//
// The buffer is sized from [MaxNumDevices], the bound DCGM writes up to. The
// returned slice is freshly allocated and belongs to the caller.
func (h Handle) Devices() ([]uint32, error) {
	var buf [MaxNumDevices]uint32
	n, err := GetAllDevices(h, &buf[0])
	if err != nil {
		return nil, err
	}
	if n < 0 || int(n) > len(buf) {
		return nil, fmt.Errorf("dcgm: GetAllDevices reported %d devices, outside 0..%d", n, len(buf))
	}
	out := make([]uint32, n)
	copy(out, buf[:n])
	return out, nil
}

// SupportedDevices returns the GPU ids DCGM supports on this host. It is a
// subset of [Handle.Devices]; a GPU DCGM does not support has no field data.
func (h Handle) SupportedDevices() ([]uint32, error) {
	var buf [MaxNumDevices]uint32
	n, err := GetAllSupportedDevices(h, &buf[0])
	if err != nil {
		return nil, err
	}
	if n < 0 || int(n) > len(buf) {
		return nil, fmt.Errorf("dcgm: GetAllSupportedDevices reported %d devices, outside 0..%d", n, len(buf))
	}
	out := make([]uint32, n)
	copy(out, buf[:n])
	return out, nil
}

// Groups returns the ids of every entity group on the host engine, including
// groups other processes created. Destroying one this process did not create
// would take it away from its owner; see [Session] for the ownership rule.
func (h Handle) Groups() ([]GpuGrp, error) {
	var buf [MaxNumGroups]GpuGrp
	n, err := GroupGetAllIds(h, &buf[0])
	if err != nil {
		return nil, err
	}
	if int(n) > len(buf) {
		return nil, fmt.Errorf("dcgm: GroupGetAllIds reported %d groups, more than the %d DCGM can hold", n, len(buf))
	}
	out := make([]GpuGrp, n)
	copy(out, buf[:n])
	return out, nil
}

// GroupEntities returns the entities in a group.
//
// It stamps the version field [GroupInfo] requires before the call. A caller
// that uses the generated [Handle.GroupGetInfo] directly must stamp it: DCGM
// rejects an unstamped structure with [ErrVerMismatch], which reads at run
// time like a broken install rather than a missing assignment.
func (h Handle) GroupEntities(groupID GpuGrp) ([]GroupEntityPair, error) {
	info := GroupInfo{Version: GroupInfoVersion3}
	if err := GroupGetInfo(h, groupID, &info); err != nil {
		return nil, err
	}
	if int(info.Count) > len(info.EntityList) {
		return nil, fmt.Errorf("dcgm: group reports %d entities, more than the %d it can hold", info.Count, len(info.EntityList))
	}
	out := make([]GroupEntityPair, info.Count)
	copy(out, info.EntityList[:info.Count])
	return out, nil
}

// LatestValues reads the most recent sample of each field for one GPU.
//
// The fields must already be watched; see [Session.Watch]. A field that is
// watched but has not been sampled comes back with a nil Err and Blank true,
// which is why the result keeps [FieldValue] rather than bare numbers.
//
// fields is read during the call and not retained. The returned slice is
// freshly allocated and belongs to the caller.
func (h Handle) LatestValues(gpuID uint32, fields []FieldID) ([]FieldValueV1, error) {
	if len(fields) == 0 {
		return nil, errors.New("dcgm: LatestValues: no fields requested")
	}
	if len(fields) > MaxFieldIDsPerFieldGroup {
		return nil, ErrTooManyFields
	}
	values := make([]FieldValueV1, len(fields))
	for i := range values {
		values[i].Version = FieldValueVersion1
	}
	ids := fieldIDsToUint16(fields)
	err := GetLatestValuesForFields(h, int32(gpuID), &ids[0], uint32(len(ids)), &values[0])
	runtime.KeepAlive(ids)
	if err != nil {
		return nil, err
	}
	return values, nil
}

// EntityLatestValues reads the most recent sample of each field for each of
// several entities, which may be GPUs, GPU instances, compute instances,
// NvSwitches or links.
//
// The result has one reading per entity-field pair, in entity-major order, and
// each keeps its own status and timestamp. entities and fields are read during
// the call and not retained.
func (h Handle) EntityLatestValues(entities []GroupEntityPair, fields []FieldID, flags uint32) ([]FieldValueV2, error) {
	if len(entities) == 0 {
		return nil, errors.New("dcgm: EntityLatestValues: no entities requested")
	}
	if len(fields) == 0 {
		return nil, errors.New("dcgm: EntityLatestValues: no fields requested")
	}
	if len(fields) > MaxFieldIDsPerFieldGroup {
		return nil, ErrTooManyFields
	}
	if len(entities) > GroupMaxEntitiesV2 {
		return nil, fmt.Errorf("dcgm: %d entities, more than the %d DCGM_GROUP_MAX_ENTITIES_V2 allows", len(entities), GroupMaxEntitiesV2)
	}
	values := make([]FieldValueV2, len(entities)*len(fields))
	for i := range values {
		values[i].Version = FieldValueVersion2
	}
	ids := fieldIDsToUint16(fields)
	err := EntitiesGetLatestValues(h, &entities[0], uint32(len(entities)), &ids[0], uint32(len(ids)), flags, &values[0])
	runtime.KeepAlive(entities)
	runtime.KeepAlive(ids)
	if err != nil {
		return nil, err
	}
	return values, nil
}
