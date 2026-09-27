package cudalmvet

import (
	"encoding/binary"
	"math"
)

// FNV-1a 64-bit parameters.
const (
	// FNVOffsetBasis is the 64-bit FNV-1a offset basis (14695981039346656037).
	FNVOffsetBasis uint64 = 14695981039346656037

	// FNVPrime is the 64-bit FNV-1a prime multiplier (1099511628211).
	FNVPrime uint64 = 1099511628211
)

// Canonical quiet-NaN bit representations.
const (
	CanonicalQuietNaN32Bits uint32 = 0x7fc00000
	CanonicalQuietNaN64Bits uint64 = 0x7ff8000000000000
)

// FloatPolicy defines how floating-point representations are treated prior to hashing.
type FloatPolicy int

const (
	// PolicyRawBits preserves exact IEEE-754 bit representations without canonicalization:
	// +0.0 and -0.0 produce distinct hashes, and differing NaN payloads produce distinct hashes.
	PolicyRawBits FloatPolicy = iota

	// PolicyCanonicalizeNaN collapses all NaN bit patterns to a canonical quiet NaN.
	// Signed zeros remain distinct.
	PolicyCanonicalizeNaN

	// PolicyCanonicalizeZero maps -0.0 to +0.0 before hashing.
	// NaN payloads remain distinct.
	PolicyCanonicalizeZero

	// PolicyCanonicalizeAll normalizes both signed zeros and NaNs.
	PolicyCanonicalizeAll
)

// TraceEvent records a single fold operation for diff triage.
type TraceEvent struct {
	Index    int
	NodeName string
	Value    float64
	Bits     uint64
	Hash     uint64
}

// Accumulator folds scalar values and tensor intermediate outputs into a single
// 64-bit FNV-1a scalar.
type Accumulator struct {
	hash      uint64
	count     int
	policy    FloatPolicy
	recording bool
	events    []TraceEvent
}

// AccumulatorOption configures an Accumulator.
type AccumulatorOption func(*Accumulator)

// WithPolicy sets the FloatPolicy on the accumulator.
func WithPolicy(p FloatPolicy) AccumulatorOption {
	return func(a *Accumulator) {
		a.policy = p
	}
}

// WithRecording enables recording each fold event for triage inspection.
func WithRecording(record bool) AccumulatorOption {
	return func(a *Accumulator) {
		a.recording = record
	}
}

// NewAccumulator creates an Accumulator initialized to FNVOffsetBasis.
func NewAccumulator(opts ...AccumulatorOption) *Accumulator {
	a := &Accumulator{
		hash:   FNVOffsetBasis,
		policy: PolicyRawBits,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Reset restores the accumulator to FNVOffsetBasis and clears event history.
func (a *Accumulator) Reset() {
	a.hash = FNVOffsetBasis
	a.count = 0
	if a.recording {
		a.events = a.events[:0]
	}
}

// Hash returns the current 64-bit FNV-1a hash.
func (a *Accumulator) Hash() uint64 {
	return a.hash
}

// Count returns the number of fold events performed.
func (a *Accumulator) Count() int {
	return a.count
}

// Events returns the recorded trace events (empty if recording was disabled).
func (a *Accumulator) Events() []TraceEvent {
	return append([]TraceEvent(nil), a.events...)
}

// FoldByte mixes a single byte into the accumulator.
func (a *Accumulator) FoldByte(b byte) {
	a.hash = (a.hash ^ uint64(b)) * FNVPrime
}

// FoldBytes mixes a slice of bytes into the accumulator.
func (a *Accumulator) FoldBytes(p []byte) {
	for _, b := range p {
		a.hash = (a.hash ^ uint64(b)) * FNVPrime
	}
}

// FoldBool mixes a boolean into the accumulator as 1 or 0 byte.
func (a *Accumulator) FoldBool(v bool) {
	if v {
		a.FoldByte(1)
	} else {
		a.FoldByte(0)
	}
	a.recordEvent("", 0, uint64(a.hash))
}

// FoldUint8 mixes an unsigned 8-bit integer into the accumulator.
func (a *Accumulator) FoldUint8(v uint8) {
	a.FoldByte(v)
	a.recordEvent("", float64(v), uint64(v))
}

// FoldInt8 mixes a signed 8-bit integer into the accumulator.
func (a *Accumulator) FoldInt8(v int8) {
	a.FoldUint8(uint8(v))
}

// FoldUint16 mixes an unsigned 16-bit integer in little-endian order.
func (a *Accumulator) FoldUint16(v uint16) {
	a.FoldByte(byte(v & 0xff))
	a.FoldByte(byte((v >> 8) & 0xff))
	a.recordEvent("", float64(v), uint64(v))
}

// FoldInt16 mixes a signed 16-bit integer in little-endian order.
func (a *Accumulator) FoldInt16(v int16) {
	a.FoldUint16(uint16(v))
}

// FoldUint32 mixes an unsigned 32-bit integer in little-endian order.
func (a *Accumulator) FoldUint32(v uint32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], v)
	a.FoldBytes(buf[:])
	a.recordEvent("", float64(v), uint64(v))
}

// FoldInt32 mixes a signed 32-bit integer in little-endian order.
func (a *Accumulator) FoldInt32(v int32) {
	a.FoldUint32(uint32(v))
}

// FoldUint64 mixes an unsigned 64-bit integer in little-endian order.
func (a *Accumulator) FoldUint64(v uint64) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	a.FoldBytes(buf[:])
	a.recordEvent("", float64(v), v)
}

// FoldInt64 mixes a signed 64-bit integer in little-endian order.
func (a *Accumulator) FoldInt64(v int64) {
	a.FoldUint64(uint64(v))
}

// FoldFloat32 mixes a 32-bit float into the accumulator according to the configured FloatPolicy.
func (a *Accumulator) FoldFloat32(v float32) {
	a.FoldFloat32Named("", v)
}

// FoldFloat32Named mixes a 32-bit float with an associated node name for triage.
func (a *Accumulator) FoldFloat32Named(nodeName string, v float32) {
	bits := math.Float32bits(v)

	switch a.policy {
	case PolicyCanonicalizeNaN:
		if math.IsNaN(float64(v)) {
			bits = CanonicalQuietNaN32Bits
		}
	case PolicyCanonicalizeZero:
		if v == 0 {
			bits = 0
		}
	case PolicyCanonicalizeAll:
		if math.IsNaN(float64(v)) {
			bits = CanonicalQuietNaN32Bits
		} else if v == 0 {
			bits = 0
		}
	}

	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], bits)
	for _, b := range buf {
		a.hash = (a.hash ^ uint64(b)) * FNVPrime
	}

	a.recordEvent(nodeName, float64(v), uint64(bits))
}

// FoldFloat32s mixes a slice of 32-bit floats in flat row-major order.
// Empty slices fold nothing.
func (a *Accumulator) FoldFloat32s(vals []float32) {
	a.FoldFloat32sNamed("", vals)
}

// FoldFloat32sNamed mixes a slice of 32-bit floats with an associated node name.
func (a *Accumulator) FoldFloat32sNamed(nodeName string, vals []float32) {
	for _, v := range vals {
		a.FoldFloat32Named(nodeName, v)
	}
}

// FoldFloat64 mixes a 64-bit float into the accumulator according to the configured FloatPolicy.
func (a *Accumulator) FoldFloat64(v float64) {
	a.FoldFloat64Named("", v)
}

// FoldFloat64Named mixes a 64-bit float with an associated node name.
func (a *Accumulator) FoldFloat64Named(nodeName string, v float64) {
	bits := math.Float64bits(v)

	switch a.policy {
	case PolicyCanonicalizeNaN:
		if math.IsNaN(v) {
			bits = CanonicalQuietNaN64Bits
		}
	case PolicyCanonicalizeZero:
		if v == 0 {
			bits = 0
		}
	case PolicyCanonicalizeAll:
		if math.IsNaN(v) {
			bits = CanonicalQuietNaN64Bits
		} else if v == 0 {
			bits = 0
		}
	}

	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], bits)
	for _, b := range buf {
		a.hash = (a.hash ^ uint64(b)) * FNVPrime
	}

	a.recordEvent(nodeName, v, bits)
}

func (a *Accumulator) recordEvent(nodeName string, val float64, bits uint64) {
	idx := a.count
	a.count++
	if a.recording {
		a.events = append(a.events, TraceEvent{
			Index:    idx,
			NodeName: nodeName,
			Value:    val,
			Bits:     bits,
			Hash:     a.hash,
		})
	}
}
