package perfetto

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"sync"
)

// Standard Perfetto wire-format constants
const (
	FieldTracePacket            = 1
	FieldPacketClockSnapshot    = 6
	FieldPacketTimestamp        = 8
	FieldPacketSequenceID       = 10
	FieldPacketTrackEvent       = 11
	FieldPacketSequenceFlags    = 13
	FieldPacketTimestampClockId = 58
	FieldPacketTrackDesc        = 60

	FieldClockSnapshotClocks = 1
	FieldClockID             = 1
	FieldClockTimestamp      = 2

	ClockRealtime            = 1
	ClockMonotonic           = 3
	ClockGPU                 = 64
	FieldTrackDescUUID       = 1
	FieldTrackDescName       = 2
	FieldTrackDescParentUUID = 5

	FieldTrackDescProcessUUID = 3
	FieldTrackDescThreadUUID  = 4
	FieldTrackDescCounter     = 8

	FieldCounterDescUnit           = 3
	FieldCounterDescUnitName       = 6
	FieldCounterDescUnitMultiplier = 4

	FieldProcessDescPID  = 1
	FieldProcessDescName = 6

	// ThreadDescriptor field numbers, per
	// protos/perfetto/trace/track_event/thread_descriptor.proto:
	// "optional int32 pid = 1", "optional int64 tid = 2",
	// "optional string thread_name = 5".
	FieldThreadDescPID  = 1
	FieldThreadDescName = 5
	FieldThreadDescTID  = 2

	FieldTrackEventCategories       = 22
	FieldTrackEventName             = 23
	FieldTrackEventType             = 9
	FieldTrackEventTrackUUID        = 11
	FieldTrackEventCounterValue     = 30
	FieldTrackEventDoubleCounterVal = 44
	FieldTrackEventDebugAnnotations = 4
	FieldTrackEventSourceLocation   = 33

	DefaultSequenceID                   = 1
	SequenceFlagIncrementalStateCleared = 1

	FieldDebugAnnotationName        = 10
	FieldDebugAnnotationBoolValue   = 2
	FieldDebugAnnotationUintValue   = 3
	FieldDebugAnnotationIntValue    = 4
	FieldDebugAnnotationDoubleValue = 5
	FieldDebugAnnotationStringValue = 6

	FieldSourceLocationFileName     = 2
	FieldSourceLocationFunctionName = 3
	FieldSourceLocationLineNumber   = 4

	TrackEventTypeSliceBegin = 1
	TrackEventTypeSliceEnd   = 2
	TrackEventTypeInstant    = 3
	TrackEventTypeCounter    = 4
)

// AnnotationType denotes the value type stored in a DebugAnnotation.
type AnnotationType int

const (
	AnnotationTypeString AnnotationType = iota
	AnnotationTypeInt
	AnnotationTypeUint
	AnnotationTypeDouble
	AnnotationTypeBool
)

// DebugAnnotation represents a structured metadata key-value annotation attached to a Perfetto slice.
type DebugAnnotation struct {
	Name        string
	StringValue string
	IntValue    int64
	UintValue   uint64
	DoubleValue float64
	BoolValue   bool
	Type        AnnotationType
}

// SourceLocation specifies source file and line coordinates for timeline slice inspection.
type SourceLocation struct {
	FileName     string
	FunctionName string
	LineNumber   uint32
}

// NewStringAnnotation creates a string debug annotation.
func NewStringAnnotation(name, val string) DebugAnnotation {
	return DebugAnnotation{Name: name, StringValue: val, Type: AnnotationTypeString}
}

// NewIntAnnotation creates an integer debug annotation.
func NewIntAnnotation(name string, val int64) DebugAnnotation {
	return DebugAnnotation{Name: name, IntValue: val, Type: AnnotationTypeInt}
}

// NewUintAnnotation creates an unsigned integer debug annotation.
func NewUintAnnotation(name string, val uint64) DebugAnnotation {
	return DebugAnnotation{Name: name, UintValue: val, Type: AnnotationTypeUint}
}

// NewDoubleAnnotation creates a floating-point debug annotation.
func NewDoubleAnnotation(name string, val float64) DebugAnnotation {
	return DebugAnnotation{Name: name, DoubleValue: val, Type: AnnotationTypeDouble}
}

// NewBoolAnnotation creates a boolean debug annotation.
func NewBoolAnnotation(name string, val bool) DebugAnnotation {
	return DebugAnnotation{Name: name, BoolValue: val, Type: AnnotationTypeBool}
}

// Writer encodes and writes Perfetto binary trace files.
type Writer struct {
	mu                 sync.Mutex
	w                  io.Writer
	buf                bytes.Buffer
	hasEmittedSeqFlags bool
}

// NewWriter returns a new Perfetto trace writer writing into w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{
		w: w,
	}
}

func (pw *Writer) writePacket(packetBody []byte, clearIncrementalState bool) error {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	emitClear := clearIncrementalState && !pw.hasEmittedSeqFlags
	if emitClear {
		packetBody = append(packetBody, encodeVarint(FieldPacketSequenceFlags, SequenceFlagIncrementalStateCleared)...)
	}
	packet := encodeLengthDelimited(FieldTracePacket, packetBody)
	if err := writeFull(pw.w, packet); err != nil {
		return err
	}
	if emitClear {
		pw.hasEmittedSeqFlags = true
	}
	return nil
}

func writeFull(w io.Writer, p []byte) error {
	n, err := w.Write(p)
	if err == nil && n != len(p) {
		return io.ErrShortWrite
	}
	return err
}

func encodeVarint(fieldNum int, val uint64) []byte {
	tag := uint64((fieldNum << 3) | 0) // wire type 0 = varint
	var buf [20]byte
	n1 := binary.PutUvarint(buf[:], tag)
	n2 := binary.PutUvarint(buf[n1:], val)
	return buf[:n1+n2]
}

func encodeLengthDelimited(fieldNum int, data []byte) []byte {
	tag := uint64((fieldNum << 3) | 2) // wire type 2 = length delimited
	var buf [10]byte
	n1 := binary.PutUvarint(buf[:], tag)
	var lenBuf [10]byte
	n2 := binary.PutUvarint(lenBuf[:], uint64(len(data)))

	out := make([]byte, 0, n1+n2+len(data))
	out = append(out, buf[:n1]...)
	out = append(out, lenBuf[:n2]...)
	out = append(out, data...)
	return out
}

func encodeString(fieldNum int, s string) []byte {
	return encodeLengthDelimited(fieldNum, []byte(s))
}

// WriteClockSnapshot writes correlated host monotonic clock and GPU device clock timestamps
// to establish absolute timeline calibration across disparate hardware domains.
func (pw *Writer) WriteClockSnapshot(hostMonotonicNs, gpuClockNs uint64) error {
	var clock1 []byte
	clock1 = append(clock1, encodeVarint(FieldClockID, ClockMonotonic)...)
	clock1 = append(clock1, encodeVarint(FieldClockTimestamp, hostMonotonicNs)...)

	var clock2 []byte
	clock2 = append(clock2, encodeVarint(FieldClockID, ClockGPU)...)
	clock2 = append(clock2, encodeVarint(FieldClockTimestamp, gpuClockNs)...)

	var snapshotBody []byte
	snapshotBody = append(snapshotBody, encodeLengthDelimited(FieldClockSnapshotClocks, clock1)...)
	snapshotBody = append(snapshotBody, encodeLengthDelimited(FieldClockSnapshotClocks, clock2)...)

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketClockSnapshot, snapshotBody)...)
	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	return writeFull(pw.w, packet)
}

// WriteTrackDescriptor registers a track in the trace timeline.
func (pw *Writer) WriteTrackDescriptor(uuid uint64, name string, parentUUID uint64) error {
	var body []byte
	body = append(body, encodeVarint(FieldTrackDescUUID, uuid)...)
	if name != "" {
		body = append(body, encodeString(FieldTrackDescName, name)...)
	}
	if parentUUID != 0 {
		body = append(body, encodeVarint(FieldTrackDescParentUUID, parentUUID)...)
	}

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackDesc, body)...)
	return pw.writePacket(packetBody, true)
}

// WriteProcessDescriptor registers a process track.
func (pw *Writer) WriteProcessDescriptor(uuid uint64, pid int32, processName string) error {
	var procBody []byte
	if pid != 0 {
		procBody = append(procBody, encodeVarint(FieldProcessDescPID, uint64(pid))...)
	}
	if processName != "" {
		procBody = append(procBody, encodeString(FieldProcessDescName, processName)...)
	}

	var body []byte
	body = append(body, encodeVarint(FieldTrackDescUUID, uuid)...)
	body = append(body, encodeLengthDelimited(FieldTrackDescProcessUUID, procBody)...)

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackDesc, body)...)
	return pw.writePacket(packetBody, true)
}

// WriteThreadDescriptor registers a thread track grouped under a process track.
func (pw *Writer) WriteThreadDescriptor(uuid uint64, pid int32, tid int32, threadName string) error {
	var thBody []byte
	if tid != 0 {
		thBody = append(thBody, encodeVarint(FieldThreadDescTID, uint64(tid))...)
	}
	if pid != 0 {
		thBody = append(thBody, encodeVarint(FieldThreadDescPID, uint64(pid))...)
	}
	if threadName != "" {
		thBody = append(thBody, encodeString(FieldThreadDescName, threadName)...)
	}

	var body []byte
	body = append(body, encodeVarint(FieldTrackDescUUID, uuid)...)
	body = append(body, encodeLengthDelimited(FieldTrackDescThreadUUID, thBody)...)

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackDesc, body)...)
	return pw.writePacket(packetBody, true)
}

// WriteCounterDescriptor registers a continuous numerical counter track.
func (pw *Writer) WriteCounterDescriptor(uuid uint64, name string, unitName string, parentUUID uint64) error {
	var counterBody []byte
	if unitName != "" {
		counterBody = append(counterBody, encodeString(FieldCounterDescUnitName, unitName)...)
	}

	var body []byte
	body = append(body, encodeVarint(FieldTrackDescUUID, uuid)...)
	if name != "" {
		body = append(body, encodeString(FieldTrackDescName, name)...)
	}
	if parentUUID != 0 {
		body = append(body, encodeVarint(FieldTrackDescParentUUID, parentUUID)...)
	}
	body = append(body, encodeLengthDelimited(FieldTrackDescCounter, counterBody)...)

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackDesc, body)...)
	return pw.writePacket(packetBody, true)
}

// WriteCounterValue records an integer counter value on a counter track at tsNs.
func (pw *Writer) WriteCounterValue(trackUUID uint64, tsNs uint64, val int64) error {
	var body []byte
	body = append(body, encodeVarint(FieldTrackEventType, TrackEventTypeCounter)...)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)
	body = append(body, encodeVarint(FieldTrackEventCounterValue, uint64(val))...)

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackEvent, body)...)

	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	return writeFull(pw.w, packet)
}

// WriteDoubleCounterValue records a floating point counter value on a counter track at tsNs.
func (pw *Writer) WriteDoubleCounterValue(trackUUID uint64, tsNs uint64, val float64) error {
	var body []byte
	body = append(body, encodeVarint(FieldTrackEventType, TrackEventTypeCounter)...)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)

	bits := math.Float64bits(val)
	tag := uint64((FieldTrackEventDoubleCounterVal << 3) | 1)
	var b [16]byte
	n := binary.PutUvarint(b[:], tag)
	binary.LittleEndian.PutUint64(b[n:], bits)
	body = append(body, b[:n+8]...)

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackEvent, body)...)

	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	return writeFull(pw.w, packet)
}

// WriteSliceBegin records the start of a timed slice on trackUUID at timestamp ns.
func (pw *Writer) WriteSliceBegin(trackUUID uint64, tsNs uint64, name string, category string) error {
	return pw.WriteSliceBeginWithAnnotations(trackUUID, tsNs, name, category, nil, nil)
}

// WriteSliceEnd records the end of the active slice on trackUUID at timestamp ns.
func (pw *Writer) WriteSliceEnd(trackUUID uint64, tsNs uint64) error {
	var body []byte
	body = append(body, encodeVarint(FieldTrackEventType, TrackEventTypeSliceEnd)...)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackEvent, body)...)

	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	return writeFull(pw.w, packet)
}

// WriteInstant records an instant marker event on trackUUID at timestamp ns.
func (pw *Writer) WriteInstant(trackUUID uint64, tsNs uint64, name string, category string) error {
	var body []byte
	body = append(body, encodeVarint(FieldTrackEventType, TrackEventTypeInstant)...)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)
	if name != "" {
		body = append(body, encodeString(FieldTrackEventName, name)...)
	}
	if category != "" {
		body = append(body, encodeString(FieldTrackEventCategories, category)...)
	}

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackEvent, body)...)

	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	return writeFull(pw.w, packet)
}

func encodeDebugAnnotation(da DebugAnnotation) []byte {
	var body []byte
	if da.Name != "" {
		body = append(body, encodeString(FieldDebugAnnotationName, da.Name)...)
	}
	switch da.Type {
	case AnnotationTypeBool:
		val := uint64(0)
		if da.BoolValue {
			val = 1
		}
		body = append(body, encodeVarint(FieldDebugAnnotationBoolValue, val)...)
	case AnnotationTypeUint:
		body = append(body, encodeVarint(FieldDebugAnnotationUintValue, da.UintValue)...)
	case AnnotationTypeInt:
		body = append(body, encodeVarint(FieldDebugAnnotationIntValue, uint64(da.IntValue))...)
	case AnnotationTypeDouble:
		bits := math.Float64bits(da.DoubleValue)
		tag := uint64((FieldDebugAnnotationDoubleValue << 3) | 1)
		var b [16]byte
		n := binary.PutUvarint(b[:], tag)
		binary.LittleEndian.PutUint64(b[n:], bits)
		body = append(body, b[:n+8]...)
	case AnnotationTypeString:
		body = append(body, encodeString(FieldDebugAnnotationStringValue, da.StringValue)...)
	}
	return encodeLengthDelimited(FieldTrackEventDebugAnnotations, body)
}

func encodeSourceLocation(loc SourceLocation) []byte {
	var body []byte
	if loc.FileName != "" {
		body = append(body, encodeString(FieldSourceLocationFileName, loc.FileName)...)
	}
	if loc.FunctionName != "" {
		body = append(body, encodeString(FieldSourceLocationFunctionName, loc.FunctionName)...)
	}
	if loc.LineNumber != 0 {
		body = append(body, encodeVarint(FieldSourceLocationLineNumber, uint64(loc.LineNumber))...)
	}
	return encodeLengthDelimited(FieldTrackEventSourceLocation, body)
}

// WriteSliceBeginWithAnnotations records the start of a timed slice with debug annotations and optional source location.
func (pw *Writer) WriteSliceBeginWithAnnotations(trackUUID uint64, tsNs uint64, name string, category string, annotations []DebugAnnotation, loc *SourceLocation) error {
	var body []byte
	body = append(body, encodeVarint(FieldTrackEventType, TrackEventTypeSliceBegin)...)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)
	if name != "" {
		body = append(body, encodeString(FieldTrackEventName, name)...)
	}
	if category != "" {
		body = append(body, encodeString(FieldTrackEventCategories, category)...)
	}
	for _, da := range annotations {
		body = append(body, encodeDebugAnnotation(da)...)
	}
	if loc != nil {
		body = append(body, encodeSourceLocation(*loc)...)
	}

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packetBody = append(packetBody, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackEvent, body)...)
	return pw.writePacket(packetBody, true)
}
