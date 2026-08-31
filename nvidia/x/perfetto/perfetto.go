package perfetto

import (
	"bytes"
	"encoding/binary"
	"io"
	"sync"
)

// Standard Perfetto wire-format constants
const (
	FieldTracePacket         = 1
	FieldPacketTimestamp     = 8
	FieldPacketTrackEvent    = 11
	FieldPacketClockSnapshot = 35
	FieldPacketTrackDesc     = 60
	FieldPacketSequenceFlags = 89

	FieldClockSnapshotClocks = 1
	FieldClockID             = 1
	FieldClockTimestamp      = 2

	ClockMonotonic = 1
	ClockGPU       = 6

	FieldTrackDescUUID       = 1
	FieldTrackDescName       = 2
	FieldTrackDescParentUUID = 5

	FieldTrackEventCategories = 22
	FieldTrackEventName       = 2
	FieldTrackEventType       = 9
	FieldTrackEventTrackUUID  = 11

	TrackEventTypeSliceBegin = 1
	TrackEventTypeSliceEnd   = 2
	TrackEventTypeInstant    = 3
)

// Writer encodes and writes Perfetto binary trace files.
type Writer struct {
	mu  sync.Mutex
	w   io.Writer
	buf bytes.Buffer
}

// NewWriter returns a new Perfetto trace writer writing into w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{
		w: w,
	}
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

	packetBody := encodeLengthDelimited(FieldPacketClockSnapshot, snapshotBody)
	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	_, err := pw.w.Write(packet)
	return err
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

	packetBody := encodeLengthDelimited(FieldPacketTrackDesc, body)
	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	_, err := pw.w.Write(packet)
	return err
}

// WriteSliceBegin records the start of a timed slice on trackUUID at timestamp ns.
func (pw *Writer) WriteSliceBegin(trackUUID uint64, tsNs uint64, name string, category string) error {
	var body []byte
	body = append(body, encodeVarint(FieldTrackEventType, TrackEventTypeSliceBegin)...)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)
	if name != "" {
		body = append(body, encodeString(FieldTrackEventName, name)...)
	}
	if category != "" {
		body = append(body, encodeString(FieldTrackEventCategories, category)...)
	}

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackEvent, body)...)

	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	_, err := pw.w.Write(packet)
	return err
}

// WriteSliceEnd records the end of the active slice on trackUUID at timestamp ns.
func (pw *Writer) WriteSliceEnd(trackUUID uint64, tsNs uint64) error {
	var body []byte
	body = append(body, encodeVarint(FieldTrackEventType, TrackEventTypeSliceEnd)...)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)

	var packetBody []byte
	packetBody = append(packetBody, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackEvent, body)...)

	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	_, err := pw.w.Write(packet)
	return err
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
	packetBody = append(packetBody, encodeLengthDelimited(FieldPacketTrackEvent, body)...)

	packet := encodeLengthDelimited(FieldTracePacket, packetBody)

	pw.mu.Lock()
	defer pw.mu.Unlock()
	_, err := pw.w.Write(packet)
	return err
}
