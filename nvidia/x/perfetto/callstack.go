package perfetto

import "fmt"

// StackFrame is a symbolized frame. Call stacks use leaf-first order.
type StackFrame struct {
	Function string
	File     string
	Line     uint32
}

// InternCallstack writes interned symbols and returns a sequence-local stack ID.
// Equal stacks share an ID. Perfetto stores frame IDs root first.
func (pw *Writer) InternCallstack(frames []StackFrame) (uint64, error) {
	if len(frames) == 0 || len(frames) > 128 {
		return 0, fmt.Errorf("callstack must contain 1 to 128 frames")
	}
	for _, f := range frames {
		if f.Function == "" {
			return 0, fmt.Errorf("callstack frame has no function")
		}
	}
	key := fmt.Sprintf("%#v", frames)
	pw.stackMu.Lock()
	defer pw.stackMu.Unlock()
	if err := pw.checkCallstackLocked(0); err != nil {
		return 0, err
	}
	if pw.stackSequence == 0 {
		pw.stackSequence = pw.sequenceID()
	}
	if id := pw.callstacks[key]; id != 0 {
		return id, nil
	}
	if pw.callstacks == nil {
		pw.callstacks = make(map[string]uint64)
		pw.stackFrames = make(map[StackFrame]uint64)
		pw.stackStrings = make(map[string]uint64)
	}
	if len(pw.callstacks) >= 100000 || len(pw.stackFrames)+len(frames) > 200000 || len(pw.stackStrings)+2*len(frames) > 400000 {
		return 0, fmt.Errorf("callstack limit exceeded")
	}
	var data []byte
	ids := make([]uint64, len(frames))
	for i, f := range frames {
		if f.Function == "" {
			return 0, fmt.Errorf("callstack frame has no function")
		}
		id := pw.stackFrames[f]
		if id == 0 {
			intern := func(field int, s string) uint64 {
				key := fmt.Sprintf("%d:%s", field, s)
				n := pw.stackStrings[key]
				if n == 0 {
					n = uint64(len(pw.stackStrings) + 1)
					b := encodeVarint(1, n)
					b = append(b, encodeString(2, s)...)
					data = append(data, encodeLengthDelimited(field, b)...)
					pw.stackStrings[key] = n
				}
				return n
			}
			name := intern(5, f.Function)
			var path uint64
			if f.File != "" {
				path = intern(18, f.File)
			}
			id = uint64(len(pw.stackFrames) + 1)
			b := encodeVarint(1, id)
			b = append(b, encodeVarint(2, name)...)
			if path != 0 {
				b = append(b, encodeVarint(5, path)...)
			}
			if f.Line != 0 {
				b = append(b, encodeVarint(6, uint64(f.Line))...)
			}
			data = append(data, encodeLengthDelimited(6, b)...)
			pw.stackFrames[f] = id
		}
		ids[len(frames)-1-i] = id
	}
	id := uint64(len(pw.callstacks) + 1)
	b := encodeVarint(1, id)
	for _, f := range ids {
		b = append(b, encodeVarint(2, f)...)
	}
	data = append(data, encodeLengthDelimited(7, b)...)
	packet := encodeVarint(FieldPacketSequenceID, pw.sequenceID())
	packet = append(packet, encodeLengthDelimited(12, data)...)
	if err := pw.writePacket(packet, true); err != nil {
		pw.stackErr = err
		return 0, err
	}
	pw.callstacks[key] = id
	return id, nil
}

// WriteSliceBeginWithCallstack records a slice with an interned language stack.
// callstackID must have been returned by this writer's InternCallstack.
func (pw *Writer) WriteSliceBeginWithCallstack(trackUUID, tsNs uint64, name, category string, annotations []DebugAnnotation, callstackID uint64) error {
	if err := pw.checkCallstack(callstackID); err != nil {
		return err
	}
	return pw.writeSliceBegin(trackUUID, tsNs, name, category, annotations, nil, callstackID)
}

// sequenceID returns the configured packet sequence.
func (pw *Writer) sequenceID() uint64 {
	if pw.SequenceID != 0 {
		return pw.SequenceID
	}
	return DefaultSequenceID
}

// WriteInstantWithCallstack records a timestamped sample with an interned stack.
func (pw *Writer) WriteInstantWithCallstack(trackUUID, tsNs uint64, name, category string, annotations []DebugAnnotation, callstackID uint64) error {
	if err := pw.checkCallstack(callstackID); err != nil {
		return err
	}
	body := encodeVarint(FieldTrackEventType, TrackEventTypeInstant)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)
	body = append(body, encodeString(FieldTrackEventName, name)...)
	body = append(body, encodeString(FieldTrackEventCategories, category)...)
	for _, a := range annotations {
		body = append(body, encodeDebugAnnotation(a)...)
	}
	if callstackID != 0 {
		body = append(body, encodeVarint(56, callstackID)...)
	}
	packet := encodeVarint(FieldPacketSequenceID, pw.sequenceID())
	packet = append(packet, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packet = append(packet, encodeLengthDelimited(FieldPacketTrackEvent, body)...)
	return pw.writePacket(packet, true)
}

func (pw *Writer) checkCallstack(id uint64) error {
	pw.stackMu.Lock()
	defer pw.stackMu.Unlock()
	return pw.checkCallstackLocked(id)
}
func (pw *Writer) checkCallstackLocked(id uint64) error {
	if pw.stackErr != nil {
		return pw.stackErr
	}
	if pw.stackSequence != 0 && pw.sequenceID() != pw.stackSequence {
		return fmt.Errorf("callstack packet sequence changed")
	}
	if id > uint64(len(pw.callstacks)) {
		return fmt.Errorf("unknown callstack ID %d", id)
	}
	return nil
}
