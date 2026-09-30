package perfetto

import (
	"encoding/binary"
	"fmt"
)

// WriteFlowInstant records an instant with explicit outgoing and terminating
// flow IDs. IDs are trace-global; each must be unique to the caller's flow.
// The caller supplies ordering evidence. A flow does not infer causality or
// establish a dependency from timestamp overlap or submission correlation.
// An ID cannot both originate and terminate on the same event.
func (pw *Writer) WriteFlowInstant(trackUUID, tsNs uint64, name, category string, annotations []DebugAnnotation, flowIDs, terminatingFlowIDs []uint64) error {
	seen := make(map[uint64]bool, len(flowIDs))
	for _, id := range flowIDs {
		seen[id] = true
	}
	for _, id := range terminatingFlowIDs {
		if seen[id] {
			return fmt.Errorf("perfetto: flow id %d both originates and terminates on one event", id)
		}
	}
	var body []byte
	body = append(body, encodeVarint(FieldTrackEventType, TrackEventTypeInstant)...)
	body = append(body, encodeVarint(FieldTrackEventTrackUUID, trackUUID)...)
	if name != "" {
		body = append(body, encodeString(FieldTrackEventName, name)...)
	}
	if category != "" {
		body = append(body, encodeString(FieldTrackEventCategories, category)...)
	}
	for _, annotation := range annotations {
		body = append(body, encodeDebugAnnotation(annotation)...)
	}
	// TrackEvent.flow_ids (47) and terminating_flow_ids (48) are fixed64,
	// not varints. Keep the exact width, including IDs above 2^53.
	// https://github.com/google/perfetto/blob/main/protos/perfetto/trace/track_event/track_event.proto
	for _, id := range flowIDs {
		body = appendFlowID(body, 47, id)
	}
	for _, id := range terminatingFlowIDs {
		body = appendFlowID(body, 48, id)
	}
	var packet []byte
	packet = append(packet, encodeVarint(FieldPacketTimestamp, tsNs)...)
	packet = append(packet, encodeVarint(FieldPacketSequenceID, DefaultSequenceID)...)
	packet = append(packet, encodeLengthDelimited(FieldPacketTrackEvent, body)...)
	return pw.writePacket(packet, true)
}

func appendFlowID(body []byte, field, id uint64) []byte {
	body = binary.AppendUvarint(body, field<<3|1)
	return binary.LittleEndian.AppendUint64(body, id)
}
