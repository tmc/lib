package cudaflightrecorder

import (
	"maps"
	"time"
)

// PushPolicyRecord records a driver policy observation. It has the shape of a
// policy record sink, so a policy package can use it without this package
// importing that one. Call it after draining policy events, never from a
// driver fault handler.
//
// timestampNs is a host timestamp in Unix nanoseconds. Zero uses collector
// receipt time, marked in the record labels. The record has no kernel name or
// device duration; duration_status marks the zero DurationNs as unavailable.
// The recorder must be started. Capacity and drop accounting are those of Push.
// The caller may reuse labels after this method returns.
func (r *Recorder) PushPolicyRecord(timestampNs int64, hook string, labels map[string]string) {
	labels = maps.Clone(labels)
	if labels == nil {
		labels = make(map[string]string)
	}
	labels["record_kind"] = "driver_policy"
	labels["hook"] = hook
	labels["duration_status"] = "unavailable"
	labels["timestamp_source"] = "host_event"
	if timestampNs == 0 {
		timestampNs = time.Now().UnixNano()
		labels["timestamp_source"] = "collector_receipt"
	}
	r.Push(Record{TimestampNs: timestampNs, Labels: labels})
}
