// Package journaltest writes synthetic capture journals and captures for
// tests, so journal readers are exercised without hardware or a real capture.
package journaltest

import (
	"bytes"
	"compress/gzip"
	"encoding/json"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// A Builder writes a CDW1 journal one record at a time.
type Builder struct {
	buf      bytes.Buffer
	sequence uint64
	session  [16]byte
}

// New returns a Builder that has written a bootstrap and a capability record.
func New() *Builder {
	return NewWithCapability(map[string]any{"kind": "capability"})
}

// NewWithCapability is like [New] but writes the given capability payload.
func NewWithCapability(capability map[string]any) *Builder {
	b := &Builder{}
	b.session[0] = 0x42
	b.Append(wire.KindBootstrap, map[string]any{"kind": "bootstrap"})
	b.Append(wire.KindCapability, capability)
	return b
}

// Append writes one record of the given kind with payload marshaled as JSON.
func (b *Builder) Append(kind uint16, payload map[string]any) *Builder {
	data, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}
	b.sequence++
	rec := wire.Record{
		Kind:     kind,
		Identity: wire.Identity{Session: b.session, Process: 4242, Generation: 1},
		Sequence: b.sequence,
		Payload:  data,
	}
	if err := wire.Write(&b.buf, rec, wire.Limits{}); err != nil {
		panic(err)
	}
	return b
}

// Kernel appends one kernel activity on device 0.
func (b *Builder) Kernel(start, end uint64, correlation uint32) *Builder {
	return b.Activity("kernel", 0, start, end, correlation)
}

// Activity appends one activity of the given subtype.
func (b *Builder) Activity(subtype string, device uint32, start, end uint64, correlation uint32) *Builder {
	return b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": subtype, "valid": true,
		"start": start, "end": end, "device": device, "correlation": correlation,
	})
}

// Named appends one named activity of the given subtype on device 0.
func (b *Builder) Named(subtype, name string, start, end uint64, correlation uint32) *Builder {
	return b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": subtype, "valid": true,
		"name": name, "start": start, "end": end, "device": 0, "correlation": correlation,
	})
}

// KernelOn appends one kernel on an explicit device and context. The context
// is what a synchronization record is resolved through.
func (b *Builder) KernelOn(device, context uint32, start, end uint64, correlation uint32) *Builder {
	return b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "kernel", "valid": true,
		"start": start, "end": end, "device": device, "context": context, "correlation": correlation,
	})
}

// GraphNode appends one kernel that names the graph it ran as a node of.
func (b *Builder) GraphNode(device, context uint32, start, end uint64, correlation, graphID uint32) *Builder {
	return b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "kernel", "valid": true,
		"start": start, "end": end, "device": device, "context": context,
		"correlation": correlation, "graph_id": graphID,
	})
}

// GraphTrace appends one graph replay envelope.
func (b *Builder) GraphTrace(device uint32, start, end uint64, graphID, correlation uint32) *Builder {
	return b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "graph_trace", "valid": true,
		"start": start, "end": end, "device": device, "graph_id": graphID, "correlation": correlation,
	})
}

// Sync appends one synchronization span. It names a context and no device,
// as the producer writes it.
func (b *Builder) Sync(context uint32, start, end uint64, syncType, correlation uint32) *Builder {
	return b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "sync", "valid": true,
		"start": start, "end": end, "context": context, "sync_type": syncType,
		"cuda_event_id": 3, "correlation": correlation,
	})
}

// Overhead appends one CUPTI overhead span. It carries no device, as the
// producer writes it.
func (b *Builder) Overhead(start, end uint64, overheadKind uint32) *Builder {
	return b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "overhead", "valid": true,
		"start": start, "end": end, "overhead_kind": overheadKind, "object_kind": 2,
		"process_id": 4242, "thread_id": 99,
	})
}

// API appends one driver launch record correlated to an activity.
func (b *Builder) API(start, end uint64, correlation uint32) *Builder {
	return b.Append(wire.KindAPI, map[string]any{
		"kind": "api", "domain": "driver", "start": start, "end": end, "correlation": correlation,
	})
}

// Call appends one host API call of the given domain and callback ID.
func (b *Builder) Call(domain string, cbid, thread uint32, start, end uint64, correlation uint32) *Builder {
	return b.Append(wire.KindAPI, map[string]any{
		"kind": "api", "domain": domain, "callback_id": cbid, "thread_id": thread,
		"start": start, "end": end, "correlation": correlation,
	})
}

// Marker appends one NVTX marker record with the given flags.
func (b *Builder) Marker(name string, id, flags uint32, time uint64) *Builder {
	return b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "marker", "valid": true, "name": name,
		"marker_domain": "", "marker_id": id, "marker_flags": flags,
		"start": time, "end": time,
	})
}

// GEMM appends one cuBLASLt call as the agent writes it for --gemms.
func (b *Builder) GEMM(id uint64, api, log string) *Builder {
	return b.Append(wire.KindGEMM, map[string]any{"kind": "gemm", "id": id, "thread": 7, "api": api, "log": log})
}

// GEMMLaunch appends the link from cuBLASLt call id to a launch.
func (b *Builder) GEMMLaunch(id uint64, correlation uint32) *Builder {
	return b.Append(wire.KindGEMM, map[string]any{"kind": "gemm_launch", "id": id, "correlation": correlation})
}

// Finalize appends a finalized record and returns the journal.
func (b *Builder) Finalize() []byte {
	b.Append(wire.KindFinalized, map[string]any{"kind": "finalized", "state": "collected"})
	return b.Unfinalized()
}

// Unfinalized returns the records written so far without a finalized record.
func (b *Builder) Unfinalized() []byte { return append([]byte(nil), b.buf.Bytes()...) }

// Gzip returns data gzip-compressed.
func Gzip(data []byte) []byte {
	var out bytes.Buffer
	zw := gzip.NewWriter(&out)
	if _, err := zw.Write(data); err != nil {
		panic(err)
	}
	if err := zw.Close(); err != nil {
		panic(err)
	}
	return out.Bytes()
}

// Capture returns a capture holding files plus, when status is not nil, a
// capture-status.json wrapping it.
func Capture(files map[string][]byte, status map[string]any) *cudacapture.Capture {
	all := map[string][]byte{}
	for k, v := range files {
		all[k] = v
	}
	if status != nil {
		data, err := json.Marshal(map[string]any{"status": status})
		if err != nil {
			panic(err)
		}
		all["capture-status.json"] = data
	}
	return &cudacapture.Capture{
		Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}}},
		Files:   all,
	}
}

// CompleteStatus returns the status of a capture that lost nothing.
func CompleteStatus() map[string]any {
	return map[string]any{
		"capture_success":  true,
		"run_completeness": "complete",
		"dropped_records":  0,
		"detailed_records": "per-record activity and API arrays are not embedded in this status",
	}
}
