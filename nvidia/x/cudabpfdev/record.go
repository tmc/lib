package cudabpfdev

import (
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

type boundary struct {
	timestamp             int64
	digest, entry, target string
	active                uint32
}

// Record describes successful snapshot publication for the flight recorder.
// Call Recorder.Push with its result after Execute returns. Failed execution
// has no completed snapshot record, even if subsequent cleanup succeeds.
//
// TimestampNs is the host's snapshot-publication time, after synchronization,
// readback and context restoration. GPU duration and CUPTI correlation are
// unavailable. Labels identify the loaded image and summarize the snapshot;
// the full values remain available from Snapshot. Each call returns new labels.
func (e *Execution) Record() (cudaflightrecorder.Record, error) {
	if e == nil {
		return cudaflightrecorder.Record{}, fmt.Errorf("cudabpfdev: no completed snapshot")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.ready {
		return cudaflightrecorder.Record{}, fmt.Errorf("cudabpfdev: no completed snapshot")
	}
	return cudaflightrecorder.Record{
		TimestampNs: e.boundary.timestamp,
		KernelName:  e.boundary.entry,
		Labels: map[string]string{
			"record_kind":                      "device_bpf_snapshot",
			"timestamp_source":                 "host_snapshot_publication",
			"duration_status":                  "unavailable",
			"correlation_status":               "unavailable",
			"device_bpf.execution":             "standalone_oracle",
			"device_bpf.image_digest":          e.boundary.digest,
			"device_bpf.target":                e.boundary.target,
			"device_bpf.active_mask":           strconv.FormatUint(uint64(e.boundary.active), 10),
			"device_bpf.map_slots":             strconv.Itoa(len(e.snapshot.Map)),
			"device_bpf.map_misses":            strconv.FormatUint(e.snapshot.Misses, 10),
			"device_bpf.consistency":           "snapshot_at_boundary",
			"device_bpf.compiled_registers":    strconv.FormatInt(int64(e.resources.registers), 10),
			"device_bpf.compiled_shared_bytes": strconv.FormatInt(int64(e.resources.sharedBytes), 10),
			"device_bpf.compiled_local_bytes":  strconv.FormatInt(int64(e.resources.localBytes), 10),
			"device_bpf.resource_source":       "loaded_function_attributes",
		},
	}, nil
}
