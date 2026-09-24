package cudabpfdev

import (
	"context"
	"encoding/binary"
	"fmt"
	"runtime"
	"slices"
	"strconv"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// MapSnapshot contains the instrumentation counters read after one completed
// kernel. Values is a fixed-capacity counter array; Misses counts out-of-range
// keys. This host copy is not coherent with a running kernel.
type MapSnapshot struct {
	Values []uint32
	Misses uint64
}

// Launch runs the selected image once using the original parameter ABI and
// geometry on a private nonblocking stream. Inputs must already be ready in the
// device's context; this call does not inherit dependencies from another stream.
// Device arguments remain caller-owned and must stay live until Launch succeeds
// or Close succeeds after an error. Host argument values are copied before launch.
//
// Candidate globals are zeroed on the same stream, then the selected kernel is
// launched and synchronized. Only then is the instrumentation map read back.
// Original fallback launches publish no instrumentation snapshot. Execution
// errors are returned; Launch never reruns the original after possible submission.
// Cancellation is checked between calls and cannot interrupt native work.
//
// A loaded owner admits one invocation; use another Load for another launch.
// Invalid arguments or a cancelled context at entry leave it unused. Once
// context entry is attempted, even a failed invocation consumes the owner. Close is
// always required. Concurrent calls serialize, with at most one invocation.
func (l *LoadedInjection) Launch(ctx context.Context, args ...any) error {
	if l == nil {
		return fmt.Errorf("cudabpfdev: no loaded selection")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.ready || l.closed || l.modules[0] == 0 {
		return fmt.Errorf("cudabpfdev: no open loaded selection")
	}
	if l.used {
		return fmt.Errorf("cudabpfdev: injection invocation already used")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	packed, err := l.schema.Pack(args...)
	if err != nil {
		return fmt.Errorf("cudabpfdev: launch arguments: %w", err)
	}
	l.used = true
	var snapshot MapSnapshot
	err = l.ops.run(func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		var err error
		l.stream, err = l.ops.stream()
		if err != nil {
			return err
		}
		if l.selection.Candidate {
			for n, size := range []uint{uint(max(l.slots, 1)) * 4, 8} {
				if err := ctx.Err(); err != nil {
					return err
				}
				l.pending = true
				if err := l.ops.zero(l.globals[n], size, l.stream); err != nil {
					return err
				}
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		l.pending = true
		if err := l.ops.launch(l.function, l.geometry, l.stream, l.schema, packed); err != nil {
			return err
		}
		if err := l.ops.synchronize(l.stream); err != nil {
			return err
		}
		l.pending = false
		if err := ctx.Err(); err != nil {
			return err
		}
		if l.selection.Candidate {
			data := make([]byte, uint(max(l.slots, 1))*4)
			misses := make([]byte, 8)
			if err := l.ops.read(l.globals[0], data); err != nil {
				return err
			}
			if err := l.ops.read(l.globals[1], misses); err != nil {
				return err
			}
			snapshot.Values = make([]uint32, l.slots)
			for n := range snapshot.Values {
				snapshot.Values[n] = binary.LittleEndian.Uint32(data[n*4:])
			}
			snapshot.Misses = binary.LittleEndian.Uint64(misses)
		}
		return ctx.Err()
	})
	if err != nil {
		return fmt.Errorf("cudabpfdev: launch injection: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	l.complete, l.snapshot, l.timestamp = true, snapshot, time.Now().UnixNano()
	return nil
}

// Snapshot returns a detached instrumentation map after a successful candidate
// launch. It is unavailable for fallback or failed invocations and survives Close.
func (l *LoadedInjection) Snapshot() (MapSnapshot, bool) {
	if l == nil {
		return MapSnapshot{}, false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.complete || !l.selection.Candidate {
		return MapSnapshot{}, false
	}
	s := l.snapshot
	s.Values = slices.Clone(s.Values)
	return s, true
}

// Record returns a successful invocation boundary for Recorder.Push, including
// the original fallback decision. Failed invocations produce no success record.
// The timestamp is host completion/readback publication; GPU duration and CUPTI
// correlation are unavailable. Full counter values are available from Snapshot.
func (l *LoadedInjection) Record() (cudaflightrecorder.Record, error) {
	if l == nil {
		return cudaflightrecorder.Record{}, fmt.Errorf("cudabpfdev: no completed invocation")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.complete {
		return cudaflightrecorder.Record{}, fmt.Errorf("cudabpfdev: no completed invocation")
	}
	labels := map[string]string{
		"record_kind": "device_bpf_boundary", "timestamp_source": "host_snapshot_publication",
		"duration_status": "unavailable", "correlation_status": "unavailable",
		"device_bpf.image_digest": l.selection.Kernel.Digest, "device_bpf.target": l.selection.Kernel.Target,
		"device_bpf.original_digest": l.originalKernel.Digest,
		"device_bpf.candidate":       strconv.FormatBool(l.selection.Candidate), "device_bpf.fallback_reason": l.selection.Reason,
		"device_bpf.snapshot": "unavailable",
	}
	if l.selection.Candidate {
		labels["device_bpf.snapshot"] = "snapshot_at_boundary"
		labels["device_bpf.map_slots"] = strconv.Itoa(len(l.snapshot.Values))
		labels["device_bpf.map_misses"] = strconv.FormatUint(l.snapshot.Misses, 10)
		labels["device_bpf.compiled_registers"] = strconv.FormatInt(int64(l.resources[1].registers), 10)
		labels["device_bpf.original_registers"] = strconv.FormatInt(int64(l.resources[0].registers), 10)
		labels["device_bpf.compiled_shared_bytes"] = strconv.FormatInt(int64(l.resources[1].sharedBytes), 10)
		labels["device_bpf.compiled_local_bytes"] = strconv.FormatInt(int64(l.resources[1].localBytes), 10)
		labels["device_bpf.original_shared_bytes"] = strconv.FormatInt(int64(l.resources[0].sharedBytes), 10)
		labels["device_bpf.original_local_bytes"] = strconv.FormatInt(int64(l.resources[0].localBytes), 10)
		labels["device_bpf.resource_source"] = "loaded_function_attributes"
	}
	return cudaflightrecorder.Record{TimestampNs: l.timestamp, KernelName: l.selection.Kernel.EntryPoint, Labels: labels}, nil
}

func launchInjection(fn driver.Function, g Geometry, stream driver.Stream, schema cudakernel.Schema, packed []byte) error {
	var pins runtime.Pinner
	defer pins.Unpin()
	if len(packed) > 0 {
		pins.Pin(&packed[0])
	}
	ptrs := make([]unsafe.Pointer, len(schema.Params))
	for n, param := range schema.Params {
		ptrs[n] = unsafe.Pointer(&packed[param.Offset])
	}
	if len(ptrs) > 0 {
		pins.Pin(&ptrs[0])
	}
	err := fn.LaunchKernel(driver.Dim3{X: g.Grid[0], Y: g.Grid[1], Z: g.Grid[2]}, driver.Dim3{X: g.Block[0], Y: g.Block[1], Z: g.Block[2]}, g.SharedBytes, stream, ptrs)
	runtime.KeepAlive(packed)
	runtime.KeepAlive(ptrs)
	return err
}
