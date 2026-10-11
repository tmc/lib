package cudaprobe

import (
	"fmt"
	"strconv"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

// Record converts an event to a flight-recorder observation. received is the
// collector's Unix receipt time used for retention; the source's monotonic
// timestamp remains in labels. Neither timestamp measures GPU execution.
// Call recorder.Push with the result after draining the capture source.
// Kernel names, tensor shapes/dtypes and device duration remain unavailable.
func (e Event) Record(received time.Time) (cudaflightrecorder.Record, error) {
	if received.IsZero() {
		return cudaflightrecorder.Record{}, fmt.Errorf("cudaprobe: missing collector receipt time")
	}
	u := func(v uint64) string { return strconv.FormatUint(v, 10) }
	labels := map[string]string{
		"record_kind": "cuda_probe", "provenance": e.Identity.Provenance,
		"probe.interposed": strconv.FormatBool(e.Identity.Interposed),
		"duration_status":  "unavailable", "timestamp_source": "collector_receipt",
		"probe.timestamp_ns": u(e.Timestamp), "probe.clock": "host_monotonic",
		"probe.sequence": u(e.Sequence), "probe.tid": u(uint64(e.Identity.ThreadID)), "probe.thread_generation": u(e.Identity.ThreadGeneration), "probe.thread_generation_known": strconv.FormatBool(e.Identity.ThreadGenerationKnown), "probe.pid": u(uint64(e.Identity.Process.PID)),
		"probe.process_generation": u(e.Identity.Process.Generation),
		"probe.address_space":      u(e.Identity.AddressSpace),
		"probe.library":            e.Identity.Library, "probe.build_id": e.Identity.BuildID,
		"probe.library_version": e.Identity.Version, "probe.symbol": e.Identity.Symbol,
		"probe.abi": e.Identity.ABI, "probe.decoded": strconv.FormatBool(e.Decoded),
		"probe.return_known": strconv.FormatBool(e.ReturnKnown),
		"tensor_shapes":      "unavailable", "tensor_dtypes": "unavailable", "kernel_identity": "unavailable",
	}
	if labels["provenance"] == "" {
		labels["provenance"] = "unavailable"
	}
	if e.Reason != "" {
		labels["probe.reason"] = e.Reason
	}
	if e.ReturnKnown {
		labels["probe.status"] = strconv.FormatInt(int64(e.Status), 10)
	}
	if e.Decoded {
		if e.Launch != nil {
			labels["probe.function"] = u(e.Launch.Function)
			labels["probe.grid"] = fmt.Sprint(e.Launch.Grid)
			labels["probe.block"] = fmt.Sprint(e.Launch.Block)
			labels["probe.shared_bytes"] = u(e.Launch.SharedBytes)
			labels["probe.stream"] = u(e.Launch.Stream)
		} else if e.AsyncMemory != nil {
			labels["probe.address"] = u(e.AsyncMemory.Address)
			labels["probe.bytes"] = u(e.AsyncMemory.Bytes)
			labels["probe.stream"] = u(e.AsyncMemory.Stream)
			labels["probe.memory.async"] = "true"
			labels["probe.memory.free"] = strconv.FormatBool(e.AsyncMemory.Free)
		} else if e.Control != nil {
			labels["probe.control.scope"] = e.Control.Scope
			labels["probe.control.operation"] = e.Control.Operation
			labels["probe.control.handle"] = u(e.Control.Handle)
			labels["probe.control.flags"] = u(uint64(e.Control.Flags))
			labels["probe.control.device"] = strconv.FormatInt(int64(e.Control.Device), 10)
		} else if e.Copy != nil {
			labels["probe.copy.destination"] = u(e.Copy.Destination)
			labels["probe.copy.source"] = u(e.Copy.Source)
			labels["probe.bytes"] = u(e.Copy.Bytes)
			labels["probe.stream"] = u(e.Copy.Stream)
			labels["probe.copy.kind"] = strconv.FormatInt(int64(e.Copy.Kind), 10)
			labels["probe.copy.direction"] = e.Copy.Direction
			labels["probe.copy.async"] = strconv.FormatBool(e.Copy.Async)
		} else if e.Sync != nil {
			labels["probe.sync.scope"] = e.Sync.Scope
			labels["probe.stream"] = u(e.Sync.Stream)
		} else if e.Graph != nil {
			labels["probe.graph_exec"] = u(e.Graph.Exec)
			labels["probe.stream"] = u(e.Graph.Stream)
		} else {
			labels["probe.address"] = u(e.Address)
			labels["probe.bytes"] = u(e.Bytes)
		}
	}
	return cudaflightrecorder.Record{TimestampNs: received.UnixNano(), Labels: labels}, nil
}

// RecordWithBindings records a launch with an explicit artifact registration.
// When Identify fails it returns the unjoined record and the error, so callers
// can retain the observation without claiming a kernel identity. An identified
// kernel still has no measured device duration.
func (e Event) RecordWithBindings(received time.Time, bindings []Binding) (cudaflightrecorder.Record, error) {
	r, err := e.Record(received)
	if err != nil {
		return r, err
	}
	k, err := Identify(e, bindings)
	if err != nil {
		r.Labels["kernel_identity_reason"] = err.Error()
		return r, err
	}
	r.KernelName = k.EntryPoint
	r.Labels["kernel_identity"] = "collector_registration"
	r.Labels["kernel.digest"] = k.Digest
	r.Labels["kernel.target"] = k.Target
	r.Labels["kernel.module_generation"] = strconv.FormatUint(k.ModuleGeneration, 10)
	r.Labels["kernel.binding_evidence"] = k.Evidence
	return r, nil
}

// Count is a caller-supplied count of launch API observations, including failed
// launches, from one source and one explicitly identified measurement interval.
// It is not a count of GPU executions. Interval must identify the same boundaries
// for both sources, not just two similarly named workloads.
type Count struct {
	Process          Process
	Interval, Source string
	Value            uint64
}

// Comparison retains independent counts and their unreconciled difference.
// Exactly one of Excess or Missing can be nonzero; subtraction cannot overflow.
type Comparison struct {
	Probe, Independent Count
	Excess, Missing    uint64
}

// CompareCounts compares distinct sources over the same process and interval.
// It does not use agreement as proof that either source has complete coverage.
func CompareCounts(probe, independent Count) (Comparison, error) {
	if probe.Process.PID == 0 || probe.Process.Generation == 0 || probe.Process != independent.Process || probe.Interval == "" || probe.Interval != independent.Interval || probe.Source == "" || independent.Source == "" || probe.Source == independent.Source {
		return Comparison{}, fmt.Errorf("cudaprobe: counts need matching process and interval with distinct sources")
	}
	c := Comparison{Probe: probe, Independent: independent}
	if probe.Value > independent.Value {
		c.Excess = probe.Value - independent.Value
	} else {
		c.Missing = independent.Value - probe.Value
	}
	return c, nil
}
