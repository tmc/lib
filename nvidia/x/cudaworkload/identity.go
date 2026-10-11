package cudaworkload

import (
	"fmt"
	"maps"
)

// OpID is a non-zero, monotonically increasing identifier allocated for an
// application operation scope. Zero indicates an invalid or unassigned ID.
type OpID uint64

// IsValid reports whether the OpID is non-zero.
func (id OpID) IsValid() bool {
	return id != 0
}

// String formats the OpID as a decimal string.
func (id OpID) String() string {
	if id == 0 {
		return "invalid"
	}
	return fmt.Sprintf("op:%d", uint64(id))
}

// ProcessGenID identifies a specific process execution instance. Pairing the
// operating system PID with a monotonic generation counter (or boot timestamp)
// guarantees that PID recycling after process termination cannot cause false joins.
type ProcessGenID struct {
	PID        uint32 `json:"pid"`
	Generation uint64 `json:"generation"`
}

// IsZero reports whether the ProcessGenID has default/zero values.
func (p ProcessGenID) IsZero() bool {
	return p.PID == 0 && p.Generation == 0
}

// String formats the ProcessGenID.
func (p ProcessGenID) String() string {
	return fmt.Sprintf("%d#%d", p.PID, p.Generation)
}

// WorkloadIdentity encapsulates the identifying metadata of an execution context,
// including the process generation, workload name, run identifier, and tags.
type WorkloadIdentity struct {
	Process      ProcessGenID      `json:"process"`
	WorkloadName string            `json:"workload_name"`
	RunID        string            `json:"run_id,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

// Clone returns a deep copy of the WorkloadIdentity.
func (w WorkloadIdentity) Clone() WorkloadIdentity {
	c := w
	if w.Labels != nil {
		c.Labels = make(map[string]string, len(w.Labels))
		maps.Copy(c.Labels, w.Labels)
	}
	return c
}

// CorrelationProvenance records how the correlation link between host and
// device was established.
type CorrelationProvenance string

const (
	// ProvenanceUninstrumented indicates no external-correlation instrumentation
	// was enabled. It does not establish a runtime intercept or native ID link.
	ProvenanceUninstrumented CorrelationProvenance = "uninstrumented_native_call"
	// ProvenanceCUPTI indicates correlation was established via CUPTI external
	// correlation push/pop APIs.
	ProvenanceCUPTI CorrelationProvenance = "cupti_external_correlation"
	// ProvenanceUprobe indicates association via an independently verified host
	// uprobe call window with confirmed PID and TID during a locked thread interval.
	ProvenanceUprobe CorrelationProvenance = "uprobe_call_window"
	// ProvenanceRuntimeAPI indicates correlation established by a direct CUDA
	// runtime or driver API intercept shim.
	ProvenanceRuntimeAPI CorrelationProvenance = "runtime_api_intercept"
	// ProvenanceGraphNode indicates correlation linked through explicit CUDA
	// graph node capture and replay instance tracking.
	ProvenanceGraphNode CorrelationProvenance = "graph_node_instance"
	// ProvenanceSynthetic indicates a fixture or synthetic test event.
	ProvenanceSynthetic CorrelationProvenance = "synthetic_fixture"
)

// ContextStreamID identifies the target CUDA execution context and stream.
type ContextStreamID struct {
	Context uint64 `json:"context"`
	Stream  uint64 `json:"stream"`
}

// String formats the context and stream identifiers.
func (cs ContextStreamID) String() string {
	return fmt.Sprintf("ctx:%x/stream:%x", cs.Context, cs.Stream)
}

// GraphReplayGeneration identifies an execution instance of a captured CUDA graph.
// Capturing a graph assigns structure and node IDs; each execution of the graph via
// cudaGraphLaunch constitutes a discrete replay generation.
type GraphReplayGeneration struct {
	GraphID          uint64 `json:"graph_id"`
	ExecGraphID      uint64 `json:"exec_graph_id"`
	ReplayGeneration uint64 `json:"replay_generation"`
}

// IsZero reports whether the GraphReplayGeneration indicates non-graph work.
func (g GraphReplayGeneration) IsZero() bool {
	return g.GraphID == 0 && g.ExecGraphID == 0 && g.ReplayGeneration == 0
}

// String formats the graph replay generation.
func (g GraphReplayGeneration) String() string {
	if g.IsZero() {
		return "nongraph"
	}
	return fmt.Sprintf("graph:%x/exec:%x#%d", g.GraphID, g.ExecGraphID, g.ReplayGeneration)
}

// RawIDs preserves verbatim identifiers from capture sources without alteration.
type RawIDs struct {
	ProcessGenID          ProcessGenID          `json:"process_gen_id"`
	NativeCorrelationID   uint32                `json:"native_correlation_id"`
	RuntimeCorrelationID  uint32                `json:"runtime_correlation_id,omitempty"`
	ExternalCorrelationID uint64                `json:"external_correlation_id,omitempty"`
	ContextStream         ContextStreamID       `json:"context_stream"`
	GraphReplay           GraphReplayGeneration `json:"graph_replay,omitempty"`
	TID                   uint64                `json:"tid,omitempty"`
}
