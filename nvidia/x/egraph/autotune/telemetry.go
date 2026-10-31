package autotune

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"
	"time"
)

const (
	telemetryPID          = 1
	tidForegroundStream   = 1
	tidCUPTIPCSampling    = 2
	tidPTXASCompileLoop   = 3
	tidHoldoutHotSwapGate = 4
)

// ChromeTraceEvent represents a single event in the Chrome Trace / Perfetto
// JSON trace format.
type ChromeTraceEvent struct {
	Name string         `json:"name"`
	Cat  string         `json:"cat"`
	Ph   string         `json:"ph"`
	Ts   float64        `json:"ts"`
	Dur  float64        `json:"dur,omitempty"`
	Pid  int            `json:"pid"`
	Tid  int            `json:"tid"`
	S    string         `json:"s,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

// ChromeTraceDocument is the top-level JSON container consumed by
// chrome://tracing and Perfetto UI (ui.perfetto.dev).
type ChromeTraceDocument struct {
	TraceEvents     []ChromeTraceEvent `json:"traceEvents"`
	DisplayTimeUnit string             `json:"displayTimeUnit,omitempty"`
}

// FlightTelemetryRecorder records timestamped foreground megakernel stage
// spans, CUPTI PC-sampling stall attributions, zero-GPU ptxas compile-loop
// iterations, holdout gate / CUgraphExec hot-swap epochs, and speculative
// bucket transitions, exporting them as Chrome Trace / Perfetto JSON.
type FlightTelemetryRecorder struct {
	mu     sync.Mutex
	origin time.Time
	events []ChromeTraceEvent
}

// NewFlightTelemetryRecorder creates a new [FlightTelemetryRecorder].
func NewFlightTelemetryRecorder() *FlightTelemetryRecorder {
	return &FlightTelemetryRecorder{}
}

func (r *FlightTelemetryRecorder) relMicrosLocked(t time.Time) float64 {
	if t.IsZero() {
		t = time.Now().UTC()
	}
	if r.origin.IsZero() {
		r.origin = t
		return 0
	}
	return float64(t.Sub(r.origin).Nanoseconds()) / 1000.0
}

// RecordStageSpan records a complete duration event (ph: "X") for a foreground
// megakernel stage execution span.
func (r *FlightTelemetryRecorder) RecordStageSpan(
	stageName string,
	epoch uint64,
	start time.Time,
	dur time.Duration,
	blockBase, blocks int,
) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	ts := r.relMicrosLocked(start)
	durUS := max(float64(dur.Nanoseconds())/1000.0, 0.001)
	r.events = append(r.events, ChromeTraceEvent{
		Name: stageName,
		Cat:  "megakernel.stage",
		Ph:   "X",
		Ts:   ts,
		Dur:  durUS,
		Pid:  telemetryPID,
		Tid:  tidForegroundStream,
		Args: map[string]any{
			"epoch":      epoch,
			"block_base": blockBase,
			"blocks":     blocks,
		},
	})
}

// RecordPCStallAttribution records an instant event (ph: "i") attributing
// hardware CUPTI PC-sampling stalls (`stall_long_score`, `stall_barrier`,
// `stall_mio`) to stageName.
func (r *FlightTelemetryRecorder) RecordPCStallAttribution(
	stageName string,
	ts time.Time,
	stallLongScore, stallBarrier, stallMIO, totalSamples uint64,
) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	relTS := r.relMicrosLocked(ts)
	r.events = append(r.events, ChromeTraceEvent{
		Name: "pc_stall:" + stageName,
		Cat:  "cupti.pcsampling",
		Ph:   "i",
		Ts:   relTS,
		Pid:  telemetryPID,
		Tid:  tidCUPTIPCSampling,
		S:    "t",
		Args: map[string]any{
			"stage":            stageName,
			"stall_long_score": stallLongScore,
			"stall_barrier":    stallBarrier,
			"stall_mio":        stallMIO,
			"total_samples":    totalSamples,
		},
	})
}

// RecordPTXASIteration records a complete duration event (ph: "X") for one
// zero-GPU ptxas compile-loop iteration with register and spill byte counts.
func (r *FlightTelemetryRecorder) RecordPTXASIteration(
	candidateName string,
	iter int,
	start time.Time,
	dur time.Duration,
	regs, spillBytes int,
) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	ts := r.relMicrosLocked(start)
	durUS := max(float64(dur.Nanoseconds())/1000.0, 0.001)
	r.events = append(r.events, ChromeTraceEvent{
		Name: fmt.Sprintf("ptxas:%s:iter%d", candidateName, iter),
		Cat:  "ptxas.compile_loop",
		Ph:   "X",
		Ts:   ts,
		Dur:  durUS,
		Pid:  telemetryPID,
		Tid:  tidPTXASCompileLoop,
		Args: map[string]any{
			"candidate":   candidateName,
			"iter":        iter,
			"regs":        regs,
			"spill_bytes": spillBytes,
		},
	})
}

// RecordHoldoutAndHotSwap records a complete duration event (ph: "X") for
// held-out ABBA gate evaluation and an instant global marker (ph: "i") at the
// CUgraphExec hot-swap epoch boundary.
func (r *FlightTelemetryRecorder) RecordHoldoutAndHotSwap(
	candidateName string,
	epoch uint64,
	start time.Time,
	dur time.Duration,
	holdoutRatio, pValue float64,
	promoted bool,
) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	ts := r.relMicrosLocked(start)
	durUS := max(float64(dur.Nanoseconds())/1000.0, 0.001)
	r.events = append(r.events, ChromeTraceEvent{
		Name: "holdout_eval:" + candidateName,
		Cat:  "qualify.holdout",
		Ph:   "X",
		Ts:   ts,
		Dur:  durUS,
		Pid:  telemetryPID,
		Tid:  tidHoldoutHotSwapGate,
		Args: map[string]any{
			"candidate":     candidateName,
			"epoch":         epoch,
			"holdout_ratio": holdoutRatio,
			"p_value":       pValue,
			"promoted":      promoted,
		},
	})

	r.events = append(r.events, ChromeTraceEvent{
		Name: fmt.Sprintf("CUgraphExec_hotswap_epoch_%d", epoch),
		Cat:  "cudahotswap.epoch",
		Ph:   "i",
		Ts:   ts + durUS,
		Pid:  telemetryPID,
		Tid:  tidHoldoutHotSwapGate,
		S:    "g",
		Args: map[string]any{
			"candidate":     candidateName,
			"epoch":         epoch,
			"holdout_ratio": holdoutRatio,
			"p_value":       pValue,
			"promoted":      promoted,
		},
	})
}

// RecordBucketTransition records an instant marker (ph: "i") when the
// speculative / continuous-batching controller transitions between shape
// buckets (e.g. "M=1 -> M=4" or "M=4 -> M=8").
func (r *FlightTelemetryRecorder) RecordBucketTransition(
	ts time.Time,
	fromBucketM, toBucketM int,
	acceptanceEMA float64,
) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	relTS := r.relMicrosLocked(ts)
	label := fmt.Sprintf("M=%d -> M=%d", fromBucketM, toBucketM)
	r.events = append(r.events, ChromeTraceEvent{
		Name: label,
		Cat:  "specdecode.bucket",
		Ph:   "i",
		Ts:   relTS,
		Pid:  telemetryPID,
		Tid:  tidForegroundStream,
		S:    "g",
		Args: map[string]any{
			"from_m":         fromBucketM,
			"to_m":           toBucketM,
			"transition":     label,
			"acceptance_ema": acceptanceEMA,
		},
	})
}

// Events returns a snapshot of all recorded [ChromeTraceEvent] items.
func (r *FlightTelemetryRecorder) Events() []ChromeTraceEvent {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.events)
}

// ExportChromeTraceJSON serializes all recorded telemetry events as valid
// Chrome Trace / Perfetto JSON to w.
func (r *FlightTelemetryRecorder) ExportChromeTraceJSON(w io.Writer) error {
	if r == nil {
		return errors.New("export chrome trace json: nil recorder")
	}
	if w == nil {
		return errors.New("export chrome trace json: nil writer")
	}
	r.mu.Lock()
	doc := ChromeTraceDocument{
		TraceEvents:     slices.Clone(r.events),
		DisplayTimeUnit: "ns",
	}
	r.mu.Unlock()

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(doc)
}
