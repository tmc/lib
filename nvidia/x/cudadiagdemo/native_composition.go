package cudadiagdemo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// NativeRequestLog records application request controls independently of the
// diagnostic observations. Attempt distinguishes repetitions of a request ID.
// GroundTruth uses the existing service log format; its mode and durations do
// not generate diagnostic intervals, gates, clock maps or scheduler states.
// Incomplete requests remain in the population, including failed/cancelled work.
type NativeRequestLog struct {
	Attempt     string         `json:"attempt"`
	StartedNS   int64          `json:"started_ns"`
	FinishedNS  int64          `json:"finished_ns"`
	Complete    bool           `json:"complete"`
	GroundTruth GroundTruthLog `json:"ground_truth"`
}

// NativeEvidenceRecord associates supplied diagnostic observations with one
// independently logged operation. Phase is pre_entry or api_scheduler and must
// agree with Input; it does not come from an injected ground-truth mode.
// Input retains existing receipt, raw artifact, clock, host/device and scheduler
// contracts. Its raw source artifacts must remain separately retained.
type NativeEvidenceRecord struct {
	RequestID string                   `json:"request_id"`
	Attempt   string                   `json:"attempt"`
	OpID      cudaworkload.OpID        `json:"op_id"`
	Phase     string                   `json:"phase"`
	Input     cudadiag.DiagnosticInput `json:"input"`
}

// NativeEvidence is the bounded version 1 composition envelope. Synthetic is
// explicit, must match the request producer, and cannot qualify native work.
type NativeEvidence struct {
	Version   int                    `json:"version"`
	Synthetic bool                   `json:"synthetic"`
	Records   []NativeEvidenceRecord `json:"records"`
}

// NativeReplay selects one admitted host-delay request for the existing plan
// and replay protocol. The caller owns the runner and immutable artifact closure;
// no alternate runner, training workload or device collector is constructed.
type NativeReplay struct {
	RequestID, Attempt string
	Capture            cudaregress.ArtifactRef
	Environment        cudaregress.Environment
	PlanOptions        cudaregress.PlanOptions
	Runner             cudaregress.ReplayRunner
	ReplayOptions      cudaregress.ReplayOptions
}

// NativeCompositionOptions supplies original evidence and optional existing
// replay closure. Zero limits mean 16MiB and 512 requests; hard caps are 32MiB
// and 4096 requests. Evidence nil preserves missing observations as withheld.
// Readers and replay resources remain caller-owned; cancellation cannot interrupt
// a caller's blocking Read. RunNative invokes this composition after cleanup.
type NativeCompositionOptions struct {
	Evidence    io.Reader
	Replay      *NativeReplay
	MaxBytes    int64
	MaxRequests int
}

// NativeRequestDiagnosis retains one logged attempt and its independent finding.
// Evidence absence/refusal cannot remove that attempt from the population.
type NativeRequestDiagnosis struct {
	RequestID    string                   `json:"request_id"`
	Attempt      string                   `json:"attempt"`
	OpID         cudaworkload.OpID        `json:"op_id"`
	Observations cudadiag.DiagnosticInput `json:"observations"`
	Diagnosis    cudadiag.DiagnosisResult `json:"diagnosis"`
	Refusal      string                   `json:"refusal,omitempty"`
}

// NativeComposition retains actual diagnosis -> incident -> plan -> replay
// results. Synthetic host composition remains distinct from native qualification.
// EvidenceSHA256 binds the supplied envelope, not authenticity of its assertions.
type NativeComposition struct {
	Synthetic      bool                           `json:"synthetic"`
	Qualification  string                         `json:"qualification"`
	EvidenceSHA256 string                         `json:"evidence_sha256,omitempty"`
	Requests       []NativeRequestDiagnosis       `json:"requests"`
	Incident       *cudaregress.Incident          `json:"incident,omitempty"`
	Plan           *cudaregress.ExperimentPlan    `json:"plan,omitempty"`
	PlanReceipt    *cudaregress.PlanReceipt       `json:"plan_receipt,omitempty"`
	Execution      *cudaregress.ExecutionReceipt  `json:"execution,omitempty"`
	Comparison     *cudaregress.ComparisonReceipt `json:"comparison,omitempty"`
	ReplayRefusal  string                         `json:"replay_refusal,omitempty"`
}

// ComposeNativeEvidence joins supplied observations to independently logged
// request/attempt/operation identities, invokes the real public diagnoser, and
// optionally calls FormulatePlan and ExecuteReplay. Ground truth never supplies
// a missing diagnostic gate or measurement. Current RunNative capture is partial:
// this consumer cannot turn it into admitted native evidence or replay.
func ComposeNativeEvidence(ctx context.Context, r *NativeResult, o NativeCompositionOptions) (NativeComposition, error) {
	var out NativeComposition
	if ctx == nil || r == nil {
		return out, errors.New("native composition: missing context or result")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if o.MaxBytes == 0 {
		o.MaxBytes = 16 << 20
	}
	if o.MaxRequests == 0 {
		o.MaxRequests = 512
	}
	if o.MaxBytes < 1 || o.MaxBytes > 32<<20 || o.MaxRequests < 1 || o.MaxRequests > 4096 || len(r.Requests) > o.MaxRequests {
		return out, errors.New("native composition: request or byte bound exceeded")
	}
	out.Synthetic = r.Synthetic
	out.Qualification = "partial native observations; admission unavailable"
	if r.Synthetic {
		out.Qualification = "supplied synthetic host composition; not native qualification"
	}
	type key struct{ request, attempt string }
	expected := map[key]NativeRequestLog{}
	type operationKey struct {
		process cudaworkload.ProcessGenID
		op      cudaworkload.OpID
	}
	operations := map[operationKey]bool{}
	for _, log := range r.Requests {
		g := log.GroundTruth
		k := key{g.RequestID, log.Attempt}
		if k.request == "" || k.attempt == "" || len(k.request) > 256 || len(k.attempt) > 256 || len(g.WorkloadName) > 256 || len(g.Workload.WorkloadName) > 256 || len(g.Workload.RunID) > 256 || len(g.Workload.Labels) > 32 || log.StartedNS < 0 || log.FinishedNS < log.StartedNS {
			return out, errors.New("native composition: invalid request identity or interval")
		}
		for k, v := range g.Workload.Labels {
			if len(k) > 256 || len(v) > 256 {
				return out, errors.New("native composition: workload label bound exceeded")
			}
		}
		if _, ok := expected[k]; ok {
			return out, errors.New("native composition: duplicate request attempt")
		}
		if g.OpID != 0 && !g.ProcessGen.IsZero() {
			operation := operationKey{g.ProcessGen, g.OpID}
			if operations[operation] {
				return out, errors.New("native composition: operation assigned to multiple request attempts")
			}
			operations[operation] = true
		}
		expected[k] = log
	}
	evidence := map[key]NativeEvidenceRecord{}
	if o.Evidence != nil {
		raw, err := io.ReadAll(io.LimitReader(o.Evidence, o.MaxBytes+1))
		if err != nil {
			return out, err
		}
		if int64(len(raw)) > o.MaxBytes {
			return out, errors.New("native composition: evidence byte bound exceeded")
		}
		out.EvidenceSHA256 = digestBytes(raw)
		if err := nativeCompositionJSON(raw); err != nil {
			return out, err
		}
		var packet NativeEvidence
		d := json.NewDecoder(bytes.NewReader(raw))
		d.DisallowUnknownFields()
		if err := d.Decode(&packet); err != nil {
			return out, fmt.Errorf("native composition: %w", err)
		}
		if packet.Version != 1 || packet.Synthetic != r.Synthetic || packet.Records == nil || len(packet.Records) > o.MaxRequests {
			return out, errors.New("native composition: evidence version, scope or population mismatch")
		}
		for _, record := range packet.Records {
			if err := ctx.Err(); err != nil {
				return out, err
			}
			k := key{record.RequestID, record.Attempt}
			log, ok := expected[k]
			if !ok {
				return out, errors.New("native composition: unexpected request attempt")
			}
			if _, ok := evidence[k]; ok {
				return out, errors.New("native composition: duplicate request evidence")
			}
			g := log.GroundTruth
			if record.OpID == 0 || record.OpID != g.OpID || g.ProcessGen.IsZero() || g.ProcessGen != g.Workload.Process {
				return out, errors.New("native composition: operation/process identity mismatch")
			}
			if record.Phase == "pre_entry" {
				if len(record.Input.PreEntries) != 1 {
					return out, errors.New("native composition: pre-entry phase evidence missing")
				}
				sub := record.Input.PreEntries[0].Submission
				if sub.OpID != g.OpID || sub.Process != g.ProcessGen || !sameTimingWorkload(sub.Workload, g.Workload) || sub.TID != g.TID {
					return out, errors.New("native composition: observed request/thread identity mismatch")
				}
				source := "native"
				if r.Synthetic {
					source = "synthetic"
				}
				if record.Input.PreEntries[0].SourceClass != source {
					return out, errors.New("native composition: phase source class mismatch")
				}
			} else if record.Phase != "api_scheduler" || len(record.Input.PreEntries) != 0 {
				return out, errors.New("native composition: wrong diagnostic phase")
			} else if record.Input.Sched.Process != g.ProcessGen || record.Input.Sched.TID != g.TID || g.TID == 0 || record.Input.Thresholds.TargetTID != 0 && record.Input.Thresholds.TargetTID != g.TID {
				return out, errors.New("native composition: scheduler target thread mismatch")
			}
			if len(record.Input.HostCalls) > 4096 || len(record.Input.Kernels) > 4096 || len(record.Input.Sched.Intervals) > 4096 {
				return out, errors.New("native composition: observation count exceeded")
			}
			evidence[k] = record
		}
	}
	for _, log := range r.Requests {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		g := log.GroundTruth
		record, ok := evidence[key{g.RequestID, log.Attempt}]
		row := NativeRequestDiagnosis{RequestID: g.RequestID, Attempt: log.Attempt, OpID: g.OpID}
		var input cudadiag.DiagnosticInput
		if !ok && !r.Synthetic {
			var err error
			input, err = nativeRequestObservations(r, log, o.MaxRequests)
			if err != nil {
				return out, err
			}
		}
		if ok {
			input = record.Input
		} else {
			row.Refusal = "request observations unavailable"
		}
		if !log.Complete {
			row.Refusal = "request did not complete"
		}
		if !r.Synthetic {
			row.Refusal = "native host/device clock mapping, scheduler/loss and independent baseline qualification unavailable"
		}
		row.Observations = input
		if row.Refusal != "" {
			// The current native producer has no qualified host/CUPTI mapping;
			// an incomplete request likewise cannot inherit a clock admission.
			// Preserve original facts above; never promote a supplied bound.
			input.Receipt.ClockUncertaintyNS = nil
		}
		target := g.Workload
		target.Labels = make(map[string]string, len(g.Workload.Labels))
		for k, v := range g.Workload.Labels {
			target.Labels[k] = v
		}
		row.Diagnosis = cudadiag.Diagnose(target, g.OpID, input)
		out.Requests = append(out.Requests, row)
	}
	if o.Replay == nil {
		return out, nil
	}
	replay := o.Replay
	var selected *NativeRequestDiagnosis
	for i := range out.Requests {
		row := &out.Requests[i]
		if row.RequestID == replay.RequestID && row.Attempt == replay.Attempt {
			selected = row
			break
		}
	}
	if selected == nil || !selected.Diagnosis.Admitted || selected.Diagnosis.Mechanism != cudadiag.MechanismHostSubmissionDelay {
		out.ReplayRefusal = "selected request lacks an admitted host-delay finding"
		return out, nil
	}
	if replay.Runner == nil {
		return out, errors.New("native composition: missing existing replay runner")
	}
	// Require the closed source artifact to be the supplied original envelope.
	if replay.Capture.Digest != out.EvidenceSHA256 {
		return out, errors.New("native composition: replay evidence digest mismatch")
	}
	digest, size, err := cudaregress.ComputeFileDigest(replay.Capture.Path)
	if err != nil {
		return out, err
	}
	if digest != out.EvidenceSHA256 || size != replay.Capture.SizeBytes {
		return out, errors.New("native composition: replay evidence file mismatch")
	}
	diagnosis := selected.Diagnosis
	incident := cudaregress.Incident{ID: "request-" + out.EvidenceSHA256 + "-" + selected.Attempt, Admitted: diagnosis.Admitted, Diagnosis: diagnosis, PrimaryFinding: diagnosis.PrimaryFinding, Workload: diagnosis.Workload, CaptureReceiptRef: replay.Capture, ObservedEnvironment: replay.Environment, CreatedAt: time.Now()}
	out.Incident = &incident
	out.Plan, out.PlanReceipt, err = FormulatePlan(incident, replay.PlanOptions)
	if err != nil {
		return out, err
	}
	out.Execution, out.Comparison, err = ExecuteReplay(ctx, out.Plan, replay.Runner, replay.ReplayOptions)
	return out, err
}

// Refuse duplicate keys and bound JSON nesting before decoding the fixed wire
// format. No data field is interpreted as a path, command or source receipt.
func nativeCompositionJSON(raw []byte) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var value func(int) error
	value = func(depth int) error {
		if depth > 64 {
			return errors.New("native composition: JSON nesting exceeded")
		}
		token, err := d.Token()
		if err != nil {
			return err
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := map[string]bool{}
			for d.More() {
				key, err := d.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok || seen[name] {
					return errors.New("native composition: duplicate JSON field")
				}
				seen[name] = true
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		case '[':
			for d.More() {
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		default:
			return errors.New("native composition: unexpected JSON delimiter")
		}
		_, err = d.Token()
		return err
	}
	if err := value(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("native composition: trailing evidence JSON")
	}
	return nil
}

// Preserve actual wrapper observations from the native producer. CUPTI kernel
// times stay in Activity/ActivityAnalysis: no conversion into the host epoch is
// possible here. The receipt records unavailable gates, never successful ones.
func nativeRequestObservations(r *NativeResult, log NativeRequestLog, max int) (cudadiag.DiagnosticInput, error) {
	var input cudadiag.DiagnosticInput
	if len(r.Trials) > max || len(r.HostClockDomain) > 256 || len(r.HostClockEpochID) > 256 {
		return input, errors.New("native composition: native observation bounds exceeded")
	}
	for _, trial := range r.Trials {
		if trial.RequestID != log.GroundTruth.RequestID || trial.Attempt != log.Attempt {
			continue
		}
		if len(trial.Calls) > 4096 {
			return input, errors.New("native composition: wrapper count exceeded")
		}
		for _, call := range trial.Calls {
			if len(call.Name) > 256 || len(call.Error) > 4096 || call.StartNS < 0 || call.EndNS < call.StartNS {
				return input, errors.New("native composition: invalid wrapper observation")
			}
		}
		raw, err := json.Marshal(trial)
		if err != nil {
			return input, err
		}
		clock := r.HostClockDomain
		input.WallInterval = tracecontract.Interval{Start: log.StartedNS, End: log.FinishedNS, Clock: clock}
		input.Receipt = tracecontract.Receipt{Version: 1, Command: "RunNative request wrapper observations", InputPath: "native-receipt.json#trials", InputSHA256: digestBytes(raw), Kind: "capture", Partial: true, ClockDomain: clock, ClockProvenance: "time.Since process epoch " + r.HostClockEpochID + "; CUPTI mapping unavailable", Backends: []tracecontract.Backend{{Name: "native application wrappers"}}, Gates: map[string]tracecontract.Gate{}}
		for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
			input.Receipt.Gates[name] = tracecontract.Gate{Disposition: "unavailable", Reason: "native wrapper facts do not establish diagnosis qualification", Provenance: "RunNative partial producer"}
		}
		for _, call := range trial.Calls {
			input.HostCalls = append(input.HostCalls, cudadiag.HostCallEvent{Name: call.Name, Span: tracecontract.Interval{Start: call.StartNS, End: call.EndNS, Clock: clock}, HasReturn: true})
		}
		return input, nil
	}
	return input, nil
}
