package cudadiagdemo

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// Native request recipe and timing bounds.
const (
	// nativeRecipe orders every attempt: open its owners, enqueue its
	// upload, observe launch-ready, wait (baseline only), then launch. All
	// arms of round r upload nativeInput(elements, r-1).
	nativeRecipe     = "launch-ready-v1"
	nativeTimingName = "native-timing.jsonl"
	// nativeMaxTimingBytes bounds the encoded timing artifact.
	nativeMaxTimingBytes = 32 << 20
	// nativeMaxTimingCalls bounds the planned wrapper observations.
	nativeMaxTimingCalls = 65536
)

// errNativeTimingBound reports a planned wrapper population or an encoded
// timing artifact above its bound.
var errNativeTimingBound = errors.New("cudadiagdemo: native timing exceeds its bound")

// NativeSample is one raw CUPTI timestamp read bracketed by host times in the
// process monotonic epoch, taken in order: HostBeforeNS, the read,
// HostAfterNS. CUPTI is "observed" with the value in CUPTINS, "zero" when the
// read returned zero, or "error" with Error; only an observed sample has a
// CUPTI value. A sample is a raw bracket, not a host/CUPTI clock mapping, an
// uncertainty bound or the time of a CUDA API entry.
type NativeSample struct {
	HostBeforeNS int64  `json:"host_before_ns"`
	HostAfterNS  int64  `json:"host_after_ns"`
	CUPTI        string `json:"cupti"`
	CUPTINS      uint64 `json:"cupti_ns,omitempty"`
	Error        string `json:"error,omitempty"`
}

// NativeReady is an attempt's host launch-ready observation: its input is
// constructed, its stream and buffers exist and its upload is enqueued on
// that stream, so its next application step is the first targeted launch,
// after the baseline's intentional wait. It does not mean the upload has
// completed on the device. HostNS is in the process monotonic epoch; CUPTI
// is a raw sample taken after it when a timestamp source exists.
type NativeReady struct {
	HostNS int64         `json:"host_ns"`
	CUPTI  *NativeSample `json:"cupti,omitempty"`
}

// NativeTiming is the raw timing record of one dispatched request attempt
// under Recipe. A trial without one was not observed.
//
// OpID and Workload identify the attempt's application operation, created
// after its upload and before Ready; both are absent without capture. TID is
// the request worker's thread observed at Ready, zero where the platform
// reports none. It is not CUDA API entry: the trial's Correlation and
// annotation record entry. Ready is nil when the attempt stopped before
// launch-ready, as after a failed open or upload.
//
// TimestampAPI names the CUPTI timestamp function sampled at Ready and around
// each of the trial's Calls, or is "unavailable". TimestampMethod is
// "unknown": neither the loaded library nor its timestamp method is
// authenticated. These observations establish no clock mapping, API entry
// time, scheduler evidence or diagnosis.
type NativeTiming struct {
	Recipe          string                         `json:"recipe"`
	OpID            cudaworkload.OpID              `json:"op_id,omitempty"`
	Workload        *cudaworkload.WorkloadIdentity `json:"workload,omitempty"`
	TID             uint64                         `json:"tid,omitempty"`
	TimestampAPI    string                         `json:"timestamp_api"`
	TimestampMethod string                         `json:"timestamp_method,omitempty"`
	Ready           *NativeReady                   `json:"ready,omitempty"`
}

// nativeTimingCalls returns the wrapper observations that rounds of planned
// attempts record with deviceIterations device launches: an upload,
// synchronize and download per attempt plus every planned launch, which is
// rounds*(deviceIterations+11). It refuses a population above
// nativeMaxTimingCalls rather than dropping any of it.
func nativeTimingCalls(rounds, deviceIterations int) (int, error) {
	if rounds < 0 || deviceIterations < 0 || rounds > nativeMaxTimingCalls || deviceIterations > nativeMaxTimingCalls {
		return 0, fmt.Errorf("%w: invalid population", errNativeTimingBound)
	}
	n := int64(rounds) * (int64(deviceIterations) + 11)
	if n > nativeMaxTimingCalls {
		return 0, fmt.Errorf("%w: %d planned wrapper observations exceed %d", errNativeTimingBound, n, nativeMaxTimingCalls)
	}
	return int(n), nil
}

// sample reads e.timestamp between two host times. A failed or zero read is
// recorded as such, never as a value.
func (e *nativeExecutor) sample() *NativeSample {
	s := &NativeSample{HostBeforeNS: e.since()}
	ts, err := e.timestamp()
	s.HostAfterNS = e.since()
	switch {
	case err != nil:
		s.CUPTI, s.Error = "error", err.Error()
	case ts == 0:
		s.CUPTI = "zero"
	default:
		s.CUPTI, s.CUPTINS = "observed", ts
	}
	return s
}

// observeReady records t's launch-ready point, once the attempt's upload is
// enqueued. A submission probe of scope supplies only the worker's thread ID:
// its wall-clock timestamp is not ready evidence and it is not retained as
// an annotation.
func (e *nativeExecutor) observeReady(t *NativeTiming, scope *cudaworkload.Scope) {
	if scope != nil {
		t.TID = scope.NewSubmission(0, cudaworkload.ContextStreamID{}, cudaworkload.GraphReplayGeneration{}, "launch-ready", e.evidence).TID
	}
	t.Ready = &NativeReady{HostNS: e.since()}
	if e.timestamp != nil {
		t.Ready.CUPTI = e.sample()
	}
}

// nativeTimingLine is one line of native-timing.jsonl: a dispatched attempt's
// identity, status, timing and every wrapper it called, in call order.
type nativeTimingLine struct {
	HostClockDomain  string        `json:"host_clock_domain"`
	HostClockEpochID string        `json:"host_clock_epoch_id"`
	ActivityRun      string        `json:"activity_run,omitempty"`
	RequestID        string        `json:"request_id"`
	Attempt          string        `json:"attempt"`
	Round            int           `json:"round"`
	Worker           int           `json:"worker"`
	Correlation      string        `json:"correlation,omitempty"`
	KernelLaunches   int           `json:"kernel_launches"`
	Error            string        `json:"error,omitempty"`
	Timing           *NativeTiming `json:"timing"`
	Calls            []NativeCall  `json:"calls"`
}

// encodeNativeTiming returns one JSON line for each trial of r, in plan
// order. It refuses more than limit bytes rather than truncating.
func encodeNativeTiming(r *NativeResult, limit int) ([]byte, error) {
	var b []byte
	for _, t := range r.Trials {
		line, err := json.Marshal(nativeTimingLine{
			HostClockDomain:  r.HostClockDomain,
			HostClockEpochID: r.HostClockEpochID,
			ActivityRun:      r.ActivityRun,
			RequestID:        t.RequestID,
			Attempt:          t.Attempt,
			Round:            t.Round,
			Worker:           t.Worker,
			Correlation:      t.Correlation,
			KernelLaunches:   t.KernelLaunches,
			Error:            t.Error,
			Timing:           t.Timing,
			Calls:            t.Calls,
		})
		if err != nil {
			return nil, err
		}
		if len(line)+1 > limit-len(b) {
			return nil, fmt.Errorf("%w: encoded timing exceeds %d bytes", errNativeTimingBound, limit)
		}
		b = append(append(b, line...), '\n')
	}
	return b, nil
}

// retainNativeTiming writes native-timing.jsonl to dir, one line per trial of
// r, and binds its exact size and digest in r.Timing; the file never holds
// its own digest. An encoding, bound, write or close failure leaves r.Timing
// nil, records the failure in r.TimingError and adds a remaining gate: a
// file that may be incomplete is never referenced. Without trials it writes
// nothing.
func retainNativeTiming(dir string, r *NativeResult, create func(path string) (io.WriteCloser, error)) error {
	r.Timing, r.TimingError = nil, ""
	if len(r.Trials) == 0 {
		return nil
	}
	path := filepath.Join(dir, nativeTimingName)
	data, err := encodeNativeTiming(r, nativeMaxTimingBytes)
	if err == nil {
		err = writeNativeFile(create, path, data)
	}
	if err != nil {
		err = fmt.Errorf("retain %s: %w", nativeTimingName, err)
		r.TimingError = err.Error()
		addNativeGate(r, "complete native timing retention")
		return err
	}
	r.Timing = &cudaregress.ArtifactRef{Name: nativeTimingName, Path: path, Digest: digestBytes(data), SizeBytes: int64(len(data)), Role: "trace"}
	return nil
}
