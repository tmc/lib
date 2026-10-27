package cudadiag

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"github.com/tmc/lib/nvidia/x/cudatracehost"
)

// ProbeReportOptions declares the context of a drained collector JSONL stream.
// Manifest supplies process, backend, monotonic clock and boot epoch. Other
// manifest fields do not establish coverage or authenticate this input.
// ThreadGeneration, ExecID, AddressSpace and ModuleGeneration are independent
// caller assertions; zero leaves kernel attribution unavailable. Capture
// AddressSpace is never filled from these assertions or from a binding.
//
// FirstSequence and LastSequence identify the inclusive captured-call range,
// not raw entry/scheduler line numbers. LossKnown and Dropped describe the
// caller's post-drain loss receipt; they cannot establish complete coverage.
// Every resulting report remains partial. Threshold selects host API
// entry-to-return durations, never GPU execution or pre-entry launch latency.
//
// All limits must be positive. MaxBytes is at most 64 MiB, MaxLineBytes at
// most 256 KiB, MaxRecords at most 65536 and MaxOutliers at most 4096.
// Bindings admits at most 256 entries and 16 MiB of artifact bytes in total.
// Demangler is optional; give it capacity for MaxOutliers. Its actual process
// is invoked only after the entire stream has been consumed and validated,
// with a finite context deadline. Bindings and the demangler must not be
// concurrently mutated or used during this call.
type ProbeReportOptions struct {
	Manifest                                                 cudatracehost.Manifest
	ThreadID                                                 uint32
	ThreadGeneration, ExecID, AddressSpace, ModuleGeneration uint64
	FirstSequence, LastSequence                              uint64
	LossKnown                                                bool
	Dropped                                                  uint64
	Gap                                                      bool
	Received                                                 time.Time
	Threshold                                                time.Duration
	MaxBytes                                                 int64
	MaxLineBytes, MaxRecords, MaxOutliers                    int
	Bindings                                                 []cudaprobe.Binding
	Demangler                                                *cudaprobe.Demangler
}

// ProbeOutlier is one observed long host API interval. Record retains the
// existing probe identity/binding/demangler labels. A missing KernelName is
// accompanied by kernel_identity_reason; a name establishes caller consistency
// only, not physical loading, successful execution or a GPU duration.
type ProbeOutlier struct {
	Sequence, EntryTimestamp, ReturnTimestamp, HostDurationNanos uint64
	Record                                                       cudaflightrecorder.Record
}

// ProbeReport describes a bounded post-drain input. RawSHA256 hashes the exact
// consumed JSONL bytes. Gap also includes sequence and pairing discontinuities.
// Partial is always true: received observations cannot prove complete capture.
type ProbeReport struct {
	RawSHA256      string
	Records        int
	Gap, Partial   bool
	Process        cudaprobe.Process
	ThreadID       uint32
	Clock, Epoch   string
	ThresholdNanos uint64
	Outliers       []ProbeOutlier
}

type probeReportCall struct {
	event cudaprobe.Event
	call  cudatracehost.CallInterval
}

// ReadProbeReport selects host API outliers from newline-terminated JSON
// cudatracehost.Result records, as emitted by CollectHostEvidence or the
// collector CLI. It uses Decode, RecordWithBindings/Identify and the supplied
// Demangler.Records rather than inferring a kernel from a function address.
// Invalid framing/context returns no report. A demangler failure returns the
// report with retained names/failure labels and an error. It never attaches a
// probe, loads an artifact, reads process mappings or calls a GPU API.
// The reader must supply finite post-drain input; context cancellation cannot
// interrupt a blocked call to an arbitrary Reader's Read method.
func ReadProbeReport(ctx context.Context, input io.Reader, o ProbeReportOptions) (*ProbeReport, error) {
	if err := probeReportOptions(o); err != nil {
		return nil, err
	}
	if input == nil {
		return nil, fmt.Errorf("probe report: missing JSONL reader")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r := &ProbeReport{Partial: true, Gap: o.Gap, Process: o.Manifest.Process, ThreadID: o.ThreadID, Clock: o.Manifest.Clock, Epoch: o.Manifest.Epoch, ThresholdNanos: uint64(o.Threshold), Outliers: []ProbeOutlier{}}
	hash := sha256.New()
	scanner := bufio.NewReaderSize(io.LimitReader(input, o.MaxBytes+1), o.MaxLineBytes+1)
	var bytesRead int64
	var pending []probeReportCall
	expected := o.FirstSequence
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		line, err := scanner.ReadSlice('\n')
		bytesRead += int64(len(line))
		if bytesRead > o.MaxBytes || len(line) > o.MaxLineBytes {
			return nil, fmt.Errorf("probe report: input byte limit")
		}
		if err == io.EOF && len(line) == 0 {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("probe report: truncated or excessive JSONL line: %w", err)
		}
		r.Records++
		if r.Records > o.MaxRecords {
			return nil, fmt.Errorf("probe report: record limit")
		}
		hash.Write(line)
		var result cudatracehost.Result
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		if err := d.Decode(&result); err != nil {
			return nil, fmt.Errorf("probe report: JSONL record %d: %w", r.Records, err)
		}
		if err := d.Decode(new(any)); err != io.EOF {
			return nil, fmt.Errorf("probe report: trailing JSON value")
		}
		if result.PairingStatus != "" || result.ThreadRescan != nil {
			r.Gap = true
		}
		if result.Native != nil && len(result.Native.Stack) > 8 {
			return nil, fmt.Errorf("probe report: excessive native stack")
		}
		if result.Capture == nil {
			if result.Call != nil {
				return nil, fmt.Errorf("probe report: call has no capture")
			}
			continue
		}
		// encoding/json otherwise silently zero-fills short fixed arrays.
		var wire struct{ Capture *struct{ Data []byte } }
		if err := json.Unmarshal(line, &wire); err != nil || wire.Capture == nil || len(wire.Capture.Data) != 128 {
			return nil, fmt.Errorf("probe report: incomplete capture data")
		}
		e, err := cudaprobe.Decode(*result.Capture)
		if err != nil {
			return nil, fmt.Errorf("probe report: decode: %w", err)
		}
		if e.Identity.Process != o.Manifest.Process || e.Sequence == 0 || e.Sequence == math.MaxUint64 {
			return nil, fmt.Errorf("probe report: capture process or sequence mismatch")
		}
		if e.Sequence != expected {
			r.Gap = true
		}
		expected = e.Sequence + 1
		for _, s := range []string{e.Identity.Library, e.Identity.BuildID, e.Identity.Version, e.Identity.Symbol, e.Identity.ABI, e.Identity.Provenance} {
			if len(s) > 4096 {
				return nil, fmt.Errorf("probe report: identity string limit")
			}
		}
		if result.Call == nil {
			r.Gap = true
			continue
		}
		c := *result.Call
		observed, err := observedHostCall(e.Identity.Symbol, &c, o.Manifest)
		if err != nil {
			return nil, err
		}
		if c.EntryTimestamp != e.Timestamp || c.TID != e.Identity.ThreadID || c.ThreadGeneration != e.Identity.ThreadGeneration || c.ThreadGenerationKnown != e.Identity.ThreadGenerationKnown {
			return nil, fmt.Errorf("probe report: call and capture disagree")
		}
		if !observed.HasReturn {
			r.Gap = true
			continue
		}
		if c.TID != o.ThreadID || c.ReturnTimestamp-c.EntryTimestamp < uint64(o.Threshold) {
			continue
		}
		if len(pending) == o.MaxOutliers {
			return nil, fmt.Errorf("probe report: outlier limit")
		}
		pending = append(pending, probeReportCall{e, c})
	}
	if expected != o.LastSequence+1 {
		r.Gap = true
	}
	r.RawSHA256 = hex.EncodeToString(hash.Sum(nil))
	var records []cudaflightrecorder.Record
	for _, p := range pending {
		record, err := probeReportRecord(p, o, r.Gap)
		if err != nil {
			return nil, err
		}
		records = append(records, record)
		r.Outliers = append(r.Outliers, ProbeOutlier{p.event.Sequence, p.call.EntryTimestamp, p.call.ReturnTimestamp, p.call.ReturnTimestamp - p.call.EntryTimestamp, record})
	}
	if o.Demangler != nil {
		for _, record := range records {
			if record.KernelName != "" {
				if _, ok := ctx.Deadline(); !ok {
					return r, fmt.Errorf("probe report: demangler requires finite deadline")
				}
				break
			}
		}
		out, err := o.Demangler.Records(ctx, records)
		if len(out) == len(records) {
			for i := range out {
				r.Outliers[i].Record = out[i]
			}
		} else if err != nil {
			for i := range r.Outliers {
				r.Outliers[i].Record.Labels["demangler.failure"] = err.Error()
			}
		}
		return r, err
	}
	return r, nil
}

func probeReportOptions(o ProbeReportOptions) error {
	if o.Manifest.Process.PID == 0 || o.Manifest.Process.Generation == 0 || o.Manifest.Clock != "linux_monotonic" || len(o.Manifest.Epoch) == 0 || len(o.Manifest.Epoch) > 256 || (o.Manifest.Backend != "bpf" && o.Manifest.Backend != "tracefs") || o.ThreadID == 0 || o.Received.IsZero() || o.Received.UnixNano() <= 0 || o.Threshold <= 0 {
		return fmt.Errorf("probe report: missing process, thread, clock or receipt context")
	}
	if !time.Unix(0, o.Received.UnixNano()).Equal(o.Received) {
		return fmt.Errorf("probe report: receipt time overflows nanoseconds")
	}
	if o.MaxBytes <= 0 || o.MaxBytes > 64<<20 || o.MaxLineBytes <= 0 || o.MaxLineBytes > 256<<10 || int64(o.MaxLineBytes) > o.MaxBytes || o.MaxRecords <= 0 || o.MaxRecords > 65536 || o.MaxOutliers <= 0 || o.MaxOutliers > 4096 || o.FirstSequence == 0 || o.LastSequence < o.FirstSequence || o.LastSequence == math.MaxUint64 || o.LastSequence-o.FirstSequence >= uint64(o.MaxRecords) || len(o.Bindings) > 256 {
		return fmt.Errorf("probe report: invalid bounds")
	}
	remaining := 16 << 20
	for _, b := range o.Bindings {
		if len(b.Evidence) > 4096 {
			return fmt.Errorf("probe report: binding evidence limit")
		}
		if b.Artifact != nil {
			n := b.Artifact.Size()
			if n < 0 || n > remaining {
				return fmt.Errorf("probe report: binding artifact limit")
			}
			remaining -= n
		}
	}
	return nil
}

func probeReportRecord(p probeReportCall, o ProbeReportOptions, gap bool) (cudaflightrecorder.Record, error) {
	r, err := p.event.Record(o.Received)
	if err != nil {
		return r, err
	}
	reason := ""
	switch {
	case gap || o.Dropped != 0:
		reason = "capture gap or dropped observations"
	case !o.LossKnown:
		reason = "post-drain loss unavailable"
	case !p.event.Identity.ThreadGenerationKnown || o.ThreadGeneration == 0 || p.call.ThreadGeneration != o.ThreadGeneration:
		reason = "thread lifetime unavailable or mismatched"
	case o.ExecID == 0 || p.call.ExecID != o.ExecID:
		reason = "execution lifetime unavailable or mismatched"
	case p.event.Identity.AddressSpace == 0:
		reason = "capture address space unavailable"
	case o.AddressSpace == 0 || p.event.Identity.AddressSpace != o.AddressSpace:
		reason = "address space context unavailable or mismatched"
	case o.ModuleGeneration == 0:
		reason = "module generation unavailable"
	default:
		joined, joinErr := p.event.RecordWithBindings(o.Received, o.Bindings)
		if joinErr != nil {
			reason = joinErr.Error()
		} else if joined.Labels["kernel.module_generation"] != strconv.FormatUint(o.ModuleGeneration, 10) {
			reason = "module generation mismatch"
		} else {
			r = joined
		}
	}
	if reason != "" {
		r.Labels["kernel_identity_reason"] = reason
	}
	r.Labels["kernel.binding_scope"] = "caller_consistency_only"
	r.Labels["probe.clock_epoch"] = o.Manifest.Epoch
	r.Labels["host_api.duration_ns"] = strconv.FormatUint(p.call.ReturnTimestamp-p.call.EntryTimestamp, 10)
	r.Labels["host_api.pairing_status"] = p.call.PairingStatus
	r.Labels["demangler.status"] = "unavailable"
	return r, nil
}
