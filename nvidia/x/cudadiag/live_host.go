package cudadiag

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/tmc/lib/ebpfhost"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudatracehost"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// HostEvidence retains an actual collector's host observations. Receipt.Partial
// remains true: late attachment and unquantified tracefs transport loss prevent
// exact totals and causal GPU attribution. HostCalls are API intervals, never
// GPU execution intervals. RawSHA256 hashes the JSON lines written to the sink.
type HostEvidence struct {
	Manifest  cudatracehost.Manifest
	Process   cudaworkload.ProcessGenID
	TID       uint64
	RawSHA256 string
	Records   int
	HostCalls []HostCallEvent
	Scheduler SchedEvidence
	Receipt   tracecontract.Receipt
	Findings  []tracecontract.Finding
}

// CollectHostEvidence attaches real CUDA API and scheduler probes for cfg.PID.
// A finite context deadline is required. Raw bytes are capped at 64 MiB. The optional
// running flight recorder receives decoded host observations with explicit raw
// clock labels; it does not receive fabricated kernel names or GPU durations.
// Library permissions/ABI failures are returned, never replaced by fixtures.
func CollectHostEvidence(ctx context.Context, cfg cudatracehost.Config, tid uint64, maxRecords int, raw io.Writer, recorder *cudaflightrecorder.Recorder) (out HostEvidence, err error) {
	if raw == nil || tid == 0 || maxRecords <= 0 || maxRecords > 65536 || len(cfg.Symbols) == 0 {
		return out, fmt.Errorf("host evidence: invalid sink, thread or record limit")
	}
	if _, ok := ctx.Deadline(); !ok {
		return out, fmt.Errorf("host evidence: finite capture deadline required")
	}
	if e := ctx.Err(); e != nil {
		return out, e
	}
	cfg.Scheduler = true
	collector, err := cudatracehost.Open(cfg)
	if err != nil {
		return out, err
	}
	defer func() { err = errors.Join(err, collector.Close()) }()
	generation, err := ebpfhost.ProcessStartTime(cfg.PID)
	if err != nil {
		return out, err
	}
	if _, err := os.Stat(fmt.Sprintf("/proc/%d/task/%d", cfg.PID, tid)); err != nil {
		return out, fmt.Errorf("host evidence: target thread membership: %w", err)
	}
	out.Process = cudaworkload.ProcessGenID{PID: uint32(cfg.PID), Generation: generation}
	out.TID = tid
	out.Manifest = collector.Manifest()
	if out.Manifest.Process.PID != out.Process.PID || out.Manifest.Process.Generation != out.Process.Generation {
		return out, fmt.Errorf("host evidence: attachment process generation mismatch")
	}
	hash := sha256.New()
	encoder := json.NewEncoder(io.MultiWriter(&hostRawWriter{writer: raw}, hash))
	var sched []SchedulerEvent
	var warnings []string
	for out.Records < maxRecords {
		result, nextErr := collector.Next(ctx)
		if result.Raw != "" {
			if e := encoder.Encode(result); e != nil {
				return out, fmt.Errorf("host evidence: write raw record: %w", e)
			}
			out.Records++
		}
		if nextErr != nil {
			err = nextErr
			break
		}
		if result.Scheduler {
			ev, e := cudatracehost.ParseScheduler(result.Raw)
			if e != nil || ev.Timestamp > math.MaxInt64 {
				if e == nil {
					e = fmt.Errorf("scheduler timestamp overflows signed interval")
				}
				warnings = append(warnings, e.Error())
				continue
			}
			// Events from other threads remain raw but cannot alter target intervals.
			if uint64(ev.PreviousTID) == tid || uint64(ev.NextTID) == tid || uint64(ev.WakeTID) == tid {
				sched = append(sched, SchedulerEvent{Kind: ev.Kind, Timestamp: int64(ev.Timestamp), CPU: int(ev.CPU), PreviousTID: uint64(ev.PreviousTID), NextTID: uint64(ev.NextTID), WakeTID: uint64(ev.WakeTID), PreviousState: ev.PrevState})
			}
			continue
		}
		if result.Capture == nil {
			continue
		}
		event, e := cudaprobe.Decode(*result.Capture)
		if e != nil {
			warnings = append(warnings, e.Error())
			continue
		}
		if event.Identity.Process.PID != out.Process.PID || event.Identity.Process.Generation != out.Process.Generation {
			return out, fmt.Errorf("host evidence: process generation mismatch")
		}
		if result.Call != nil && uint64(result.Call.TID) == tid {
			call, e := observedHostCall(event.Identity.Symbol, result.Call, out.Manifest)
			if e != nil {
				return out, e
			}
			out.HostCalls = append(out.HostCalls, call)
		}
		if recorder != nil {
			record, e := event.Record(time.Now())
			if e != nil {
				return out, e
			}
			record.Labels["probe.clock"] = out.Manifest.Clock
			record.Labels["probe.clock_epoch"] = out.Manifest.Epoch
			record.Labels["probe.backend"] = out.Manifest.Backend
			record.Labels["probe.exact_gpu_join"] = "unavailable"
			recorder.Push(record)
		}
	}
	if out.Records == maxRecords {
		err = errors.Join(err, fmt.Errorf("host evidence: record limit reached"))
	}
	out.RawSHA256 = hex.EncodeToString(hash.Sum(nil))
	var e error
	out.Scheduler, e = SchedulerEvidence(out.Process, tid, "linux_monotonic", sched, 0, true)
	if e != nil {
		warnings = append(warnings, e.Error())
		err = errors.Join(err, e)
	}
	out.Receipt = hostReceipt(out, warnings)
	if named, ok := raw.(interface{ Name() string }); ok && named.Name() != "" {
		out.Receipt.InputPath = named.Name()
	}
	value := float64(out.Records)
	out.Findings = []tracecontract.Finding{tracecontract.Admit("received-host-records", "received_host_records", tracecontract.FindingObserved, "raw transport records retained; not a complete API or GPU count", &value, "received_records", []tracecontract.Requirement{{Name: "raw_artifact", ArtifactDigest: out.RawSHA256}}, tracecontract.EvidenceContext{Receipt: out.Receipt, Artifacts: []string{out.RawSHA256}})}
	return out, err
}

func hostReceipt(out HostEvidence, warnings []string) tracecontract.Receipt {
	gates := map[string]tracecontract.Gate{}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		gates[name] = tracecontract.Gate{Disposition: "unavailable", Reason: "host tracefs observations do not establish native GPU qualification", Provenance: "actual host collector"}
	}
	gates["collector"] = tracecontract.Gate{Disposition: "unavailable", Reason: "late attach and transport loss unquantified", Provenance: "linux tracefs"}
	gates["import"] = tracecontract.Gate{Disposition: "passed", Reason: "raw bytes retained and hashed", Provenance: "actual host collector"}
	return tracecontract.Receipt{Version: 1, Command: "CollectHostEvidence", InputPath: "caller raw JSONL sink", InputSHA256: out.RawSHA256, Kind: "capture", Backends: []tracecontract.Backend{{Name: "linux_" + out.Manifest.Backend + "_uprobe", Version: "host-abi-v1"}}, Gates: gates, Warnings: warnings, Partial: true, ClockDomain: "linux_monotonic", ClockProvenance: out.Manifest.Clock + " epoch=" + out.Manifest.Epoch, LostRecords: map[string]tracecontract.Loss{"call": {Unknown: true}, "sched": {Unknown: true}}, ProbeCoverage: map[string]tracecontract.ProbeCoverage{"sched": {Present: true}, "cuda_api": {Present: true}}}
}

// hostRawWriter also bounds raw bytes, independently of event count.
type hostRawWriter struct {
	writer  io.Writer
	written int64
}

func (w *hostRawWriter) Write(p []byte) (int, error) {
	const maxBytes = 64 << 20
	if int64(len(p)) > maxBytes-w.written {
		return 0, fmt.Errorf("host evidence: raw byte limit reached")
	}
	n, err := w.writer.Write(p)
	w.written += int64(n)
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	return n, err
}

// observedHostCall accepts structured transport timing without interpreting raw
// BPF JSON as tracefs text. Unverified thread lifetimes retain candidate spans
// but cannot satisfy diagnosis return-timing admission.
func observedHostCall(name string, call *cudatracehost.CallInterval, manifest cudatracehost.Manifest) (HostCallEvent, error) {
	if call == nil || call.Process != manifest.Process || call.Clock != manifest.Clock || call.Epoch != manifest.Epoch || call.Clock != "linux_monotonic" || call.Epoch == "" || call.ReturnTimestamp < call.EntryTimestamp || call.ReturnTimestamp > math.MaxInt64 {
		return HostCallEvent{}, fmt.Errorf("host evidence: invalid structured call identity or clock")
	}
	verified := call.PairingStatus == "verified_thread_lifetime" && call.ThreadGenerationKnown
	if call.PairingStatus != "unverified_thread_lifetime" && !verified {
		return HostCallEvent{}, fmt.Errorf("host evidence: contradictory thread pairing status")
	}
	return HostCallEvent{Name: name, HasReturn: verified, Span: tracecontract.Interval{Start: int64(call.EntryTimestamp), End: int64(call.ReturnTimestamp), Clock: call.Clock}}, nil
}
