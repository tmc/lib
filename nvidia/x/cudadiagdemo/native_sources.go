package cudadiagdemo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// NativeActivitySource is one caller-retained raw activity export in the only
// supported format: cuptiactivity.Record JSON lines, as RunNative writes
// activity.jsonl. Ref declares the exact bytes with Role "trace"; Reader
// supplies them and remains caller-owned. Neither Ref.Path nor any payload
// string is opened. Batch declares the producer session, run, clock epoch and
// process, with ExternalKind cupti.ExternalCorrelationKindCustom0. Its Records
// must be empty and its EvidenceRef empty or Ref.Digest; MaxRecords keeps the
// reader's meaning. The result's Annotations are joined to the records.
//
// Composition reads Reader synchronously, checking its context before each
// Read; it cannot interrupt a blocking Read. A matching length and SHA-256 show
// only that these are the bytes read, not their provenance or authenticity.
type NativeActivitySource struct {
	Ref    cudaregress.ArtifactRef
	Reader io.Reader
	Batch  cudaworkload.ActivityBatch
}

// bindNativeActivity verifies the declared raw activity bytes against the
// result and its logged request population, then parses them with the
// existing reader. Annotation timestamps keep their declared post-call clock;
// no record is selected, dropped, relabelled or converted to the host epoch.
func bindNativeActivity(ctx context.Context, r *NativeResult, s NativeActivitySource, maxBytes int64, envelope string) (cudaregress.ArtifactRef, cudadiag.ActivityAnalysis, error) {
	var none cudadiag.ActivityAnalysis
	ref, batch := s.Ref, s.Batch
	if s.Reader == nil {
		return ref, none, errors.New("native composition: missing raw activity reader")
	}
	if ref.Name == "" || len(ref.Name) > 256 || len(ref.Path) > 4096 || ref.Role != "trace" || !nativeDigest(ref.Digest) || ref.SizeBytes < 1 || ref.SizeBytes > maxBytes {
		return ref, none, errors.New("native composition: invalid raw activity reference")
	}
	if len(batch.Records) != 0 || batch.Session == "" || batch.Run == "" || batch.ClockEpoch == "" || len(batch.Session) > 256 || len(batch.Run) > 256 || len(batch.ClockEpoch) > 256 || batch.Process.PID == 0 || batch.Process.Generation == 0 || batch.ExternalKind != cupti.ExternalCorrelationKindCustom0 || batch.EvidenceRef != "" && batch.EvidenceRef != ref.Digest {
		return ref, none, errors.New("native composition: invalid raw activity batch identity")
	}
	batch.EvidenceRef = ref.Digest
	// RunNative declares one run that is also its activity session and epoch.
	if r.ActivitySHA256 != "" && r.ActivitySHA256 != ref.Digest || r.ActivityRun != "" && (batch.Session != r.ActivityRun || batch.Run != r.ActivityRun || batch.ClockEpoch != r.ActivityRun) {
		return ref, none, errors.New("native composition: raw activity contradicts declared result activity")
	}
	if ref.Digest == envelope {
		return ref, none, errors.New("native composition: raw activity duplicates evidence envelope")
	}
	type operation struct {
		process cudaworkload.ProcessGenID
		op      cudaworkload.OpID
	}
	logged := map[operation]GroundTruthLog{}
	bound := false
	for _, log := range r.Requests {
		g := log.GroundTruth
		if g.ProcessGen == batch.Process && g.Workload.RunID == batch.Run {
			bound = true
		}
		if g.OpID != 0 && !g.ProcessGen.IsZero() {
			logged[operation{g.ProcessGen, g.OpID}] = g
		}
	}
	if !bound {
		return ref, none, errors.New("native composition: raw activity outside logged request population")
	}
	if len(r.Annotations) > 65536 || len(r.Activity.Records) > 65536 {
		return ref, none, errors.New("native composition: raw activity record bound exceeded")
	}
	for _, sub := range r.Annotations {
		if len(sub.Symbol) > 256 || len(sub.EvidenceRef) > 256 || len(sub.TimestampClock) > 256 || len(sub.TimestampEpoch) > 256 {
			return ref, none, errors.New("native composition: annotation string bound exceeded")
		}
		g, ok := logged[operation{sub.Process, sub.OpID}]
		if !ok {
			return ref, none, errors.New("native composition: annotation outside logged request population")
		}
		if sub.TID != g.TID || !sameTimingWorkload(sub.Workload, g.Workload) {
			return ref, none, errors.New("native composition: annotation contradicts logged request")
		}
	}
	raw, err := readNativeActivity(ctx, s.Reader, ref.SizeBytes)
	if err != nil {
		return ref, none, err
	}
	if digestBytes(raw) != ref.Digest {
		return ref, none, errors.New("native composition: raw activity digest mismatch")
	}
	if len(r.Activity.Records) > 0 && !nativeActivityEncodes(r.Activity.Records, raw) {
		return ref, none, errors.New("native composition: declared activity records contradict raw activity")
	}
	if err := ctx.Err(); err != nil {
		return ref, none, err
	}
	gates := make(map[string]tracecontract.Gate)
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		gates[name] = tracecontract.Gate{Disposition: "unchecked", Reason: "raw activity bound by declared length and digest only", Provenance: "ComposeNativeEvidence"}
	}
	receipt := tracecontract.Receipt{Version: 1, Command: "ComposeNativeEvidence raw activity source", InputPath: ref.Name, InputSHA256: ref.Digest, Kind: "capture", Partial: true, ClockDomain: "cupti", ClockProvenance: "CUPTI timestamps as recorded; no host mapping", Backends: []tracecontract.Backend{{Name: "cuptiactivity"}}, Gates: gates}
	analysis, err := cudadiag.AnalyzeActivityJSONL(bytes.NewReader(raw), batch, r.Annotations, receipt)
	if err != nil {
		return ref, none, fmt.Errorf("native composition: raw activity: %w", err)
	}
	for _, join := range analysis.Joins {
		if len(join.Record.Name) > 4096 || len(join.Record.DisplayName) > 4096 || len(join.Record.Source) > 4096 {
			return ref, none, errors.New("native composition: raw activity string bound exceeded")
		}
	}
	if r.ActivityAnalysisError != "" || r.ActivityAnalysis != nil && !sameNativeActivityAnalysis(*r.ActivityAnalysis, analysis) {
		return ref, none, errors.New("native composition: declared activity analysis contradicts raw activity")
	}
	return ref, analysis, nil
}

// readNativeActivity reads the declared size from the caller's reader, checking
// ctx before every Read. It starts no goroutine, so cancellation takes effect
// only between reads. It refuses one byte more or less than declared.
func readNativeActivity(ctx context.Context, r io.Reader, size int64) ([]byte, error) {
	buf := make([]byte, size+1)
	n, empty := 0, 0
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(n+32<<10, len(buf))
		m, err := r.Read(buf[n:end])
		if m < 0 || m > end-n {
			return nil, errors.New("native composition: invalid raw activity read")
		}
		n += m
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("native composition: raw activity: %w", err)
		}
		if n == len(buf) {
			return nil, errors.New("native composition: raw activity length mismatch")
		}
		if m > 0 {
			empty = 0
			continue
		}
		if empty++; empty == 100 {
			return nil, fmt.Errorf("native composition: raw activity: %w", io.ErrNoProgress)
		}
	}
	if int64(n) != size {
		return nil, errors.New("native composition: raw activity length mismatch")
	}
	return buf[:n], nil
}

// checkNativeActivityProjection refuses an envelope observation that names the
// raw activity digest but differs from the complete exact joins parsed for its
// operation. Only host API and kernel roles derive from activity records, in
// their recorded clock. The reader has no application-ready, scheduler or
// power role, so those stay unsupported and a post-call time is never an entry.
func checkNativeActivityProjection(record NativeEvidenceRecord, digest string, analysis cudadiag.ActivityAnalysis) error {
	in := record.Input
	if in.Receipt.InputSHA256 != digest {
		return nil
	}
	if len(in.PreEntries) != 0 || in.Sched.ProbePresent || len(in.Sched.Intervals) != 0 || in.PowerClock.ClockThrottled || in.PowerClock.PowerCapped || len(in.PowerClock.Intervals) != 0 {
		return errors.New("native composition: unsupported raw activity projection role")
	}
	var calls []cudadiag.HostCallEvent
	var kernels []cudadiag.KernelEvent
	for _, join := range analysis.Joins {
		a := join.Record
		if join.Join.Outcome == cudaworkload.OutcomeAmbiguous && join.Join.HostSubmission != nil && join.Join.HostSubmission.OpID == record.OpID {
			return errors.New("native composition: unsupported raw activity projection role")
		}
		if join.Join.Outcome != cudaworkload.OutcomeExact || join.Join.OpID != record.OpID {
			continue
		}
		if a.Start > math.MaxInt64 || a.End > math.MaxInt64 {
			return errors.New("native composition: unsupported raw activity projection role")
		}
		span := tracecontract.Interval{Start: int64(a.Start), End: int64(a.End), Clock: join.Clock}
		switch a.Kind {
		case cupti.ActivityKindDriver, cupti.ActivityKindRuntime:
			calls = append(calls, cudadiag.HostCallEvent{Span: span, Name: a.Name, HasReturn: a.Complete})
		case cupti.ActivityKindConcurrentKernel:
			kernels = append(kernels, cudadiag.KernelEvent{Span: span, Name: a.Name, StreamID: uint64(a.StreamID), ContextID: uint64(a.ContextID), DeviceID: a.DeviceID})
		}
	}
	if !slices.Equal(in.HostCalls, calls) || !slices.Equal(in.Kernels, kernels) {
		return errors.New("native composition: raw activity projection contradicts raw records")
	}
	return nil
}

// nativeActivityEncodes reports whether raw is exactly the RunNative
// JSON-lines encoding of records.
func nativeActivityEncodes(records []cuptiactivity.Record, raw []byte) bool {
	for _, record := range records {
		line, err := json.Marshal(record)
		if err != nil || len(raw) <= len(line) || !bytes.Equal(raw[:len(line)], line) || raw[len(line)] != '\n' {
			return false
		}
		raw = raw[len(line)+1:]
	}
	return len(raw) == 0
}

// sameNativeActivityAnalysis compares the record-derived parts of two
// analyses: version, every join and each finding's identity and value.
// Finding admissions depend on the caller's receipt and are not compared.
func sameNativeActivityAnalysis(a, b cudadiag.ActivityAnalysis) bool {
	if a.Version != b.Version || len(a.Findings) != len(b.Findings) {
		return false
	}
	x, xerr := json.Marshal(a.Joins)
	y, yerr := json.Marshal(b.Joins)
	if xerr != nil || yerr != nil || !bytes.Equal(x, y) {
		return false
	}
	for i, f := range a.Findings {
		g := b.Findings[i]
		if f.ID != g.ID || (f.Value == nil) != (g.Value == nil) || f.Value != nil && *f.Value != *g.Value {
			return false
		}
	}
	return true
}

func nativeDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}
