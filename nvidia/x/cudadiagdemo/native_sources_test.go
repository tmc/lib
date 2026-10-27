package cudadiagdemo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// compositionActivity returns constructed retained activity bytes, in the
// RunNative activity.jsonl encoding, for the synthetic composition fixture.
// They are a synthetic host contract, not a native capture. The device request
// makes two launches and one kernel is unannotated. It sets r's annotations,
// whose post-call timestamps keep their own unix_wall clock. The source has
// no Reader.
func compositionActivity(t *testing.T, r *NativeResult) ([]cuptiactivity.Record, []byte, NativeActivitySource) {
	t.Helper()
	g0, g1 := r.Requests[0].GroundTruth, r.Requests[1].GroundTruth
	r.Annotations = nil
	for i, g := range []GroundTruthLog{g0, g1} {
		r.Annotations = append(r.Annotations, cudaworkload.HostSubmission{Process: g.ProcessGen, OpID: g.OpID, Workload: g.Workload.Clone(), Provenance: cudaworkload.ProvenanceCUPTI, TID: g.TID, TimestampNs: 1_700_000_000_000_000_000 + uint64(i), TimestampClock: "unix_wall", TimestampEpoch: "unix_epoch", Symbol: "cuLaunchKernel", EvidenceRef: "synthetic host contract"})
	}
	pid := g0.ProcessGen.PID
	link := func(correlation uint32, op cudaworkload.OpID) cuptiactivity.Record {
		return cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom0, CorrelationID: correlation, ExternalID: uint64(op)}
	}
	api := func(correlation uint32, start, end uint64) cuptiactivity.Record {
		return cuptiactivity.Record{Kind: cupti.ActivityKindDriver, Name: "cuLaunchKernel", CorrelationID: correlation, Start: start, End: end, Complete: true, ProcessID: pid}
	}
	kernel := func(correlation uint32, start, end uint64) cuptiactivity.Record {
		return cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "transform", CorrelationID: correlation, Start: start, End: end, Complete: true, ProcessID: pid}
	}
	records := []cuptiactivity.Record{
		link(1, g0.OpID), api(1, 1000, 1100), kernel(1, 1200, 1500),
		link(2, g1.OpID), api(2, 2000, 2100), kernel(2, 2200, 4000),
		link(3, g1.OpID), api(3, 2150, 2250), kernel(3, 4000, 5800),
		kernel(9, 6000, 6100),
	}
	var raw bytes.Buffer
	encoder := json.NewEncoder(&raw)
	for _, record := range records {
		if err := encoder.Encode(record); err != nil {
			t.Fatal(err)
		}
	}
	data := raw.Bytes()
	source := NativeActivitySource{
		Ref:   cudaregress.ArtifactRef{Name: "activity.jsonl", Digest: digestBytes(data), SizeBytes: int64(len(data)), Role: "trace"},
		Batch: cudaworkload.ActivityBatch{Session: "synthetic-session", Run: g0.Workload.RunID, ClockEpoch: "synthetic-cupti-epoch", Process: g0.ProcessGen, ExternalKind: cupti.ExternalCorrelationKindCustom0},
	}
	return records, data, source
}

// activityProjection builds the host API and kernel observations supported by
// the raw records with the given correlation IDs, in their recorded clock.
func activityProjection(records []cuptiactivity.Record, correlations ...uint32) ([]cudadiag.HostCallEvent, []cudadiag.KernelEvent) {
	var calls []cudadiag.HostCallEvent
	var kernels []cudadiag.KernelEvent
	for _, r := range records {
		if !slices.Contains(correlations, r.CorrelationID) {
			continue
		}
		span := tracecontract.Interval{Start: int64(r.Start), End: int64(r.End), Clock: "cupti"}
		switch r.Kind {
		case cupti.ActivityKindDriver:
			calls = append(calls, cudadiag.HostCallEvent{Span: span, Name: r.Name, HasReturn: true})
		case cupti.ActivityKindConcurrentKernel:
			kernels = append(kernels, cudadiag.KernelEvent{Span: span, Name: r.Name})
		}
	}
	return calls, kernels
}

func TestNativeCompositionRawActivitySource(t *testing.T) {
	_, r, packet, _ := compositionFixture(t)
	records, raw, source := compositionActivity(t, r)
	envelope := compositionRaw(t, packet)
	base, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(envelope)})
	if err != nil || base.ActivitySource != nil || base.ActivityAnalysis != nil {
		t.Fatal("no-option composition", base.ActivitySource, base.ActivityAnalysis, err)
	}
	var fields map[string]json.RawMessage
	envelopeOnly := base
	envelopeOnly.Requests = nil
	data, err := json.Marshal(envelopeOnly)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	if _, ok := fields["activity_source"]; ok {
		t.Fatal("no-option JSON gained activity_source")
	}
	if _, ok := fields["activity_analysis"]; ok {
		t.Fatal("no-option JSON gained activity_analysis")
	}

	source.Reader = bytes.NewReader(raw)
	out, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(envelope), Activity: &source})
	if err != nil || out.ActivitySource == nil || *out.ActivitySource != source.Ref || out.ActivityAnalysis == nil {
		t.Fatal("raw activity not bound", out.ActivitySource, out.ActivityAnalysis, err)
	}
	if !reflect.DeepEqual(base.Requests, out.Requests) || out.Qualification != base.Qualification || out.Synthetic != base.Synthetic {
		t.Fatal("raw activity changed request diagnoses or qualification", out.Requests)
	}

	// Every API and kernel record is retained in order; external-correlation
	// records only link IDs.
	var want []cuptiactivity.Record
	for _, record := range records {
		if record.Kind != cupti.ActivityKindExternalCorrelation {
			want = append(want, record)
		}
	}
	joins := out.ActivityAnalysis.Joins
	if len(joins) != len(want) || len(out.ActivityAnalysis.Findings) != len(want) {
		t.Fatalf("got %d joins and %d findings, want %d", len(joins), len(out.ActivityAnalysis.Findings), len(want))
	}
	posted := map[cudaworkload.OpID]uint64{}
	for _, sub := range r.Annotations {
		posted[sub.OpID] = sub.TimestampNs
	}
	exact := map[cudaworkload.OpID]int{}
	launches := map[cudaworkload.OpID]int{}
	for i, join := range joins {
		if !reflect.DeepEqual(join.Record, want[i]) || join.Clock != "cupti" || join.EvidenceRef != source.Ref.Digest || join.Session != source.Batch.Session || join.Run != source.Batch.Run || join.ClockEpoch != source.Batch.ClockEpoch {
			t.Fatalf("join %d is not the raw record in its producer identity: %+v", i, join)
		}
		if join.Join.Outcome != cudaworkload.OutcomeExact {
			continue
		}
		exact[join.Join.OpID]++
		if join.Record.Kind == cupti.ActivityKindConcurrentKernel {
			launches[join.Join.OpID]++
		}
		sub, activity := join.Join.HostSubmission, join.Join.DeviceActivity
		if sub == nil || sub.TimestampNs != posted[sub.OpID] || sub.TimestampClock != "unix_wall" || activity == nil || activity.StartNs != join.Record.Start || activity.EndNs != join.Record.End {
			t.Fatalf("join %d converted a clock or relabelled the post-call annotation: %+v", i, join.Join)
		}
	}
	g0, g1 := r.Requests[0].GroundTruth, r.Requests[1].GroundTruth
	if exact[g0.OpID] != 2 || exact[g1.OpID] != 4 || launches[g1.OpID] != 2 {
		t.Fatalf("exact joins %v, kernel launches %v; want every launch retained", exact, launches)
	}
	if last := joins[len(joins)-1]; last.Join.Outcome != cudaworkload.OutcomeMissing || last.Record.CorrelationID != 9 {
		t.Fatalf("unannotated kernel not retained as missing: %+v", last.Join)
	}

	// Declared result activity that agrees with the bytes is accepted.
	r.ActivitySHA256 = source.Ref.Digest
	r.Activity.Records = records
	r.ActivityAnalysis = out.ActivityAnalysis
	source.Reader = bytes.NewReader(raw)
	declared, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(envelope), Activity: &source})
	if err != nil || !reflect.DeepEqual(declared.ActivityAnalysis, out.ActivityAnalysis) {
		t.Fatal("agreeing declared activity refused", err)
	}
	r.ActivityAnalysis = nil
	r.ActivityRun = source.Batch.Run
	source.Batch.Session, source.Batch.ClockEpoch = r.ActivityRun, r.ActivityRun
	source.Reader = bytes.NewReader(raw)
	if _, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(envelope), Activity: &source}); err != nil {
		t.Fatal("declared RunNative run refused", err)
	}
}

func TestNativeCompositionRawActivityRefusals(t *testing.T) {
	type rawCase struct {
		r             *NativeResult
		s             *NativeActivitySource
		records       []cuptiactivity.Record
		raw, envelope []byte
		analysis      *cudadiag.ActivityAnalysis
		noReader      bool
	}
	for _, tt := range []struct {
		name string
		edit func(c *rawCase)
		want string
	}{
		{"missing reader", func(c *rawCase) { c.noReader = true }, "missing raw activity reader"},
		{"hash", func(c *rawCase) { c.s.Ref.Digest = digestBytes([]byte("other")) }, "raw activity digest mismatch"},
		{"declared longer", func(c *rawCase) { c.s.Ref.SizeBytes++ }, "raw activity length mismatch"},
		{"declared shorter", func(c *rawCase) { c.s.Ref.SizeBytes-- }, "raw activity length mismatch"},
		{"role", func(c *rawCase) { c.s.Ref.Role = "source" }, "invalid raw activity reference"},
		{"batch evidence", func(c *rawCase) { c.s.Batch.EvidenceRef = digestBytes([]byte("other")) }, "invalid raw activity batch identity"},
		{"batch records", func(c *rawCase) { c.s.Batch.Records = c.records[:1] }, "invalid raw activity batch identity"},
		{"correlation kind", func(c *rawCase) { c.s.Batch.ExternalKind++ }, "invalid raw activity batch identity"},
		{"session", func(c *rawCase) { c.r.ActivityRun = c.s.Batch.Run; c.s.Batch.ClockEpoch = c.s.Batch.Run }, "contradicts declared result activity"},
		{"epoch", func(c *rawCase) { c.r.ActivityRun = c.s.Batch.Run; c.s.Batch.Session = c.s.Batch.Run }, "contradicts declared result activity"},
		{"declared run", func(c *rawCase) { c.r.ActivityRun = "other-run" }, "contradicts declared result activity"},
		{"declared digest", func(c *rawCase) { c.r.ActivitySHA256 = digestBytes([]byte("other")) }, "contradicts declared result activity"},
		{"run", func(c *rawCase) { c.s.Batch.Run = "other-run" }, "raw activity outside logged request population"},
		{"process", func(c *rawCase) { c.s.Batch.Process.Generation++ }, "raw activity outside logged request population"},
		{"op", func(c *rawCase) {
			c.r.Annotations[0].OpID = c.r.Requests[0].GroundTruth.OpID + c.r.Requests[1].GroundTruth.OpID + 1
		}, "annotation outside logged request population"},
		{"annotation process", func(c *rawCase) { c.r.Annotations[0].Process.Generation++ }, "annotation outside logged request population"},
		{"thread", func(c *rawCase) { c.r.Annotations[0].TID++ }, "annotation contradicts logged request"},
		{"annotation string", func(c *rawCase) { c.r.Annotations[0].Symbol = strings.Repeat("x", 257) }, "annotation string bound exceeded"},
		{"duplicate envelope", func(c *rawCase) {
			c.raw = c.envelope
			c.s.Ref.Digest, c.s.Ref.SizeBytes = digestBytes(c.envelope), int64(len(c.envelope))
		}, "raw activity duplicates evidence envelope"},
		{"declared records", func(c *rawCase) { c.r.Activity.Records = c.records[:len(c.records)-1] }, "declared activity records contradict raw activity"},
		{"declared analysis", func(c *rawCase) {
			a := *c.analysis
			a.Joins = a.Joins[:len(a.Joins)-1]
			c.r.ActivityAnalysis = &a
		}, "declared activity analysis contradicts raw activity"},
		{"declared analysis error", func(c *rawCase) { c.r.ActivityAnalysisError = "refused" }, "declared activity analysis contradicts raw activity"},
		{"record bound", func(c *rawCase) { c.s.Batch.MaxRecords = 3 }, "raw activity: activity findings: record limit reached"},
		{"truncated", func(c *rawCase) {
			c.raw = c.raw[:len(c.raw)-1]
			c.s.Ref.Digest, c.s.Ref.SizeBytes = digestBytes(c.raw), int64(len(c.raw))
		}, "truncated export"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, r, packet, _ := compositionFixture(t)
			records, raw, source := compositionActivity(t, r)
			envelope := compositionRaw(t, packet)
			source.Reader = bytes.NewReader(raw)
			good, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(envelope), Activity: &source})
			if err != nil || good.ActivityAnalysis == nil {
				t.Fatal("positive control", err)
			}
			source.Reader = nil
			c := &rawCase{r: r, s: &source, records: records, raw: raw, envelope: envelope, analysis: good.ActivityAnalysis}
			tt.edit(c)
			if !c.noReader {
				source.Reader = bytes.NewReader(c.raw)
			}
			out, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(envelope), Activity: &source})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ComposeNativeEvidence error = %v, want %q", err, tt.want)
			}
			if out.ActivitySource != nil || out.ActivityAnalysis != nil {
				t.Fatal("refused raw activity exposed")
			}
		})
	}
}

func TestNativeCompositionRawActivityProjection(t *testing.T) {
	for _, mode := range []string{"exact", "dropped launch", "post-call entry", "clock relabel", "cross-request", "cross-process", "pre-entry role", "scheduler role"} {
		t.Run(mode, func(t *testing.T) {
			_, r, packet, _ := compositionFixture(t)
			records, raw, source := compositionActivity(t, r)
			// Off Linux the portable fixture logs no thread; the constructed
			// host contract names one logged thread for the device request.
			const thread = 4242
			r.Requests[1].GroundTruth.TID = thread
			r.Annotations[1].TID = thread
			g1 := r.Requests[1].GroundTruth
			calls0, kernels0 := activityProjection(records, 1)
			calls1, kernels1 := activityProjection(records, 2, 3)
			if len(calls1) != 2 || len(kernels1) != 2 {
				t.Fatal("device request projection", calls1, kernels1)
			}
			// The device request's envelope record claims derivation from the
			// raw activity by naming its digest.
			record := &packet.Records[1]
			preEntries := record.Input.PreEntries
			record.Phase = "api_scheduler"
			record.Input.PreEntries = nil
			record.Input.Sched = cudadiag.SchedEvidence{Process: g1.ProcessGen, TID: g1.TID}
			record.Input.Thresholds.TargetTID = g1.TID
			record.Input.Receipt.InputSHA256 = source.Ref.Digest
			record.Input.HostCalls = slices.Clone(calls1)
			record.Input.Kernels = slices.Clone(kernels1)
			want := "raw activity projection contradicts raw records"
			switch mode {
			case "exact":
				want = ""
			case "dropped launch":
				record.Input.Kernels = record.Input.Kernels[:1]
			case "post-call entry":
				record.Input.HostCalls[0].Span.Start = int64(r.Annotations[1].TimestampNs)
			case "clock relabel":
				record.Input.HostCalls[0].Span.Clock = "process_monotonic"
			case "cross-request":
				record.Input.HostCalls, record.Input.Kernels = calls0, kernels0
			case "cross-process":
				// The device request runs in another process but its record
				// copies the exact raw joins of the same-numbered op, which a
				// separately logged raw-process request owns.
				same := r.Requests[1]
				same.Attempt = "raw-process-owner"
				other := cudaworkload.ProcessGenID{PID: g1.ProcessGen.PID + 1, Generation: g1.ProcessGen.Generation}
				r.Requests[1].GroundTruth.ProcessGen = other
				r.Requests[1].GroundTruth.Workload.Process = other
				r.Requests = append(r.Requests, same)
				record.Input.Sched.Process = other
			case "pre-entry role":
				record.Phase = "pre_entry"
				record.Input.Sched = cudadiag.SchedEvidence{}
				record.Input.HostCalls, record.Input.Kernels = nil, nil
				record.Input.PreEntries = preEntries
				record.Input.PreEntries[0].Submission.TID = thread
				record.Input.PreEntries[0].Span.End = int64(r.Annotations[1].TimestampNs)
				want = "unsupported raw activity projection role"
			case "scheduler role":
				record.Input.Sched.Intervals = make([]cudadiag.SchedInterval, 1)
				want = "unsupported raw activity projection role"
			}
			source.Reader = bytes.NewReader(raw)
			out, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(compositionRaw(t, packet)), Activity: &source})
			if want == "" {
				if err != nil || out.ActivityAnalysis == nil || len(out.Requests) != 2 || !slices.Equal(out.Requests[1].Observations.Kernels, kernels1) || !slices.Equal(out.Requests[1].Observations.HostCalls, calls1) {
					t.Fatal("exact raw projection refused or changed", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("ComposeNativeEvidence error = %v, want %q", err, want)
			}
		})
	}
}

func TestNativeCompositionRawActivityNativeWithheld(t *testing.T) {
	_, r, _, _ := compositionFixture(t)
	_, raw, source := compositionActivity(t, r)
	r.Synthetic = false
	r.HostClockDomain = "process_monotonic"
	r.HostClockEpochID = "unqualified-host-control"
	r.Trials = []NativeTrial{{RequestID: r.Requests[0].GroundTruth.RequestID, Attempt: r.Requests[0].Attempt, Calls: []NativeCall{{Name: "cuLaunchKernel", StartNS: r.Requests[0].StartedNS, EndNS: r.Requests[0].StartedNS + 1}}}}
	// A cancelled request without an operation stays in the population.
	r.Requests = append(r.Requests, NativeRequestLog{Attempt: "cancelled", StartedNS: r.Requests[0].StartedNS, FinishedNS: r.Requests[0].StartedNS, GroundTruth: GroundTruthLog{RequestID: "cancelled", WorkloadName: "same-name"}})
	base, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	source.Reader = bytes.NewReader(raw)
	out, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Activity: &source})
	if err != nil || out.ActivityAnalysis == nil || out.Synthetic || out.Qualification != base.Qualification || len(out.Requests) != 3 || !reflect.DeepEqual(base.Requests, out.Requests) {
		t.Fatal("raw activity changed native composition", out, err)
	}
	for _, row := range out.Requests {
		if row.Diagnosis.Admitted || row.Refusal != "native host/device clock mapping, scheduler/loss and independent baseline qualification unavailable" || row.Observations.Receipt.ClockUncertaintyNS != nil || len(row.Observations.Kernels) != 0 {
			t.Fatalf("native row admitted or given activity observations: %+v", row)
		}
	}
}

// rawActivityReader counts calls and bytes and runs before ahead of each Read.
type rawActivityReader struct {
	r           io.Reader
	calls, read int
	before      func()
}

func (r *rawActivityReader) Read(p []byte) (int, error) {
	r.calls++
	if r.before != nil {
		r.before()
	}
	n, err := r.r.Read(p)
	r.read += n
	return n, err
}

type stalledReader struct{}

func (stalledReader) Read([]byte) (int, error) { return 0, nil }

func TestNativeCompositionRawActivityBoundedRead(t *testing.T) {
	for _, mode := range []string{"byte bound", "canceled", "cancel during read", "longer than declared", "no progress"} {
		t.Run(mode, func(t *testing.T) {
			_, r, packet, _ := compositionFixture(t)
			_, raw, source := compositionActivity(t, r)
			envelope := compositionRaw(t, packet)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			reader := &rawActivityReader{r: bytes.NewReader(raw)}
			options := NativeCompositionOptions{Evidence: bytes.NewReader(envelope), Activity: &source}
			switch mode {
			case "byte bound":
				options.MaxBytes = int64(len(envelope) + len(raw) - 1)
			case "canceled":
				cancel()
			case "cancel during read":
				reader.before = cancel
			case "longer than declared":
				reader.r = io.MultiReader(bytes.NewReader(raw), strings.NewReader(strings.Repeat("x", 1<<20)))
			case "no progress":
				reader.r = stalledReader{}
			}
			source.Reader = reader
			out, err := ComposeNativeEvidence(ctx, r, options)
			if err == nil || out.ActivityAnalysis != nil {
				t.Fatal("raw activity admitted", err)
			}
			switch mode {
			case "byte bound":
				if !strings.Contains(err.Error(), "invalid raw activity reference") || reader.calls != 0 {
					t.Fatalf("aggregate bound: %v after %d reads", err, reader.calls)
				}
			case "canceled":
				if !errors.Is(err, context.Canceled) || reader.calls != 0 {
					t.Fatalf("canceled: %v after %d reads", err, reader.calls)
				}
			case "cancel during read":
				if !errors.Is(err, context.Canceled) || reader.calls != 1 {
					t.Fatalf("cancel during read: %v after %d reads", err, reader.calls)
				}
			case "longer than declared":
				if !strings.Contains(err.Error(), "raw activity length mismatch") || reader.read > len(raw)+1 {
					t.Fatalf("over-long reader: %v after %d bytes", err, reader.read)
				}
			case "no progress":
				if !errors.Is(err, io.ErrNoProgress) || reader.calls != 100 {
					t.Fatalf("stalled reader: %v after %d reads", err, reader.calls)
				}
			}
		})
	}
}

func TestNativeCompositionRawActivityReplayClosure(t *testing.T) {
	sf, r, packet, captured := compositionFixture(t)
	_, raw, source := compositionActivity(t, r)
	envelope := compositionRaw(t, packet)
	dir := t.TempDir()
	capture := compositionRef(t, dir, "observations.json", envelope, "trace")
	sourceRef := compositionRef(t, dir, "service-capture.tar", captured, "source")
	source.Ref = compositionRef(t, dir, "activity.jsonl", raw, "trace")
	renamed := source.Ref
	renamed.Name = "renamed.jsonl"
	counter := 0
	compose := func(sources ...cudaregress.ArtifactRef) (NativeComposition, error) {
		source.Reader = bytes.NewReader(raw)
		replay := compositionReplay(t, sf, dir, r.Requests[0].Attempt, capture, sources, &counter)
		return ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(envelope), Activity: &source, Replay: replay})
	}
	for _, sources := range [][]cudaregress.ArtifactRef{{capture, sourceRef}, {capture, sourceRef, renamed}} {
		out, err := compose(sources...)
		if err == nil || !strings.Contains(err.Error(), "replay closure lacks raw activity source") || out.Plan != nil || counter != 0 {
			t.Fatal("incomplete raw closure replayed", sources, err, counter)
		}
	}
	out, err := compose(capture, sourceRef, source.Ref)
	if err != nil || out.Plan == nil || !slices.Contains(out.Plan.SourceArtifacts, source.Ref) || out.PlanReceipt == nil || !out.PlanReceipt.ArtifactClosureValid || !slices.Contains(out.PlanReceipt.ImmutableRefs, source.Ref) || out.Execution == nil || !out.Execution.OutputsVerified || counter == 0 {
		t.Fatal("complete raw closure not replayed", err, counter)
	}
	// The reader's bytes do not vouch for the retained file; existing plan
	// validation checks it.
	corrupt := bytes.Clone(raw)
	corrupt[0] ^= 1
	if err := os.WriteFile(source.Ref.Path, corrupt, 0600); err != nil {
		t.Fatal(err)
	}
	before := counter
	out, err = compose(capture, sourceRef, source.Ref)
	if !errors.Is(err, cudaregress.ErrArtifactCorrupted) || out.Plan != nil || counter != before {
		t.Fatal("corrupt retained raw source replayed", err, counter)
	}
}
