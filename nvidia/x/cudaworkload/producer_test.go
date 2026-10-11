package cudaworkload

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestActivityProducerFrozenLinks(t *testing.T) {
	p := ProcessGenID{1, 2}
	sub := HostSubmission{Process: p, OpID: 9, Symbol: "native", Workload: WorkloadIdentity{Process: p, RunID: "run"}, Provenance: ProvenanceCUPTI, Context: ContextStreamID{3, 4}, EvidenceRef: "annotation"}
	raw := cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7, ContextID: 3, StreamID: 4, Start: 10, End: 20, Complete: true}
	link := cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalID: 9, ExternalKind: cupti.ExternalCorrelationKindCustom0}
	b := ActivityBatch{Session: "s", Run: "run", ClockEpoch: "epoch", EvidenceRef: "raw", Process: p, ExternalKind: cupti.ExternalCorrelationKindCustom0, Records: []cuptiactivity.Record{raw, link}}
	out, err := JoinActivities(b, []HostSubmission{sub})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].Join.Outcome != OutcomeExact || out[0].Record != raw || out[0].ClockEpoch != "epoch" {
		t.Fatalf("unexpected join: %+v", out)
	}
	bad := link
	bad.ExternalID = 10
	b.Records = append(b.Records, bad)
	out, err = JoinActivities(b, []HostSubmission{sub})
	if err != nil || out[0].Join.Outcome != OutcomeAmbiguous {
		t.Fatalf("contradiction admitted: %+v %v", out, err)
	}
	b.Records = []cuptiactivity.Record{raw}
	out, err = JoinActivities(b, []HostSubmission{sub})
	if err != nil || out[0].Join.Outcome != OutcomeMissing {
		t.Fatal("unmapped identity admitted")
	}
	b.Records = []cuptiactivity.Record{raw, link}
	raw.GraphID = 3
	b.Records[0] = raw
	out, err = JoinActivities(b, []HostSubmission{sub})
	if err != nil || out[0].Join.Outcome == OutcomeExact {
		t.Fatal("graph replay inferred")
	}
	b.MaxRecords = 1
	if _, err = JoinActivities(b, []HostSubmission{sub}); err == nil {
		t.Fatal("bound ignored")
	}
}

type balancedHook struct {
	id        uint64
	push, pop int
}

func (h *balancedHook) Push(_ cupti.ExternalCorrelationKind, id uint64) error {
	h.id = id
	h.push++
	return nil
}
func (h *balancedHook) Pop(_ cupti.ExternalCorrelationKind) (uint64, error) {
	h.pop++
	return h.id, nil
}
func TestNativeOperationCancellation(t *testing.T) {
	h := new(balancedHook)
	a := NewAnnotator(ProcessGenID{1, 2}, "work", WithRunID("run"), WithCorrelationHook(h))
	_, s := a.Begin(context.Background(), "op")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := s.NativeOperation(ctx, ContextStreamID{}, "native", "ref", func() error { t.Fatal("called after cancellation"); return nil })
	if !errors.Is(err, context.Canceled) || h.push != 1 || h.pop != 1 {
		t.Fatalf("unbalanced cancellation: %v %+v", err, h)
	}
}

func TestActivityIdentityContradictions(t *testing.T) {
	p := ProcessGenID{1, 2}
	sub := HostSubmission{Process: p, OpID: 9, Symbol: "native", Workload: WorkloadIdentity{Process: p, RunID: "run"}, Provenance: ProvenanceCUPTI, EvidenceRef: "annotation", Context: ContextStreamID{3, 4}, TID: 5}
	raw := cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7, ContextID: 3, StreamID: 4}
	link := cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalID: 9, ExternalKind: cupti.ExternalCorrelationKindCustom0}
	b := ActivityBatch{Session: "s", Run: "run", ClockEpoch: "epoch", EvidenceRef: "raw", Process: p, ExternalKind: cupti.ExternalCorrelationKindCustom0, Records: []cuptiactivity.Record{raw, link}}
	for _, tt := range []struct {
		name   string
		mutate func(*ActivityBatch, *HostSubmission)
	}{
		{"process", func(b *ActivityBatch, _ *HostSubmission) { b.Records[0].ProcessID = 99 }},
		{"complete_zero", func(b *ActivityBatch, _ *HostSubmission) { b.Records[0].Complete = true }},
		{"symbol", func(_ *ActivityBatch, s *HostSubmission) { s.Symbol = "" }},
		{"evidence", func(_ *ActivityBatch, s *HostSubmission) { s.EvidenceRef = "" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			copyBatch := b
			copyBatch.Records = append([]cuptiactivity.Record(nil), b.Records...)
			s := sub
			tt.mutate(&copyBatch, &s)
			if _, err := JoinActivities(copyBatch, []HostSubmission{s}); err == nil {
				t.Fatal("contradiction accepted")
			}
		})
	}
	b.Records[0].ThreadID = 6
	out, err := JoinActivities(b, []HostSubmission{sub})
	if err != nil || out[0].Join.Outcome == OutcomeExact {
		t.Fatalf("thread contradiction: %+v %v", out, err)
	}
}

func TestNativeProducerAdmission(t *testing.T) {
	var typedNil *balancedHook
	for _, tt := range []struct {
		name string
		hook CorrelationHook
		kind cupti.ExternalCorrelationKind
	}{
		{"typed_nil", typedNil, cupti.ExternalCorrelationKindCustom0},
		{"invalid_kind", new(balancedHook), cupti.ExternalCorrelationKindSize},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := NewAnnotator(ProcessGenID{1, 2}, "work", WithCorrelationHook(tt.hook), WithCorrelationKind(tt.kind))
			_, s := a.Begin(context.Background(), "op")
			if err := s.NativeCall(func() error { t.Fatal("invalid producer called"); return nil }); err == nil {
				t.Fatal("invalid producer accepted")
			}
			if _, err := s.NativeOperation(context.Background(), ContextStreamID{}, "native", "ref", func() error { t.Fatal("invalid operation called"); return nil }); err == nil {
				t.Fatal("invalid operation accepted")
			}
		})
	}
	a := NewAnnotator(ProcessGenID{1, 2}, "work", WithCorrelationHook(nil))
	_, s := a.Begin(context.Background(), "op")
	sub := s.NewSubmission(1, ContextStreamID{}, GraphReplayGeneration{}, "native", "ref")
	if sub.Provenance == ProvenanceCUPTI || sub.TimestampClock != "unix_wall" || sub.TimestampEpoch != "unix_epoch" {
		t.Fatalf("false provenance: %+v", sub)
	}
	a = NewAnnotator(ProcessGenID{1, 2}, "work", WithCorrelationHook(new(balancedHook)))
	_, s = a.Begin(context.Background(), "op")
	for _, v := range [][2]string{{"", "ref"}, {"native", ""}} {
		if _, err := s.NativeOperation(context.Background(), ContextStreamID{}, v[0], v[1], func() error { t.Fatal("missing identity called"); return nil }); err == nil {
			t.Fatal("missing identity accepted")
		}
	}
}
