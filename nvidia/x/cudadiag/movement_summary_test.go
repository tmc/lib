package cudadiag_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const movementBytes = 1<<53 + 1

func movementSHA(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

// movementCase holds the separately produced inputs before they are encoded,
// so each control mutates one source of evidence.
type movementCase struct {
	context     nvtelemetry.MovementContext
	records     []cuptiactivity.Record
	annotations []cudaworkload.HostSubmission
	health      *nvtelemetry.HealthSnapshot
	artifacts   map[string][]byte
	telemetry   []byte
	telemetryOK cudadiag.TelemetryOptions
	// activityRun overrides the activity producer's run when set.
	activityRun string
}

func movementHealthLimits() nvtelemetry.HealthLimits {
	return nvtelemetry.HealthLimits{MaxBytes: 1 << 20, MaxDevices: 8, MaxSources: 8, MaxObservations: 8, MaxEdges: 8}
}

func newMovementCase() *movementCase {
	p := cudaworkload.ProcessGenID{PID: 1, Generation: 2}
	n := uint64(movementBytes)
	enq, start, done, unc := int64(100), int64(110), int64(200), int64(5)
	path := []byte("independent NVLink transfer path artifact")
	return &movementCase{
		context: nvtelemetry.MovementContext{
			Version: 1, RunID: "run", Producer: "app", ProducerVersion: "1", ProducerSchema: "app.movement.v1", SourceClass: "imported",
			LossByFamily: map[string]tracecontract.Loss{"cuda_memcpy": {Total: 2}},
			Transfers: []nvtelemetry.MovementTransfer{{
				ID: "t1", Family: "cuda_memcpy", RawSHA256: movementSHA([]byte("raw t1")), RequestID: "r1", Attempt: "a1",
				Source:      nvtelemetry.MovementEndpoint{Entity: "GPU-0", Kind: "device"},
				Destination: nvtelemetry.MovementEndpoint{Entity: "GPU-1", Kind: "device"},
				Direction:   "peer_to_peer", Bytes: &n, Unit: "bytes", State: "completed",
				EnqueueNanos: &enq, StartNanos: &start, CompleteNanos: &done,
				ClockDomain: "monotonic", ClockEpoch: "boot", UncertaintyNS: &unc, TopologyEpoch: "topo",
				Path: nvtelemetry.MovementPath{Mechanism: "nvlink", Basis: "independent_artifact", Evidence: movementSHA(path)},
			}, {
				ID: "t2", Family: "cuda_memcpy", RawSHA256: movementSHA([]byte("raw t2")), RequestID: "r2", Attempt: "a1", OpID: 9,
				Source:      nvtelemetry.MovementEndpoint{Entity: "host-numa0", Kind: "host_pinned"},
				Destination: nvtelemetry.MovementEndpoint{Entity: "GPU-0", Kind: "device"},
				Direction:   "host_to_device", Bytes: &n, Unit: "bytes", State: "completed",
				EnqueueNanos: &enq, StartNanos: &start, CompleteNanos: &done,
				ClockDomain: "monotonic", ClockEpoch: "boot", UncertaintyNS: &unc,
				Path: nvtelemetry.MovementPath{Mechanism: "unknown"},
			}},
		},
		records: []cuptiactivity.Record{
			{Kind: cupti.ActivityKindMemcpy, Name: "memcpy", CorrelationID: 7, ContextID: 3, StreamID: 4, Start: 120, End: 190, Complete: true, Bytes: movementBytes, CopyKind: cupti.ActivityMemcpyKindHtod, SourceMemoryKind: cupti.ActivityMemoryKindPinned, DestinationMemoryKind: cupti.ActivityMemoryKindDevice, CopyCount: 1},
			{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalID: 9, ExternalKind: cupti.ExternalCorrelationKindCustom0},
		},
		annotations: []cudaworkload.HostSubmission{{Process: p, OpID: 9, Symbol: "cudaMemcpyAsync", Workload: cudaworkload.WorkloadIdentity{Process: p, RunID: "run"}, Provenance: cudaworkload.ProvenanceCUPTI, Context: cudaworkload.ContextStreamID{Context: 3, Stream: 4}, EvidenceRef: "application-annotation"}},
		health: &nvtelemetry.HealthSnapshot{
			Version: 1, NodeID: "node", TopologyEpoch: "topo", ClockDomain: "monotonic", ClockEpoch: "boot", AtNanos: 500,
			DeviceStates: map[string]string{"GPU-0": "present", "GPU-1": "present", "GPU-2": "present"},
			Sources:      []nvtelemetry.HealthSource{{ID: "nvml", Kind: "native", Version: "1", State: "available", HasLastSuccess: true, LastSuccessNanos: 500}},
			Edges:        []nvtelemetry.TopologyEdge{{SourceID: "nvml", From: "GPU-0", To: "GPU-1", Kind: "nvlink_remote_pci", Scope: "link/0/0000:02:00.0", State: "available", ValidFromNanos: 0, ValidUntilNanos: 1000}},
			Complete:     true,
		},
		artifacts: map[string][]byte{movementSHA(path): path},
	}
}

func (m *movementCase) input(t *testing.T) cudadiag.MovementInput {
	t.Helper()
	in, err := m.build()
	if err != nil {
		t.Fatal(err)
	}
	return in
}

// build encodes each input with its public writer.
func (m *movementCase) build() (cudadiag.MovementInput, error) {
	var movement, activities, health bytes.Buffer
	if err := nvtelemetry.WriteMovementContext(&movement, m.context, nvtelemetry.WorkloadLimits{}); err != nil {
		return cudadiag.MovementInput{}, err
	}
	for _, r := range m.records {
		if err := json.NewEncoder(&activities).Encode(r); err != nil {
			return cudadiag.MovementInput{}, err
		}
	}
	receipt := cudadiag.BuildSyntheticFixture(cudadiag.InjectionNone).Input.Receipt
	receipt.InputSHA256 = movementSHA(activities.Bytes())
	receipt.ClockDomain = "cupti"
	receipt.ClockProvenance = "producer fixture epoch"
	p := cudaworkload.ProcessGenID{PID: 1, Generation: 2}
	run := "run"
	annotations := m.annotations
	if m.activityRun != "" {
		run = m.activityRun
		annotations = append([]cudaworkload.HostSubmission(nil), m.annotations...)
		for i := range annotations {
			annotations[i].Workload.RunID = run
		}
	}
	in := cudadiag.MovementInput{
		Movement: &movement, Source: nvtelemetry.WorkloadSource{Provider: "application", Version: "movement-context-v1", RunID: "run"},
		Activities: &activities, Batch: cudaworkload.ActivityBatch{Session: "session", Run: run, ClockEpoch: "producer-epoch", Process: p, EvidenceRef: receipt.InputSHA256, ExternalKind: cupti.ExternalCorrelationKindCustom0},
		Annotations: annotations, Receipt: receipt, Artifacts: m.artifacts, HealthLimits: movementHealthLimits(),
	}
	if m.health != nil {
		if err := nvtelemetry.WriteHealthSnapshot(&health, *m.health, movementHealthLimits()); err != nil {
			return cudadiag.MovementInput{}, err
		}
		in.Health = &health
	}
	if m.telemetry != nil {
		in.Telemetry = bytes.NewReader(m.telemetry)
		in.TelemetryOptions = m.telemetryOK
	}
	return in, nil
}

func summarizeMovement(t *testing.T, m *movementCase) cudadiag.MovementSummary {
	t.Helper()
	s, err := cudadiag.SummarizeMovement(context.Background(), m.input(t))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// withheldRules returns the sorted rules withheld for transfer i, after
// checking that every finding serializes under the existing contract.
func withheldRules(t *testing.T, s cudadiag.MovementSummary, i int) []string {
	t.Helper()
	var out []string
	for _, f := range s.Transfers[i].Findings {
		if _, err := f.JSON(); err != nil {
			t.Fatalf("finding %s: %v", f.ID, err)
		}
		if !f.Admission.Admitted {
			if f.Class != tracecontract.FindingUnavailable || f.Value != nil || f.Admission.ReasonCode == "" {
				t.Fatalf("withheld finding keeps a claim: %+v", f)
			}
			out = append(out, f.Rule)
		}
	}
	sort.Strings(out)
	return out
}

func movementFindingFor(s cudadiag.MovementSummary, i int, rule string) tracecontract.Finding {
	for _, f := range s.Transfers[i].Findings {
		if f.Rule == rule {
			return f
		}
	}
	return tracecontract.Finding{}
}

var (
	baseT1 = []string{cudadiag.RuleMovementActivity, cudadiag.RuleMovementCongestion}
	baseT2 = []string{cudadiag.RuleMovementCongestion, cudadiag.RuleMovementPath}
)

func with(base []string, rules ...string) []string {
	out := append(append([]string(nil), base...), rules...)
	sort.Strings(out)
	return out
}

func TestMovementSummaryAdmitsSuppliedTransfer(t *testing.T) {
	s := summarizeMovement(t, newMovementCase())
	if got := withheldRules(t, s, 0); !reflect.DeepEqual(got, baseT1) {
		t.Fatalf("t1 withheld %v", got)
	}
	if got := withheldRules(t, s, 1); !reflect.DeepEqual(got, baseT2) {
		t.Fatalf("t2 withheld %v", got)
	}
	t1, t2 := s.Transfers[0], s.Transfers[1]
	if t1.Bytes == nil || *t1.Bytes != movementBytes || t1.ActivityIndex != nil {
		t.Fatalf("t1 exact bytes lost or activity invented: %+v", t1)
	}
	if t2.ActivityIndex == nil || s.Activities.Joins[*t2.ActivityIndex].Record.Bytes != movementBytes || s.Activities.Joins[*t2.ActivityIndex].Join.OpID != 9 {
		t.Fatalf("t2 activity link lost: %+v", t2)
	}
	if t1.RequestID != "r1" || t2.RequestID != "r2" || t2.OpID != 9 || t1.RawSHA256 != s.Context.Transfers[0].RawSHA256 {
		t.Fatal("supplied identity or raw lineage lost")
	}
	for rule, want := range map[string]tracecontract.FindingClass{
		cudadiag.RuleMovementBytes:   tracecontract.FindingAssociated,
		cudadiag.RuleMovementPath:    tracecontract.FindingAssociated,
		cudadiag.RuleMovementAttempt: tracecontract.FindingAssociated,
	} {
		if f := movementFindingFor(s, 0, rule); f.Class != want {
			t.Errorf("%s class %s, want %s", rule, f.Class, want)
		}
	}
	if f := movementFindingFor(s, 1, cudadiag.RuleMovementActivity); f.Class != tracecontract.FindingAssociated {
		t.Errorf("activity link class %s", f.Class)
	}
	if d := movementFindingFor(s, 0, cudadiag.RuleMovementDuration); d.Value == nil || *d.Value != 90 || d.Intervals[0].Clock != "monotonic" {
		t.Fatalf("duration %+v", d)
	}
	if p := movementFindingFor(s, 0, cudadiag.RuleMovementPath); len(p.Artifacts) != 2 || p.Artifacts[1] != s.Context.Transfers[0].Path.Evidence {
		t.Fatal("path evidence digest not retained")
	}
	if len(s.Families) != 1 || !s.Families[0].Qualified || s.Families[0].Completed != 2 || s.TopologyEpoch != "topo" {
		t.Fatalf("population %+v", s.Families)
	}
	if s.ContextSHA256 == "" || s.Activities.Version != 1 || len(s.Activities.Findings) == 0 {
		t.Fatal("consumed analyses not retained")
	}

	native := newMovementCase()
	native.context.SourceClass = "native"
	ns := summarizeMovement(t, native)
	if f := movementFindingFor(ns, 0, cudadiag.RuleMovementBytes); f.Class != tracecontract.FindingObserved {
		t.Fatalf("native bytes class %s", f.Class)
	}
	if f := movementFindingFor(ns, 0, cudadiag.RuleMovementAttempt); f.Class != tracecontract.FindingAssociated {
		t.Fatal("supplied attempt annotation promoted to observed")
	}
	if f := movementFindingFor(ns, 0, cudadiag.RuleMovementPath); f.Class != tracecontract.FindingAssociated {
		t.Fatal("topology-consistent path promoted to an observed route")
	}
}

func TestMovementSummaryWithholdsOnlyAffectedClaims(t *testing.T) {
	tests := []struct {
		name         string
		edit         func(*movementCase)
		want1, want2 []string
	}{
		{"swapped peer", func(m *movementCase) { m.context.Transfers[0].Destination.Entity = "GPU-2" }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"swapped peer direction", func(m *movementCase) { m.context.Transfers[0].Direction = "device_to_host" }, with(baseT1, cudadiag.RuleMovementDirection, cudadiag.RuleMovementPath), baseT2},
		{"swapped copy direction", func(m *movementCase) { m.context.Transfers[1].Direction = "device_to_host" }, baseT1, with(baseT2, cudadiag.RuleMovementActivity, cudadiag.RuleMovementDirection)},
		{"mixed topology epoch", func(m *movementCase) { m.context.Transfers[0].TopologyEpoch = "topo-2" }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"missing topology epoch", func(m *movementCase) { m.context.Transfers[0].TopologyEpoch = "" }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"no topology snapshot", func(m *movementCase) { m.health = nil }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"unmapped clock", func(m *movementCase) { m.context.Transfers[0].ClockEpoch = "other-boot" }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"edge outside transfer", func(m *movementCase) { m.health.Edges[0].ValidFromNanos = 150 }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"missing uncertainty", func(m *movementCase) { m.context.Transfers[0].UncertaintyNS = nil }, with(baseT1, cudadiag.RuleMovementDuration, cudadiag.RuleMovementPath), baseT2},
		{"excessive uncertainty", func(m *movementCase) { v := int64(51); m.context.Transfers[0].UncertaintyNS = &v }, with(baseT1, cudadiag.RuleMovementDuration, cudadiag.RuleMovementPath), baseT2},
		{"edge inside uncertainty", func(m *movementCase) { m.health.Edges[0].ValidFromNanos = 108 }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"reverse directed edge", func(m *movementCase) { m.health.Edges[0].From, m.health.Edges[0].To = "GPU-1", "GPU-0" }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"loose edge kind", func(m *movementCase) { m.health.Edges[0].Kind = "nvlink_guess" }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"unsupported pcie topology", func(m *movementCase) { m.context.Transfers[0].Path.Mechanism = "pcie" }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"foreign activity run", func(m *movementCase) { m.activityRun = "other-run" }, baseT1, with(baseT2, cudadiag.RuleMovementActivity)},
		{"batched memcpy record", func(m *movementCase) { m.records[0].CopyCount = 2 }, baseT1, with(baseT2, cudadiag.RuleMovementActivity)},
		{"absent completion", func(m *movementCase) {
			tr := &m.context.Transfers[0]
			p, b := int64(150), uint64(10)
			tr.State, tr.CompleteNanos, tr.ProgressNanos, tr.ProgressBytes = "in_progress", nil, &p, &b
		}, with(baseT1, cudadiag.RuleMovementCompletion, cudadiag.RuleMovementDuration), baseT2},
		{"incomplete activity", func(m *movementCase) { m.records[0].Complete, m.records[0].Start, m.records[0].End = false, 0, 0 }, baseT1, with(baseT2, cudadiag.RuleMovementCompletion, cudadiag.RuleMovementDuration)},
		{"wrong bytes", func(m *movementCase) { v := uint64(movementBytes - 1); m.context.Transfers[1].Bytes = &v }, baseT1, with(baseT2, cudadiag.RuleMovementActivity, cudadiag.RuleMovementBytes)},
		{"unreported bytes", func(m *movementCase) { m.context.Transfers[0].Bytes, m.context.Transfers[0].Unit = nil, "" }, with(baseT1, cudadiag.RuleMovementBytes), baseT2},
		{"missing operation", func(m *movementCase) { m.context.Transfers[1].OpID = 0 }, baseT1, with(baseT2, cudadiag.RuleMovementActivity)},
		{"missing annotation", func(m *movementCase) { m.annotations = nil }, baseT1, with(baseT2, cudadiag.RuleMovementActivity)},
		{"wrong memory kind", func(m *movementCase) { m.records[0].SourceMemoryKind = cupti.ActivityMemoryKindPageable }, baseT1, with(baseT2, cudadiag.RuleMovementActivity)},
		{"missing attempt", func(m *movementCase) { m.context.Transfers[0].RequestID, m.context.Transfers[0].Attempt = "", "" }, with(baseT1, cudadiag.RuleMovementAttempt), baseT2},
		{"path evidence not retained", func(m *movementCase) { m.artifacts = nil }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"unknown path", func(m *movementCase) { m.context.Transfers[0].Path = nvtelemetry.MovementPath{Mechanism: "unknown"} }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
		{"storage path on peer copy", func(m *movementCase) { m.context.Transfers[0].Path.Mechanism = "gds_direct" }, with(baseT1, cudadiag.RuleMovementPath), baseT2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := newMovementCase()
			tt.edit(m)
			s := summarizeMovement(t, m)
			if got := withheldRules(t, s, 0); !reflect.DeepEqual(got, tt.want1) {
				t.Errorf("t1 withheld %v, want %v", got, tt.want1)
			}
			if got := withheldRules(t, s, 1); !reflect.DeepEqual(got, tt.want2) {
				t.Errorf("t2 withheld %v, want %v", got, tt.want2)
			}
		})
	}
}

func TestMovementSummaryCUDAActivityIsNotAPath(t *testing.T) {
	m := newMovementCase()
	var activity bytes.Buffer
	for _, r := range m.records {
		if err := json.NewEncoder(&activity).Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	// The memcpy export itself is offered as the path artifact.
	m.context.Transfers[0].Path.Evidence = movementSHA(activity.Bytes())
	m.artifacts = map[string][]byte{movementSHA(activity.Bytes()): activity.Bytes()}
	s := summarizeMovement(t, m)
	if f := movementFindingFor(s, 0, cudadiag.RuleMovementPath); f.Admission.Admitted {
		t.Fatal("CUDA memcpy activity admitted as path evidence")
	}
	if f := movementFindingFor(s, 1, cudadiag.RuleMovementActivity); !f.Admission.Admitted {
		t.Fatal("activity link lost")
	}
}

func TestMovementSummaryUncertaintyOverflow(t *testing.T) {
	m := newMovementCase()
	start, done, unc := int64(1<<62), int64(1<<62+10), int64(1<<62-1)
	tr := &m.context.Transfers[0]
	tr.EnqueueNanos, tr.StartNanos, tr.CompleteNanos, tr.UncertaintyNS = &start, &start, &done, &unc
	in := m.input(t)
	in.MaxUncertaintyNS = math.MaxInt64
	s, err := cudadiag.SummarizeMovement(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if f := movementFindingFor(s, 0, cudadiag.RuleMovementPath); f.Admission.Admitted {
		t.Fatal("path admitted after uncertainty widening overflowed")
	}
}

func TestMovementSummaryStoragePathMode(t *testing.T) {
	for _, tt := range []struct {
		mechanism, basis string
		admit            bool
	}{
		{"gds_direct", "independent_artifact", true},
		{"gds_direct", "producer_reported", false},
		{"gds_fallback", "independent_artifact", true},
		{"gds_fallback", "producer_reported", false},
		{"host_staged", "independent_artifact", true},
		{"host_staged", "producer_reported", false},
		{"network_rdma", "independent_artifact", false},
	} {
		m := newMovementCase()
		tr := &m.context.Transfers[0]
		tr.Source = nvtelemetry.MovementEndpoint{Entity: "nvme0n1/file-17", Kind: "storage"}
		tr.Direction = "storage_to_device"
		tr.Path.Mechanism, tr.Path.Basis = tt.mechanism, tt.basis
		s := summarizeMovement(t, m)
		if f := movementFindingFor(s, 0, cudadiag.RuleMovementPath); f.Admission.Admitted != tt.admit {
			t.Errorf("%s/%s: path admitted=%v", tt.mechanism, tt.basis, f.Admission.Admitted)
		}
	}
	m := newMovementCase()
	tr := &m.context.Transfers[0]
	tr.Destination = nvtelemetry.MovementEndpoint{Entity: "mlx5_0/port1", Kind: "network"}
	tr.Direction, tr.Path.Mechanism = "network_send", "network_rdma"
	if f := movementFindingFor(summarizeMovement(t, m), 0, cudadiag.RuleMovementPath); !f.Admission.Admitted {
		t.Error("independent RDMA artifact withheld")
	}
	tr.Path.Basis = "producer_reported"
	if f := movementFindingFor(summarizeMovement(t, m), 0, cudadiag.RuleMovementPath); f.Admission.Admitted {
		t.Error("producer-reported RDMA admitted")
	}
}

func TestMovementSummaryLatencyIsNotCongestion(t *testing.T) {
	readings, opt := mixedProtocol()
	m := newMovementCase()
	m.telemetry = collectMixed(t, readings, opt)
	m.telemetryOK = opt
	tr := &m.context.Transfers[0]
	// A very slow transfer on an endpoint with admitted telemetry.
	start, done := opt.WallInterval.Start, opt.WallInterval.End
	tr.Source.Entity, tr.ClockDomain, tr.ClockEpoch = "GPU-protocol", "monotonic", "protocol-epoch"
	tr.EnqueueNanos, tr.StartNanos, tr.CompleteNanos = &start, &start, &done
	tr.Path = nvtelemetry.MovementPath{Mechanism: "unknown"}
	s := summarizeMovement(t, m)
	if s.Telemetry == nil || len(s.Telemetry.Observations) == 0 {
		t.Fatal("telemetry diagnosis not consumed")
	}
	if len(s.Transfers[0].Symptoms) == 0 {
		t.Fatal("admitted endpoint telemetry not attached as symptom")
	}
	f := movementFindingFor(s, 0, cudadiag.RuleMovementCongestion)
	if f.Admission.Admitted || f.Admission.ReasonCode != tracecontract.ReasonControlMissing {
		t.Fatalf("latency and symptoms admitted congestion: %+v", f.Admission)
	}
	if d := movementFindingFor(s, 0, cudadiag.RuleMovementDuration); !d.Admission.Admitted {
		t.Fatal("observed latency withheld")
	}

	tr.ClockEpoch = "unmapped"
	s = summarizeMovement(t, m)
	if len(s.Transfers[0].Symptoms) != 0 || movementFindingFor(s, 0, cudadiag.RuleMovementCongestion).Admission.ReasonCode != tracecontract.ReasonClockDomainMismatch {
		t.Fatal("symptoms attached across unmapped clocks")
	}
}

func TestMovementSummaryPopulationAndSourceClass(t *testing.T) {
	for _, tt := range []struct {
		name string
		loss map[string]tracecontract.Loss
		want bool
	}{
		{"zero", map[string]tracecontract.Loss{"cuda_memcpy": {Total: 2}}, true},
		{"unassessed", nil, false},
		{"unknown", map[string]tracecontract.Loss{"cuda_memcpy": {Total: 2, Unknown: true}}, false},
		{"dropped", map[string]tracecontract.Loss{"cuda_memcpy": {Total: 2, Dropped: 1}}, false},
	} {
		m := newMovementCase()
		m.context.LossByFamily = tt.loss
		s := summarizeMovement(t, m)
		if s.Families[0].Qualified != tt.want {
			t.Errorf("%s: qualified=%v", tt.name, s.Families[0].Qualified)
		}
		if got := withheldRules(t, s, 0); !reflect.DeepEqual(got, baseT1) {
			t.Errorf("%s: family loss withheld transfer claims %v", tt.name, got)
		}
	}
	m := newMovementCase()
	m.context.SourceClass = "modeled"
	s := summarizeMovement(t, m)
	if got := withheldRules(t, s, 0); len(got) != len(s.Transfers[0].Findings) || s.Transfers[0].Bytes != nil || s.Families[0].Qualified {
		t.Fatalf("modeled source admitted %v", got)
	}
	if s.Transfers[1].ActivityIndex != nil || len(s.Transfers[1].Symptoms) != 0 {
		t.Fatal("modeled source kept claim-bearing links")
	}
}

func TestMovementSummaryRefusesInputs(t *testing.T) {
	tests := []struct {
		name string
		edit func(*cudadiag.MovementInput)
	}{
		{"nil movement", func(in *cudadiag.MovementInput) { in.Movement = nil }},
		{"nil activities", func(in *cudadiag.MovementInput) { in.Activities = nil }},
		{"wrong source", func(in *cudadiag.MovementInput) { in.Source.Version = "movement-context-v2" }},
		{"activity hash", func(in *cudadiag.MovementInput) { in.Receipt.InputSHA256 = movementSHA([]byte("other")) }},
		{"artifact digest", func(in *cudadiag.MovementInput) {
			in.Artifacts = map[string][]byte{movementSHA([]byte("a")): []byte("b")}
		}},
		{"health", func(in *cudadiag.MovementInput) { in.Health = bytes.NewReader([]byte("{}")) }},
		{"uncertainty bound", func(in *cudadiag.MovementInput) { in.MaxUncertaintyNS = -1 }},
	}
	for _, tt := range tests {
		in := newMovementCase().input(t)
		tt.edit(&in)
		if _, err := cudadiag.SummarizeMovement(context.Background(), in); err == nil {
			t.Errorf("%s accepted", tt.name)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudadiag.SummarizeMovement(ctx, newMovementCase().input(t)); err == nil {
		t.Error("cancelled context accepted")
	}
}
