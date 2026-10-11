package ncclinspector

import (
	"bytes"
	"errors"
	"os"
	"testing"
)

func TestRecordConsumerMapping(t *testing.T) {
	raw, err := os.ReadFile("testdata/verbose.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	records, err := Read(bytes.NewReader(raw), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	record := records[0]
	// The original vendor example has a historically truncated hash. Mutation
	// makes an explicitly synthetic full-identity control, not native evidence.
	record.Header.ID = "0x1234567890abcdef"
	record.Header.Rank = 0
	record.Header.Ranks = 2
	clock := ClockMapping{SourceBootEpoch: "fixture-boot", SourceProcessEpoch: "fixture-process", CalibrationKnown: true, UncertaintyKnown: true, ValidityKnown: true, ValidFromNS: 0, ValidUntilNS: 1 << 62, Domain: ClockDomainHostRealtime, OffsetNS: 1000, UncertaintyNS: 50, ReferenceDomain: ClockDomainHostMonotonic, ReferenceEpoch: "synthetic-reference", EvidenceRef: "fixture-calibration"}
	obs, err := ObservationFromRecord(record, clock)
	if err != nil {
		t.Fatal(err)
	}
	if obs.HostEnqueueStart.Nanoseconds != int64(record.Collective.EventTraceTS.CollectiveStartTS)*1000+1000 || obs.HostEnqueueStart.UncertaintyNS != 50 || obs.HostEnqueueStart.Domain != ClockDomainHostMonotonic {
		t.Fatalf("mapping not applied: %+v", obs.HostEnqueueStart)
	}
	if obs.HasCompletion {
		t.Fatal("publication promoted to completion")
	}
	a := NewTwoRankAttributor()
	if err := a.AddObservation(obs); err != nil {
		t.Fatal(err)
	}
	peer := obs
	peer.Rank = 1
	peer.HostPID++
	peer.Hostname = "peer"
	peer.HostEnqueueStart.Nanoseconds += 1_000_000
	peer.HostEnqueueEnd.Nanoseconds += 1_000_000
	if err := a.AddObservation(peer); err != nil {
		t.Fatal(err)
	}
	joined, err := a.Attribute(record.Header.ID, obs.Sequence)
	if err != nil || !joined.Admitted || joined.Diagnosis != DiagnosisHostEnqueueSkew || joined.DelayedRank != 1 {
		t.Fatalf("record consumer: %+v %v", joined, err)
	}
	clock.EvidenceRef = ""
	if _, err := ObservationFromRecord(record, clock); !errors.Is(err, ErrIncompatibleClocks) {
		t.Fatalf("incomplete mapping admitted: %v", err)
	}
	record.Collective.EventTraceTS = nil
	clock = ClockMapping{Domain: ClockDomainHostRealtime, UncertaintyNS: 50}
	aggregate, err := ObservationFromRecord(record, clock)
	if err != nil {
		t.Fatal(err)
	}
	if aggregate.HasDeviceActivity || aggregate.HasCompletion {
		t.Fatal("aggregate duration manufactured interval")
	}
}

func TestAttributorRetention(t *testing.T) {
	a := NewTwoRankAttributor()
	a.limit = 2
	obs := CollectiveObservation{CommunicatorHash: "0x1234567890abcdef", TotalRanks: 2, HostPID: 1, Operation: "AllReduce"}
	if err := a.AddObservation(obs); err != nil {
		t.Fatal(err)
	}
	obs.Rank = 1
	obs.HostPID = 2
	if err := a.AddObservation(obs); err != nil {
		t.Fatal(err)
	}
	obs.Sequence = 1
	if err := a.AddObservation(obs); !errors.Is(err, ErrObservationCapacity) {
		t.Fatalf("capacity: %v", err)
	}
	a.Forget(obs.CommunicatorHash, 0)
	obs.Sequence = 0
	if err := a.AddObservation(obs); !errors.Is(err, ErrDuplicateSequence) {
		t.Fatalf("forgotten duplicate admitted: %v", err)
	}
	obs.Sequence = 1
	if err := a.AddObservation(obs); err != nil {
		t.Fatal(err)
	}
	obs.Sequence = 2
	obs.Hostname = "different-node"
	if err := a.AddObservation(obs); !errors.Is(err, ErrRankRestarted) {
		t.Fatalf("same-PID different host admitted: %v", err)
	}
}
