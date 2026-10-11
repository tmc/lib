package ncclinspector

import (
	"errors"
	"testing"
)

func validStallClock() ClockMapping {
	return ClockMapping{Domain: ClockDomainHostMonotonic, SourceBootEpoch: "boot-a", SourceProcessEpoch: "process-a", CalibrationKnown: true, UncertaintyKnown: true, ValidityKnown: true, ValidFromNS: 0, ValidUntilNS: 200, ReferenceDomain: ClockDomainHostRealtime, ReferenceEpoch: "reference-a", EvidenceRef: "calibration-a", OffsetNS: 1000, UncertaintyNS: 5}
}

func TestAlignedValidity(t *testing.T) {
	m := validStallClock()
	ts := BoundedTimestamp{Nanoseconds: 100, UncertaintyNS: 2, Domain: m.Domain, Epoch: m.SourceProcessEpoch, BootEpoch: m.SourceBootEpoch}
	got, err := ts.Aligned(m)
	if err != nil || got.Nanoseconds != 1100 || got.UncertaintyNS != 7 || got.Epoch != "reference-a" {
		t.Fatalf("alignment: %+v %v", got, err)
	}
	tests := []struct {
		name   string
		change func(*ClockMapping, *BoundedTimestamp)
	}{
		{"unknown calibration", func(m *ClockMapping, ts *BoundedTimestamp) { m.CalibrationKnown = false }},
		{"unknown uncertainty", func(m *ClockMapping, ts *BoundedTimestamp) { m.UncertaintyKnown = false }},
		{"unknown validity", func(m *ClockMapping, ts *BoundedTimestamp) { m.ValidityKnown = false }},
		{"expired", func(m *ClockMapping, ts *BoundedTimestamp) { ts.Nanoseconds = 201 }},
		{"uncertainty crosses zero", func(m *ClockMapping, ts *BoundedTimestamp) { ts.Nanoseconds = 0 }},
		{"uncertainty crosses end", func(m *ClockMapping, ts *BoundedTimestamp) { ts.Nanoseconds = 200 }},
		{"process swap", func(m *ClockMapping, ts *BoundedTimestamp) { ts.Epoch = "process-b" }},
		{"boot swap", func(m *ClockMapping, ts *BoundedTimestamp) { ts.BootEpoch = "boot-b" }},
		{"missing reference", func(m *ClockMapping, ts *BoundedTimestamp) { m.ReferenceEpoch = "" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, ts := m, ts
			tt.change(&m, &ts)
			if _, err := ts.Aligned(m); !errors.Is(err, ErrIncompatibleClocks) {
				t.Fatalf("invalid calibration admitted: %v", err)
			}
		})
	}
	ts.Nanoseconds = 0
	ts.UncertaintyNS = 0
	if _, err := ts.Aligned(m); err != nil {
		t.Fatalf("explicit zero-start validity refused: %v", err)
	}
}

func TestSourceGenerationRestart(t *testing.T) {
	for _, field := range []string{"boot", "process"} {
		t.Run(field, func(t *testing.T) {
			a := NewTwoRankAttributor()
			obs := CollectiveObservation{CommunicatorHash: "0x1234567890abcdef", TotalRanks: 2, HostPID: 1, Hostname: "node", ProducerRevision: "revision", SourceBootEpoch: "boot-a", SourceProcessEpoch: "process-a"}
			if err := a.AddObservation(obs); err != nil {
				t.Fatal(err)
			}
			a.Forget(obs.CommunicatorHash, 0)
			obs.Sequence = 1
			if field == "boot" {
				obs.SourceBootEpoch = "boot-b"
			} else {
				obs.SourceProcessEpoch = "process-b"
			}
			if err := a.AddObservation(obs); !errors.Is(err, ErrRankRestarted) {
				t.Fatalf("same-PID generation reuse admitted: %v", err)
			}
		})
	}
}

func TestCrossHostPublicClockLabelsDoNotEstablishMapping(t *testing.T) {
	a := NewTwoRankAttributor()
	obs := CollectiveObservation{CommunicatorHash: "0x1234567890abcdef", TotalRanks: 2, HostPID: 1, Hostname: "node-a", Operation: "AllReduce", ClockEpoch: "shared", ClockEvidenceRef: "label", HasHostEnqueue: true, HostEnqueueStart: BoundedTimestamp{Nanoseconds: 10, Domain: ClockDomainHostMonotonic}, HostEnqueueEnd: BoundedTimestamp{Nanoseconds: 20, Domain: ClockDomainHostMonotonic}}
	if err := a.AddObservation(obs); err != nil {
		t.Fatal(err)
	}
	obs.Rank = 1
	obs.HostPID = 2
	obs.Hostname = "node-b"
	if err := a.AddObservation(obs); err != nil {
		t.Fatal(err)
	}
	report, err := a.Attribute(obs.CommunicatorHash, 0)
	if err != nil || report.Admitted || report.Diagnosis != DiagnosisIncompatibleClocks {
		t.Fatalf("public labels admitted cross-host claim: %+v %v", report, err)
	}
}

func TestMappedLineageRejectsMutation(t *testing.T) {
	m := validStallClock()
	ts, err := (BoundedTimestamp{Nanoseconds: 100, Domain: m.Domain, Epoch: m.SourceProcessEpoch, BootEpoch: m.SourceBootEpoch}).Aligned(m)
	if err != nil {
		t.Fatal(err)
	}
	obs := CollectiveObservation{SourceBootEpoch: m.SourceBootEpoch, SourceProcessEpoch: m.SourceProcessEpoch, ClockEpoch: m.ReferenceEpoch, ClockEvidenceRef: m.EvidenceRef, mapping: &m, HasHostEnqueue: true, HostEnqueueStart: ts, HostEnqueueEnd: ts}
	if !validObservationMapping(obs) {
		t.Fatal("valid lineage refused")
	}
	for _, change := range []func(*CollectiveObservation){
		func(o *CollectiveObservation) { o.HostEnqueueEnd.Nanoseconds = 1201 },
		func(o *CollectiveObservation) { o.HostEnqueueStart.Epoch = "changed" },
		func(o *CollectiveObservation) { o.SourceProcessEpoch = "restarted" },
		func(o *CollectiveObservation) { o.ClockEpoch = "changed" },
	} {
		mutated := obs
		change(&mutated)
		if validObservationMapping(mutated) {
			t.Fatal("mutated lineage admitted")
		}
	}
}

func TestUnknownHostDoesNotEstablishSharedClock(t *testing.T) {
	for _, hosts := range [][2]string{{"", ""}, {"node", ""}, {"", "node"}} {
		a := NewTwoRankAttributor()
		for rank, host := range hosts {
			obs := CollectiveObservation{CommunicatorHash: "0x1234567890abcdef", TotalRanks: 2, Rank: rank, HostPID: rank + 1, Hostname: host, Operation: "AllReduce", HasHostEnqueue: true, HostEnqueueStart: BoundedTimestamp{Nanoseconds: int64(10 + rank*10), Domain: ClockDomainHostMonotonic}, HostEnqueueEnd: BoundedTimestamp{Nanoseconds: int64(15 + rank*10), Domain: ClockDomainHostMonotonic}}
			if err := a.AddObservation(obs); err != nil {
				t.Fatal(err)
			}
		}
		result, err := a.Attribute("0x1234567890abcdef", 0)
		if err != nil || result.Admitted || result.Diagnosis != DiagnosisIncompatibleClocks {
			t.Fatalf("hosts %q result %+v err %v", hosts, result, err)
		}
	}
}
